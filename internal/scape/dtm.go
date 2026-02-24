package scape

import (
	"context"
	"fmt"
	"hash/fnv"
	"strings"

	protoio "protogonos/internal/io"
)

// DTMScape mirrors the delayed T-maze benchmark behavior from scape.erl.
type DTMScape struct{}

func (DTMScape) Name() string {
	return "dtm"
}

func (DTMScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return DTMScape{}.EvaluateMode(ctx, agent, "gt")
}

func (DTMScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := dtmConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateDTMWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateDTMWithStep(ctx, runner, cfg)
}

type dtmCoord struct {
	x int
	y int
}

type dtmView struct {
	next       dtmCoord
	hasNext    bool
	rangeLeft  float64
	rangeFwd   float64
	rangeRight float64
}

type dtmSector struct {
	reward float64
	views  map[int]dtmView
}

type dtmEpisode struct {
	sectors     map[dtmCoord]dtmSector
	position    dtmCoord
	direction   int
	totalRuns   int
	runIndex    int
	stepIndex   int
	switchEvent int
	switched    bool
	fitnessAcc  float64
}

type dtmModeConfig struct {
	mode           string
	totalRuns      int
	maxStepsPerRun int
	switchFloor    int
	switchSpread   int
}

func dtmConfigForMode(mode string) (dtmModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return dtmModeConfig{
			mode:           "gt",
			totalRuns:      100,
			maxStepsPerRun: 0,
			switchFloor:    35,
			switchSpread:   30,
		}, nil
	case "validation":
		return dtmModeConfig{
			mode:           "validation",
			totalRuns:      72,
			maxStepsPerRun: 0,
			switchFloor:    18,
			switchSpread:   20,
		}, nil
	case "test":
		return dtmModeConfig{
			mode:           "test",
			totalRuns:      72,
			maxStepsPerRun: 0,
			switchFloor:    42,
			switchSpread:   20,
		}, nil
	case "benchmark":
		return dtmModeConfig{
			mode:           "benchmark",
			totalRuns:      72,
			maxStepsPerRun: 0,
			switchFloor:    42,
			switchSpread:   20,
		}, nil
	default:
		return dtmModeConfig{}, fmt.Errorf("unsupported dtm mode: %s", mode)
	}
}

func evaluateDTMWithStep(ctx context.Context, runner StepAgent, cfg dtmModeConfig) (Fitness, Trace, error) {
	return evaluateDTM(
		ctx,
		runner.ID(),
		cfg,
		func(ctx context.Context, sense []float64) (float64, error) {
			out, err := runner.RunStep(ctx, sense)
			if err != nil {
				return 0, err
			}
			if len(out) != 1 {
				return 0, fmt.Errorf("dtm requires one output, got %d", len(out))
			}
			return out[0], nil
		},
	)
}

func evaluateDTMWithTick(ctx context.Context, ticker TickAgent, cfg dtmModeConfig) (Fitness, Trace, error) {
	ioBindings, err := dtmIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateDTM(
		ctx,
		ticker.ID(),
		cfg,
		func(ctx context.Context, sense []float64) (float64, error) {
			if ioBindings.leftSetter != nil {
				ioBindings.leftSetter.Set(sense[0])
			}
			if ioBindings.frontSetter != nil {
				ioBindings.frontSetter.Set(sense[1])
			}
			if ioBindings.rightSetter != nil {
				ioBindings.rightSetter.Set(sense[2])
			}
			if ioBindings.rewardSetter != nil {
				ioBindings.rewardSetter.Set(sense[3])
			}

			out, err := ticker.Tick(ctx)
			if err != nil {
				return 0, err
			}

			move := 0.0
			last := ioBindings.moveOutput.Last()
			if len(last) > 0 {
				move = last[0]
			} else if len(out) > 0 {
				move = out[0]
			}
			return move, nil
		},
	)
}

func evaluateDTM(
	ctx context.Context,
	agentID string,
	cfg dtmModeConfig,
	chooseMove func(context.Context, []float64) (float64, error),
) (Fitness, Trace, error) {
	episode := newDTMEpisode(agentID, cfg)
	terminalRuns := 0
	crashRuns := 0
	timeoutRuns := 0
	steps := 0
	runStepsAcc := 0
	maxRunStepIndex := 0
	terminalRewardTotal := 0.0
	leftTerminalRuns := 0
	rightTerminalRuns := 0
	switchTriggeredAt := -1

	for episode.runIndex < episode.totalRuns {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		if episode.runIndex == episode.switchEvent && !episode.switched {
			episode.swapRewardSectors()
			episode.switched = true
			switchTriggeredAt = episode.runIndex
		}

		runSteps := 0
		for {
			if err := ctx.Err(); err != nil {
				return 0, nil, err
			}
			if cfg.maxStepsPerRun > 0 && runSteps >= cfg.maxStepsPerRun {
				episode.fitnessAcc -= 0.4
				episode.resetRun()
				episode.runIndex++
				crashRuns++
				timeoutRuns++
				runStepsAcc += runSteps
				break
			}

			sense, err := episode.sense()
			if err != nil {
				return 0, nil, err
			}

			move, err := chooseMove(ctx, sense)
			if err != nil {
				return 0, nil, err
			}

			done, crashed, reachedTerminal, reward, terminalPosition, err := episode.applyMove(move)
			if err != nil {
				return 0, nil, err
			}

			steps++
			runSteps++
			if runSteps > maxRunStepIndex {
				maxRunStepIndex = runSteps
			}
			if done {
				runStepsAcc += runSteps
				if crashed {
					crashRuns++
				}
				if reachedTerminal {
					terminalRuns++
					terminalRewardTotal += reward
					switch terminalPosition {
					case (dtmCoord{x: -1, y: 1}):
						leftTerminalRuns++
					case (dtmCoord{x: 1, y: 1}):
						rightTerminalRuns++
					}
				}
				break
			}
		}
	}

	avgStepsPerRun := 0.0
	if episode.totalRuns > 0 {
		avgStepsPerRun = float64(runStepsAcc) / float64(episode.totalRuns)
	}
	fitnessStart := 50.0
	fitnessDelta := episode.fitnessAcc - fitnessStart

	return Fitness(episode.fitnessAcc), Trace{
		"fitness_acc":            episode.fitnessAcc,
		"fitness_start":          fitnessStart,
		"fitness_delta":          fitnessDelta,
		"total_runs":             episode.totalRuns,
		"switch_event":           episode.switchEvent,
		"switch_triggered_at":    switchTriggeredAt,
		"switched":               episode.switched,
		"terminal_runs":          terminalRuns,
		"left_terminal_runs":     leftTerminalRuns,
		"right_terminal_runs":    rightTerminalRuns,
		"terminal_reward_total":  terminalRewardTotal,
		"crash_runs":             crashRuns,
		"timeout_runs":           timeoutRuns,
		"steps_executed":         steps,
		"max_run_step_index":     maxRunStepIndex,
		"avg_steps_per_run":      avgStepsPerRun,
		"mode":                   cfg.mode,
		"max_steps":              cfg.maxStepsPerRun,
		"last_run_index":         episode.runIndex,
		"last_step_index":        episode.stepIndex,
		"switch_spread_floor":    cfg.switchFloor,
		"switch_spread_interval": cfg.switchSpread,
	}, nil
}

func newDTMEpisode(agentID string, cfg dtmModeConfig) dtmEpisode {
	totalRuns := cfg.totalRuns
	if totalRuns <= 0 {
		totalRuns = 1
	}
	return dtmEpisode{
		sectors:     buildTMazeSectors(),
		position:    dtmCoord{x: 0, y: 0},
		direction:   90,
		totalRuns:   totalRuns,
		runIndex:    0,
		stepIndex:   0,
		switchEvent: deterministicDTMSwitch(agentID, totalRuns, cfg.switchFloor, cfg.switchSpread),
		fitnessAcc:  50,
	}
}

func deterministicDTMSwitch(agentID string, totalRuns, floor, spread int) int {
	if totalRuns <= 1 {
		return 0
	}
	if floor < 0 {
		floor = 0
	}
	if spread <= 0 {
		spread = 1
	}

	h := fnv.New32a()
	_, _ = h.Write([]byte(agentID))
	// Reference-style switch window approximation:
	// floor + uniform(spread), where uniform is 1..spread.
	switchEvent := floor + int(h.Sum32()%uint32(spread)) + 1
	if switchEvent >= totalRuns {
		return totalRuns - 1
	}
	return switchEvent
}

func buildTMazeSectors() map[dtmCoord]dtmSector {
	return map[dtmCoord]dtmSector{
		{x: 0, y: 0}: {
			reward: 0,
			views: map[int]dtmView{
				0:   {hasNext: false, rangeLeft: 1, rangeFwd: 0, rangeRight: 0},
				90:  {hasNext: true, next: dtmCoord{x: 0, y: 1}, rangeLeft: 0, rangeFwd: 1, rangeRight: 0},
				180: {hasNext: false, rangeLeft: 0, rangeFwd: 0, rangeRight: 1},
				270: {hasNext: false, rangeLeft: 0, rangeFwd: 0, rangeRight: 0},
			},
		},
		{x: 0, y: 1}: {
			reward: 0,
			views: map[int]dtmView{
				0:   {hasNext: true, next: dtmCoord{x: 1, y: 1}, rangeLeft: 0, rangeFwd: 1, rangeRight: 1},
				90:  {hasNext: false, rangeLeft: 1, rangeFwd: 0, rangeRight: 1},
				180: {hasNext: true, next: dtmCoord{x: -1, y: 1}, rangeLeft: 1, rangeFwd: 1, rangeRight: 0},
				270: {hasNext: true, next: dtmCoord{x: 0, y: 0}, rangeLeft: 1, rangeFwd: 1, rangeRight: 1},
			},
		},
		{x: 1, y: 1}: {
			reward: 1,
			views: map[int]dtmView{
				0:   {hasNext: false, rangeLeft: 0, rangeFwd: 0, rangeRight: 0},
				90:  {hasNext: false, rangeLeft: 2, rangeFwd: 0, rangeRight: 0},
				180: {hasNext: true, next: dtmCoord{x: 0, y: 1}, rangeLeft: 0, rangeFwd: 2, rangeRight: 0},
				270: {hasNext: false, rangeLeft: 0, rangeFwd: 0, rangeRight: 2},
			},
		},
		{x: -1, y: 1}: {
			reward: 0.2,
			views: map[int]dtmView{
				0:   {hasNext: true, next: dtmCoord{x: 0, y: 1}, rangeLeft: 0, rangeFwd: 2, rangeRight: 0},
				90:  {hasNext: false, rangeLeft: 0, rangeFwd: 0, rangeRight: 2},
				180: {hasNext: false, rangeLeft: 0, rangeFwd: 0, rangeRight: 0},
				270: {hasNext: false, rangeLeft: 2, rangeFwd: 0, rangeRight: 0},
			},
		},
	}
}

func (e *dtmEpisode) swapRewardSectors() {
	left := dtmCoord{x: 1, y: 1}
	right := dtmCoord{x: -1, y: 1}
	ls := e.sectors[left]
	rs := e.sectors[right]
	ls.reward, rs.reward = rs.reward, ls.reward
	e.sectors[left] = ls
	e.sectors[right] = rs
}

func (e *dtmEpisode) sense() ([]float64, error) {
	sector, ok := e.sectors[e.position]
	if !ok {
		return nil, fmt.Errorf("dtm missing sector at position=%+v", e.position)
	}
	view, ok := sector.views[e.direction]
	if !ok {
		return nil, fmt.Errorf("dtm missing view for direction=%d at position=%+v", e.direction, e.position)
	}
	return []float64{view.rangeLeft, view.rangeFwd, view.rangeRight, sector.reward}, nil
}

func (e *dtmEpisode) applyMove(move float64) (done bool, crashed bool, reachedTerminal bool, reward float64, terminalPosition dtmCoord, err error) {
	e.stepIndex++

	if e.position == (dtmCoord{x: 1, y: 1}) || e.position == (dtmCoord{x: -1, y: 1}) {
		sector := e.sectors[e.position]
		reward = sector.reward
		terminalPosition = e.position
		e.fitnessAcc += reward
		e.resetRun()
		e.runIndex++
		return true, false, true, reward, terminalPosition, nil
	}

	sector, ok := e.sectors[e.position]
	if !ok {
		return false, false, false, 0, dtmCoord{}, fmt.Errorf("dtm missing sector at position=%+v", e.position)
	}

	if move > 0.33 {
		e.direction = (e.direction + 270) % 360
		view, ok := sector.views[e.direction]
		if !ok {
			return false, false, false, 0, dtmCoord{}, fmt.Errorf("dtm missing clockwise view for direction=%d at position=%+v", e.direction, e.position)
		}
		return e.applyTransition(view)
	}
	if move < -0.33 {
		e.direction = (e.direction + 90) % 360
		view, ok := sector.views[e.direction]
		if !ok {
			return false, false, false, 0, dtmCoord{}, fmt.Errorf("dtm missing counterclockwise view for direction=%d at position=%+v", e.direction, e.position)
		}
		return e.applyTransition(view)
	}

	view, ok := sector.views[e.direction]
	if !ok {
		return false, false, false, 0, dtmCoord{}, fmt.Errorf("dtm missing forward view for direction=%d at position=%+v", e.direction, e.position)
	}
	return e.applyTransition(view)
}

func (e *dtmEpisode) applyTransition(view dtmView) (done bool, crashed bool, reachedTerminal bool, reward float64, terminalPosition dtmCoord, err error) {
	if !view.hasNext {
		e.fitnessAcc -= 0.4
		e.resetRun()
		e.runIndex++
		return true, true, false, 0, dtmCoord{}, nil
	}
	e.position = view.next
	return false, false, false, 0, dtmCoord{}, nil
}

func (e *dtmEpisode) resetRun() {
	e.position = dtmCoord{x: 0, y: 0}
	e.direction = 90
	e.stepIndex = 0
}

type dtmIOBindings struct {
	leftSetter   protoio.ScalarSensorSetter
	frontSetter  protoio.ScalarSensorSetter
	rightSetter  protoio.ScalarSensorSetter
	rewardSetter protoio.ScalarSensorSetter
	moveOutput   protoio.SnapshotActuator
}

func dtmIO(agent TickAgent) (dtmIOBindings, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return dtmIOBindings{}, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	leftSetter, hasLeft, err := resolveOptionalDTMSetter(typed, protoio.DTMRangeLeftSensorName)
	if err != nil {
		return dtmIOBindings{}, err
	}

	frontSetter, hasFront, err := resolveOptionalDTMSetter(typed, protoio.DTMRangeFrontSensorName)
	if err != nil {
		return dtmIOBindings{}, err
	}

	rightSetter, hasRight, err := resolveOptionalDTMSetter(typed, protoio.DTMRangeRightSensorName)
	if err != nil {
		return dtmIOBindings{}, err
	}

	rangeCount := 0
	if hasLeft {
		rangeCount++
	}
	if hasFront {
		rangeCount++
	}
	if hasRight {
		rangeCount++
	}

	if rangeCount != 0 && rangeCount != 3 {
		return dtmIOBindings{}, fmt.Errorf("agent %s has partial dtm range sensor set (%d/3); expected all or none", agent.ID(), rangeCount)
	}

	rewardSetter, hasReward, err := resolveOptionalDTMSetter(typed, protoio.DTMRewardSensorName)
	if err != nil {
		return dtmIOBindings{}, err
	}
	if rangeCount == 0 && !hasReward {
		return dtmIOBindings{}, fmt.Errorf("agent %s missing dtm sensing surface: expected range_sense and/or reward sensor", agent.ID())
	}

	actuator, ok := typed.RegisteredActuator(protoio.DTMMoveActuatorName)
	if !ok {
		return dtmIOBindings{}, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.DTMMoveActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return dtmIOBindings{}, fmt.Errorf("actuator %s does not support output snapshot", protoio.DTMMoveActuatorName)
	}

	return dtmIOBindings{
		leftSetter:   leftSetter,
		frontSetter:  frontSetter,
		rightSetter:  rightSetter,
		rewardSetter: rewardSetter,
		moveOutput:   output,
	}, nil
}

func resolveOptionalDTMSetter(
	typed interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
	},
	sensorID string,
) (protoio.ScalarSensorSetter, bool, error) {
	sensor, ok := typed.RegisteredSensor(sensorID)
	if !ok {
		return nil, false, nil
	}
	setter, ok := sensor.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, false, fmt.Errorf("sensor %s does not support scalar set", sensorID)
	}
	return setter, true, nil
}
