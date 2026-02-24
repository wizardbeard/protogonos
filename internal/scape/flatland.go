package scape

import (
	"context"
	"fmt"
	"math"
	"strings"

	protoio "protogonos/internal/io"
)

type FlatlandScape struct{}

func (FlatlandScape) Name() string {
	return "flatland"
}

func (FlatlandScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return FlatlandScape{}.EvaluateMode(ctx, agent, "gt")
}

func (FlatlandScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := flatlandConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateFlatlandWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateFlatlandWithStep(ctx, runner, cfg)
}

type flatlandModeConfig struct {
	mode            string
	maxAge          int
	forageGoal      int
	foodPositions   []int
	poisonPositions []int
}

func flatlandConfigForMode(mode string) (flatlandModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return flatlandModeConfig{
			mode:            "gt",
			maxAge:          flatlandDefaultMaxAge,
			forageGoal:      flatlandDefaultForageGoal,
			foodPositions:   []int{5, 11, 19, 27, 35, 43},
			poisonPositions: []int{14, 30},
		}, nil
	case "validation":
		return flatlandModeConfig{
			mode:            "validation",
			maxAge:          180,
			forageGoal:      6,
			foodPositions:   []int{4, 10, 18, 26, 34, 42},
			poisonPositions: []int{13, 29},
		}, nil
	case "test":
		return flatlandModeConfig{
			mode:            "test",
			maxAge:          180,
			forageGoal:      6,
			foodPositions:   []int{6, 12, 20, 28, 36, 44},
			poisonPositions: []int{15, 31},
		}, nil
	case "benchmark":
		return flatlandModeConfig{
			mode:            "benchmark",
			maxAge:          180,
			forageGoal:      6,
			foodPositions:   []int{6, 12, 20, 28, 36, 44},
			poisonPositions: []int{15, 31},
		}, nil
	default:
		return flatlandModeConfig{}, fmt.Errorf("unsupported flatland mode: %s", mode)
	}
}

func evaluateFlatlandWithStep(ctx context.Context, runner StepAgent, cfg flatlandModeConfig) (Fitness, Trace, error) {
	return evaluateFlatland(ctx, cfg, func(ctx context.Context, distance, energy float64) (float64, error) {
		out, err := runner.RunStep(ctx, []float64{distance, energy})
		if err != nil {
			return 0, err
		}
		if len(out) != 1 {
			return 0, fmt.Errorf("flatland requires one output, got %d", len(out))
		}
		return out[0], nil
	})
}

func evaluateFlatlandWithTick(ctx context.Context, ticker TickAgent, cfg flatlandModeConfig) (Fitness, Trace, error) {
	distanceSetter, energySetter, moveOutput, err := flatlandIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateFlatland(ctx, cfg, func(ctx context.Context, distance, energy float64) (float64, error) {
		distanceSetter.Set(distance)
		energySetter.Set(energy)
		out, err := ticker.Tick(ctx)
		if err != nil {
			return 0, err
		}
		last := moveOutput.Last()
		if len(last) > 0 {
			return last[0], nil
		}
		if len(out) > 0 {
			return out[0], nil
		}
		return 0, nil
	})
}

func evaluateFlatland(
	ctx context.Context,
	cfg flatlandModeConfig,
	chooseMove func(context.Context, float64, float64) (float64, error),
) (Fitness, Trace, error) {
	episode := newFlatlandEpisode(cfg)
	movementSteps := 0
	foodCollisions := 0
	poisonCollisions := 0
	lastDistance := 0.0
	terminalReason := "age_limit"

	for episode.age < cfg.maxAge && episode.energy > 0 {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		episode.advanceRespawns()
		distance := episode.senseDistanceToFood()
		lastDistance = distance
		energySignal := episode.normalizedEnergy()

		move, err := chooseMove(ctx, distance, energySignal)
		if err != nil {
			return 0, nil, err
		}

		moveStep, hitFood, hitPoison, reason := episode.step(move)
		if moveStep != 0 {
			movementSteps++
		}
		if hitFood {
			foodCollisions++
		}
		if hitPoison {
			poisonCollisions++
		}
		if reason != "" {
			terminalReason = reason
			break
		}
	}

	if episode.energy <= 0 {
		terminalReason = "depleted"
	}
	if episode.foodCollected >= episode.forageGoal {
		terminalReason = "forage_goal"
	}

	age := episode.age
	if age <= 0 {
		age = 1
	}

	totalCollisions := foodCollisions + poisonCollisions
	avgReward := episode.rewardAcc / float64(age)
	survival := float64(episode.age) / float64(cfg.maxAge)
	energyTerm := clamp(episode.normalizedEnergy(), 0, 1)
	forageDenom := float64(totalCollisions + 2)
	forageBalance := float64(episode.foodCollected-episode.poisonHits) / forageDenom
	forageTerm := 0.5 + 0.5*clamp(forageBalance, -1, 1)
	rewardTerm := 0.5 + 0.5*clamp(avgReward, -1, 1)

	fitness := 0.35*survival + 0.25*energyTerm + 0.25*forageTerm + 0.15*rewardTerm
	if episode.foodCollected >= episode.forageGoal {
		fitness += 0.1
	}
	fitness = clamp(fitness, 0, 1.4)

	return Fitness(fitness), Trace{
		"position":           float64(episode.position),
		"energy":             episode.energy,
		"energy_norm":        episode.normalizedEnergy(),
		"reward":             avgReward,
		"reward_total":       episode.rewardAcc,
		"age":                episode.age,
		"max_age":            cfg.maxAge,
		"forage_goal":        episode.forageGoal,
		"food_collected":     episode.foodCollected,
		"poison_hits":        episode.poisonHits,
		"collisions":         totalCollisions,
		"movement_steps":     movementSteps,
		"terminal_reason":    terminalReason,
		"last_food_distance": lastDistance,
		"mode":               cfg.mode,
	}, nil
}

const (
	flatlandWorldSize         = 48
	flatlandDefaultMaxAge     = 220
	flatlandInitialEnergy     = 1.2
	flatlandEnergyCap         = 2.0
	flatlandBaseMetabolic     = 0.012
	flatlandMoveMetabolic     = 0.018
	flatlandFoodEnergy        = 0.32
	flatlandPoisonDamage      = 0.38
	flatlandSurvivalReward    = 0.01
	flatlandFoodReward        = 0.25
	flatlandPoisonPenalty     = 0.30
	flatlandFoodRespawn       = 12
	flatlandPoisonRespawn     = 16
	flatlandDefaultForageGoal = 8
	flatlandSensorDistanceLo  = -1.0
	flatlandSensorDistanceHi  = 1.0
)

type flatlandEpisode struct {
	position      int
	energy        float64
	age           int
	maxAge        int
	forageGoal    int
	foodCollected int
	poisonHits    int
	rewardAcc     float64
	food          map[int]int
	poison        map[int]int
}

func newFlatlandEpisode(cfg flatlandModeConfig) *flatlandEpisode {
	maxAge := cfg.maxAge
	if maxAge <= 0 {
		maxAge = flatlandDefaultMaxAge
	}
	forageGoal := cfg.forageGoal
	if forageGoal <= 0 {
		forageGoal = flatlandDefaultForageGoal
	}

	foodPositions := cfg.foodPositions
	if len(foodPositions) == 0 {
		foodPositions = []int{5, 11, 19, 27, 35, 43}
	}
	poisonPositions := cfg.poisonPositions
	if len(poisonPositions) == 0 {
		poisonPositions = []int{14, 30}
	}

	food := make(map[int]int, len(foodPositions))
	for _, position := range foodPositions {
		food[wrapFlatlandPosition(position)] = 0
	}
	poison := make(map[int]int, len(poisonPositions))
	for _, position := range poisonPositions {
		poison[wrapFlatlandPosition(position)] = 0
	}

	return &flatlandEpisode{
		position:   0,
		energy:     flatlandInitialEnergy,
		maxAge:     maxAge,
		forageGoal: forageGoal,
		food:       food,
		poison:     poison,
	}
}

func (e *flatlandEpisode) advanceRespawns() {
	for position, cooldown := range e.food {
		if cooldown > 0 {
			e.food[position] = cooldown - 1
		}
	}
	for position, cooldown := range e.poison {
		if cooldown > 0 {
			e.poison[position] = cooldown - 1
		}
	}
}

func (e *flatlandEpisode) normalizedEnergy() float64 {
	if flatlandInitialEnergy <= 0 {
		return 0
	}
	return clamp(e.energy/flatlandInitialEnergy, 0, flatlandEnergyCap/flatlandInitialEnergy)
}

func (e *flatlandEpisode) senseDistanceToFood() float64 {
	half := flatlandWorldSize / 2
	if half <= 0 {
		return 0
	}

	bestDelta := 0
	bestDistance := flatlandWorldSize + 1
	for targetPosition, cooldown := range e.food {
		if cooldown > 0 {
			continue
		}
		delta := signedRingDistance(e.position, targetPosition, flatlandWorldSize)
		distance := absInt(delta)
		if distance < bestDistance {
			bestDistance = distance
			bestDelta = delta
		}
	}
	if bestDistance > flatlandWorldSize {
		return 0
	}
	return clamp(float64(bestDelta)/float64(half), flatlandSensorDistanceLo, flatlandSensorDistanceHi)
}

func (e *flatlandEpisode) step(move float64) (int, bool, bool, string) {
	move = clamp(move, -1, 1)
	moveStep := 0
	if move > 0.33 {
		moveStep = 1
	} else if move < -0.33 {
		moveStep = -1
	}

	if moveStep != 0 {
		e.position = wrapFlatlandPosition(e.position + moveStep)
	}

	hitFood := false
	if cooldown, ok := e.food[e.position]; ok && cooldown == 0 {
		hitFood = true
		e.food[e.position] = flatlandFoodRespawn
		e.energy += flatlandFoodEnergy
		if e.energy > flatlandEnergyCap {
			e.energy = flatlandEnergyCap
		}
		e.foodCollected++
		e.rewardAcc += flatlandFoodReward
	}

	hitPoison := false
	if cooldown, ok := e.poison[e.position]; ok && cooldown == 0 {
		hitPoison = true
		e.poison[e.position] = flatlandPoisonRespawn
		e.energy -= flatlandPoisonDamage
		e.poisonHits++
		e.rewardAcc -= flatlandPoisonPenalty
	}

	e.energy -= flatlandBaseMetabolic + flatlandMoveMetabolic*math.Abs(float64(moveStep))
	if e.energy < 0 {
		e.energy = 0
	}
	e.rewardAcc += flatlandSurvivalReward
	e.age++

	if e.energy <= 0 {
		return moveStep, hitFood, hitPoison, "depleted"
	}
	if e.foodCollected >= e.forageGoal {
		return moveStep, hitFood, hitPoison, "forage_goal"
	}
	if e.age >= e.maxAge {
		return moveStep, hitFood, hitPoison, "age_limit"
	}
	return moveStep, hitFood, hitPoison, ""
}

func wrapFlatlandPosition(position int) int {
	wrapped := position % flatlandWorldSize
	if wrapped < 0 {
		wrapped += flatlandWorldSize
	}
	return wrapped
}

func signedRingDistance(from, to, size int) int {
	if size <= 0 {
		return 0
	}
	delta := (to - from) % size
	half := size / 2
	if delta > half {
		delta -= size
	}
	if delta < -half {
		delta += size
	}
	return delta
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

func flatlandIO(agent TickAgent) (
	protoio.ScalarSensorSetter,
	protoio.ScalarSensorSetter,
	protoio.SnapshotActuator,
	error,
) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	distance, ok := typed.RegisteredSensor(protoio.FlatlandDistanceSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.FlatlandDistanceSensorName)
	}
	distanceSetter, ok := distance.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.FlatlandDistanceSensorName)
	}

	energy, ok := typed.RegisteredSensor(protoio.FlatlandEnergySensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.FlatlandEnergySensorName)
	}
	energySetter, ok := energy.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.FlatlandEnergySensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.FlatlandMoveActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.FlatlandMoveActuatorName)
	}
	moveOutput, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.FlatlandMoveActuatorName)
	}
	return distanceSetter, energySetter, moveOutput, nil
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
