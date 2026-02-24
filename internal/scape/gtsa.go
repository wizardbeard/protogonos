package scape

import (
	"context"
	"fmt"
	"math"
	"strings"

	protoio "protogonos/internal/io"
)

type GTSAScape struct{}

func (GTSAScape) Name() string {
	return "gtsa"
}

func (GTSAScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return GTSAScape{}.EvaluateMode(ctx, agent, "gt")
}

func (GTSAScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := gtsaConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateGTSAWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateGTSAWithStep(ctx, runner, cfg)
}

func evaluateGTSAWithStep(ctx context.Context, runner StepAgent, cfg gtsaModeConfig) (Fitness, Trace, error) {
	return evaluateGTSA(ctx, cfg, func(ctx context.Context, current float64) (float64, error) {
		out, err := runner.RunStep(ctx, []float64{current})
		if err != nil {
			return 0, err
		}
		if len(out) != 1 {
			return 0, fmt.Errorf("gtsa requires one output, got %d", len(out))
		}
		return out[0], nil
	})
}

func evaluateGTSAWithTick(ctx context.Context, ticker TickAgent, cfg gtsaModeConfig) (Fitness, Trace, error) {
	inputSetter, predictOutput, err := gtsaIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateGTSA(ctx, cfg, func(ctx context.Context, current float64) (float64, error) {
		inputSetter.Set(current)
		out, err := ticker.Tick(ctx)
		if err != nil {
			return 0, err
		}
		lastOutput := predictOutput.Last()
		if len(lastOutput) > 0 {
			return lastOutput[0], nil
		}
		if len(out) > 0 {
			return out[0], nil
		}
		return 0, nil
	})
}

func evaluateGTSA(
	ctx context.Context,
	cfg gtsaModeConfig,
	predict func(context.Context, float64) (float64, error),
) (Fitness, Trace, error) {
	table := defaultGTSATable()
	state, err := newGTSAWindowState(cfg, table)
	if err != nil {
		return 0, nil, err
	}

	warmupSteps := 0
	for i := 0; i < cfg.warmupSteps; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		current, err := state.getPercept()
		if err != nil {
			return 0, nil, err
		}
		if _, err := predict(ctx, current); err != nil {
			return 0, nil, err
		}

		_, done, err := state.applyPrediction()
		if err != nil {
			return 0, nil, err
		}
		warmupSteps++
		if done {
			break
		}
	}

	if cfg.scoreSteps <= 0 {
		return 0, Trace{
			"mse":                0.0,
			"mae":                0.0,
			"direction_accuracy": 0.0,
			"prediction_jitter":  0.0,
			"mode":               cfg.mode,
			"start_index":        state.indexStart - 1,
			"start_t":            float64(state.indexStart - 1),
			"warmup_steps":       warmupSteps,
			"steps":              0,
			"window":             warmupSteps,
			"window_length":      state.windowLength,
			"window_rows":        state.totRows,
			"table_name":         state.info.name,
			"index_start":        state.indexStart,
			"index_current":      state.indexCurrent,
			"index_end":          state.indexEnd,
		}, nil
	}

	squaredErr := 0.0
	absErr := 0.0
	directionalCorrect := 0
	predictionJitter := 0.0
	prevPrediction := 0.0
	hasPrevPrediction := false
	scoredSteps := 0

	for i := 0; i < cfg.scoreSteps; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		current, err := state.getPercept()
		if err != nil {
			return 0, nil, err
		}

		predicted, err := predict(ctx, current)
		if err != nil {
			return 0, nil, err
		}

		expected, done, err := state.applyPrediction()
		if err != nil {
			return 0, nil, err
		}
		if done {
			break
		}

		delta := predicted - expected
		squaredErr += delta * delta
		absErr += math.Abs(delta)

		if gtsaDirectionalMatch(current, predicted, expected) {
			directionalCorrect++
		}
		if hasPrevPrediction {
			predictionJitter += math.Abs(predicted - prevPrediction)
		}
		prevPrediction = predicted
		hasPrevPrediction = true
		scoredSteps++
	}

	if scoredSteps <= 0 {
		return 0, Trace{
			"mse":                0.0,
			"mae":                0.0,
			"direction_accuracy": 0.0,
			"prediction_jitter":  0.0,
			"mode":               cfg.mode,
			"start_index":        state.indexStart - 1,
			"start_t":            float64(state.indexStart - 1),
			"warmup_steps":       warmupSteps,
			"steps":              0,
			"window":             warmupSteps,
			"window_length":      state.windowLength,
			"window_rows":        state.totRows,
			"table_name":         state.info.name,
			"index_start":        state.indexStart,
			"index_current":      state.indexCurrent,
			"index_end":          state.indexEnd,
		}, nil
	}

	mse := squaredErr / float64(scoredSteps)
	mae := absErr / float64(scoredSteps)
	directionAccuracy := float64(directionalCorrect) / float64(scoredSteps)
	avgJitter := 0.0
	if scoredSteps > 1 {
		avgJitter = predictionJitter / float64(scoredSteps-1)
	}

	base := 1.0 / (1.0 + mae + 0.5*mse)
	directionTerm := 0.75 + 0.25*directionAccuracy
	stabilityTerm := 1.0 / (1.0 + avgJitter)
	fitness := clampGTSA(base*directionTerm*stabilityTerm, 0, 1.5)

	return Fitness(fitness), Trace{
		"mse":                mse,
		"mae":                mae,
		"direction_accuracy": directionAccuracy,
		"prediction_jitter":  avgJitter,
		"mode":               cfg.mode,
		"start_index":        state.indexStart - 1,
		"start_t":            float64(state.indexStart - 1),
		"warmup_steps":       warmupSteps,
		"steps":              scoredSteps,
		"window":             warmupSteps + scoredSteps,
		"window_length":      state.windowLength,
		"window_rows":        state.totRows,
		"table_name":         state.info.name,
		"index_start":        state.indexStart,
		"index_current":      state.indexCurrent,
		"index_end":          state.indexEnd,
	}, nil
}

func gtsaDirectionalMatch(current, predicted, expectedNext float64) bool {
	predictedDelta := predicted - current
	expectedDelta := expectedNext - current
	if math.Abs(expectedDelta) < 1e-9 {
		return math.Abs(predictedDelta) < 0.05
	}
	return predictedDelta*expectedDelta > 0
}

func gtsaSeries(index int) float64 {
	t := float64(index)
	seasonal := math.Sin(t*0.17) + 0.45*math.Sin(t*0.043+0.6)
	trend := 0.0018 * t
	regime := 0.0
	switch {
	case index >= 180 && index < 360:
		regime = -0.24
	case index >= 360 && index < 540:
		regime = 0.18
	case index >= 540:
		regime = -0.1
	}
	shock := 0.0
	if index%97 == 0 {
		shock += 0.2
	}
	if index%131 == 0 {
		shock -= 0.15
	}
	return trend + seasonal + regime + shock
}

func gtsaIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	input, ok := typed.RegisteredSensor(protoio.GTSAInputSensorName)
	if !ok {
		return nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.GTSAInputSensorName)
	}
	inputSetter, ok := input.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.GTSAInputSensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.GTSAPredictActuatorName)
	if !ok {
		return nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.GTSAPredictActuatorName)
	}
	predictOutput, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.GTSAPredictActuatorName)
	}
	return inputSetter, predictOutput, nil
}

type gtsaModeConfig struct {
	mode        string
	startIndex  int
	warmupSteps int
	scoreSteps  int
	windowRows  int
}

func gtsaConfigForMode(mode string) (gtsaModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return gtsaModeConfig{mode: "gt", startIndex: 0, warmupSteps: 8, scoreSteps: 40, windowRows: 8}, nil
	case "validation":
		return gtsaModeConfig{mode: "validation", startIndex: 320, warmupSteps: 8, scoreSteps: 32, windowRows: 8}, nil
	case "test":
		return gtsaModeConfig{mode: "test", startIndex: 640, warmupSteps: 8, scoreSteps: 32, windowRows: 8}, nil
	case "benchmark":
		return gtsaModeConfig{mode: "benchmark", startIndex: 640, warmupSteps: 8, scoreSteps: 32, windowRows: 8}, nil
	default:
		return gtsaModeConfig{}, fmt.Errorf("unsupported gtsa mode: %s", mode)
	}
}

type gtsaInfo struct {
	name   string
	ivl    int
	ovl    int
	trnEnd int
	valEnd int
	tstEnd int
}

type gtsaTable struct {
	info   gtsaInfo
	values []float64
}

func defaultGTSATable() gtsaTable {
	info := gtsaInfo{
		name:   "gtsa.synthetic.v2",
		ivl:    1,
		ovl:    1,
		trnEnd: 320,
		valEnd: 640,
		tstEnd: 960,
	}

	values := make([]float64, info.tstEnd+1) // 1-based indexing for reference-style parity.
	for idx := 1; idx <= info.tstEnd; idx++ {
		values[idx] = gtsaSeries(idx - 1)
	}
	return gtsaTable{info: info, values: values}
}

type gtsaWindowState struct {
	info         gtsaInfo
	values       []float64
	indexStart   int
	indexCurrent int
	indexEnd     int
	windowLength int
	window       []float64
	totRows      int
}

func newGTSAWindowState(cfg gtsaModeConfig, table gtsaTable) (*gtsaWindowState, error) {
	if len(table.values) <= 1 {
		return nil, fmt.Errorf("gtsa table %s is empty", table.info.name)
	}

	start, end, err := gtsaBoundsForMode(cfg.mode, table.info)
	if err != nil {
		return nil, err
	}

	if cfg.startIndex >= 0 {
		desired := cfg.startIndex + 1
		if desired >= start && desired <= end {
			start = desired
		}
	}

	rows := cfg.windowRows
	if rows <= 0 {
		rows = 1
	}
	windowLength := rows * maxGTSA(1, table.info.ivl)

	return &gtsaWindowState{
		info:         table.info,
		values:       table.values,
		indexStart:   start,
		indexCurrent: start,
		indexEnd:     end,
		windowLength: windowLength,
		window:       make([]float64, 0, windowLength),
		totRows:      rows,
	}, nil
}

func gtsaBoundsForMode(mode string, info gtsaInfo) (start int, end int, err error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		start, end = 1, info.trnEnd
	case "validation":
		if info.valEnd > info.trnEnd {
			start, end = info.trnEnd+1, info.valEnd
		} else {
			start, end = 1, info.trnEnd
		}
	case "test", "benchmark":
		if info.tstEnd > info.valEnd && info.valEnd > 0 {
			start, end = info.valEnd+1, info.tstEnd
		} else if info.valEnd > info.trnEnd {
			start, end = info.trnEnd+1, info.valEnd
		} else {
			start, end = 1, info.trnEnd
		}
	default:
		return 0, 0, fmt.Errorf("unsupported gtsa mode: %s", mode)
	}
	if start <= 0 || end < start {
		return 0, 0, fmt.Errorf("invalid gtsa bounds for mode=%s start=%d end=%d", mode, start, end)
	}
	return start, end, nil
}

func (s *gtsaWindowState) getPercept() (float64, error) {
	if s.indexCurrent <= 0 || s.indexCurrent >= len(s.values) {
		return 0, fmt.Errorf("gtsa index out of range: %d", s.indexCurrent)
	}
	value := s.values[s.indexCurrent]
	s.window = append([]float64{value}, s.window...)
	if len(s.window) > s.windowLength {
		s.window = s.window[:s.windowLength]
	}
	return value, nil
}

func (s *gtsaWindowState) applyPrediction() (expected float64, done bool, err error) {
	next := s.indexCurrent + 1
	if next > s.indexEnd {
		return 0, true, nil
	}
	if next <= 0 || next >= len(s.values) {
		return 0, false, fmt.Errorf("gtsa next index out of range: %d", next)
	}
	expected = s.values[next]
	s.indexCurrent = next
	return expected, false, nil
}

func maxGTSA(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func clampGTSA(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
