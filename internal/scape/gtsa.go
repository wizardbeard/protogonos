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
	index := cfg.startIndex
	current := gtsaSeries(index)

	for i := 0; i < cfg.warmupSteps; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}
		if _, err := predict(ctx, current); err != nil {
			return 0, nil, err
		}
		index++
		current = gtsaSeries(index)
	}

	if cfg.scoreSteps <= 0 {
		return 0, Trace{
			"mse":                0.0,
			"mae":                0.0,
			"direction_accuracy": 0.0,
			"prediction_jitter":  0.0,
			"mode":               cfg.mode,
			"start_index":        cfg.startIndex,
			"start_t":            float64(cfg.startIndex),
			"warmup_steps":       cfg.warmupSteps,
			"steps":              0,
			"window":             cfg.warmupSteps,
		}, nil
	}

	squaredErr := 0.0
	absErr := 0.0
	directionalCorrect := 0
	predictionJitter := 0.0
	prevPrediction := 0.0
	hasPrevPrediction := false

	for i := 0; i < cfg.scoreSteps; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		predicted, err := predict(ctx, current)
		if err != nil {
			return 0, nil, err
		}
		next := gtsaSeries(index + 1)
		delta := predicted - next
		squaredErr += delta * delta
		absErr += math.Abs(delta)

		if gtsaDirectionalMatch(current, predicted, next) {
			directionalCorrect++
		}
		if hasPrevPrediction {
			predictionJitter += math.Abs(predicted - prevPrediction)
		}
		prevPrediction = predicted
		hasPrevPrediction = true

		index++
		current = next
	}

	mse := squaredErr / float64(cfg.scoreSteps)
	mae := absErr / float64(cfg.scoreSteps)
	directionAccuracy := float64(directionalCorrect) / float64(cfg.scoreSteps)
	avgJitter := 0.0
	if cfg.scoreSteps > 1 {
		avgJitter = predictionJitter / float64(cfg.scoreSteps-1)
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
		"start_index":        cfg.startIndex,
		"start_t":            float64(cfg.startIndex),
		"warmup_steps":       cfg.warmupSteps,
		"steps":              cfg.scoreSteps,
		"window":             cfg.warmupSteps + cfg.scoreSteps,
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
}

func gtsaConfigForMode(mode string) (gtsaModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return gtsaModeConfig{mode: "gt", startIndex: 0, warmupSteps: 8, scoreSteps: 40}, nil
	case "validation":
		return gtsaModeConfig{mode: "validation", startIndex: 320, warmupSteps: 8, scoreSteps: 32}, nil
	case "test":
		return gtsaModeConfig{mode: "test", startIndex: 640, warmupSteps: 8, scoreSteps: 32}, nil
	case "benchmark":
		return gtsaModeConfig{mode: "benchmark", startIndex: 640, warmupSteps: 8, scoreSteps: 32}, nil
	default:
		return gtsaModeConfig{}, fmt.Errorf("unsupported gtsa mode: %s", mode)
	}
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
