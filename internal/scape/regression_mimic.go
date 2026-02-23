package scape

import (
	"context"
	"fmt"
	"strings"

	protoio "protogonos/internal/io"
)

// RegressionMimicScape evaluates a one-dimensional regression target y=x.
type RegressionMimicScape struct{}

func (RegressionMimicScape) Name() string {
	return "regression-mimic"
}

func (RegressionMimicScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return RegressionMimicScape{}.EvaluateMode(ctx, agent, "gt")
}

func (RegressionMimicScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := regressionConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateRegressionMimicWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateRegressionMimicWithStep(ctx, runner, cfg)
}

type regressionModeConfig struct {
	mode   string
	inputs []float64
}

func regressionConfigForMode(mode string) (regressionModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return regressionModeConfig{
			mode:   "gt",
			inputs: []float64{0.0, 0.25, 0.5, 0.75, 1.0},
		}, nil
	case "validation":
		return regressionModeConfig{
			mode:   "validation",
			inputs: []float64{-1.0, -0.5, 0.0, 0.5, 1.0},
		}, nil
	case "test":
		return regressionModeConfig{
			mode:   "test",
			inputs: []float64{-0.9, -0.45, -0.1, 0.3, 0.7, 0.95},
		}, nil
	default:
		return regressionModeConfig{}, fmt.Errorf("unsupported regression-mimic mode: %s", mode)
	}
}

func evaluateRegressionMimicWithStep(ctx context.Context, runner StepAgent, cfg regressionModeConfig) (Fitness, Trace, error) {
	return evaluateRegressionMimic(
		ctx,
		cfg,
		func(ctx context.Context, x float64) (float64, error) {
			out, err := runner.RunStep(ctx, []float64{x})
			if err != nil {
				return 0, err
			}
			if len(out) != 1 {
				return 0, fmt.Errorf("regression-mimic requires one output, got %d", len(out))
			}
			return out[0], nil
		},
	)
}

func evaluateRegressionMimicWithTick(ctx context.Context, ticker TickAgent, cfg regressionModeConfig) (Fitness, Trace, error) {
	setter, output, err := regressionMimicIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateRegressionMimic(
		ctx,
		cfg,
		func(ctx context.Context, x float64) (float64, error) {
			setter.Set(x)
			out, err := ticker.Tick(ctx)
			if err != nil {
				return 0, err
			}
			last := output.Last()
			if len(last) > 0 {
				return last[0], nil
			}
			if len(out) > 0 {
				return out[0], nil
			}
			return 0, fmt.Errorf("regression-mimic requires one output, got 0")
		},
	)
}

func evaluateRegressionMimic(
	ctx context.Context,
	cfg regressionModeConfig,
	predict func(context.Context, float64) (float64, error),
) (Fitness, Trace, error) {
	predictions := make([]float64, 0, len(cfg.inputs))
	var squaredErr float64
	for _, x := range cfg.inputs {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}
		predicted, err := predict(ctx, x)
		if err != nil {
			return 0, nil, err
		}
		predictions = append(predictions, predicted)
		delta := predicted - x
		squaredErr += delta * delta
	}

	if len(cfg.inputs) == 0 {
		return 0, Trace{"mse": 0.0, "predictions": predictions, "mode": cfg.mode, "samples": 0}, nil
	}
	mse := squaredErr / float64(len(cfg.inputs))
	fitness := Fitness(1.0 - mse)
	return fitness, Trace{
		"mse":         mse,
		"predictions": predictions,
		"mode":        cfg.mode,
		"samples":     len(cfg.inputs),
	}, nil
}

func regressionMimicIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	sensor, ok := typed.RegisteredSensor(protoio.ScalarInputSensorName)
	if !ok {
		return nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.ScalarInputSensorName)
	}
	setter, ok := sensor.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.ScalarInputSensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.ScalarOutputActuatorName)
	if !ok {
		return nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.ScalarOutputActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.ScalarOutputActuatorName)
	}
	return setter, output, nil
}
