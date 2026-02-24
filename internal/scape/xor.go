package scape

import (
	"context"
	"fmt"
	"strings"

	protoio "protogonos/internal/io"
)

type StepAgent interface {
	Agent
	RunStep(ctx context.Context, input []float64) ([]float64, error)
}

type XORScape struct{}

func (XORScape) Name() string {
	return "xor"
}

func (XORScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return XORScape{}.EvaluateMode(ctx, agent, "gt")
}

func (XORScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := xorConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateXORWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateXORWithStep(ctx, runner, cfg)
}

type xorCase struct {
	in   []float64
	want float64
}

type xorModeConfig struct {
	mode  string
	cases []xorCase
}

func xorConfigForMode(mode string) (xorModeConfig, error) {
	base := []xorCase{
		{in: []float64{0, 0}, want: 0},
		{in: []float64{0, 1}, want: 1},
		{in: []float64{1, 0}, want: 1},
		{in: []float64{1, 1}, want: 0},
	}

	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return xorModeConfig{mode: "gt", cases: base}, nil
	case "validation":
		return xorModeConfig{
			mode: "validation",
			cases: []xorCase{
				base[1], base[2], base[0], base[3], base[1], base[2],
			},
		}, nil
	case "test":
		return xorModeConfig{
			mode: "test",
			cases: []xorCase{
				base[3], base[2], base[1], base[0], base[3], base[0], base[2], base[1],
			},
		}, nil
	case "benchmark":
		return xorModeConfig{
			mode: "benchmark",
			cases: []xorCase{
				base[3], base[2], base[1], base[0], base[3], base[0], base[2], base[1],
			},
		}, nil
	default:
		return xorModeConfig{}, fmt.Errorf("unsupported xor mode: %s", mode)
	}
}

func evaluateXORWithStep(ctx context.Context, runner StepAgent, cfg xorModeConfig) (Fitness, Trace, error) {
	return evaluateXOR(
		ctx,
		cfg,
		func(ctx context.Context, in []float64) (float64, error) {
			out, err := runner.RunStep(ctx, in)
			if err != nil {
				return 0, err
			}
			if len(out) != 1 {
				return 0, fmt.Errorf("xor requires one output, got %d", len(out))
			}
			return out[0], nil
		},
	)
}

func evaluateXORWithTick(ctx context.Context, ticker TickAgent, cfg xorModeConfig) (Fitness, Trace, error) {
	leftSetter, rightSetter, output, err := xorIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateXOR(
		ctx,
		cfg,
		func(ctx context.Context, in []float64) (float64, error) {
			leftSetter.Set(in[0])
			rightSetter.Set(in[1])

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
			return 0, fmt.Errorf("xor requires one output, got 0")
		},
	)
}

func evaluateXOR(
	ctx context.Context,
	cfg xorModeConfig,
	predict func(context.Context, []float64) (float64, error),
) (Fitness, Trace, error) {
	var squaredErr float64
	predictions := make([]float64, 0, len(cfg.cases))
	for _, c := range cfg.cases {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}
		predicted, err := predict(ctx, c.in)
		if err != nil {
			return 0, nil, err
		}
		predictions = append(predictions, predicted)
		delta := predicted - c.want
		squaredErr += delta * delta
	}

	if len(cfg.cases) == 0 {
		return 0, Trace{"mse": 0.0, "sse": 0.0, "predictions": predictions, "mode": cfg.mode, "cases": 0}, nil
	}

	sse := squaredErr
	mse := sse / float64(len(cfg.cases))
	// Mirror reference scape.erl xor fitness semantics: reciprocal SSE with epsilon.
	fitness := Fitness(1.0 / (sse + 0.000001))
	return fitness, Trace{
		"mse":         mse,
		"sse":         sse,
		"predictions": predictions,
		"mode":        cfg.mode,
		"cases":       len(cfg.cases),
	}, nil
}

func xorIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	left, ok := typed.RegisteredSensor(protoio.XORInputLeftSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.XORInputLeftSensorName)
	}
	leftSetter, ok := left.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.XORInputLeftSensorName)
	}

	right, ok := typed.RegisteredSensor(protoio.XORInputRightSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.XORInputRightSensorName)
	}
	rightSetter, ok := right.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.XORInputRightSensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.XOROutputActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.XOROutputActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.XOROutputActuatorName)
	}
	return leftSetter, rightSetter, output, nil
}
