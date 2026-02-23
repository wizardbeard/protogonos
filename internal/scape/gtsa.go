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
	return evaluateGTSA(ctx, cfg, func(ctx context.Context, last float64) (float64, error) {
		out, err := runner.RunStep(ctx, []float64{last})
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

	return evaluateGTSA(ctx, cfg, func(ctx context.Context, last float64) (float64, error) {
		inputSetter.Set(last)
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
	t := cfg.startT
	last := gtsaSignal(t)
	mse := 0.0

	for i := 0; i < cfg.steps; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}
		predicted, err := predict(ctx, last)
		if err != nil {
			return 0, nil, err
		}
		t += 1.0
		next := gtsaSignal(t)
		delta := predicted - next
		mse += delta * delta
		last = next
	}

	mse /= float64(cfg.steps)
	fitness := 1.0 / (1.0 + mse)
	return Fitness(fitness), Trace{
		"mse":     mse,
		"mode":    cfg.mode,
		"steps":   cfg.steps,
		"start_t": cfg.startT,
	}, nil
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
	mode   string
	steps  int
	startT float64
}

func gtsaConfigForMode(mode string) (gtsaModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return gtsaModeConfig{mode: "gt", steps: 40, startT: 0}, nil
	case "validation":
		return gtsaModeConfig{mode: "validation", steps: 32, startT: 120}, nil
	case "test":
		return gtsaModeConfig{mode: "test", steps: 32, startT: 240}, nil
	default:
		return gtsaModeConfig{}, fmt.Errorf("unsupported gtsa mode: %s", mode)
	}
}

func gtsaSignal(t float64) float64 {
	return math.Sin(t*0.2) + 0.5*math.Sin(t*0.05)
}
