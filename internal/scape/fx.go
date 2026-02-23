package scape

import (
	"context"
	"fmt"
	"math"
	"strings"

	protoio "protogonos/internal/io"
)

type FXScape struct{}

func (FXScape) Name() string {
	return "fx"
}

func (FXScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return FXScape{}.EvaluateMode(ctx, agent, "gt")
}

func (FXScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := fxConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateFXWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateFXWithStep(ctx, runner, cfg)
}

func evaluateFXWithStep(ctx context.Context, runner StepAgent, cfg fxModeConfig) (Fitness, Trace, error) {
	return evaluateFX(ctx, cfg, func(ctx context.Context, price, signal float64) (float64, error) {
		out, err := runner.RunStep(ctx, []float64{price, signal})
		if err != nil {
			return 0, err
		}
		if len(out) != 1 {
			return 0, fmt.Errorf("fx requires one output, got %d", len(out))
		}
		return out[0], nil
	})
}

func evaluateFXWithTick(ctx context.Context, ticker TickAgent, cfg fxModeConfig) (Fitness, Trace, error) {
	priceSetter, signalSetter, tradeOutput, err := fxIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateFX(ctx, cfg, func(ctx context.Context, price, signal float64) (float64, error) {
		priceSetter.Set(price)
		signalSetter.Set(signal)
		out, err := ticker.Tick(ctx)
		if err != nil {
			return 0, err
		}
		lastOutput := tradeOutput.Last()
		if len(lastOutput) > 0 {
			return lastOutput[0], nil
		}
		if len(out) > 0 {
			return out[0], nil
		}
		return 0, nil
	})
}

func evaluateFX(
	ctx context.Context,
	cfg fxModeConfig,
	chooseTrade func(context.Context, float64, float64) (float64, error),
) (Fitness, Trace, error) {
	price := fxPrice(cfg.startStep)
	prevPrice := price
	equity := 1.0
	position := 0.0
	turnover := 0.0

	for i := 0; i < cfg.steps; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}
		step := cfg.startStep + i
		price = fxPrice(step)
		signal := fxSignal(step)
		targetPosition, err := chooseTrade(ctx, price, signal)
		if err != nil {
			return 0, nil, err
		}

		targetPosition = clampFX(targetPosition, -1, 1)
		turnover += math.Abs(targetPosition - position)
		position = targetPosition

		returnRate := (price - prevPrice) / prevPrice
		equity *= 1.0 + position*returnRate - 0.0005*turnover
		if equity < 0.1 {
			equity = 0.1
		}
		prevPrice = price
	}

	fitness := math.Log(equity) - 0.002*turnover
	return Fitness(1.0 / (1.0 + math.Exp(-fitness))), Trace{
		"equity":     equity,
		"turnover":   turnover,
		"mode":       cfg.mode,
		"steps":      cfg.steps,
		"start_step": cfg.startStep,
	}, nil
}

func fxIO(agent TickAgent) (
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

	price, ok := typed.RegisteredSensor(protoio.FXPriceSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.FXPriceSensorName)
	}
	priceSetter, ok := price.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.FXPriceSensorName)
	}

	signal, ok := typed.RegisteredSensor(protoio.FXSignalSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.FXSignalSensorName)
	}
	signalSetter, ok := signal.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.FXSignalSensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.FXTradeActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.FXTradeActuatorName)
	}
	tradeOutput, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.FXTradeActuatorName)
	}
	return priceSetter, signalSetter, tradeOutput, nil
}

type fxModeConfig struct {
	mode      string
	steps     int
	startStep int
}

func fxConfigForMode(mode string) (fxModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return fxModeConfig{mode: "gt", steps: 64, startStep: 0}, nil
	case "validation":
		return fxModeConfig{mode: "validation", steps: 48, startStep: 128}, nil
	case "test":
		return fxModeConfig{mode: "test", steps: 48, startStep: 256}, nil
	default:
		return fxModeConfig{}, fmt.Errorf("unsupported fx mode: %s", mode)
	}
}

func fxPrice(step int) float64 {
	t := float64(step)
	return 1.0 + 0.08*math.Sin(t*0.25) + 0.03*math.Sin(t*0.07)
}

func fxSignal(step int) float64 {
	current := fxPrice(step)
	next := fxPrice(step + 1)
	return (next - current) / current
}

func clampFX(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
