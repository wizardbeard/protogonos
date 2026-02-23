package scape

import (
	"context"
	"fmt"
	"math"
	"strings"
)

type FXScape struct{}

func (FXScape) Name() string {
	return "fx"
}

func (FXScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return FXScape{}.EvaluateMode(ctx, agent, "gt")
}

func (FXScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	cfg, err := fxConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	price := fxPrice(cfg.startStep)
	prevPrice := price
	equity := 1.0
	position := 0.0
	turnover := 0.0

	for i := 0; i < cfg.steps; i++ {
		step := cfg.startStep + i
		price = fxPrice(step)
		signal := fxSignal(step)
		out, err := runner.RunStep(ctx, []float64{price, signal})
		if err != nil {
			return 0, nil, err
		}
		if len(out) != 1 {
			return 0, nil, fmt.Errorf("fx requires one output, got %d", len(out))
		}

		targetPosition := clampFX(out[0], -1, 1)
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
