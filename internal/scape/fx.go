package scape

import (
	"context"
	"fmt"
	"math"
)

type FXScape struct{}

func (FXScape) Name() string {
	return "fx"
}

func (FXScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	const steps = 64
	price := 1.0
	prevPrice := price
	equity := 1.0
	position := 0.0
	turnover := 0.0

	for i := 0; i < steps; i++ {
		price = fxPrice(i)
		signal := fxSignal(i)
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
		"equity":   equity,
		"turnover": turnover,
	}, nil
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
