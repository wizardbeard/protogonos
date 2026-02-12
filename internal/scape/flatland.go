package scape

import (
	"context"
	"fmt"
	"math"
)

type FlatlandScape struct{}

func (FlatlandScape) Name() string {
	return "flatland"
}

func (FlatlandScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	const steps = 24
	position := 0.0
	target := 1.0
	energy := 1.0
	reward := 0.0

	for i := 0; i < steps; i++ {
		distance := target - position
		input := []float64{distance, energy}
		out, err := runner.RunStep(ctx, input)
		if err != nil {
			return 0, nil, err
		}
		if len(out) != 1 {
			return 0, nil, fmt.Errorf("flatland requires one output, got %d", len(out))
		}
		move := clamp(out[0], -1, 1)
		position += move * 0.08
		energy -= 0.02 + 0.03*math.Abs(move)
		if energy < 0 {
			energy = 0
		}
		reward += 1.0 - math.Abs(target-position)
		if energy <= 0 {
			break
		}
	}

	normalizedReward := reward / float64(steps)
	fitness := clamp(normalizedReward, -1, 1)*0.8 + energy*0.2
	if math.Abs(target-position) < 0.15 {
		fitness += 0.2
	}
	return Fitness(clamp(fitness, 0, 1.2)), Trace{
		"position": position,
		"energy":   energy,
		"reward":   normalizedReward,
	}, nil
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
