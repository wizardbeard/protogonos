package scape

import (
	"context"
	"fmt"
	"math"
)

type GTSAScape struct{}

func (GTSAScape) Name() string {
	return "gtsa"
}

func (GTSAScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	const steps = 40
	t := 0.0
	last := gtsaSignal(t)
	mse := 0.0

	for i := 0; i < steps; i++ {
		input := []float64{last}
		out, err := runner.RunStep(ctx, input)
		if err != nil {
			return 0, nil, err
		}
		if len(out) != 1 {
			return 0, nil, fmt.Errorf("gtsa requires one output, got %d", len(out))
		}
		t += 1.0
		next := gtsaSignal(t)
		delta := out[0] - next
		mse += delta * delta
		last = next
	}

	mse /= float64(steps)
	fitness := 1.0 / (1.0 + mse)
	return Fitness(fitness), Trace{"mse": mse}, nil
}

func gtsaSignal(t float64) float64 {
	return math.Sin(t*0.2) + 0.5*math.Sin(t*0.05)
}
