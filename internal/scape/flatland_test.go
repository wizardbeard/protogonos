package scape

import (
	"context"
	"testing"
)

type scriptedStepAgent struct {
	id string
	fn func(input []float64) []float64
}

func (a scriptedStepAgent) ID() string { return a.id }

func (a scriptedStepAgent) RunStep(_ context.Context, input []float64) ([]float64, error) {
	return a.fn(input), nil
}

func TestFlatlandScapeRewardsForwardMotion(t *testing.T) {
	scape := FlatlandScape{}
	stationary := scriptedStepAgent{
		id: "stationary",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	forward := scriptedStepAgent{
		id: "forward",
		fn: func(input []float64) []float64 {
			if len(input) == 0 {
				return []float64{0}
			}
			if input[0] > 0 {
				return []float64{1}
			}
			return []float64{-1}
		},
	}

	stationaryFitness, _, err := scape.Evaluate(context.Background(), stationary)
	if err != nil {
		t.Fatalf("evaluate stationary: %v", err)
	}
	forwardFitness, _, err := scape.Evaluate(context.Background(), forward)
	if err != nil {
		t.Fatalf("evaluate forward: %v", err)
	}
	if forwardFitness <= stationaryFitness {
		t.Fatalf("expected forward policy to outperform stationary, got forward=%f stationary=%f", forwardFitness, stationaryFitness)
	}
}
