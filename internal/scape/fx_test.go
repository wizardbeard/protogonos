package scape

import (
	"context"
	"testing"
)

func TestFXScapeRewardsSignalFollowingPolicy(t *testing.T) {
	scape := FXScape{}
	flat := scriptedStepAgent{
		id: "flat",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	follow := scriptedStepAgent{
		id: "follow",
		fn: func(input []float64) []float64 {
			return []float64{input[1]}
		},
	}

	flatFitness, _, err := scape.Evaluate(context.Background(), flat)
	if err != nil {
		t.Fatalf("evaluate flat: %v", err)
	}
	followFitness, _, err := scape.Evaluate(context.Background(), follow)
	if err != nil {
		t.Fatalf("evaluate follow: %v", err)
	}
	if followFitness <= flatFitness {
		t.Fatalf("expected signal-following strategy to outperform flat, got follow=%f flat=%f", followFitness, flatFitness)
	}
}

func TestFXScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := FXScape{}
	follow := scriptedStepAgent{
		id: "follow",
		fn: func(input []float64) []float64 {
			return []float64{input[1]}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), follow, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), follow, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
}
