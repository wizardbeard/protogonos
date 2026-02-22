package genotype

import (
	"math/rand"
	"testing"
)

func TestCreateWeightRange(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 50; i++ {
		weight := CreateWeight(rng)
		if weight < -0.5 || weight > 0.5 {
			t.Fatalf("expected weight in [-0.5, 0.5], got=%f", weight)
		}
	}
}

func TestCreateNeuralWeightsPCountAndPlasticityWidth(t *testing.T) {
	weights := CreateNeuralWeightsP("hebbian_w", 3, rand.New(rand.NewSource(7)))
	if len(weights) != 3 {
		t.Fatalf("expected 3 neural weights, got=%d", len(weights))
	}
	for _, weight := range weights {
		if len(weight.PlasticityParams) != 1 {
			t.Fatalf("expected hebbian_w plasticity width 1, got=%v", weight.PlasticityParams)
		}
	}
}

func TestCreateInputIDPsBuildsOrderedSpecs(t *testing.T) {
	inputs := CreateInputIDPs(
		"none",
		[]InputSpec{
			{FromID: "s1", Width: 2},
			{FromID: " ", Width: 1},
			{FromID: "s2", Width: 1},
			{FromID: "s3", Width: 0},
		},
		rand.New(rand.NewSource(9)),
	)
	if len(inputs) != 2 {
		t.Fatalf("expected two valid input specs, got=%d", len(inputs))
	}
	if inputs[0].FromID != "s1" || len(inputs[0].Weights) != 2 {
		t.Fatalf("unexpected first input weights spec: %+v", inputs[0])
	}
	if inputs[1].FromID != "s2" || len(inputs[1].Weights) != 1 {
		t.Fatalf("unexpected second input weights spec: %+v", inputs[1])
	}
}

func TestCreateNeuralWeightsPHandlesNonPositiveCount(t *testing.T) {
	if got := CreateNeuralWeightsP("none", 0, rand.New(rand.NewSource(1))); got != nil {
		t.Fatalf("expected nil for non-positive count, got=%v", got)
	}
}
