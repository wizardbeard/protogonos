package evo

import (
	"math"
	"testing"
)

func TestSizeProportionalPostprocessorUsesReferenceEfficiencyExponent(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("small", 1), Fitness: 1.0},
		{Genome: newComplexLinearGenome("large", 1), Fitness: 1.0},
	}
	out := SizeProportionalPostprocessor{}.Process(scored)

	smallComplexity := float64(len(scored[0].Genome.Neurons) + len(scored[0].Genome.Synapses))
	largeComplexity := float64(len(scored[1].Genome.Neurons) + len(scored[1].Genome.Synapses))
	wantSmall := 1.0 / math.Pow(smallComplexity, sizeProportionalEfficiency)
	wantLarge := 1.0 / math.Pow(largeComplexity, sizeProportionalEfficiency)

	if math.Abs(out[0].Fitness-wantSmall) > 1e-9 {
		t.Fatalf("unexpected small adjusted fitness: got=%f want=%f", out[0].Fitness, wantSmall)
	}
	if math.Abs(out[1].Fitness-wantLarge) > 1e-9 {
		t.Fatalf("unexpected large adjusted fitness: got=%f want=%f", out[1].Fitness, wantLarge)
	}
}

func TestSizeProportionalPostprocessorKeepsCloneIsolation(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("g", 1), Fitness: 1.0},
	}
	out := SizeProportionalPostprocessor{}.Process(scored)
	out[0].Fitness = 999
	if scored[0].Fitness == 999 {
		t.Fatal("expected postprocessor output to be cloned from input")
	}
}
