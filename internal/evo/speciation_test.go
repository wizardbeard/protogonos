package evo

import (
	"testing"

	"protogonos/internal/model"
)

func TestGenomeCompatibilityDistance(t *testing.T) {
	a := newLinearGenome("a", 1.0)
	b := newLinearGenome("b", 0.5)
	c := newComplexLinearGenome("c", 1.0)

	ab := GenomeCompatibilityDistance(a, b)
	ac := GenomeCompatibilityDistance(a, c)
	if ab != 0 {
		t.Fatalf("expected identical-topology genomes to have zero distance, got %f", ab)
	}
	if ac <= 0 {
		t.Fatalf("expected different-topology genomes to have positive distance, got %f", ac)
	}
}

func TestAdaptiveSpeciationAdjustsThresholdTowardTarget(t *testing.T) {
	spec := &AdaptiveSpeciation{
		TargetSpeciesCount: 1,
		Threshold:          0.05,
		MinThreshold:       0.01,
		MaxThreshold:       5.0,
		AdjustStep:         0.5,
	}
	genomes := []model.Genome{
		newLinearGenome("g0", 1.0),
		newComplexLinearGenome("g1", 1.0),
	}

	_, stats := spec.Assign(genomes)
	if stats.SpeciesCount <= 1 {
		t.Fatalf("expected at least 2 species with low threshold, got %d", stats.SpeciesCount)
	}
	if stats.Threshold <= 0.05 {
		t.Fatalf("expected threshold to increase when species exceed target, got %f", stats.Threshold)
	}

	spec.TargetSpeciesCount = 5
	before := spec.Threshold
	_, stats = spec.Assign(genomes)
	if stats.Threshold >= before {
		t.Fatalf("expected threshold to decrease when species below target, before=%f after=%f", before, stats.Threshold)
	}
}
