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
		nextSpeciesID:      1,
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

func TestAdaptiveSpeciationMaintainsSpeciesIdentityAcrossGenerations(t *testing.T) {
	spec := NewAdaptiveSpeciation(8)
	spec.Threshold = 0.6
	spec.MinThreshold = 0.1
	spec.MaxThreshold = 2.0
	spec.AdjustStep = 0.1

	gen1 := []model.Genome{
		newLinearGenome("a0", 1.0),
		newLinearGenome("a1", 0.8),
		newComplexLinearGenome("b0", 1.0),
		newComplexLinearGenome("b1", 0.7),
	}
	s1, _ := spec.Assign(gen1)
	if len(s1) != 2 {
		t.Fatalf("expected 2 species in generation 1, got %d", len(s1))
	}

	gen2 := []model.Genome{
		newLinearGenome("a2", 0.9),
		newLinearGenome("a3", 0.85),
		newComplexLinearGenome("b2", 0.9),
		newComplexLinearGenome("b3", 0.75),
	}
	s2, _ := spec.Assign(gen2)
	if len(s2) != 2 {
		t.Fatalf("expected 2 species in generation 2, got %d", len(s2))
	}

	commonKeys := 0
	for key := range s1 {
		if _, ok := s2[key]; ok {
			commonKeys++
		}
	}
	if commonKeys < 2 {
		t.Fatalf("expected species keys continuity across generations, got common=%d", commonKeys)
	}
}
