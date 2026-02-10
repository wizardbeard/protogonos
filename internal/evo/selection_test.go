package evo

import (
	"math/rand"
	"testing"
)

func TestTopologySpecieIdentifier(t *testing.T) {
	id := TopologySpecieIdentifier{}
	a := newLinearGenome("a", 1.0)
	b := newLinearGenome("b", 0.5)
	c := newComplexLinearGenome("c", 1.0)

	if id.Identify(a) != id.Identify(b) {
		t.Fatal("expected same topology species key")
	}
	if id.Identify(a) == id.Identify(c) {
		t.Fatal("expected different topology species key")
	}
}

func TestSpeciesTournamentSelectorProducesMultiSpeciesParents(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.99},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.98},
		{Genome: newLinearGenome("a2", 1), Fitness: 0.97},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.60},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.59},
		{Genome: newComplexLinearGenome("b2", 1), Fitness: 0.58},
	}

	selector := SpeciesTournamentSelector{
		Identifier:     TopologySpecieIdentifier{},
		PoolSize:       len(scored),
		TournamentSize: 2,
	}
	rng := rand.New(rand.NewSource(42))
	seenSpecies := map[string]struct{}{}
	id := TopologySpecieIdentifier{}

	for i := 0; i < 40; i++ {
		parent, err := selector.PickParent(rng, scored, 1)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		seenSpecies[id.Identify(parent)] = struct{}{}
	}
	if len(seenSpecies) < 2 {
		t.Fatalf("expected at least 2 species in selected parents, got %d", len(seenSpecies))
	}
}
