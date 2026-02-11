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

func TestSpeciesSharedTournamentSelectorBiasesTowardHigherMeanFitnessSpecies(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.9},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.8},
		{Genome: newLinearGenome("a2", 1), Fitness: 0.7},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.2},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.15},
		{Genome: newComplexLinearGenome("b2", 1), Fitness: 0.1},
	}

	selector := &SpeciesSharedTournamentSelector{
		Identifier:     TopologySpecieIdentifier{},
		PoolSize:       len(scored),
		TournamentSize: 1,
	}
	rng := rand.New(rand.NewSource(11))
	id := TopologySpecieIdentifier{}
	countBySpecies := map[string]int{}

	for i := 0; i < 500; i++ {
		parent, err := selector.PickParentForGeneration(rng, scored, 1, 1)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		countBySpecies[id.Identify(parent)]++
	}

	if len(countBySpecies) < 2 {
		t.Fatalf("expected 2 species in picks, got %d", len(countBySpecies))
	}
	highFitnessSpecies := id.Identify(scored[0].Genome)
	lowFitnessSpecies := id.Identify(scored[len(scored)-1].Genome)
	if countBySpecies[highFitnessSpecies] <= countBySpecies[lowFitnessSpecies] {
		t.Fatalf("expected high-fitness species to be picked more often: high=%d low=%d", countBySpecies[highFitnessSpecies], countBySpecies[lowFitnessSpecies])
	}
}

func TestSpeciesSharedTournamentSelectorFiltersStagnantSpecies(t *testing.T) {
	selector := &SpeciesSharedTournamentSelector{
		Identifier:            TopologySpecieIdentifier{},
		PoolSize:              6,
		TournamentSize:        1,
		StagnationGenerations: 1,
	}
	rng := rand.New(rand.NewSource(19))
	id := TopologySpecieIdentifier{}

	gen1 := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.5},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.4},
		{Genome: newLinearGenome("a2", 1), Fitness: 0.3},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.4},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.3},
		{Genome: newComplexLinearGenome("b2", 1), Fitness: 0.2},
	}
	if _, err := selector.PickParentForGeneration(rng, gen1, 1, 1); err != nil {
		t.Fatalf("generation 1 pick parent: %v", err)
	}

	gen2 := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.8},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.7},
		{Genome: newLinearGenome("a2", 1), Fitness: 0.6},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.4},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.3},
		{Genome: newComplexLinearGenome("b2", 1), Fitness: 0.2},
	}
	if _, err := selector.PickParentForGeneration(rng, gen2, 1, 2); err != nil {
		t.Fatalf("generation 2 pick parent: %v", err)
	}

	gen3 := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.81},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.71},
		{Genome: newLinearGenome("a2", 1), Fitness: 0.61},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.4},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.3},
		{Genome: newComplexLinearGenome("b2", 1), Fitness: 0.2},
	}
	if _, err := selector.PickParentForGeneration(rng, gen3, 1, 3); err != nil {
		t.Fatalf("generation 3 pick parent: %v", err)
	}

	gen4 := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.81},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.71},
		{Genome: newLinearGenome("a2", 1), Fitness: 0.61},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.4},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.3},
		{Genome: newComplexLinearGenome("b2", 1), Fitness: 0.2},
	}

	keptSpecies := map[string]struct{}{}
	for i := 0; i < 100; i++ {
		parent, err := selector.PickParentForGeneration(rng, gen4, 1, 4)
		if err != nil {
			t.Fatalf("generation 4 pick parent: %v", err)
		}
		keptSpecies[id.Identify(parent)] = struct{}{}
	}

	if len(keptSpecies) != 1 {
		t.Fatalf("expected only one non-stagnant species to be selected, got %d", len(keptSpecies))
	}
	activeSpecies := id.Identify(gen4[0].Genome)
	if _, ok := keptSpecies[activeSpecies]; !ok {
		t.Fatalf("expected active species %q to be selected", activeSpecies)
	}
}
