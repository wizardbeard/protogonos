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

func TestSpeciesTournamentSelectorUsesProvidedSpeciesAssignments(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 1.0},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.9},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.8},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.7},
	}
	speciesByGenomeID := map[string]string{
		"a0": "sp-1",
		"b0": "sp-1",
		"a1": "sp-2",
		"b1": "sp-2",
	}

	selector := SpeciesTournamentSelector{
		Identifier:     TopologySpecieIdentifier{},
		PoolSize:       len(scored),
		TournamentSize: 1,
	}
	rng := rand.New(rand.NewSource(33))
	seen := map[string]struct{}{}

	for i := 0; i < 40; i++ {
		parent, err := selector.PickParentForGenerationWithSpecies(rng, scored, 1, 1, speciesByGenomeID)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		seen[speciesByGenomeID[parent.ID]] = struct{}{}
	}
	if len(seen) != 2 {
		t.Fatalf("expected picks from both provided species, got %d", len(seen))
	}
}

func TestSpeciesSharedTournamentSelectorUsesProvidedSpeciesAssignments(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.95},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.85},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.25},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.15},
	}
	speciesByGenomeID := map[string]string{
		"a0": "sp-1",
		"b0": "sp-1",
		"a1": "sp-2",
		"b1": "sp-2",
	}

	selector := &SpeciesSharedTournamentSelector{
		Identifier:     TopologySpecieIdentifier{},
		PoolSize:       len(scored),
		TournamentSize: 1,
	}
	rng := rand.New(rand.NewSource(37))
	countBySpecies := map[string]int{}

	for i := 0; i < 200; i++ {
		parent, err := selector.PickParentForGenerationWithSpecies(rng, scored, 1, 1, speciesByGenomeID)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		countBySpecies[speciesByGenomeID[parent.ID]]++
	}
	if countBySpecies["sp-1"] <= countBySpecies["sp-2"] {
		t.Fatalf("expected fitter provided species to dominate picks: sp-1=%d sp-2=%d", countBySpecies["sp-1"], countBySpecies["sp-2"])
	}
}

func TestRankSelectorBiasesTowardTopRankedGenome(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("g0", 1), Fitness: 0.9},
		{Genome: newLinearGenome("g1", 1), Fitness: 0.7},
		{Genome: newLinearGenome("g2", 1), Fitness: 0.5},
		{Genome: newLinearGenome("g3", 1), Fitness: 0.3},
	}
	selector := RankSelector{PoolSize: len(scored)}
	rng := rand.New(rand.NewSource(55))

	counts := map[string]int{}
	for i := 0; i < 500; i++ {
		parent, err := selector.PickParent(rng, scored, 1)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		counts[parent.ID]++
	}
	if counts["g0"] <= counts["g3"] {
		t.Fatalf("expected top-ranked genome to be picked more than lowest-ranked: g0=%d g3=%d", counts["g0"], counts["g3"])
	}
}

func TestEfficiencySelectorFavorsMoreEfficientGenome(t *testing.T) {
	highComplexity := newComplexLinearGenome("complex", 1)
	highComplexity.Neurons = append(highComplexity.Neurons, highComplexity.Neurons...)
	highComplexity.Synapses = append(highComplexity.Synapses, highComplexity.Synapses...)

	scored := []ScoredGenome{
		{Genome: highComplexity, Fitness: 1.0},
		{Genome: newLinearGenome("efficient", 1), Fitness: 0.8},
	}
	selector := EfficiencySelector{PoolSize: len(scored)}
	rng := rand.New(rand.NewSource(77))

	counts := map[string]int{}
	for i := 0; i < 400; i++ {
		parent, err := selector.PickParent(rng, scored, 1)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		counts[parent.ID]++
	}
	if counts["efficient"] <= counts["complex"] {
		t.Fatalf("expected efficient genome to be selected more often: efficient=%d complex=%d", counts["efficient"], counts["complex"])
	}
}

func TestRandomSelectorSelectsMultipleCandidates(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("a", 1), Fitness: 1},
		{Genome: newLinearGenome("b", 1), Fitness: 0.9},
		{Genome: newLinearGenome("c", 1), Fitness: 0.8},
	}
	selector := RandomSelector{PoolSize: len(scored)}
	rng := rand.New(rand.NewSource(91))
	seen := map[string]struct{}{}
	for i := 0; i < 80; i++ {
		parent, err := selector.PickParent(rng, scored, 1)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		seen[parent.ID] = struct{}{}
	}
	if len(seen) < 2 {
		t.Fatalf("expected random selector to sample multiple parents, got %d", len(seen))
	}
}

func TestTopKFitnessSelectorRestrictsSelectionToTopK(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("g0", 1), Fitness: 0.90},
		{Genome: newLinearGenome("g1", 1), Fitness: 0.80},
		{Genome: newLinearGenome("g2", 1), Fitness: 0.70},
		{Genome: newLinearGenome("g3", 1), Fitness: 0.60},
		{Genome: newLinearGenome("g4", 1), Fitness: 0.50},
	}
	selector := TopKFitnessSelector{K: 3}
	rng := rand.New(rand.NewSource(123))
	seen := map[string]struct{}{}

	for i := 0; i < 250; i++ {
		parent, err := selector.PickParent(rng, scored, 1)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		seen[parent.ID] = struct{}{}
	}
	if _, ok := seen["g3"]; ok {
		t.Fatalf("selector picked genome outside top-k: g3")
	}
	if _, ok := seen["g4"]; ok {
		t.Fatalf("selector picked genome outside top-k: g4")
	}
}

func TestTopKFitnessSelectorBiasesTowardBestInTopK(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("best", 1), Fitness: 1.0},
		{Genome: newLinearGenome("mid", 1), Fitness: 0.2},
		{Genome: newLinearGenome("low", 1), Fitness: 0.1},
		{Genome: newLinearGenome("outside", 1), Fitness: 0.9},
	}
	selector := TopKFitnessSelector{K: 3}
	rng := rand.New(rand.NewSource(321))
	counts := map[string]int{}

	for i := 0; i < 600; i++ {
		parent, err := selector.PickParent(rng, scored, 1)
		if err != nil {
			t.Fatalf("pick parent: %v", err)
		}
		counts[parent.ID]++
	}
	if counts["best"] <= counts["mid"] || counts["best"] <= counts["low"] {
		t.Fatalf("expected best to dominate in top-k pool: best=%d mid=%d low=%d", counts["best"], counts["mid"], counts["low"])
	}
	if counts["outside"] != 0 {
		t.Fatalf("expected outside top-k genome to never be selected, got %d", counts["outside"])
	}
}
