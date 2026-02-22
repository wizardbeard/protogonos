package genotype

import (
	"testing"

	"protogonos/internal/model"
)

func TestSpeciateByFingerprintGroupsExactMatches(t *testing.T) {
	a := model.Genome{
		ID:          "a",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Weight: 0.25, Enabled: true},
		},
	}
	b := CloneGenome(a)
	b.ID = "b"
	b.Synapses[0].Weight = -0.75 // fingerprint ignores concrete weights.
	c := CloneGenome(a)
	c.ID = "c"
	c.Neurons = append(c.Neurons, model.Neuron{ID: "h", Activation: "tanh"})
	c.Synapses = append(c.Synapses, model.Synapse{ID: "s2", From: "i", To: "h", Weight: 0.1, Enabled: true})

	grouped := SpeciateByFingerprint([]model.Genome{a, b, c})
	if len(grouped) != 2 {
		t.Fatalf("expected 2 species buckets, got=%d", len(grouped))
	}

	keyAB := fingerprintSpeciesKey(a)
	membersAB := grouped[keyAB]
	if len(membersAB) != 2 {
		t.Fatalf("expected 2 members in shared species, got=%d", len(membersAB))
	}
	if membersAB[0].ID != "a" || membersAB[1].ID != "b" {
		t.Fatalf("unexpected species member ordering: %+v", []string{membersAB[0].ID, membersAB[1].ID})
	}
}

func TestSpeciateByFingerprintDifferentiatesDistinctIOSets(t *testing.T) {
	a := model.Genome{
		ID:          "a",
		SensorIDs:   []string{"sensor:left"},
		ActuatorIDs: []string{"actuator:go"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Weight: 0.25, Enabled: true},
		},
	}
	b := CloneGenome(a)
	b.ID = "b"
	b.SensorIDs = []string{"sensor:right"}
	b.ActuatorIDs = []string{"actuator:turn"}

	grouped := SpeciateByFingerprint([]model.Genome{a, b})
	if len(grouped) != 2 {
		t.Fatalf("expected 2 distinct species for different io identity, got=%d", len(grouped))
	}
}

func TestSpeciateByFingerprintWithHistoryDifferentiatesMutationTrajectories(t *testing.T) {
	base := model.Genome{
		ID:          "a",
		SensorIDs:   []string{"sensor:left"},
		ActuatorIDs: []string{"actuator:go"},
		Neurons: []model.Neuron{
			{ID: "L0:n0", Activation: "identity"},
			{ID: "L1:n1", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "L0:n0", To: "L1:n1", Weight: 0.25, Enabled: true},
		},
	}
	peer := CloneGenome(base)
	peer.ID = "b"

	withoutHistory := SpeciateByFingerprint([]model.Genome{base, peer})
	if len(withoutHistory) != 1 {
		t.Fatalf("expected identical genomes to share one species without history, got=%d", len(withoutHistory))
	}

	historyByGenomeID := map[string][]EvoHistoryEvent{
		"a": {{Mutation: "add_link"}},
		"b": {{Mutation: "remove_link"}},
	}
	withHistory := SpeciateByFingerprintWithHistory([]model.Genome{base, peer}, historyByGenomeID)
	if len(withHistory) != 2 {
		t.Fatalf("expected divergent histories to split species, got=%d", len(withHistory))
	}
}

func TestAssignToFingerprintSpeciesAppendsIncrementally(t *testing.T) {
	base := model.Genome{
		ID:          "base",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Weight: 0.5, Enabled: true},
		},
	}
	var species map[string][]model.Genome

	k1, species := AssignToFingerprintSpecies(base, species)
	if len(species) != 1 {
		t.Fatalf("expected one species after first assign, got=%d", len(species))
	}

	peer := CloneGenome(base)
	peer.ID = "peer"
	k2, species := AssignToFingerprintSpecies(peer, species)
	if k2 != k1 {
		t.Fatalf("expected identical fingerprint species keys, got %q != %q", k2, k1)
	}
	if len(species[k1]) != 2 {
		t.Fatalf("expected two members in first species, got=%d", len(species[k1]))
	}

	other := CloneGenome(base)
	other.ID = "other"
	other.Neurons = append(other.Neurons, model.Neuron{ID: "h", Activation: "tanh"})
	k3, species := AssignToFingerprintSpecies(other, species)
	if k3 == k1 {
		t.Fatalf("expected distinct species key for changed topology, got=%q", k3)
	}
	if len(species) != 2 {
		t.Fatalf("expected two species after distinct topology assign, got=%d", len(species))
	}
}

func TestComputeSpeciationFingerprintKeyUsesReferenceFingerprint(t *testing.T) {
	genome := model.Genome{
		ID:          "g",
		SensorIDs:   []string{"sensor:right", "sensor:left", "sensor:left"},
		ActuatorIDs: []string{"actuator:out"},
		Neurons: []model.Neuron{
			{ID: "L0:n0", Activation: "identity", Aggregator: "dot_product"},
			{ID: "L1:n1", Activation: "tanh", Aggregator: "dot_product"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "L0:n0", To: "L1:n1", Enabled: true},
		},
	}
	history := []EvoHistoryEvent{
		{Mutation: "add_link", IDs: []string{"L0:n0", "L1:n1"}},
	}

	want := "fp:" + ComputeReferenceFingerprint(genome, history)
	got := ComputeSpeciationFingerprintKey(genome, history)
	if got != want {
		t.Fatalf("expected reference-based speciation fingerprint key, got=%q want=%q", got, want)
	}
}

func TestSpeciateInPopulationWithHistoryUsesProvidedHistory(t *testing.T) {
	genome := model.Genome{
		ID:          "g1",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "L0:n0", Activation: "identity"},
			{ID: "L1:n1", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "L0:n0", To: "L1:n1", Enabled: true},
		},
	}
	population := map[string]FingerprintSpecies{}
	historyA := []EvoHistoryEvent{{Mutation: "add_link"}}
	keyA, population := SpeciateInPopulationWithHistory(genome, historyA, population, nil)
	if keyA == "" {
		t.Fatal("expected non-empty species id with history")
	}

	peer := CloneGenome(genome)
	peer.ID = "g2"
	historyB := []EvoHistoryEvent{{Mutation: "remove_link"}}
	keyB, population := SpeciateInPopulationWithHistory(peer, historyB, population, nil)
	if keyB == keyA {
		t.Fatalf("expected distinct history to allocate a distinct species id, got keyA=%q keyB=%q", keyA, keyB)
	}
	if len(population) != 2 {
		t.Fatalf("expected two species after divergent-history insertion, got=%d", len(population))
	}
}

func TestSpeciateAssignsFingerprintSpeciesForNonTestGenome(t *testing.T) {
	genome := model.Genome{
		ID:          "agent-1",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Enabled: true},
		},
	}
	key, species := Speciate(genome, nil)
	if key == "" {
		t.Fatal("expected non-empty species key for regular genome")
	}
	if len(species[key]) != 1 {
		t.Fatalf("expected one species member, got=%d", len(species[key]))
	}
}

func TestSpeciateSkipsTestGenomeAssignment(t *testing.T) {
	genome := model.Genome{ID: "test"}
	key, species := Speciate(genome, nil)
	if key != "" {
		t.Fatalf("expected empty species key for test genome, got=%q", key)
	}
	if len(species) != 0 {
		t.Fatalf("expected no species assignment for test genome, got=%v", species)
	}
}

func TestSpeciateInPopulationCreatesAndReusesFingerprintSpecies(t *testing.T) {
	base := model.Genome{
		ID:          "g1",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Enabled: true},
		},
	}
	peer := CloneGenome(base)
	peer.ID = "g2"
	peer.Synapses[0].Weight = -0.25 // same fingerprint topology.
	other := CloneGenome(base)
	other.ID = "g3"
	other.Neurons = append(other.Neurons, model.Neuron{ID: "h", Activation: "tanh"})

	nextCalls := 0
	nextID := func(_ string) string {
		nextCalls++
		return ""
	}

	var population map[string]FingerprintSpecies
	sp1, population := SpeciateInPopulation(base, population, nextID)
	if sp1 == "" {
		t.Fatal("expected species id for first genome")
	}
	if nextCalls != 1 {
		t.Fatalf("expected one species-id generation call, got=%d", nextCalls)
	}

	sp2, population := SpeciateInPopulation(peer, population, nextID)
	if sp2 != sp1 {
		t.Fatalf("expected peer with same fingerprint to reuse species %q, got=%q", sp1, sp2)
	}
	if got := population[sp1].GenomeIDs; len(got) != 2 {
		t.Fatalf("expected two members in reused species, got=%v", got)
	}

	sp3, population := SpeciateInPopulation(other, population, nextID)
	if sp3 == sp1 {
		t.Fatalf("expected topology-changed genome to create a new species, got=%q", sp3)
	}
	if len(population) != 2 {
		t.Fatalf("expected 2 species after distinct topology, got=%d", len(population))
	}
	if nextCalls != 2 {
		t.Fatalf("expected species-id generation call only on new species creation, got=%d", nextCalls)
	}
}

func TestSpeciateInPopulationUsesProvidedSpeciesIDAndSkipsTestGenome(t *testing.T) {
	genome := model.Genome{
		ID:          "g1",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Enabled: true},
		},
	}
	population := map[string]FingerprintSpecies{}
	key, population := SpeciateInPopulation(genome, population, func(_ string) string { return "species:custom" })
	if key != "species:custom" {
		t.Fatalf("expected explicit custom species id, got=%q", key)
	}
	if len(population[key].GenomeIDs) != 1 || population[key].GenomeIDs[0] != "g1" {
		t.Fatalf("expected custom species member to include g1, got=%v", population[key].GenomeIDs)
	}

	key2, population := SpeciateInPopulation(model.Genome{ID: "test"}, population, func(_ string) string { return "ignored" })
	if key2 != "" {
		t.Fatalf("expected empty species id for test genome, got=%q", key2)
	}
	if len(population) != 1 {
		t.Fatalf("expected population species unchanged for test genome, got=%d", len(population))
	}
}
