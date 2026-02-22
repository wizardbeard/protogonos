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
