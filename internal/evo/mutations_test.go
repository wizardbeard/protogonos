package evo

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/storage"
	"protogonos/internal/substrate"
	"protogonos/internal/tuning"
)

func TestPerturbWeightAtMatchesFixture(t *testing.T) {
	input := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "minimal_genome_v1.json"))
	expected := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_perturb_weight_v1.json"))

	op := PerturbWeightAt{Index: 0, Delta: 0.25}
	actual, err := op.Apply(context.Background(), input)
	if err != nil {
		t.Fatalf("apply operator: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("mutation mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestChangeActivationAtMatchesFixture(t *testing.T) {
	input := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "minimal_genome_v1.json"))
	expected := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_change_activation_v1.json"))

	op := ChangeActivationAt{Index: 1, Activation: "relu"}
	actual, err := op.Apply(context.Background(), input)
	if err != nil {
		t.Fatalf("apply operator: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("mutation mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestAddSynapseMatchesFixture(t *testing.T) {
	input := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "minimal_genome_v1.json"))
	expected := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_add_synapse_v1.json"))

	op := AddSynapse{ID: "s-2", From: "n-output", To: "n-input", Weight: -0.5, Enabled: true}
	actual, err := op.Apply(context.Background(), input)
	if err != nil {
		t.Fatalf("apply operator: %v", err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("mutation mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestRemoveSynapseMatchesFixture(t *testing.T) {
	input := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "minimal_genome_v1.json"))
	expected := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_remove_synapse_v1.json"))

	op := RemoveSynapse{ID: "s-1"}
	actual, err := op.Apply(context.Background(), input)
	if err != nil {
		t.Fatalf("apply operator: %v", err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("mutation mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestAddNeuronMatchesFixture(t *testing.T) {
	input := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "minimal_genome_v1.json"))
	expected := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_add_neuron_v1.json"))

	op := AddNeuronAtSynapse{SynapseIndex: 0, NeuronID: "n-hidden", Activation: "relu", Bias: 0}
	actual, err := op.Apply(context.Background(), input)
	if err != nil {
		t.Fatalf("apply operator: %v", err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("mutation mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestRemoveNeuronMatchesFixture(t *testing.T) {
	input := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_add_neuron_v1.json"))
	expected := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_remove_neuron_v1.json"))

	op := RemoveNeuron{ID: "n-hidden"}
	actual, err := op.Apply(context.Background(), input)
	if err != nil {
		t.Fatalf("apply operator: %v", err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("mutation mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestAddNeuronOnIOXORGenomeMatchesFixture(t *testing.T) {
	input := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "io_xor_genome_v1.json"))
	expected := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_io_xor_add_neuron_v1.json"))

	op := AddNeuronAtSynapse{SynapseIndex: 0, NeuronID: "h1", Activation: "relu", Bias: 0}
	actual, err := op.Apply(context.Background(), input)
	if err != nil {
		t.Fatalf("apply operator: %v", err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("mutation mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestRemoveSynapseOnIOXORGenomeMatchesFixture(t *testing.T) {
	input := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "io_xor_genome_v1.json"))
	expected := decodeGenomeFixture(t, filepath.Join("..", "..", "testdata", "fixtures", "mutations", "expected_io_xor_remove_synapse_v1.json"))

	op := RemoveSynapse{ID: "s2"}
	actual, err := op.Apply(context.Background(), input)
	if err != nil {
		t.Fatalf("apply operator: %v", err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("mutation mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestPerturbWeightAtInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(7))

	for i := 0; i < 200; i++ {
		genome := randomGenome(rng)
		idx := rng.Intn(len(genome.Synapses))
		delta := (rng.Float64() * 2) - 1

		op := PerturbWeightAt{Index: idx, Delta: delta}
		mutated, err := op.Apply(context.Background(), genome)
		if err != nil {
			t.Fatalf("apply failed: %v", err)
		}

		if len(mutated.Synapses) != len(genome.Synapses) {
			t.Fatalf("synapse count changed: got=%d want=%d", len(mutated.Synapses), len(genome.Synapses))
		}
		if len(mutated.Neurons) != len(genome.Neurons) {
			t.Fatalf("neuron count changed: got=%d want=%d", len(mutated.Neurons), len(genome.Neurons))
		}

		for j := range genome.Synapses {
			before := genome.Synapses[j]
			after := mutated.Synapses[j]
			if before.ID != after.ID || before.From != after.From || before.To != after.To || before.Enabled != after.Enabled || before.Recurrent != after.Recurrent {
				t.Fatalf("topology changed for synapse %d", j)
			}
			if math.IsNaN(after.Weight) || math.IsInf(after.Weight, 0) {
				t.Fatalf("invalid mutated weight at synapse %d", j)
			}
		}

		if mutated.Synapses[idx].Weight != genome.Synapses[idx].Weight+delta {
			t.Fatalf("weight delta not applied at index %d", idx)
		}
	}
}

func TestChangeActivationAtInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	activations := []string{"identity", "relu", "tanh", "sigmoid"}

	for i := 0; i < 200; i++ {
		genome := randomGenome(rng)
		idx := rng.Intn(len(genome.Neurons))
		nextActivation := activations[rng.Intn(len(activations))]

		op := ChangeActivationAt{Index: idx, Activation: nextActivation}
		mutated, err := op.Apply(context.Background(), genome)
		if err != nil {
			t.Fatalf("apply failed: %v", err)
		}

		if len(mutated.Synapses) != len(genome.Synapses) {
			t.Fatalf("synapse count changed: got=%d want=%d", len(mutated.Synapses), len(genome.Synapses))
		}
		if len(mutated.Neurons) != len(genome.Neurons) {
			t.Fatalf("neuron count changed: got=%d want=%d", len(mutated.Neurons), len(genome.Neurons))
		}

		for j := range genome.Neurons {
			before := genome.Neurons[j]
			after := mutated.Neurons[j]
			if before.ID != after.ID || before.Bias != after.Bias {
				t.Fatalf("neuron identity changed at index %d", j)
			}
		}

		if mutated.Neurons[idx].Activation != nextActivation {
			t.Fatalf("activation not applied at index %d", idx)
		}
	}
}

func TestAddSynapseInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for i := 0; i < 200; i++ {
		genome := randomGenome(rng)
		from := genome.Neurons[rng.Intn(len(genome.Neurons))].ID
		to := genome.Neurons[rng.Intn(len(genome.Neurons))].ID

		op := AddSynapse{
			ID:      "s-new-" + strconv.Itoa(i),
			From:    from,
			To:      to,
			Weight:  (rng.Float64() * 2) - 1,
			Enabled: true,
		}
		mutated, err := op.Apply(context.Background(), genome)
		if err != nil {
			t.Fatalf("apply failed: %v", err)
		}

		if len(mutated.Neurons) != len(genome.Neurons) {
			t.Fatalf("neuron count changed")
		}
		if len(mutated.Synapses) != len(genome.Synapses)+1 {
			t.Fatalf("synapse count mismatch")
		}
		assertNoDanglingSynapses(t, mutated)
	}
}

func TestAddRandomSynapseRejectsDuplicateWhenFullyConnected(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "n1", To: "n1", Weight: 1, Enabled: true, Recurrent: true},
		},
	}
	op := &AddRandomSynapse{
		Rand:         rand.New(rand.NewSource(271)),
		MaxAbsWeight: 1.0,
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrSynapseExists) {
		t.Fatalf("expected ErrSynapseExists, got %v", err)
	}
}

func TestRemoveSynapseInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for i := 0; i < 200; i++ {
		genome := randomGenome(rng)
		target := genome.Synapses[rng.Intn(len(genome.Synapses))]

		op := RemoveSynapse{ID: target.ID}
		mutated, err := op.Apply(context.Background(), genome)
		if err != nil {
			t.Fatalf("apply failed: %v", err)
		}

		if len(mutated.Neurons) != len(genome.Neurons) {
			t.Fatalf("neuron count changed")
		}
		if len(mutated.Synapses) != len(genome.Synapses)-1 {
			t.Fatalf("synapse count mismatch")
		}
		if hasSynapse(mutated, target.ID) {
			t.Fatalf("synapse still present after remove: %s", target.ID)
		}
		assertNoDanglingSynapses(t, mutated)
	}
}

func TestAddNeuronInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	for i := 0; i < 200; i++ {
		genome := randomGenome(rng)
		idx := rng.Intn(len(genome.Synapses))
		op := AddNeuronAtSynapse{
			SynapseIndex: idx,
			NeuronID:     "n-added-" + strconv.Itoa(i),
			Activation:   "relu",
			Bias:         0,
		}
		mutated, err := op.Apply(context.Background(), genome)
		if err != nil {
			t.Fatalf("apply failed: %v", err)
		}

		if len(mutated.Neurons) != len(genome.Neurons)+1 {
			t.Fatalf("neuron count mismatch")
		}
		if len(mutated.Synapses) != len(genome.Synapses)+1 {
			t.Fatalf("synapse count mismatch")
		}
		assertNoDanglingSynapses(t, mutated)
	}
}

func TestRemoveNeuronInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	for i := 0; i < 200; i++ {
		genome := randomGenome(rng)
		target := genome.Neurons[rng.Intn(len(genome.Neurons))].ID

		op := RemoveNeuron{ID: target}
		mutated, err := op.Apply(context.Background(), genome)
		if err != nil {
			t.Fatalf("apply failed: %v", err)
		}
		if len(mutated.Neurons) != len(genome.Neurons)-1 {
			t.Fatalf("neuron count mismatch")
		}
		if hasNeuron(mutated, target) {
			t.Fatalf("neuron still present after remove: %s", target)
		}
		assertNoDanglingSynapses(t, mutated)
	}
}

func TestRemoveRandomNeuronCancelsWhenAllProtected(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
			{ID: "n2", Activation: "identity"},
		},
	}
	op := &RemoveRandomNeuron{
		Rand: rand.New(rand.NewSource(341)),
		Protected: map[string]struct{}{
			"n1": {},
			"n2": {},
		},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected remove_random_neuron to be inapplicable when all neurons are protected")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice when all neurons are protected, got %v", err)
	}
}

func TestPerturbPlasticityRateMutation(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(1)))
	genome.Plasticity = &model.PlasticityConfig{
		Rule:            "hebbian",
		Rate:            0.4,
		SaturationLimit: 1.0,
	}
	op := &PerturbPlasticityRate{
		Rand:     rand.New(rand.NewSource(2)),
		MaxDelta: 0.2,
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if mutated.Plasticity == nil {
		t.Fatal("expected plasticity config on mutated genome")
	}
	if mutated.Plasticity.Rate == genome.Plasticity.Rate {
		t.Fatal("expected plasticity rate to change")
	}
	if mutated.Plasticity.Rate < 0 {
		t.Fatalf("expected non-negative plasticity rate, got=%f", mutated.Plasticity.Rate)
	}
}

func TestPerturbWeightsProportionalMutation(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(41)))
	op := &PerturbWeightsProportional{
		Rand:     rand.New(rand.NewSource(42)),
		MaxDelta: 0.5,
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}

	if len(mutated.Synapses) != len(genome.Synapses) {
		t.Fatalf("synapse count changed: got=%d want=%d", len(mutated.Synapses), len(genome.Synapses))
	}
	if len(mutated.Neurons) != len(genome.Neurons) {
		t.Fatalf("neuron count changed: got=%d want=%d", len(mutated.Neurons), len(genome.Neurons))
	}

	changed := 0
	for i := range genome.Synapses {
		if mutated.Synapses[i].Weight != genome.Synapses[i].Weight {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("expected at least one perturbed weight")
	}
}

func TestPerturbWeightsProportionalNoSynapses(t *testing.T) {
	op := &PerturbWeightsProportional{
		Rand:     rand.New(rand.NewSource(7)),
		MaxDelta: 0.5,
	}
	_, err := op.Apply(context.Background(), model.Genome{})
	if !errors.Is(err, ErrNoSynapses) {
		t.Fatalf("expected ErrNoSynapses, got=%v", err)
	}
}

func TestMutateWeightsUsesAllSelectionAcrossNeurons(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
			{ID: "n2", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "n1", To: "n1", Weight: 0.0, Enabled: true},
			{ID: "s2", From: "n2", To: "n2", Weight: 0.0, Enabled: true},
		},
		Strategy: &model.StrategyConfig{
			TuningSelection: tuning.CandidateSelectAll,
			AnnealingFactor: 1.0,
		},
	}
	op := &MutateWeights{
		Rand:     rand.New(rand.NewSource(43)),
		MaxDelta: 0.5,
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if mutated.Synapses[0].Weight == genome.Synapses[0].Weight {
		t.Fatal("expected synapse targeting n1 to mutate")
	}
	if mutated.Synapses[1].Weight == genome.Synapses[1].Weight {
		t.Fatal("expected synapse targeting n2 to mutate")
	}
}

func TestMutateWeightsRandomSelectionMutatesSubset(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
			{ID: "n2", Activation: "identity"},
			{ID: "n3", Activation: "identity"},
			{ID: "n4", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "n1", To: "n1", Weight: 0.0, Enabled: true},
			{ID: "s2", From: "n2", To: "n2", Weight: 0.0, Enabled: true},
			{ID: "s3", From: "n3", To: "n3", Weight: 0.0, Enabled: true},
			{ID: "s4", From: "n4", To: "n4", Weight: 0.0, Enabled: true},
		},
		Strategy: &model.StrategyConfig{
			TuningSelection: tuning.CandidateSelectAllRandom,
			AnnealingFactor: 1.0,
		},
	}
	op := &MutateWeights{
		Rand:     rand.New(rand.NewSource(7)),
		MaxDelta: 0.5,
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	changed := 0
	for i := range genome.Synapses {
		if mutated.Synapses[i].Weight != genome.Synapses[i].Weight {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("expected at least one mutated synapse")
	}
	if changed >= len(genome.Synapses) {
		t.Fatalf("expected random selection to mutate a strict subset, changed=%d total=%d", changed, len(genome.Synapses))
	}
}

func TestAddRandomInlinkPrefersInputSource(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
	}
	op := &AddRandomInlink{
		Rand:           rand.New(rand.NewSource(17)),
		MaxAbsWeight:   1.0,
		InputNeuronIDs: []string{"i1"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Synapses) != 1 {
		t.Fatalf("expected one added synapse, got=%d", len(mutated.Synapses))
	}
	if mutated.Synapses[0].From != "i1" {
		t.Fatalf("expected inlink source i1, got=%s", mutated.Synapses[0].From)
	}
	if mutated.Synapses[0].To == "i1" {
		t.Fatalf("expected inlink target to be non-input neuron, got=%s", mutated.Synapses[0].To)
	}
}

func TestAddRandomInlinkNoDirectionalCandidates(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
	}
	op := &AddRandomInlink{
		Rand:           rand.New(rand.NewSource(19)),
		MaxAbsWeight:   1.0,
		InputNeuronIDs: []string{"i1"},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected add_inlink to be inapplicable without directional candidates")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}
}

func TestAddRandomOutlinkTargetsOutput(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
	}
	op := &AddRandomOutlink{
		Rand:            rand.New(rand.NewSource(23)),
		MaxAbsWeight:    1.0,
		OutputNeuronIDs: []string{"o1"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Synapses) != 1 {
		t.Fatalf("expected one added synapse, got=%d", len(mutated.Synapses))
	}
	if mutated.Synapses[0].To != "o1" {
		t.Fatalf("expected outlink target o1, got=%s", mutated.Synapses[0].To)
	}
}

func TestAddRandomOutlinkNoDirectionalCandidates(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
		},
	}
	op := &AddRandomOutlink{
		Rand:            rand.New(rand.NewSource(29)),
		MaxAbsWeight:    1.0,
		OutputNeuronIDs: []string{"o1"},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected add_outlink to be inapplicable without directional candidates")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}
}

func TestAddRandomInlinkRejectsDuplicateDirectionalEdge(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{ID: "s_existing", From: "i1", To: "h1", Weight: 1, Enabled: true},
		},
	}
	op := &AddRandomInlink{
		Rand:           rand.New(rand.NewSource(313)),
		MaxAbsWeight:   1.0,
		InputNeuronIDs: []string{"i1"},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected add_inlink to be inapplicable when only duplicate edge candidate exists")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}
}

func TestAddRandomOutlinkRejectsDuplicateDirectionalEdge(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_existing", From: "h1", To: "o1", Weight: 1, Enabled: true},
		},
	}
	op := &AddRandomOutlink{
		Rand:            rand.New(rand.NewSource(317)),
		MaxAbsWeight:    1.0,
		OutputNeuronIDs: []string{"o1"},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected add_outlink to be inapplicable when only duplicate edge candidate exists")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}
}

func TestDirectionalMutationsRespectFeedforwardLayers(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_backward", From: "h1", To: "i1", Weight: 1, Enabled: true},
			{ID: "s_forward", From: "i1", To: "h1", Weight: 1, Enabled: true},
		},
	}

	addIn := &AddRandomInlink{
		Rand:            rand.New(rand.NewSource(211)),
		MaxAbsWeight:    1.0,
		InputNeuronIDs:  []string{"i1"},
		FeedForwardOnly: true,
	}
	added, err := addIn.Apply(context.Background(), model.Genome{Neurons: genome.Neurons})
	if err != nil {
		t.Fatalf("apply add_inlink failed: %v", err)
	}
	if len(added.Synapses) != 1 || added.Synapses[0].From != "i1" || added.Synapses[0].To == "i1" {
		t.Fatalf("unexpected feedforward add_inlink result: %+v", added.Synapses)
	}

	removeIn := &RemoveRandomInlink{
		Rand:            rand.New(rand.NewSource(223)),
		InputNeuronIDs:  []string{"i1"},
		FeedForwardOnly: true,
	}
	removed, err := removeIn.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply remove_inlink failed: %v", err)
	}
	if hasSynapse(removed, "s_forward") {
		t.Fatal("expected remove_inlink to remove only forward input edge")
	}
	if !hasSynapse(removed, "s_backward") {
		t.Fatal("expected non-feedforward edge to remain")
	}
}

func TestSpliceMutationsRespectFeedforwardLayers(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_backward", From: "h1", To: "i1", Weight: 1, Enabled: true},
			{ID: "s_out", From: "h1", To: "o1", Weight: 1, Enabled: true},
		},
	}
	op := &AddRandomOutsplice{
		Rand:            rand.New(rand.NewSource(227)),
		OutputNeuronIDs: []string{"o1"},
		FeedForwardOnly: true,
		Activations:     []string{"relu"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply outsplice failed: %v", err)
	}
	if hasSynapse(mutated, "s_out") {
		t.Fatal("expected feedforward out edge to be spliced")
	}
	if !hasSynapse(mutated, "s_backward") {
		t.Fatal("expected backward edge to remain untouched")
	}
}

func TestFeedforwardDirectionalCancellationWhenOnlyBackwardOrderingExists(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{ID: "s_backward", From: "h1", To: "i1", Weight: 1, Enabled: true},
			{ID: "s_forward", From: "i1", To: "h1", Weight: 1, Enabled: true},
		},
	}
	addIn := &AddRandomInlink{
		Rand:            rand.New(rand.NewSource(229)),
		MaxAbsWeight:    1.0,
		InputNeuronIDs:  []string{"i1"},
		FeedForwardOnly: true,
	}
	if addIn.Applicable(genome, "xor") {
		t.Fatal("expected add_inlink to be inapplicable when inferred feedforward ordering forbids candidates")
	}
	if _, err := addIn.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}

	removeIn := &RemoveRandomInlink{
		Rand:            rand.New(rand.NewSource(233)),
		InputNeuronIDs:  []string{"i1"},
		FeedForwardOnly: true,
	}
	if !removeIn.Applicable(genome, "xor") {
		t.Fatal("expected remove_inlink to remain applicable while feedforward-directed edges exist")
	}
	removed, err := removeIn.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("expected remove_inlink apply to succeed, got %v", err)
	}
	if hasSynapse(removed, "s_forward") {
		t.Fatal("expected feedforward input edge to be removed")
	}
}

func TestAddRandomOutsplicePrefersOutputEdge(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_in", From: "i1", To: "h1", Weight: 1, Enabled: true},
			{ID: "s_out", From: "h1", To: "o1", Weight: 1, Enabled: true},
		},
	}
	op := &AddRandomOutsplice{
		Rand:            rand.New(rand.NewSource(107)),
		OutputNeuronIDs: []string{"o1"},
		Activations:     []string{"relu"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if hasSynapse(mutated, "s_out") {
		t.Fatal("expected output-edge synapse to be spliced out")
	}
	if !hasSynapse(mutated, "s_in") {
		t.Fatal("expected non-output-edge synapse to remain")
	}
}

func TestAddRandomOutspliceNoDirectionalCandidates(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_in", From: "i1", To: "h1", Weight: 1, Enabled: true},
		},
	}
	op := &AddRandomOutsplice{
		Rand:            rand.New(rand.NewSource(111)),
		OutputNeuronIDs: []string{"o1"},
		Activations:     []string{"relu"},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected outsplice to be inapplicable without output-directed edges")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}
}

func TestAddRandomInsplicePrefersInputEdge(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_in", From: "i1", To: "h1", Weight: 1, Enabled: true},
			{ID: "s_out", From: "h1", To: "o1", Weight: 1, Enabled: true},
		},
	}
	op := &AddRandomInsplice{
		Rand:           rand.New(rand.NewSource(109)),
		InputNeuronIDs: []string{"i1"},
		Activations:    []string{"relu"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if hasSynapse(mutated, "s_in") {
		t.Fatal("expected input-edge synapse to be spliced out")
	}
	if !hasSynapse(mutated, "s_out") {
		t.Fatal("expected non-input-edge synapse to remain")
	}
}

func TestAddRandomInspliceNoDirectionalCandidates(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_out", From: "h1", To: "o1", Weight: 1, Enabled: true},
		},
	}
	op := &AddRandomInsplice{
		Rand:           rand.New(rand.NewSource(113)),
		InputNeuronIDs: []string{"i1"},
		Activations:    []string{"relu"},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected insplice to be inapplicable without input-directed edges")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}
}

func TestRemoveRandomInlinkPrefersInputSource(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_in", From: "i1", To: "h1", Weight: 1, Enabled: true},
			{ID: "s_other", From: "h1", To: "o1", Weight: 1, Enabled: true},
		},
	}
	op := &RemoveRandomInlink{
		Rand:           rand.New(rand.NewSource(31)),
		InputNeuronIDs: []string{"i1"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Synapses) != 1 {
		t.Fatalf("expected one removed synapse, got=%d", len(genome.Synapses)-len(mutated.Synapses))
	}
	if hasSynapse(mutated, "s_in") {
		t.Fatalf("expected input-oriented synapse to be removed")
	}
}

func TestRemoveRandomInlinkNoDirectionalCandidates(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{ID: "s_other", From: "h1", To: "i1", Weight: 1, Enabled: true},
		},
	}
	op := &RemoveRandomInlink{
		Rand:           rand.New(rand.NewSource(33)),
		InputNeuronIDs: []string{"i1"},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected remove_inlink to be inapplicable without matching edges")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}
}

func TestRemoveRandomOutlinkTargetsOutput(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_out", From: "h1", To: "o1", Weight: 1, Enabled: true},
			{ID: "s_other", From: "i1", To: "h1", Weight: 1, Enabled: true},
		},
	}
	op := &RemoveRandomOutlink{
		Rand:            rand.New(rand.NewSource(37)),
		OutputNeuronIDs: []string{"o1"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Synapses) != 1 {
		t.Fatalf("expected one removed synapse, got=%d", len(genome.Synapses)-len(mutated.Synapses))
	}
	if hasSynapse(mutated, "s_out") {
		t.Fatalf("expected output-oriented synapse to be removed")
	}
}

func TestRemoveRandomOutlinkNoDirectionalCandidates(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "h1", Activation: "tanh"},
			{ID: "o1", Activation: "sigmoid"},
		},
		Synapses: []model.Synapse{
			{ID: "s_other", From: "o1", To: "h1", Weight: 1, Enabled: true},
		},
	}
	op := &RemoveRandomOutlink{
		Rand:            rand.New(rand.NewSource(39)),
		OutputNeuronIDs: []string{"o1"},
	}
	if op.Applicable(genome, "xor") {
		t.Fatal("expected remove_outlink to be inapplicable without matching edges")
	}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice, got %v", err)
	}
}

func TestAddRandomSensorAddsCompatibleSensor(t *testing.T) {
	genome := model.Genome{
		Neurons:     []model.Neuron{{ID: "i", Activation: "identity"}, {ID: "o", Activation: "identity"}},
		SensorIDs:   []string{protoio.XORInputLeftSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
	}
	op := &AddRandomSensor{
		Rand:      rand.New(rand.NewSource(73)),
		ScapeName: "xor",
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.SensorIDs) != len(genome.SensorIDs)+1 {
		t.Fatalf("expected one added sensor, got=%d", len(mutated.SensorIDs)-len(genome.SensorIDs))
	}
	if mutated.SensorIDs[len(mutated.SensorIDs)-1] != protoio.XORInputRightSensorName {
		t.Fatalf("expected xor right sensor to be added, got=%s", mutated.SensorIDs[len(mutated.SensorIDs)-1])
	}
	if len(mutated.SensorNeuronLinks) != 1 {
		t.Fatalf("expected one sensor-neuron link for new sensor, got=%d", len(mutated.SensorNeuronLinks))
	}
	if mutated.SensorNeuronLinks[0].SensorID != protoio.XORInputRightSensorName {
		t.Fatalf("expected new sensor id on link, got=%+v", mutated.SensorNeuronLinks[0])
	}
	if mutated.SensorLinks != 1 {
		t.Fatalf("expected synchronized sensor link counter, got=%d", mutated.SensorLinks)
	}
}

func TestAddRandomActuatorAddsCompatibleActuator(t *testing.T) {
	genome := model.Genome{
		Neurons:     []model.Neuron{{ID: "i", Activation: "identity"}, {ID: "o", Activation: "identity"}},
		SensorIDs:   []string{protoio.FXPriceSensorName, protoio.FXSignalSensorName},
		ActuatorIDs: []string{},
	}
	op := &AddRandomActuator{
		Rand:      rand.New(rand.NewSource(79)),
		ScapeName: "fx",
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.ActuatorIDs) != 1 {
		t.Fatalf("expected one added actuator, got=%d", len(mutated.ActuatorIDs))
	}
	if mutated.ActuatorIDs[0] != protoio.FXTradeActuatorName {
		t.Fatalf("expected fx trade actuator to be added, got=%s", mutated.ActuatorIDs[0])
	}
	if len(mutated.NeuronActuatorLinks) != 1 {
		t.Fatalf("expected one neuron-actuator link for new actuator, got=%d", len(mutated.NeuronActuatorLinks))
	}
	if mutated.NeuronActuatorLinks[0].ActuatorID != protoio.FXTradeActuatorName {
		t.Fatalf("expected new actuator id on link, got=%+v", mutated.NeuronActuatorLinks[0])
	}
	if mutated.ActuatorLinks != 1 {
		t.Fatalf("expected synchronized actuator link counter, got=%d", mutated.ActuatorLinks)
	}
}

func TestAddRandomSensorAndActuatorRequireNeuronsForConnection(t *testing.T) {
	sensorGenome := model.Genome{
		Neurons:   nil,
		SensorIDs: []string{protoio.XORInputLeftSensorName},
	}
	sensorOp := &AddRandomSensor{
		Rand:      rand.New(rand.NewSource(261)),
		ScapeName: "xor",
	}
	if sensorOp.Applicable(sensorGenome, "xor") {
		t.Fatal("expected add_sensor to be inapplicable without neurons")
	}
	if _, err := sensorOp.Apply(context.Background(), sensorGenome); !errors.Is(err, ErrNoNeurons) {
		t.Fatalf("expected ErrNoNeurons for add_sensor without neurons, got %v", err)
	}

	actuatorGenome := model.Genome{
		Neurons:     nil,
		ActuatorIDs: []string{},
	}
	actuatorOp := &AddRandomActuator{
		Rand:      rand.New(rand.NewSource(269)),
		ScapeName: "xor",
	}
	if actuatorOp.Applicable(actuatorGenome, "xor") {
		t.Fatal("expected add_actuator to be inapplicable without neurons")
	}
	if _, err := actuatorOp.Apply(context.Background(), actuatorGenome); !errors.Is(err, ErrNoNeurons) {
		t.Fatalf("expected ErrNoNeurons for add_actuator without neurons, got %v", err)
	}
}

func TestAddRandomSensorAndActuatorCancelWhenNoCompatibleCandidates(t *testing.T) {
	sensorGenome := model.Genome{
		Neurons:   []model.Neuron{{ID: "n1", Activation: "identity"}},
		SensorIDs: []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
	}
	if _, err := (&AddRandomSensor{
		Rand:      rand.New(rand.NewSource(271)),
		ScapeName: "xor",
	}).Apply(context.Background(), sensorGenome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for exhausted sensor candidates, got %v", err)
	}

	actuatorGenome := model.Genome{
		Neurons:     []model.Neuron{{ID: "n1", Activation: "identity"}},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
	}
	if _, err := (&AddRandomActuator{
		Rand:      rand.New(rand.NewSource(273)),
		ScapeName: "xor",
	}).Apply(context.Background(), actuatorGenome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for exhausted actuator candidates, got %v", err)
	}
}

func TestAddRandomSensorLinkIncrementsUntilCapacity(t *testing.T) {
	genome := model.Genome{
		Neurons:   []model.Neuron{{ID: "n1", Activation: "identity"}},
		SensorIDs: []string{protoio.XORInputLeftSensorName},
	}
	op := &AddRandomSensorLink{
		Rand: rand.New(rand.NewSource(83)),
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if mutated.SensorLinks != 1 {
		t.Fatalf("expected one added sensor link count, got=%d", mutated.SensorLinks)
	}
	if len(mutated.SensorNeuronLinks) != 1 {
		t.Fatalf("expected one explicit sensor-neuron link, got=%d", len(mutated.SensorNeuronLinks))
	}
	if mutated.SensorNeuronLinks[0].SensorID != protoio.XORInputLeftSensorName || mutated.SensorNeuronLinks[0].NeuronID != "n1" {
		t.Fatalf("unexpected sensor-neuron link: %+v", mutated.SensorNeuronLinks[0])
	}
	if len(mutated.SensorIDs) != 1 {
		t.Fatalf("expected sensor components unchanged, got=%v", mutated.SensorIDs)
	}
	if op.Applicable(mutated, "xor") {
		t.Fatal("expected add_sensorlink to be inapplicable after reaching full connectivity")
	}
	if _, err := op.Apply(context.Background(), mutated); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice when fully connected, got %v", err)
	}
}

func TestAddRandomActuatorLinkIncrementsUntilCapacity(t *testing.T) {
	genome := model.Genome{
		Neurons:     []model.Neuron{{ID: "n1", Activation: "identity"}},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
	}
	op := &AddRandomActuatorLink{
		Rand: rand.New(rand.NewSource(89)),
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if mutated.ActuatorLinks != 1 {
		t.Fatalf("expected one added actuator link count, got=%d", mutated.ActuatorLinks)
	}
	if len(mutated.NeuronActuatorLinks) != 1 {
		t.Fatalf("expected one explicit neuron-actuator link, got=%d", len(mutated.NeuronActuatorLinks))
	}
	if mutated.NeuronActuatorLinks[0].NeuronID != "n1" || mutated.NeuronActuatorLinks[0].ActuatorID != protoio.XOROutputActuatorName {
		t.Fatalf("unexpected neuron-actuator link: %+v", mutated.NeuronActuatorLinks[0])
	}
	if len(mutated.ActuatorIDs) != 1 {
		t.Fatalf("expected actuator components unchanged, got=%v", mutated.ActuatorIDs)
	}
	if op.Applicable(mutated, "xor") {
		t.Fatal("expected add_actuatorlink to be inapplicable after reaching full connectivity")
	}
	if _, err := op.Apply(context.Background(), mutated); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice when fully connected, got %v", err)
	}
}

func TestRemoveRandomSensorRemovesOneSensor(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
	}
	op := &RemoveRandomSensor{Rand: rand.New(rand.NewSource(113))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.SensorIDs) != 1 {
		t.Fatalf("expected one remaining sensor, got=%d", len(mutated.SensorIDs))
	}
}

func TestRemoveRandomSensorAndActuatorCancelWhenEmpty(t *testing.T) {
	if _, err := (&RemoveRandomSensor{Rand: rand.New(rand.NewSource(281))}).Apply(context.Background(), model.Genome{}); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for empty sensor removal, got %v", err)
	}
	if _, err := (&RemoveRandomActuator{Rand: rand.New(rand.NewSource(283))}).Apply(context.Background(), model.Genome{}); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for empty actuator removal, got %v", err)
	}
}

func TestRemoveRandomSensorRemovesAllLinksForSelectedSensor(t *testing.T) {
	genome := model.Genome{
		Neurons:   []model.Neuron{{ID: "n1", Activation: "identity"}, {ID: "n2", Activation: "identity"}},
		SensorIDs: []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: protoio.XORInputLeftSensorName, NeuronID: "n1"},
			{SensorID: protoio.XORInputLeftSensorName, NeuronID: "n2"},
			{SensorID: protoio.XORInputRightSensorName, NeuronID: "n1"},
		},
		SensorLinks: 3,
	}
	op := &RemoveRandomSensor{Rand: rand.New(rand.NewSource(97))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.SensorIDs) != 1 {
		t.Fatalf("expected selected sensor to be removed with all links, got=%v", mutated.SensorIDs)
	}
	removedSensorID := protoio.XORInputLeftSensorName
	if mutated.SensorIDs[0] == protoio.XORInputLeftSensorName {
		removedSensorID = protoio.XORInputRightSensorName
	}
	for _, link := range mutated.SensorNeuronLinks {
		if link.SensorID == removedSensorID {
			t.Fatalf("found dangling sensor link for removed sensor %q: %+v", removedSensorID, link)
		}
	}
	if mutated.SensorLinks != len(mutated.SensorNeuronLinks) {
		t.Fatalf("expected sensor links count to match explicit links, count=%d explicit=%d", mutated.SensorLinks, len(mutated.SensorNeuronLinks))
	}
}

func TestRemoveRandomActuatorRemovesOneActuator(t *testing.T) {
	genome := model.Genome{
		ActuatorIDs: []string{protoio.XOROutputActuatorName, protoio.FXTradeActuatorName},
	}
	op := &RemoveRandomActuator{Rand: rand.New(rand.NewSource(127))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.ActuatorIDs) != 1 {
		t.Fatalf("expected one remaining actuator, got=%d", len(mutated.ActuatorIDs))
	}
}

func TestRemoveRandomActuatorRemovesAllLinksForSelectedActuator(t *testing.T) {
	genome := model.Genome{
		Neurons:     []model.Neuron{{ID: "n1", Activation: "identity"}, {ID: "n2", Activation: "identity"}},
		ActuatorIDs: []string{protoio.XOROutputActuatorName, protoio.FXTradeActuatorName},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "n1", ActuatorID: protoio.XOROutputActuatorName},
			{NeuronID: "n2", ActuatorID: protoio.XOROutputActuatorName},
			{NeuronID: "n1", ActuatorID: protoio.FXTradeActuatorName},
		},
		ActuatorLinks: 3,
	}
	op := &RemoveRandomActuator{Rand: rand.New(rand.NewSource(101))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.ActuatorIDs) != 1 {
		t.Fatalf("expected selected actuator to be removed with all links, got=%v", mutated.ActuatorIDs)
	}
	removedActuatorID := protoio.XOROutputActuatorName
	if mutated.ActuatorIDs[0] == protoio.XOROutputActuatorName {
		removedActuatorID = protoio.FXTradeActuatorName
	}
	for _, link := range mutated.NeuronActuatorLinks {
		if link.ActuatorID == removedActuatorID {
			t.Fatalf("found dangling actuator link for removed actuator %q: %+v", removedActuatorID, link)
		}
	}
	if mutated.ActuatorLinks != len(mutated.NeuronActuatorLinks) {
		t.Fatalf("expected actuator links count to match explicit links, count=%d explicit=%d", mutated.ActuatorLinks, len(mutated.NeuronActuatorLinks))
	}
}

func TestCutlinkAliasesRemoveSensorAndActuatorLinks(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName, protoio.FXTradeActuatorName},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: protoio.XORInputLeftSensorName, NeuronID: "n1"},
			{SensorID: protoio.XORInputRightSensorName, NeuronID: "n2"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "n1", ActuatorID: protoio.XOROutputActuatorName},
			{NeuronID: "n2", ActuatorID: protoio.FXTradeActuatorName},
		},
		SensorLinks:   2,
		ActuatorLinks: 2,
	}

	mutatedSensor, err := (&CutlinkFromSensorToNeuron{Rand: rand.New(rand.NewSource(139))}).Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("cutlink sensor apply failed: %v", err)
	}
	if mutatedSensor.SensorLinks != genome.SensorLinks-1 {
		t.Fatalf("expected one sensor link removed, before=%d after=%d", genome.SensorLinks, mutatedSensor.SensorLinks)
	}
	if len(mutatedSensor.SensorNeuronLinks) != len(genome.SensorNeuronLinks)-1 {
		t.Fatalf("expected one explicit sensor-neuron link removed, before=%d after=%d", len(genome.SensorNeuronLinks), len(mutatedSensor.SensorNeuronLinks))
	}
	if len(mutatedSensor.SensorIDs) != len(genome.SensorIDs) {
		t.Fatalf("expected sensor components unchanged by cutlink, before=%d after=%d", len(genome.SensorIDs), len(mutatedSensor.SensorIDs))
	}

	mutatedActuator, err := (&CutlinkFromNeuronToActuator{Rand: rand.New(rand.NewSource(149))}).Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("cutlink actuator apply failed: %v", err)
	}
	if mutatedActuator.ActuatorLinks != genome.ActuatorLinks-1 {
		t.Fatalf("expected one actuator link removed, before=%d after=%d", genome.ActuatorLinks, mutatedActuator.ActuatorLinks)
	}
	if len(mutatedActuator.NeuronActuatorLinks) != len(genome.NeuronActuatorLinks)-1 {
		t.Fatalf("expected one explicit neuron-actuator link removed, before=%d after=%d", len(genome.NeuronActuatorLinks), len(mutatedActuator.NeuronActuatorLinks))
	}
	if len(mutatedActuator.ActuatorIDs) != len(genome.ActuatorIDs) {
		t.Fatalf("expected actuator components unchanged by cutlink, before=%d after=%d", len(genome.ActuatorIDs), len(mutatedActuator.ActuatorIDs))
	}
}

func TestCutlinkAliasesCancelWhenNoEndpointLinks(t *testing.T) {
	empty := model.Genome{}
	if _, err := (&CutlinkFromSensorToNeuron{Rand: rand.New(rand.NewSource(1401))}).Apply(context.Background(), empty); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for empty sensor cutlink, got %v", err)
	}
	if _, err := (&CutlinkFromNeuronToActuator{Rand: rand.New(rand.NewSource(1403))}).Apply(context.Background(), empty); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for empty actuator cutlink, got %v", err)
	}
}

func TestCutlinkFromNeuronToNeuronRemovesSynapse(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
			{ID: "n2", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "n1", To: "n2", Weight: 0.5, Enabled: true},
			{ID: "s2", From: "n2", To: "n1", Weight: -0.5, Enabled: true},
		},
	}
	mutated, err := (&CutlinkFromNeuronToNeuron{Rand: rand.New(rand.NewSource(173))}).Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("cutlink neuron apply failed: %v", err)
	}
	if len(mutated.Synapses) != len(genome.Synapses)-1 {
		t.Fatalf("expected one neuron-neuron link removed, before=%d after=%d", len(genome.Synapses), len(mutated.Synapses))
	}
}

func TestCutlinkFromElementToElementRemovesSynapse(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i1", To: "h1", Weight: 1, Enabled: true},
		},
	}
	mutated, err := (&CutlinkFromElementToElement{Rand: rand.New(rand.NewSource(179))}).Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Synapses) != 0 {
		t.Fatalf("expected one removed synapse, got=%d", len(genome.Synapses)-len(mutated.Synapses))
	}
}

func TestLinkFromElementToElementAddsSynapse(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
		},
	}
	mutated, err := (&LinkFromElementToElement{
		Rand:         rand.New(rand.NewSource(181)),
		MaxAbsWeight: 1,
	}).Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Synapses) != 1 {
		t.Fatalf("expected one added synapse, got=%d", len(mutated.Synapses))
	}
}

func TestLinkFromNeuronToNeuronAddsSynapse(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "h1", Activation: "tanh"},
		},
	}
	mutated, err := (&LinkFromNeuronToNeuron{
		Rand:         rand.New(rand.NewSource(191)),
		MaxAbsWeight: 1,
	}).Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Synapses) != 1 {
		t.Fatalf("expected one added synapse, got=%d", len(mutated.Synapses))
	}
}

func TestAddRandomCPPRequiresSubstrateConfig(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
	}
	op := &AddRandomCPP{Rand: rand.New(rand.NewSource(83))}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice without substrate config, got %v", err)
	}
	withSubstrate := model.Genome{
		Neurons: genome.Neurons,
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: []int{2, 2},
			Parameters: map[string]float64{},
		},
	}
	mutated, err := op.Apply(context.Background(), withSubstrate)
	if len(availableCPPChoices(withSubstrate)) == 0 {
		if !errors.Is(err, ErrNoMutationChoice) {
			t.Fatalf("expected exhausted cpp choices error, got %v", err)
		}
	} else {
		if err != nil {
			t.Fatalf("apply with substrate failed: %v", err)
		}
		if mutated.Substrate == nil || mutated.Substrate.CPPName == "" {
			t.Fatal("expected cpp mutation on substrate-configured genome")
		}
	}
}

func TestAddRandomCEPRequiresSubstrateConfig(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
	}
	op := &AddRandomCEP{Rand: rand.New(rand.NewSource(89))}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice without substrate config, got %v", err)
	}
	withSubstrate := model.Genome{
		Neurons: genome.Neurons,
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: []int{2, 2},
			Parameters: map[string]float64{},
		},
	}
	mutated, err := op.Apply(context.Background(), withSubstrate)
	if len(availableCEPChoices(withSubstrate)) == 0 {
		if !errors.Is(err, ErrNoMutationChoice) {
			t.Fatalf("expected exhausted cep choices error, got %v", err)
		}
	} else {
		if err != nil {
			t.Fatalf("apply with substrate failed: %v", err)
		}
		if mutated.Substrate == nil || mutated.Substrate.CEPName == "" {
			t.Fatal("expected cep mutation on substrate-configured genome")
		}
	}
}

func TestAddRandomCPPAndCEPApplicable(t *testing.T) {
	if (&AddRandomCPP{}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected add cpp operator to be inapplicable without substrate config")
	}
	if (&AddRandomCEP{}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected add cep operator to be inapplicable without substrate config")
	}
	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: []int{2, 2},
			Parameters: map[string]float64{},
		},
	}
	if (&AddRandomCPP{}).Applicable(genome, "xor") != (len(availableCPPChoices(genome)) > 0) {
		t.Fatal("expected add cpp applicability to track alternative choice availability")
	}
	if (&AddRandomCEP{}).Applicable(genome, "xor") != (len(availableCEPChoices(genome)) > 0) {
		t.Fatal("expected add cep applicability to track alternative choice availability")
	}
	if substrate.DefaultCPPName == "" || substrate.DefaultCEPName == "" {
		t.Fatal("expected default substrate names")
	}
}

func TestAddRandomCPPAndCEPReturnErrorWhenNoAlternativeChoice(t *testing.T) {
	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: []int{2, 2},
			Parameters: map[string]float64{},
		},
	}
	if len(availableCPPChoices(genome)) == 0 {
		if _, err := (&AddRandomCPP{Rand: rand.New(rand.NewSource(241))}).Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
			t.Fatalf("expected ErrNoMutationChoice for cpp mutation exhaustion, got %v", err)
		}
	}
	if len(availableCEPChoices(genome)) == 0 {
		if _, err := (&AddRandomCEP{Rand: rand.New(rand.NewSource(251))}).Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
			t.Fatalf("expected ErrNoMutationChoice for cep mutation exhaustion, got %v", err)
		}
	}
}

func TestAddCircuitNodeMutatesDimensions(t *testing.T) {
	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: []int{2, 3, 1},
			Parameters: map[string]float64{},
		},
	}
	op := &AddCircuitNode{Rand: rand.New(rand.NewSource(97))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Substrate.Dimensions) != 3 {
		t.Fatalf("expected same layer count, got=%d", len(mutated.Substrate.Dimensions))
	}
	before := 0
	after := 0
	for _, d := range genome.Substrate.Dimensions {
		before += d
	}
	for _, d := range mutated.Substrate.Dimensions {
		after += d
	}
	if after != before+1 {
		t.Fatalf("expected exactly one added node, before=%d after=%d", before, after)
	}
}

func TestAddCircuitLayerMutatesDimensions(t *testing.T) {
	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: []int{2, 1},
			Parameters: map[string]float64{},
		},
	}
	op := &AddCircuitLayer{Rand: rand.New(rand.NewSource(101))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if len(mutated.Substrate.Dimensions) != len(genome.Substrate.Dimensions)+1 {
		t.Fatalf("expected one added layer, before=%d after=%d", len(genome.Substrate.Dimensions), len(mutated.Substrate.Dimensions))
	}
	if mutated.Substrate.Dimensions[1] != 1 {
		t.Fatalf("expected inserted hidden layer width 1, got=%v", mutated.Substrate.Dimensions)
	}
}

func TestCircuitMutationsRequireConfiguredDimensions(t *testing.T) {
	withoutSubstrate := model.Genome{}
	if (&AddCircuitNode{}).Applicable(withoutSubstrate, "xor") {
		t.Fatal("expected add circuit node to be inapplicable without substrate config")
	}
	if (&AddCircuitLayer{}).Applicable(withoutSubstrate, "xor") {
		t.Fatal("expected add circuit layer to be inapplicable without substrate config")
	}
	if _, err := (&AddCircuitNode{Rand: rand.New(rand.NewSource(321))}).Apply(context.Background(), withoutSubstrate); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for add circuit node without substrate, got %v", err)
	}
	if _, err := (&AddCircuitLayer{Rand: rand.New(rand.NewSource(323))}).Apply(context.Background(), withoutSubstrate); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for add circuit layer without substrate, got %v", err)
	}
	if _, err := (&DeleteCircuitNode{Rand: rand.New(rand.NewSource(325))}).Apply(context.Background(), withoutSubstrate); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for delete circuit node without substrate, got %v", err)
	}

	withEmptyDims := model.Genome{
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: nil,
			Parameters: map[string]float64{},
		},
	}
	if (&AddCircuitNode{}).Applicable(withEmptyDims, "xor") {
		t.Fatal("expected add circuit node to be inapplicable with empty dimensions")
	}
	if (&AddCircuitLayer{}).Applicable(withEmptyDims, "xor") {
		t.Fatal("expected add circuit layer to be inapplicable with empty dimensions")
	}
	if _, err := (&AddCircuitNode{Rand: rand.New(rand.NewSource(327))}).Apply(context.Background(), withEmptyDims); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for add circuit node with empty dimensions, got %v", err)
	}
	if _, err := (&AddCircuitLayer{Rand: rand.New(rand.NewSource(329))}).Apply(context.Background(), withEmptyDims); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for add circuit layer with empty dimensions, got %v", err)
	}
	if _, err := (&DeleteCircuitNode{Rand: rand.New(rand.NewSource(331))}).Apply(context.Background(), withEmptyDims); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for delete circuit node with empty dimensions, got %v", err)
	}
}

func TestDeleteCircuitNodeMutatesDimensions(t *testing.T) {
	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: []int{2, 3, 1},
			Parameters: map[string]float64{},
		},
	}
	op := &DeleteCircuitNode{Rand: rand.New(rand.NewSource(131))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	before := 0
	after := 0
	for _, d := range genome.Substrate.Dimensions {
		before += d
	}
	for _, d := range mutated.Substrate.Dimensions {
		after += d
	}
	if after != before-1 {
		t.Fatalf("expected exactly one removed node, before=%d after=%d", before, after)
	}
}

func TestRemoveRandomCPPAndCEP(t *testing.T) {
	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.DefaultCEPName,
			Dimensions: []int{1, 1},
			Parameters: map[string]float64{},
		},
	}
	cppMutated, err := (&RemoveRandomCPP{}).Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("remove cpp: %v", err)
	}
	if cppMutated.Substrate.CPPName != "" {
		t.Fatalf("expected cleared cpp name, got=%q", cppMutated.Substrate.CPPName)
	}
	cepMutated, err := (&RemoveRandomCEP{}).Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("remove cep: %v", err)
	}
	if cepMutated.Substrate.CEPName != "" {
		t.Fatalf("expected cleared cep name, got=%q", cepMutated.Substrate.CEPName)
	}
	if _, err := (&RemoveRandomCPP{}).Apply(context.Background(), model.Genome{}); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for remove cpp without substrate, got %v", err)
	}
	if _, err := (&RemoveRandomCEP{}).Apply(context.Background(), model.Genome{}); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for remove cep without substrate, got %v", err)
	}
}

func TestSearchParameterMutators(t *testing.T) {
	base := model.Genome{
		Strategy: &model.StrategyConfig{
			TuningSelection:  "best_so_far",
			AnnealingFactor:  1.0,
			TopologicalMode:  "const",
			TopologicalParam: 1.0,
			HeredityType:     "asexual",
		},
	}

	tuningMutated, err := (&MutateTuningSelection{Rand: rand.New(rand.NewSource(151))}).Apply(context.Background(), base)
	if err != nil {
		t.Fatalf("mutate tuning selection failed: %v", err)
	}
	if tuningMutated.Strategy == nil || tuningMutated.Strategy.TuningSelection == "" || tuningMutated.Strategy.TuningSelection == "best_so_far" {
		t.Fatalf("expected tuning selection change, got=%+v", tuningMutated.Strategy)
	}

	annealingMutated, err := (&MutateTuningAnnealing{Rand: rand.New(rand.NewSource(157))}).Apply(context.Background(), base)
	if err != nil {
		t.Fatalf("mutate tuning annealing failed: %v", err)
	}
	if annealingMutated.Strategy == nil || annealingMutated.Strategy.AnnealingFactor == 1.0 {
		t.Fatalf("expected annealing change, got=%+v", annealingMutated.Strategy)
	}

	topologyMutated, err := (&MutateTotTopologicalMutations{Rand: rand.New(rand.NewSource(163))}).Apply(context.Background(), base)
	if err != nil {
		t.Fatalf("mutate topological mutations failed: %v", err)
	}
	if topologyMutated.Strategy == nil {
		t.Fatalf("expected topological strategy mutation, got=%+v", topologyMutated.Strategy)
	}
	if topologyMutated.Strategy.TopologicalMode == "" {
		t.Fatalf("expected topological mode to be set, got=%+v", topologyMutated.Strategy)
	}
	if topologyMutated.Strategy.TopologicalParam <= 0 {
		t.Fatalf("expected topological param to be set, got=%+v", topologyMutated.Strategy)
	}
	if topologyMutated.Strategy.TopologicalMode == "const" && math.Abs(topologyMutated.Strategy.TopologicalParam-1.0) < 1e-9 {
		t.Fatalf("expected topological mode/param pair to change, got=%+v", topologyMutated.Strategy)
	}

	heredityMutated, err := (&MutateHeredityType{Rand: rand.New(rand.NewSource(167))}).Apply(context.Background(), base)
	if err != nil {
		t.Fatalf("mutate heredity type failed: %v", err)
	}
	if heredityMutated.Strategy == nil || heredityMutated.Strategy.HeredityType == "" || heredityMutated.Strategy.HeredityType == "asexual" {
		t.Fatalf("expected heredity type change, got=%+v", heredityMutated.Strategy)
	}
}

func TestMutateTuningSelectionSupportsReferenceModeSurface(t *testing.T) {
	base := model.Genome{
		Strategy: &model.StrategyConfig{
			TuningSelection: tuning.CandidateSelectBestSoFar,
		},
	}

	for _, mode := range []string{
		tuning.CandidateSelectActive,
		tuning.CandidateSelectActiveRnd,
		tuning.CandidateSelectRecent,
		tuning.CandidateSelectRecentRnd,
		tuning.CandidateSelectLastGen,
		tuning.CandidateSelectLastGenRd,
	} {
		mutated, err := (&MutateTuningSelection{
			Rand:  rand.New(rand.NewSource(187)),
			Modes: []string{tuning.CandidateSelectBestSoFar, mode},
		}).Apply(context.Background(), base)
		if err != nil {
			t.Fatalf("mutate tuning selection failed for mode %q: %v", mode, err)
		}
		if mutated.Strategy == nil || mutated.Strategy.TuningSelection != mode {
			t.Fatalf("expected mode %q, got=%+v", mode, mutated.Strategy)
		}
	}
}

func TestSearchParameterMutatorsCancelWhenNoAlternativeChoice(t *testing.T) {
	base := model.Genome{
		Strategy: &model.StrategyConfig{
			TuningSelection:  tuning.CandidateSelectBestSoFar,
			AnnealingFactor:  1.0,
			TopologicalMode:  "const",
			TopologicalParam: 1.0,
			HeredityType:     "asexual",
		},
	}

	if _, err := (&MutateTuningSelection{
		Rand:  rand.New(rand.NewSource(271)),
		Modes: []string{tuning.CandidateSelectBestSoFar},
	}).Apply(context.Background(), base); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for mutate_tuning_selection exhaustion, got %v", err)
	}

	if _, err := (&MutateTuningAnnealing{
		Rand:   rand.New(rand.NewSource(277)),
		Values: []float64{1.0},
	}).Apply(context.Background(), base); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for mutate_tuning_annealing exhaustion, got %v", err)
	}

	if _, err := (&MutateTotTopologicalMutations{
		Rand: rand.New(rand.NewSource(281)),
		Choices: []TopologicalPolicyChoice{
			{Name: "const", Param: 1.0},
		},
	}).Apply(context.Background(), base); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for mutate_tot_topological_mutations exhaustion, got %v", err)
	}

	if _, err := (&MutateHeredityType{
		Rand:  rand.New(rand.NewSource(283)),
		Types: []string{"asexual"},
	}).Apply(context.Background(), base); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for mutate_heredity_type exhaustion, got %v", err)
	}
}

func TestSearchParameterMutatorsApplicableReflectsAlternativeChoice(t *testing.T) {
	if (&MutateTuningSelection{Modes: []string{tuning.CandidateSelectBestSoFar}}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected mutate_tuning_selection to be inapplicable with a single mode")
	}
	if (&MutateTuningAnnealing{Values: []float64{1.0}}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected mutate_tuning_annealing to be inapplicable with a single value")
	}
	if (&MutateTotTopologicalMutations{
		Choices: []TopologicalPolicyChoice{{Name: "const", Param: 1.0}},
	}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected mutate_tot_topological_mutations to be inapplicable with a single choice")
	}
	if (&MutateHeredityType{Types: []string{"asexual"}}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected mutate_heredity_type to be inapplicable with a single type")
	}
}

func TestMutateTotTopologicalMutationsCanChangeParamWithinSameMode(t *testing.T) {
	base := model.Genome{
		Strategy: &model.StrategyConfig{
			TopologicalMode:  "ncount_exponential",
			TopologicalParam: 0.5,
		},
	}
	op := &MutateTotTopologicalMutations{
		Rand: rand.New(rand.NewSource(271)),
		Choices: []TopologicalPolicyChoice{
			{Name: "ncount_exponential", Param: 0.5},
			{Name: "ncount_exponential", Param: 0.8},
		},
	}
	mutated, err := op.Apply(context.Background(), base)
	if err != nil {
		t.Fatalf("mutate topological mode/param pair failed: %v", err)
	}
	if mutated.Strategy == nil {
		t.Fatalf("expected strategy after mutation, got=%+v", mutated.Strategy)
	}
	if mutated.Strategy.TopologicalMode != "ncount_exponential" {
		t.Fatalf("expected same mode with different param, got=%+v", mutated.Strategy)
	}
	if math.Abs(mutated.Strategy.TopologicalParam-0.8) > 1e-9 {
		t.Fatalf("expected parameter change to 0.8, got=%+v", mutated.Strategy)
	}
}

func TestMutationOperatorReferenceNames(t *testing.T) {
	if (&AddRandomInlink{}).Name() != "add_inlink" {
		t.Fatalf("unexpected add_inlink name")
	}
	if (&MutateWeights{}).Name() != "mutate_weights" {
		t.Fatalf("unexpected mutate_weights name")
	}
	if (&AddBias{}).Name() != "add_bias" {
		t.Fatalf("unexpected add_bias name")
	}
	if (&RemoveBias{}).Name() != "remove_bias" {
		t.Fatalf("unexpected remove_bias name")
	}
	if (&MutateAF{}).Name() != "mutate_af" {
		t.Fatalf("unexpected mutate_af name")
	}
	if (&MutateAggrF{}).Name() != "mutate_aggrf" {
		t.Fatalf("unexpected mutate_aggrf name")
	}
	if (&AddRandomOutlink{}).Name() != "add_outlink" {
		t.Fatalf("unexpected add_outlink name")
	}
	if (&RemoveRandomInlink{}).Name() != "remove_inlink" {
		t.Fatalf("unexpected remove_inlink name")
	}
	if (&RemoveRandomOutlink{}).Name() != "remove_outlink" {
		t.Fatalf("unexpected remove_outlink name")
	}
	if (&RemoveNeuronMutation{}).Name() != "remove_neuron" {
		t.Fatalf("unexpected remove_neuron name")
	}
	if (&CutlinkFromNeuronToNeuron{}).Name() != "cutlink_FromNeuronToNeuron" {
		t.Fatalf("unexpected cutlink_FromNeuronToNeuron name")
	}
	if (&CutlinkFromElementToElement{}).Name() != "cutlink_FromElementToElement" {
		t.Fatalf("unexpected cutlink_FromElementToElement name")
	}
	if (&LinkFromElementToElement{}).Name() != "link_FromElementToElement" {
		t.Fatalf("unexpected link_FromElementToElement name")
	}
	if (&LinkFromNeuronToNeuron{}).Name() != "link_FromNeuronToNeuron" {
		t.Fatalf("unexpected link_FromNeuronToNeuron name")
	}
	if (&LinkFromSensorToNeuron{}).Name() != "link_FromSensorToNeuron" {
		t.Fatalf("unexpected link_FromSensorToNeuron name")
	}
	if (&LinkFromNeuronToActuator{}).Name() != "link_FromNeuronToActuator" {
		t.Fatalf("unexpected link_FromNeuronToActuator name")
	}
	if (&AddRandomOutsplice{}).Name() != "outsplice" {
		t.Fatalf("unexpected outsplice name")
	}
	if (&AddRandomInsplice{}).Name() != "insplice" {
		t.Fatalf("unexpected insplice name")
	}
	if (&AddRandomSensor{}).Name() != "add_sensor" {
		t.Fatalf("unexpected add_sensor name")
	}
	if (&AddRandomSensorLink{}).Name() != "add_sensorlink" {
		t.Fatalf("unexpected add_sensorlink name")
	}
	if (&AddRandomActuator{}).Name() != "add_actuator" {
		t.Fatalf("unexpected add_actuator name")
	}
	if (&AddRandomActuatorLink{}).Name() != "add_actuatorlink" {
		t.Fatalf("unexpected add_actuatorlink name")
	}
	if (&RemoveRandomSensor{}).Name() != "remove_sensor" {
		t.Fatalf("unexpected remove_sensor name")
	}
	if (&RemoveRandomActuator{}).Name() != "remove_actuator" {
		t.Fatalf("unexpected remove_actuator name")
	}
	if (&CutlinkFromSensorToNeuron{}).Name() != "cutlink_FromSensorToNeuron" {
		t.Fatalf("unexpected cutlink_FromSensorToNeuron name")
	}
	if (&CutlinkFromNeuronToActuator{}).Name() != "cutlink_FromNeuronToActuator" {
		t.Fatalf("unexpected cutlink_FromNeuronToActuator name")
	}
	if (&AddRandomCPP{}).Name() != "add_cpp" {
		t.Fatalf("unexpected add_cpp name")
	}
	if (&AddRandomCEP{}).Name() != "add_cep" {
		t.Fatalf("unexpected add_cep name")
	}
	if (&RemoveRandomCPP{}).Name() != "remove_cpp" {
		t.Fatalf("unexpected remove_cpp name")
	}
	if (&RemoveRandomCEP{}).Name() != "remove_cep" {
		t.Fatalf("unexpected remove_cep name")
	}
	if (&AddCircuitNode{}).Name() != "add_circuit_node" {
		t.Fatalf("unexpected add_circuit_node name")
	}
	if (&AddCircuitLayer{}).Name() != "add_circuit_layer" {
		t.Fatalf("unexpected add_circuit_layer name")
	}
	if (&DeleteCircuitNode{}).Name() != "delete_circuit_node" {
		t.Fatalf("unexpected delete_circuit_node name")
	}
	if (&MutatePF{}).Name() != "mutate_pf" {
		t.Fatalf("unexpected mutate_pf name")
	}
	if (&MutatePlasticityParameters{}).Name() != "mutate_plasticity_parameters" {
		t.Fatalf("unexpected mutate_plasticity_parameters name")
	}
	if (&AddNeuron{}).Name() != "add_neuron" {
		t.Fatalf("unexpected add_neuron name")
	}
	if (&MutateTuningSelection{}).Name() != "mutate_tuning_selection" {
		t.Fatalf("unexpected mutate_tuning_selection name")
	}
	if (&MutateTuningAnnealing{}).Name() != "mutate_tuning_annealing" {
		t.Fatalf("unexpected mutate_tuning_annealing name")
	}
	if (&MutateTotTopologicalMutations{}).Name() != "mutate_tot_topological_mutations" {
		t.Fatalf("unexpected mutate_tot_topological_mutations name")
	}
	if (&MutateHeredityType{}).Name() != "mutate_heredity_type" {
		t.Fatalf("unexpected mutate_heredity_type name")
	}
}

func TestPerturbSubstrateParameterMutation(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(3)))
	genome.Substrate = &model.SubstrateConfig{
		CPPName:    "set_weight",
		CEPName:    "delta_weight",
		Dimensions: []int{2, 2},
		Parameters: map[string]float64{
			"scale":  1.0,
			"offset": 0.5,
		},
		WeightCount: 1,
	}
	op := &PerturbSubstrateParameter{
		Rand:     rand.New(rand.NewSource(4)),
		MaxDelta: 0.5,
		Keys:     []string{"scale"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if mutated.Substrate == nil {
		t.Fatal("expected substrate config on mutated genome")
	}
	if mutated.Substrate.Parameters["scale"] == genome.Substrate.Parameters["scale"] {
		t.Fatal("expected selected substrate parameter to change")
	}
	if mutated.Substrate.Parameters["offset"] != genome.Substrate.Parameters["offset"] {
		t.Fatal("expected non-selected substrate parameter to remain unchanged")
	}
}

func TestPerturbSubstrateParameterCancelsWhenUnavailable(t *testing.T) {
	op := &PerturbSubstrateParameter{
		Rand:     rand.New(rand.NewSource(333)),
		MaxDelta: 0.5,
	}
	if _, err := op.Apply(context.Background(), model.Genome{}); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice without substrate, got %v", err)
	}

	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CPPName:    "set_weight",
			CEPName:    "delta_weight",
			Dimensions: []int{2, 2},
			Parameters: map[string]float64{"scale": 1.0},
		},
	}
	op.Keys = []string{"offset"}
	if _, err := op.Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for missing substrate keys, got %v", err)
	}
}

func TestChangePlasticityRuleMutation(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(5)))
	genome.Plasticity = &model.PlasticityConfig{
		Rule:            "hebbian",
		Rate:            0.2,
		SaturationLimit: 1.0,
	}
	op := &ChangePlasticityRule{
		Rand:  rand.New(rand.NewSource(6)),
		Rules: []string{"none", "hebbian", "oja"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if mutated.Plasticity == nil {
		t.Fatal("expected plasticity config on mutated genome")
	}
	if mutated.Plasticity.Rule == genome.Plasticity.Rule {
		t.Fatal("expected plasticity rule to change")
	}
}

func TestPlasticityRuleMutatorsCancelWithoutPlasticityConfig(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(335)))

	if _, err := (&ChangePlasticityRule{
		Rand: rand.New(rand.NewSource(337)),
	}).Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for change_plasticity_rule without plasticity config, got %v", err)
	}

	if _, err := (&PerturbPlasticityRate{
		Rand:     rand.New(rand.NewSource(339)),
		MaxDelta: 0.1,
	}).Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for perturb_plasticity_rate without plasticity config, got %v", err)
	}
}

func TestMutatePFMutatesNeuronPlasticityRule(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(15)))
	genome.Plasticity = &model.PlasticityConfig{
		Rule:            "hebbian",
		Rate:            0.2,
		SaturationLimit: 1.0,
	}
	op := &MutatePF{
		Rand:  rand.New(rand.NewSource(16)),
		Rules: []string{"none", "hebbian", "oja"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	changed := false
	for i := range genome.Neurons {
		if mutated.Neurons[i].PlasticityRule != genome.Neurons[i].PlasticityRule {
			changed = true
		}
	}
	if !changed {
		t.Fatalf("expected at least one neuron plasticity rule mutation, before=%+v after=%+v", genome.Neurons, mutated.Neurons)
	}
	for i := range mutated.Neurons {
		if mutated.Neurons[i].PlasticityRule == "" {
			continue
		}
		if mutated.Neurons[i].PlasticityRate <= 0 {
			t.Fatalf("expected mutated neuron plasticity rate to be set, got=%+v", mutated.Neurons[i])
		}
	}
}

func TestMutatePlasticityParametersMutatesNeuronRate(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(17)))
	genome.Neurons[0].PlasticityRate = 0.2
	op := &MutatePlasticityParameters{
		Rand:     rand.New(rand.NewSource(18)),
		MaxDelta: 0.1,
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	changed := false
	for i := range genome.Neurons {
		if mutated.Neurons[i].PlasticityRate != genome.Neurons[i].PlasticityRate {
			changed = true
		}
		if mutated.Neurons[i].PlasticityRate < 0 {
			t.Fatalf("expected non-negative neuron plasticity rate, got=%+v", mutated.Neurons[i])
		}
	}
	if !changed {
		t.Fatalf("expected one neuron plasticity rate change, before=%+v after=%+v", genome.Neurons, mutated.Neurons)
	}
}

func TestMutatePFAndPlasticityParametersApplicableWithNeuronsOnly(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(19)))
	genome.Plasticity = nil
	if !(&MutatePF{}).Applicable(genome, "xor") {
		t.Fatal("expected mutate_pf to be applicable with neuron-level mutation semantics")
	}
	if !(&MutatePlasticityParameters{}).Applicable(genome, "xor") {
		t.Fatal("expected mutate_plasticity_parameters to be applicable with neuron-level mutation semantics")
	}
}

func TestFunctionMutatorApplicableReflectsAlternativeChoiceAvailability(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity", Aggregator: "dot_product", PlasticityRule: "hebbian"},
			{ID: "n2", Activation: "identity", Aggregator: "dot_product", PlasticityRule: "hebbian"},
		},
	}
	if (&MutateAF{Activations: []string{"identity"}}).Applicable(genome, "xor") {
		t.Fatal("expected mutate_af to be inapplicable with no alternative activation choices")
	}
	if (&MutateAggrF{Aggregators: []string{"dot_product"}}).Applicable(genome, "xor") {
		t.Fatal("expected mutate_aggrf to be inapplicable with no alternative aggregator choices")
	}
	if (&MutatePF{Rules: []string{"hebbian"}}).Applicable(genome, "xor") {
		t.Fatal("expected mutate_pf to be inapplicable with no alternative plasticity-rule choices")
	}
}

func TestContextualOperatorApplicability(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(7)))
	if (&PerturbRandomBias{}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected bias operator to be inapplicable without neurons")
	}
	if (&PerturbPlasticityRate{}).Applicable(genome, "xor") {
		t.Fatal("expected plasticity operator to be inapplicable without plasticity config")
	}
	if (&ChangePlasticityRule{}).Applicable(genome, "xor") {
		t.Fatal("expected plasticity-rule operator to be inapplicable without plasticity config")
	}
	if (&PerturbSubstrateParameter{}).Applicable(genome, "xor") {
		t.Fatal("expected substrate operator to be inapplicable without substrate config")
	}

	genome.Plasticity = &model.PlasticityConfig{Rule: "hebbian", Rate: 0.1}
	genome.Substrate = &model.SubstrateConfig{
		CPPName:    "set_weight",
		CEPName:    "delta_weight",
		Parameters: map[string]float64{"scale": 1.0},
	}
	if !(&PerturbPlasticityRate{}).Applicable(genome, "xor") {
		t.Fatal("expected plasticity operator to be applicable with plasticity config")
	}
	if !(&ChangePlasticityRule{}).Applicable(genome, "xor") {
		t.Fatal("expected plasticity-rule operator to be applicable with plasticity config")
	}
	if !(&PerturbSubstrateParameter{}).Applicable(genome, "xor") {
		t.Fatal("expected substrate operator to be applicable with substrate config")
	}
}

func TestPerturbRandomBiasMutation(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(10)))
	op := &PerturbRandomBias{
		Rand:     rand.New(rand.NewSource(11)),
		MaxDelta: 0.3,
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}

	if len(mutated.Neurons) != len(genome.Neurons) {
		t.Fatalf("neuron count changed: got=%d want=%d", len(mutated.Neurons), len(genome.Neurons))
	}
	changed := false
	for i := range genome.Neurons {
		if mutated.Neurons[i].ID != genome.Neurons[i].ID {
			t.Fatalf("neuron identity changed at index %d", i)
		}
		if mutated.Neurons[i].Activation != genome.Neurons[i].Activation {
			t.Fatalf("neuron activation changed at index %d", i)
		}
		if mutated.Neurons[i].Bias != genome.Neurons[i].Bias {
			changed = true
		}
	}
	if !changed {
		t.Fatal("expected at least one neuron bias to change")
	}
}

func TestChangeRandomActivationMutation(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(12)))
	op := &ChangeRandomActivation{
		Rand:        rand.New(rand.NewSource(13)),
		Activations: []string{"identity", "relu"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}

	changed := false
	for i := range genome.Neurons {
		if mutated.Neurons[i].ID != genome.Neurons[i].ID {
			t.Fatalf("neuron identity changed at index %d", i)
		}
		if mutated.Neurons[i].Activation != genome.Neurons[i].Activation {
			changed = true
		}
	}
	if !changed {
		t.Fatal("expected one activation to change")
	}
}

func TestChangeRandomAggregatorMutation(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(14)))
	for i := range genome.Neurons {
		genome.Neurons[i].Aggregator = "dot_product"
	}
	op := &ChangeRandomAggregator{
		Rand:        rand.New(rand.NewSource(15)),
		Aggregators: []string{"dot_product", "mult_product", "diff_product"},
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}

	changed := false
	for i := range genome.Neurons {
		if mutated.Neurons[i].ID != genome.Neurons[i].ID {
			t.Fatalf("neuron identity changed at index %d", i)
		}
		if mutated.Neurons[i].Aggregator != genome.Neurons[i].Aggregator {
			changed = true
		}
	}
	if !changed {
		t.Fatal("expected one aggregator to change")
	}
}

func TestActivationAndPlasticityRuleMutatorsCancelWhenNoAlternative(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(287)))
	for i := range genome.Neurons {
		genome.Neurons[i].Activation = "identity"
		genome.Neurons[i].Aggregator = "dot_product"
		genome.Neurons[i].PlasticityRule = "hebbian"
	}
	genome.Plasticity = &model.PlasticityConfig{
		Rule:            "hebbian",
		Rate:            0.2,
		SaturationLimit: 1.0,
	}

	if _, err := (&ChangeRandomActivation{
		Rand:        rand.New(rand.NewSource(289)),
		Activations: []string{"identity"},
	}).Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for mutate_af exhaustion, got %v", err)
	}

	if _, err := (&ChangeRandomAggregator{
		Rand:        rand.New(rand.NewSource(293)),
		Aggregators: []string{"dot_product"},
	}).Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for mutate_aggrf exhaustion, got %v", err)
	}

	if _, err := (&ChangePlasticityRule{
		Rand:  rand.New(rand.NewSource(307)),
		Rules: []string{"hebbian"},
	}).Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for change_plasticity_rule exhaustion, got %v", err)
	}

	if _, err := (&MutatePF{
		Rand:  rand.New(rand.NewSource(311)),
		Rules: []string{"hebbian"},
	}).Apply(context.Background(), genome); !errors.Is(err, ErrNoMutationChoice) {
		t.Fatalf("expected ErrNoMutationChoice for mutate_pf exhaustion, got %v", err)
	}
}

func TestRemoveRandomBiasMutation(t *testing.T) {
	genome := randomGenome(rand.New(rand.NewSource(16)))
	for i := range genome.Neurons {
		genome.Neurons[i].Bias = float64(i + 1)
	}
	op := &RemoveRandomBias{
		Rand: rand.New(rand.NewSource(17)),
	}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}

	zeroed := false
	for i := range genome.Neurons {
		if mutated.Neurons[i].ID != genome.Neurons[i].ID {
			t.Fatalf("neuron identity changed at index %d", i)
		}
		if mutated.Neurons[i].Bias == 0 && genome.Neurons[i].Bias != 0 {
			zeroed = true
		}
	}
	if !zeroed {
		t.Fatal("expected one neuron bias to be removed")
	}
}

func decodeGenomeFixture(t *testing.T, path string) model.Genome {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	genome, err := storage.DecodeGenome(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	return genome
}

func randomGenome(rng *rand.Rand) model.Genome {
	neuronCount := 2 + rng.Intn(5)
	synapseCount := 1 + rng.Intn(6)

	neurons := make([]model.Neuron, 0, neuronCount)
	for i := 0; i < neuronCount; i++ {
		neurons = append(neurons, model.Neuron{
			ID:         "n" + strconv.Itoa(i),
			Activation: "identity",
			Bias:       (rng.Float64() * 2) - 1,
		})
	}

	synapses := make([]model.Synapse, 0, synapseCount)
	for i := 0; i < synapseCount; i++ {
		from := rng.Intn(neuronCount)
		to := rng.Intn(neuronCount)
		synapses = append(synapses, model.Synapse{
			ID:        "s" + strconv.Itoa(i),
			From:      "n" + strconv.Itoa(from),
			To:        "n" + strconv.Itoa(to),
			Weight:    (rng.Float64() * 4) - 2,
			Enabled:   true,
			Recurrent: from == to,
		})
	}

	return model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
		ID:              "random",
		Neurons:         neurons,
		Synapses:        synapses,
		SensorIDs:       []string{"sensor:input"},
		ActuatorIDs:     []string{"actuator:output"},
	}
}

func assertNoDanglingSynapses(t *testing.T, g model.Genome) {
	t.Helper()
	for _, s := range g.Synapses {
		if !hasNeuron(g, s.From) || !hasNeuron(g, s.To) {
			t.Fatalf("dangling synapse %+v", s)
		}
	}
}
