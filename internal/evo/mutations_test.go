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
}

func TestAddRandomCPPCreatesSubstrateConfig(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
	}
	op := &AddRandomCPP{Rand: rand.New(rand.NewSource(83))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if mutated.Substrate == nil {
		t.Fatal("expected substrate config to be created")
	}
	if mutated.Substrate.CPPName == "" {
		t.Fatal("expected cpp name to be set")
	}
	if mutated.Substrate.CEPName == "" {
		t.Fatal("expected cep default name to be set")
	}
}

func TestAddRandomCEPCreatesSubstrateConfig(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
	}
	op := &AddRandomCEP{Rand: rand.New(rand.NewSource(89))}
	mutated, err := op.Apply(context.Background(), genome)
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}
	if mutated.Substrate == nil {
		t.Fatal("expected substrate config to be created")
	}
	if mutated.Substrate.CEPName == "" {
		t.Fatal("expected cep name to be set")
	}
	if mutated.Substrate.CPPName == "" {
		t.Fatal("expected cpp default name to be set")
	}
}

func TestAddRandomCPPAndCEPApplicable(t *testing.T) {
	if !(&AddRandomCPP{}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected add cpp operator to be applicable with default registry")
	}
	if !(&AddRandomCEP{}).Applicable(model.Genome{}, "xor") {
		t.Fatal("expected add cep operator to be applicable with default registry")
	}
	if substrate.DefaultCPPName == "" || substrate.DefaultCEPName == "" {
		t.Fatal("expected default substrate names")
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

func TestMutationOperatorReferenceNames(t *testing.T) {
	if (&AddRandomInlink{}).Name() != "add_inlink" {
		t.Fatalf("unexpected add_inlink name")
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
	if (&AddRandomCPP{}).Name() != "add_cpp" {
		t.Fatalf("unexpected add_cpp name")
	}
	if (&AddRandomCEP{}).Name() != "add_cep" {
		t.Fatalf("unexpected add_cep name")
	}
	if (&AddCircuitNode{}).Name() != "add_CircuitNode" {
		t.Fatalf("unexpected add_CircuitNode name")
	}
	if (&AddCircuitLayer{}).Name() != "add_CircuitLayer" {
		t.Fatalf("unexpected add_CircuitLayer name")
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
