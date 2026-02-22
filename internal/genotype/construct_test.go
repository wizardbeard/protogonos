package genotype

import (
	"math/rand"
	"testing"

	"protogonos/internal/nn"
)

func TestGenerateNeuronAFDefaultsToTanh(t *testing.T) {
	if got := GenerateNeuronAF(rand.New(rand.NewSource(1)), nil); got != "tanh" {
		t.Fatalf("expected default tanh activation, got=%q", got)
	}
}

func TestGenerateNeuronPFDefaultsAndNormalizes(t *testing.T) {
	name, params := GenerateNeuronPF(rand.New(rand.NewSource(1)), nil)
	if name != nn.PlasticityNone {
		t.Fatalf("expected default plasticity none, got=%q", name)
	}
	if len(params) != 0 {
		t.Fatalf("expected empty default pf params, got=%v", params)
	}

	name, params = GenerateNeuronPF(rand.New(rand.NewSource(2)), []string{"  ojas  "})
	if name != nn.PlasticityOja {
		t.Fatalf("expected normalized oja rule, got=%q", name)
	}
	if len(params) != 1 {
		t.Fatalf("expected oja neural params length 1, got=%d", len(params))
	}
}

func TestGenerateNeuronAggrFDefaultsToNone(t *testing.T) {
	if got := GenerateNeuronAggrF(rand.New(rand.NewSource(1)), nil); got != "none" {
		t.Fatalf("expected default aggregator none, got=%q", got)
	}
}

func TestCalculateROIDsLayerAware(t *testing.T) {
	selfID := "L1.5:n1"
	outputIDs := []string{"L2:n2", "L1:n3", "L0.9:n4", "actuator:a"}
	got := CalculateROIDs(selfID, outputIDs)
	if len(got) != 2 {
		t.Fatalf("expected two recurrent ids, got=%v", got)
	}
	if got[0] != "L1:n3" || got[1] != "L0.9:n4" {
		t.Fatalf("unexpected recurrent ids ordering/content: %v", got)
	}
}

func TestConstructNeuronBuildsSynapsesAndPlasticity(t *testing.T) {
	neuron, synapses, roIDs, err := ConstructNeuron(
		3,
		"L1:n1",
		[]InputSpec{
			{FromID: "s1", Width: 2},
			{FromID: "n0", Width: 1},
		},
		[]string{"L0.5:n0", "L2:n2"},
		[]string{"sigmoid"},
		[]string{"hebbian_w"},
		[]string{"dot_product"},
		rand.New(rand.NewSource(7)),
	)
	if err != nil {
		t.Fatalf("construct neuron: %v", err)
	}
	if neuron.ID != "L1:n1" {
		t.Fatalf("unexpected neuron id: %s", neuron.ID)
	}
	if neuron.Generation != 3 {
		t.Fatalf("unexpected generation: %d", neuron.Generation)
	}
	if neuron.Activation != "sigmoid" {
		t.Fatalf("expected sigmoid activation, got=%q", neuron.Activation)
	}
	if neuron.Aggregator != "dot_product" {
		t.Fatalf("expected dot_product aggregation, got=%q", neuron.Aggregator)
	}
	if neuron.PlasticityRule != nn.PlasticityHebbianW {
		t.Fatalf("expected hebbian_w rule, got=%q", neuron.PlasticityRule)
	}
	if len(synapses) != 3 {
		t.Fatalf("expected 3 generated synapses, got=%d", len(synapses))
	}
	for _, synapse := range synapses {
		if synapse.To != neuron.ID {
			t.Fatalf("unexpected synapse target: %+v", synapse)
		}
		if !synapse.Enabled {
			t.Fatalf("expected enabled synapse: %+v", synapse)
		}
		if synapse.Weight < -0.5 || synapse.Weight > 0.5 {
			t.Fatalf("expected centered random weight in [-0.5,0.5], got=%f", synapse.Weight)
		}
		if len(synapse.PlasticityParams) != 1 {
			t.Fatalf("expected hebbian_w weight-parameter width=1, got=%v", synapse.PlasticityParams)
		}
	}
	if len(roIDs) != 1 || roIDs[0] != "L0.5:n0" {
		t.Fatalf("unexpected recurrent outputs: %v", roIDs)
	}
}

func TestConstructNeuronSelfModulationV6WeightWidth(t *testing.T) {
	_, synapses, _, err := ConstructNeuron(
		0,
		"L1:n1",
		[]InputSpec{{FromID: "n0", Width: 1}},
		nil,
		nil,
		[]string{"self_modulationV6"},
		nil,
		rand.New(rand.NewSource(9)),
	)
	if err != nil {
		t.Fatalf("construct neuron: %v", err)
	}
	if len(synapses) != 1 {
		t.Fatalf("expected one synapse, got=%d", len(synapses))
	}
	if len(synapses[0].PlasticityParams) != 5 {
		t.Fatalf("expected self_modulationV6 synapse parameter width 5, got=%v", synapses[0].PlasticityParams)
	}
}

func TestConstructNeuronValidatesNeuronID(t *testing.T) {
	if _, _, _, err := ConstructNeuron(0, "", nil, nil, nil, nil, nil, rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("expected validation error for empty neuron id")
	}
}
