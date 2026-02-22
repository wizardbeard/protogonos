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

func TestConstructSeedNNBuildsLayeredScaffold(t *testing.T) {
	seed, err := ConstructSeedNN(
		2,
		[]string{"s1", "s2", "s1"},
		[]string{"a1"},
		[]string{"sigmoid"},
		[]string{"hebbian_w"},
		[]string{"dot_product"},
		rand.New(rand.NewSource(13)),
	)
	if err != nil {
		t.Fatalf("construct seed nn: %v", err)
	}
	if len(seed.InputNeuronIDs) != 2 {
		t.Fatalf("expected two deduplicated input neurons, got=%v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 {
		t.Fatalf("expected one output neuron, got=%v", seed.OutputNeuronIDs)
	}
	if len(seed.Neurons) != 3 {
		t.Fatalf("expected 3 total neurons (2 input + 1 output), got=%d", len(seed.Neurons))
	}
	if len(seed.Synapses) != 2 {
		t.Fatalf("expected full input->output synapses, got=%d", len(seed.Synapses))
	}
	if len(seed.SensorNeuronLinks) != 2 {
		t.Fatalf("expected one sensor link per sensor, got=%d", len(seed.SensorNeuronLinks))
	}
	if len(seed.NeuronActuatorLinks) != 1 {
		t.Fatalf("expected one actuator link per actuator, got=%d", len(seed.NeuronActuatorLinks))
	}
	if len(seed.Pattern) != 2 {
		t.Fatalf("expected 2-layer init pattern, got=%v", seed.Pattern)
	}
	if seed.Pattern[0].Layer != 0 || seed.Pattern[1].Layer != 1 {
		t.Fatalf("unexpected layer pattern ordering: %v", seed.Pattern)
	}
	if len(seed.Pattern[0].NeuronIDs) != len(seed.InputNeuronIDs) || len(seed.Pattern[1].NeuronIDs) != len(seed.OutputNeuronIDs) {
		t.Fatalf("pattern neuron cardinality mismatch: pattern=%v", seed.Pattern)
	}
}

func TestConstructSeedNNValidatesRequiredSensorsAndActuators(t *testing.T) {
	if _, err := ConstructSeedNN(0, nil, []string{"a1"}, nil, nil, nil, rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("expected validation error for missing sensors")
	}
	if _, err := ConstructSeedNN(0, []string{"s1"}, nil, nil, nil, nil, rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("expected validation error for missing actuators")
	}
}

func TestConstructSeedNNCircuitModeBuildsRelayAndCircuitLayers(t *testing.T) {
	seed, err := ConstructSeedNN(
		0,
		[]string{"s1", "s2"},
		[]string{"a1"},
		[]string{"circuit:tanh"},
		[]string{"none"},
		[]string{"dot_product"},
		rand.New(rand.NewSource(17)),
	)
	if err != nil {
		t.Fatalf("construct seed nn circuit mode: %v", err)
	}
	if len(seed.Neurons) != 4 {
		t.Fatalf("expected 4 neurons in circuit mode (2 input + 1 relay + 1 circuit), got=%d", len(seed.Neurons))
	}
	if len(seed.Synapses) != 3 {
		t.Fatalf("expected 3 synapses in circuit mode, got=%d", len(seed.Synapses))
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "L0.99:circuit:0" {
		t.Fatalf("unexpected circuit output neuron ids: %v", seed.OutputNeuronIDs)
	}
	if len(seed.NeuronActuatorLinks) != 1 || seed.NeuronActuatorLinks[0].NeuronID != "L0.99:circuit:0" {
		t.Fatalf("expected actuator to link from circuit neuron, got=%v", seed.NeuronActuatorLinks)
	}
	if len(seed.Pattern) != 3 {
		t.Fatalf("expected 3-layer pattern in circuit mode, got=%v", seed.Pattern)
	}
	if seed.Pattern[0].Layer != 0 || seed.Pattern[1].Layer != 0.5 || seed.Pattern[2].Layer != 0.99 {
		t.Fatalf("unexpected circuit pattern layers: %v", seed.Pattern)
	}
}

func TestCreateInitPatternGroupsAndSortsLayers(t *testing.T) {
	pattern := CreateInitPattern([]string{
		"L2:n3",
		"L0:n1",
		"L1:n2",
		"L1:n4",
		"invalid",
	})
	if len(pattern) != 3 {
		t.Fatalf("expected 3 layer groups, got=%v", pattern)
	}
	if pattern[0].Layer != 0 || pattern[1].Layer != 1 || pattern[2].Layer != 2 {
		t.Fatalf("expected sorted layers [0,1,2], got=%v", pattern)
	}
	if len(pattern[1].NeuronIDs) != 2 {
		t.Fatalf("expected two neurons in layer 1 bucket, got=%v", pattern[1].NeuronIDs)
	}
}

func TestConstructSeedNNCircuitModeStripsCircuitTagFromRelayAFPool(t *testing.T) {
	seed, err := ConstructSeedNN(
		0,
		[]string{"s1"},
		[]string{"a1"},
		[]string{"circuit:tanh", "sigmoid"},
		[]string{"none"},
		[]string{"dot_product"},
		rand.New(rand.NewSource(19)),
	)
	if err != nil {
		t.Fatalf("construct seed nn mixed afs: %v", err)
	}
	byID := map[string]string{}
	for _, n := range seed.Neurons {
		byID[n.ID] = n.Activation
	}
	if byID["L0.99:circuit:0"] != "tanh" {
		t.Fatalf("expected explicit circuit activation tanh, got=%q", byID["L0.99:circuit:0"])
	}
	if byID["L0.5:relay:0"] == "circuit:tanh" {
		t.Fatalf("expected relay activation pool to exclude circuit tags, got=%q", byID["L0.5:relay:0"])
	}
}

func TestGenerateIDsReturnsCountAndUniqueValues(t *testing.T) {
	ids := GenerateIDs(4, rand.New(rand.NewSource(1)))
	if len(ids) != 4 {
		t.Fatalf("expected 4 generated ids, got=%d", len(ids))
	}
	seen := map[float64]struct{}{}
	for _, id := range ids {
		if id == 0 {
			t.Fatalf("expected non-zero id, got=%f", id)
		}
		if _, ok := seen[id]; ok {
			t.Fatalf("expected unique ids, duplicate=%f", id)
		}
		seen[id] = struct{}{}
	}
}

func TestLinkNeuronCreatesInboundAndOutboundSynapses(t *testing.T) {
	synapses, err := LinkNeuron(
		[]string{"L0:n0", "L0:n1"},
		"L1:n2",
		[]string{"L2:n3", "L0:n4"},
		rand.New(rand.NewSource(5)),
	)
	if err != nil {
		t.Fatalf("link neuron: %v", err)
	}
	if len(synapses) != 4 {
		t.Fatalf("expected 4 synapses, got=%d", len(synapses))
	}
	inbound := 0
	outbound := 0
	recurrentOutbound := 0
	for _, synapse := range synapses {
		if synapse.To == "L1:n2" {
			inbound++
		}
		if synapse.From == "L1:n2" {
			outbound++
			if synapse.Recurrent {
				recurrentOutbound++
			}
		}
		if !synapse.Enabled {
			t.Fatalf("expected enabled synapse, got=%+v", synapse)
		}
	}
	if inbound != 2 || outbound != 2 {
		t.Fatalf("expected 2 inbound and 2 outbound links, got inbound=%d outbound=%d", inbound, outbound)
	}
	if recurrentOutbound != 1 {
		t.Fatalf("expected one recurrent outbound link, got=%d", recurrentOutbound)
	}
}

func TestLinkNeuronValidatesNeuronID(t *testing.T) {
	if _, err := LinkNeuron([]string{"n0"}, "", []string{"n1"}, rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("expected validation error for empty neuron id")
	}
}
