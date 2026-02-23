package genotype

import (
	"testing"

	"protogonos/internal/model"
)

func TestBuildReferenceFingerprintIncludesGeneralizedParts(t *testing.T) {
	genome := model.Genome{
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

	fp := BuildReferenceFingerprint(genome, history)
	if len(fp.Pattern) != 2 {
		t.Fatalf("expected 2 pattern layers, got=%v", fp.Pattern)
	}
	if len(fp.EvoHistory) != 1 {
		t.Fatalf("expected generalized evo history size 1, got=%v", fp.EvoHistory)
	}
	if len(fp.Sensors) != 2 || fp.Sensors[0] != "sensor:left" || fp.Sensors[1] != "sensor:right" {
		t.Fatalf("expected sorted unique sensors, got=%v", fp.Sensors)
	}
	if len(fp.Actuators) != 1 || fp.Actuators[0] != "actuator:out" {
		t.Fatalf("expected one actuator in fingerprint, got=%v", fp.Actuators)
	}
	if fp.Topology.TotalNeurons != 2 {
		t.Fatalf("expected topology neurons 2, got=%d", fp.Topology.TotalNeurons)
	}
}

func TestComputeReferenceFingerprintDeterministicAndSensitive(t *testing.T) {
	base := model.Genome{
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
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
	gotA := ComputeReferenceFingerprint(base, history)
	gotB := ComputeReferenceFingerprint(base, history)
	if gotA != gotB {
		t.Fatalf("expected deterministic reference fingerprint, got %q != %q", gotA, gotB)
	}

	changed := CloneGenome(base)
	changed.Neurons = append(changed.Neurons, model.Neuron{ID: "L2:n2", Activation: "tanh"})
	gotC := ComputeReferenceFingerprint(changed, history)
	if gotC == gotA {
		t.Fatalf("expected topology change to alter reference fingerprint, got=%q", gotC)
	}
}

func TestBuildReferenceFingerprintInfersPatternWithoutLayerTaggedIDs(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "h", Activation: "tanh"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "h", Enabled: true},
			{ID: "s2", From: "h", To: "o", Enabled: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: "s1", NeuronID: "i"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "o", ActuatorID: "a1"},
		},
	}

	fp := BuildReferenceFingerprint(genome, nil)
	if len(fp.Pattern) != 3 {
		t.Fatalf("expected inferred 3-layer pattern, got=%v", fp.Pattern)
	}
	if fp.Pattern[0].Layer != 0 || len(fp.Pattern[0].NeuronIDs) != 1 || fp.Pattern[0].NeuronIDs[0] != "i" {
		t.Fatalf("expected layer 0 to contain input neuron, got=%v", fp.Pattern[0])
	}
	if fp.Pattern[1].Layer != 1 || len(fp.Pattern[1].NeuronIDs) != 1 || fp.Pattern[1].NeuronIDs[0] != "h" {
		t.Fatalf("expected layer 1 to contain hidden neuron, got=%v", fp.Pattern[1])
	}
	if fp.Pattern[2].Layer != 2 || len(fp.Pattern[2].NeuronIDs) != 1 || fp.Pattern[2].NeuronIDs[0] != "o" {
		t.Fatalf("expected layer 2 to contain output neuron, got=%v", fp.Pattern[2])
	}
}
