package genotype

import (
	"testing"

	"protogonos/internal/model"
)

func TestGetNodeSummaryCountsLinksAndDistribution(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "L0:n0", Activation: "identity"},
			{ID: "L1:n1", Activation: "tanh"},
			{ID: "L2:n2", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "L0:n0", To: "L1:n1", Enabled: true},
			{ID: "s2", From: "L1:n1", To: "L2:n2", Enabled: true},
			{ID: "s3", From: "L2:n2", To: "L1:n1", Enabled: true, Recurrent: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: "sensor:left", NeuronID: "L0:n0"},
			{SensorID: "sensor:right", NeuronID: "L1:n1"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "L2:n2", ActuatorID: "actuator:out"},
		},
	}

	summary := GetNodeSummary(genome)
	if summary.TotalNILs != 5 {
		t.Fatalf("expected total NILs 5, got=%d", summary.TotalNILs)
	}
	if summary.TotalNOLs != 4 {
		t.Fatalf("expected total NOLs 4, got=%d", summary.TotalNOLs)
	}
	if summary.TotalNROs != 1 {
		t.Fatalf("expected total NROs 1, got=%d", summary.TotalNROs)
	}
	if summary.ActivationDistribution["identity"] != 1 || summary.ActivationDistribution["tanh"] != 2 {
		t.Fatalf("unexpected activation distribution: %v", summary.ActivationDistribution)
	}
}

func TestGeneralizeEvoHistoryReplacesIDsWithLayerAndKind(t *testing.T) {
	history := []EvoHistoryEvent{
		{Mutation: "add_link", IDs: []string{"L0.5:n1", "actuator:out"}},
		{Mutation: "add_sensor", IDs: []string{"sensor:left"}},
		{Mutation: "noop"},
	}

	generalized := GeneralizeEvoHistory(history)
	if len(generalized) != 3 {
		t.Fatalf("expected 3 generalized events, got=%d", len(generalized))
	}
	if generalized[0].Mutation != "add_link" || len(generalized[0].Elements) != 2 {
		t.Fatalf("unexpected first generalized event: %+v", generalized[0])
	}
	firstNeuron := generalized[0].Elements[0]
	if firstNeuron.Layer == nil || *firstNeuron.Layer != 0.5 || firstNeuron.Kind != "neuron" {
		t.Fatalf("expected first ref to be neuron layer 0.5, got=%+v", firstNeuron)
	}
	firstActuator := generalized[0].Elements[1]
	if firstActuator.Layer != nil || firstActuator.Kind != "actuator" {
		t.Fatalf("expected second ref actuator w/o layer, got=%+v", firstActuator)
	}
	second := generalized[1].Elements[0]
	if second.Kind != "sensor" {
		t.Fatalf("expected sensor kind in second event, got=%+v", second)
	}
	if len(generalized[2].Elements) != 0 {
		t.Fatalf("expected no elements for id-less event, got=%+v", generalized[2].Elements)
	}
}

func TestGeneralizeEvoHistoryHandlesEmptyInput(t *testing.T) {
	if got := GeneralizeEvoHistory(nil); got != nil {
		t.Fatalf("expected nil for empty history, got=%v", got)
	}
}
