package genotype

import (
	"testing"

	"protogonos/internal/model"
)

func TestCloneComponentHelpersRemapAndPreserveUnknowns(t *testing.T) {
	neurons := []model.Neuron{
		{ID: "n1", Activation: "identity"},
		{ID: "n2", Activation: "tanh"},
	}
	synapses := []model.Synapse{
		{
			ID:               "s1",
			From:             "n1",
			To:               "n2",
			PlasticityParams: []float64{0.1, 0.2},
		},
		{ID: "s2", From: "n2", To: "n3"},
	}
	sensorLinks := []model.SensorNeuronLink{
		{SensorID: "sensor:in", NeuronID: "n1"},
	}
	actuatorLinks := []model.NeuronActuatorLink{
		{NeuronID: "n2", ActuatorID: "actuator:out"},
	}

	neuronMap := map[string]string{
		"n1": "n1c",
		"n2": "n2c",
	}
	synapseMap := map[string]string{
		"s1": "s1c",
	}

	clonedNeurons := CloneNeuronsWithIDMap(neurons, neuronMap)
	if clonedNeurons[0].ID != "n1c" || clonedNeurons[1].ID != "n2c" {
		t.Fatalf("unexpected cloned neurons: %+v", clonedNeurons)
	}

	clonedSynapses := CloneSynapsesWithIDMap(synapses, synapseMap, neuronMap)
	if clonedSynapses[0].ID != "s1c" || clonedSynapses[0].From != "n1c" || clonedSynapses[0].To != "n2c" {
		t.Fatalf("unexpected remapped first synapse: %+v", clonedSynapses[0])
	}
	if clonedSynapses[1].ID != "s2" || clonedSynapses[1].From != "n2c" || clonedSynapses[1].To != "n3" {
		t.Fatalf("expected unknown ids preserved on second synapse, got=%+v", clonedSynapses[1])
	}
	// Ensure helper deep-copies plasticity parameter slices.
	clonedSynapses[0].PlasticityParams[0] = 999
	if synapses[0].PlasticityParams[0] == 999 {
		t.Fatal("expected cloned synapse plasticity params to be deep copied")
	}

	clonedSensorLinks := CloneSensorLinksWithIDMap(sensorLinks, nil, neuronMap)
	if clonedSensorLinks[0].SensorID != "sensor:in" || clonedSensorLinks[0].NeuronID != "n1c" {
		t.Fatalf("unexpected remapped sensor link: %+v", clonedSensorLinks[0])
	}

	clonedActuatorLinks := CloneActuatorLinksWithIDMap(actuatorLinks, nil, neuronMap)
	if clonedActuatorLinks[0].NeuronID != "n2c" || clonedActuatorLinks[0].ActuatorID != "actuator:out" {
		t.Fatalf("unexpected remapped actuator link: %+v", clonedActuatorLinks[0])
	}
}
