package genotype

import "protogonos/internal/model"

// CloneNeuronsWithIDMap mirrors the ID remap role from reference clone_neurons.
func CloneNeuronsWithIDMap(neurons []model.Neuron, neuronIDMap map[string]string) []model.Neuron {
	out := append([]model.Neuron(nil), neurons...)
	for i := range out {
		if mappedID, ok := neuronIDMap[out[i].ID]; ok {
			out[i].ID = mappedID
		}
	}
	return out
}

// CloneSynapsesWithIDMap mirrors simplified remap semantics from reference
// clone_neurons input/output remapping.
func CloneSynapsesWithIDMap(
	synapses []model.Synapse,
	synapseIDMap map[string]string,
	neuronIDMap map[string]string,
) []model.Synapse {
	out := append([]model.Synapse(nil), synapses...)
	for i := range out {
		if mappedID, ok := synapseIDMap[out[i].ID]; ok {
			out[i].ID = mappedID
		}
		if mappedFrom, ok := neuronIDMap[out[i].From]; ok {
			out[i].From = mappedFrom
		}
		if mappedTo, ok := neuronIDMap[out[i].To]; ok {
			out[i].To = mappedTo
		}
		if len(out[i].PlasticityParams) > 0 {
			out[i].PlasticityParams = append([]float64(nil), out[i].PlasticityParams...)
		}
	}
	return out
}

// CloneSensorLinksWithIDMap mirrors the ID remap role from reference
// clone_sensors for simplified sensor-neuron endpoint links.
func CloneSensorLinksWithIDMap(
	links []model.SensorNeuronLink,
	sensorIDMap map[string]string,
	neuronIDMap map[string]string,
) []model.SensorNeuronLink {
	out := append([]model.SensorNeuronLink(nil), links...)
	for i := range out {
		if mappedSensorID, ok := sensorIDMap[out[i].SensorID]; ok {
			out[i].SensorID = mappedSensorID
		}
		if mappedNeuronID, ok := neuronIDMap[out[i].NeuronID]; ok {
			out[i].NeuronID = mappedNeuronID
		}
	}
	return out
}

// CloneActuatorLinksWithIDMap mirrors the ID remap role from reference
// clone_actuators for simplified neuron-actuator endpoint links.
func CloneActuatorLinksWithIDMap(
	links []model.NeuronActuatorLink,
	actuatorIDMap map[string]string,
	neuronIDMap map[string]string,
) []model.NeuronActuatorLink {
	out := append([]model.NeuronActuatorLink(nil), links...)
	for i := range out {
		if mappedActuatorID, ok := actuatorIDMap[out[i].ActuatorID]; ok {
			out[i].ActuatorID = mappedActuatorID
		}
		if mappedNeuronID, ok := neuronIDMap[out[i].NeuronID]; ok {
			out[i].NeuronID = mappedNeuronID
		}
	}
	return out
}
