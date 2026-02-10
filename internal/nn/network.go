package nn

import (
	"fmt"

	"protogonos/internal/model"
)

func Forward(genome model.Genome, inputByNeuron map[string]float64) (map[string]float64, error) {
	values := make(map[string]float64, len(genome.Neurons))
	for neuronID, value := range inputByNeuron {
		values[neuronID] = value
	}

	incoming := make(map[string][]model.Synapse, len(genome.Neurons))
	for _, synapse := range genome.Synapses {
		if !synapse.Enabled {
			continue
		}
		incoming[synapse.To] = append(incoming[synapse.To], synapse)
	}

	for _, neuron := range genome.Neurons {
		if _, fixedInput := inputByNeuron[neuron.ID]; fixedInput {
			continue
		}

		total := neuron.Bias
		for _, synapse := range incoming[neuron.ID] {
			total += values[synapse.From] * synapse.Weight
		}

		activated, err := applyActivation(neuron.Activation, total)
		if err != nil {
			return nil, fmt.Errorf("neuron %s: %w", neuron.ID, err)
		}
		values[neuron.ID] = activated
	}

	return values, nil
}

func applyActivation(name string, x float64) (float64, error) {
	fn, err := GetActivation(name)
	if err != nil {
		return 0, fmt.Errorf("unsupported activation: %s", name)
	}
	return fn(x), nil
}
