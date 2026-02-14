package nn

import (
	"fmt"
	"math"

	"protogonos/internal/model"
)

const outputSaturationLimit = 1.0

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

		total, err := aggregateIncoming(neuron.Aggregator, neuron.Bias, incoming[neuron.ID], values)
		if err != nil {
			return nil, fmt.Errorf("neuron %s: %w", neuron.ID, err)
		}

		activated, err := applyActivation(neuron.Activation, total)
		if err != nil {
			return nil, fmt.Errorf("neuron %s: %w", neuron.ID, err)
		}
		values[neuron.ID] = saturate(activated, -outputSaturationLimit, outputSaturationLimit)
	}

	return values, nil
}

func saturate(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func applyActivation(name string, x float64) (float64, error) {
	fn, err := GetActivation(name)
	if err != nil {
		return 0, fmt.Errorf("unsupported activation: %s", name)
	}
	return fn(x), nil
}

func aggregateIncoming(mode string, bias float64, synapses []model.Synapse, values map[string]float64) (float64, error) {
	switch mode {
	case "", "dot_product":
		total := bias
		for _, synapse := range synapses {
			total += values[synapse.From] * synapse.Weight
		}
		return total, nil
	case "mult_product":
		if len(synapses) == 0 {
			return bias, nil
		}
		total := 1.0
		for _, synapse := range synapses {
			total *= values[synapse.From] * synapse.Weight
		}
		return total + bias, nil
	case "diff_product":
		if len(synapses) == 0 {
			return bias, nil
		}
		total := values[synapses[0].From] * synapses[0].Weight
		for _, synapse := range synapses[1:] {
			total -= values[synapse.From] * synapse.Weight
		}
		// keep numerical behavior stable near +-Inf in pathological genomes
		if math.IsInf(total, 0) || math.IsNaN(total) {
			return 0, fmt.Errorf("invalid diff_product aggregate")
		}
		return total + bias, nil
	default:
		return 0, fmt.Errorf("unsupported aggregator: %s", mode)
	}
}
