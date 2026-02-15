package nn

import (
	"fmt"
	"math"

	"protogonos/internal/model"
)

const outputSaturationLimit = 1.0

type ForwardState struct {
	prevDiffInputs map[string][]float64
}

func NewForwardState() *ForwardState {
	return &ForwardState{prevDiffInputs: map[string][]float64{}}
}

func Forward(genome model.Genome, inputByNeuron map[string]float64) (map[string]float64, error) {
	return ForwardWithState(genome, inputByNeuron, nil)
}

func ForwardWithState(genome model.Genome, inputByNeuron map[string]float64, state *ForwardState) (map[string]float64, error) {
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

		total, err := aggregateIncoming(neuron.ID, neuron.Aggregator, neuron.Bias, incoming[neuron.ID], values, state)
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

func aggregateIncoming(
	neuronID, mode string,
	bias float64,
	synapses []model.Synapse,
	values map[string]float64,
	state *ForwardState,
) (float64, error) {
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
		// Reference mult_product is multiplicative; treat neuron bias as a
		// multiplicative factor when present.
		if bias != 0 {
			total *= bias
		}
		return total, nil
	case "diff_product":
		if len(synapses) == 0 {
			return bias, nil
		}
		rawInputs := make([]float64, len(synapses))
		for i, synapse := range synapses {
			rawInputs[i] = values[synapse.From]
		}
		diffInputs := rawInputs
		if state != nil {
			if prev, ok := state.prevDiffInputs[neuronID]; ok && len(prev) == len(rawInputs) {
				diffInputs = make([]float64, len(rawInputs))
				for i := range rawInputs {
					diffInputs[i] = rawInputs[i] - prev[i]
				}
			}
			state.prevDiffInputs[neuronID] = append([]float64(nil), rawInputs...)
		}

		total := bias
		for i, synapse := range synapses {
			total += diffInputs[i] * synapse.Weight
		}
		// keep numerical behavior stable near +-Inf in pathological genomes
		if math.IsInf(total, 0) || math.IsNaN(total) {
			return 0, fmt.Errorf("invalid diff_product aggregate")
		}
		return total, nil
	default:
		return 0, fmt.Errorf("unsupported aggregator: %s", mode)
	}
}
