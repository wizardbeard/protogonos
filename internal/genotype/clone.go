package genotype

import (
	"fmt"

	"protogonos/internal/model"
)

func CloneGenome(g model.Genome) model.Genome {
	out := g
	out.Neurons = append([]model.Neuron(nil), g.Neurons...)
	for i := range out.Neurons {
		if len(out.Neurons[i].PlasticityBiasParams) == 0 {
			continue
		}
		out.Neurons[i].PlasticityBiasParams = append([]float64(nil), out.Neurons[i].PlasticityBiasParams...)
	}
	out.Synapses = append([]model.Synapse(nil), g.Synapses...)
	for i := range out.Synapses {
		if len(out.Synapses[i].PlasticityParams) == 0 {
			continue
		}
		out.Synapses[i].PlasticityParams = append([]float64(nil), out.Synapses[i].PlasticityParams...)
	}
	out.SensorIDs = append([]string(nil), g.SensorIDs...)
	out.ActuatorIDs = append([]string(nil), g.ActuatorIDs...)
	if g.ActuatorTunables != nil {
		out.ActuatorTunables = make(map[string]float64, len(g.ActuatorTunables))
		for k, v := range g.ActuatorTunables {
			out.ActuatorTunables[k] = v
		}
	}
	if g.ActuatorGenerations != nil {
		out.ActuatorGenerations = make(map[string]int, len(g.ActuatorGenerations))
		for k, v := range g.ActuatorGenerations {
			out.ActuatorGenerations[k] = v
		}
	}
	out.SensorNeuronLinks = append([]model.SensorNeuronLink(nil), g.SensorNeuronLinks...)
	out.NeuronActuatorLinks = append([]model.NeuronActuatorLink(nil), g.NeuronActuatorLinks...)

	if g.Substrate != nil {
		sub := *g.Substrate
		sub.Dimensions = append([]int(nil), g.Substrate.Dimensions...)
		if g.Substrate.Parameters != nil {
			sub.Parameters = make(map[string]float64, len(g.Substrate.Parameters))
			for k, v := range g.Substrate.Parameters {
				sub.Parameters[k] = v
			}
		}
		out.Substrate = &sub
	}
	if g.Plasticity != nil {
		p := *g.Plasticity
		out.Plasticity = &p
	}
	if g.Strategy != nil {
		s := *g.Strategy
		out.Strategy = &s
	}
	return out
}

// CloneGenomeWithRemappedIDs clones a genome and remaps neuron/synapse IDs while
// preserving connectivity. Neuron IDs listed in preserveNeuronIDs are kept
// unchanged so callers can retain stable input/output anchors.
func CloneGenomeWithRemappedIDs(g model.Genome, newID string, preserveNeuronIDs []string) model.Genome {
	out := CloneGenome(g)
	if newID != "" {
		out.ID = newID
	}

	base := out.ID
	if base == "" {
		base = g.ID
	}
	if base == "" {
		base = "clone"
	}

	preserve := make(map[string]struct{}, len(preserveNeuronIDs))
	for _, id := range preserveNeuronIDs {
		if id == "" {
			continue
		}
		preserve[id] = struct{}{}
	}

	neuronIDMap := make(map[string]string, len(out.Neurons))
	usedNeuronIDs := make(map[string]struct{}, len(out.Neurons))
	for i := range out.Neurons {
		oldID := out.Neurons[i].ID
		if oldID == "" {
			continue
		}
		if _, keep := preserve[oldID]; keep {
			neuronIDMap[oldID] = oldID
			usedNeuronIDs[oldID] = struct{}{}
			continue
		}
		newNeuronID := nextUniqueCloneID(base, "nclone", i, usedNeuronIDs)
		neuronIDMap[oldID] = newNeuronID
		out.Neurons[i].ID = newNeuronID
	}

	synapseIDMap := make(map[string]string, len(out.Synapses))
	usedSynapseIDs := make(map[string]struct{}, len(out.Synapses))
	for i := range out.Synapses {
		oldSynapseID := out.Synapses[i].ID
		if oldSynapseID == "" {
			continue
		}
		newSynapseID := nextUniqueCloneID(base, "sclone", i, usedSynapseIDs)
		synapseIDMap[oldSynapseID] = newSynapseID
	}
	for i := range out.Synapses {
		if mappedID, ok := synapseIDMap[out.Synapses[i].ID]; ok {
			out.Synapses[i].ID = mappedID
		}
		if mappedFrom, ok := neuronIDMap[out.Synapses[i].From]; ok {
			out.Synapses[i].From = mappedFrom
		}
		if mappedTo, ok := neuronIDMap[out.Synapses[i].To]; ok {
			out.Synapses[i].To = mappedTo
		}
	}

	for i := range out.SensorNeuronLinks {
		if mappedNeuronID, ok := neuronIDMap[out.SensorNeuronLinks[i].NeuronID]; ok {
			out.SensorNeuronLinks[i].NeuronID = mappedNeuronID
		}
	}
	for i := range out.NeuronActuatorLinks {
		if mappedNeuronID, ok := neuronIDMap[out.NeuronActuatorLinks[i].NeuronID]; ok {
			out.NeuronActuatorLinks[i].NeuronID = mappedNeuronID
		}
	}

	return out
}

func nextUniqueCloneID(base, kind string, index int, used map[string]struct{}) string {
	candidate := fmt.Sprintf("%s-%s-%d", base, kind, index)
	if _, exists := used[candidate]; !exists {
		used[candidate] = struct{}{}
		return candidate
	}
	for suffix := 1; ; suffix++ {
		candidate = fmt.Sprintf("%s-%s-%d-%d", base, kind, index, suffix)
		if _, exists := used[candidate]; exists {
			continue
		}
		used[candidate] = struct{}{}
		return candidate
	}
}
