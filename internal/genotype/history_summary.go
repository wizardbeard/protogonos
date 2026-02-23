package genotype

import (
	"strings"

	"protogonos/internal/model"
)

// NodeSummary is a helper analog for reference get_NodeSummary/1 output.
type NodeSummary struct {
	TotalNILs              int
	TotalNOLs              int
	TotalNROs              int
	ActivationDistribution map[string]int
}

// GeneralizedElementRef is a generalized element descriptor used by
// GeneralizeEvoHistory as a Go analog for generalized evo-history tuples.
type GeneralizedElementRef struct {
	Layer *float64 `json:"layer,omitempty"`
	Kind  string   `json:"kind"`
}

// GeneralizedEvoHistoryEvent is a generalized evo-history event analog.
type GeneralizedEvoHistoryEvent struct {
	Mutation string                  `json:"mutation"`
	Elements []GeneralizedElementRef `json:"elements,omitempty"`
}

// GetNodeSummary mirrors genotype:get_NodeSummary/1 in the simplified model.
func GetNodeSummary(genome model.Genome) NodeSummary {
	neuronIDs := make(map[string]struct{}, len(genome.Neurons))
	activationDistribution := make(map[string]int)
	for _, neuron := range genome.Neurons {
		neuronIDs[neuron.ID] = struct{}{}
		activationDistribution[neuron.Activation]++
	}

	totalNILs := 0
	totalNOLs := 0
	totalNROs := 0
	for _, synapse := range genome.Synapses {
		if _, ok := neuronIDs[synapse.To]; ok {
			totalNILs++
		}
		if _, ok := neuronIDs[synapse.From]; ok {
			totalNOLs++
			if synapse.Recurrent {
				totalNROs++
			}
		}
	}
	totalNILs += len(genome.SensorNeuronLinks)
	totalNOLs += len(genome.NeuronActuatorLinks)

	return NodeSummary{
		TotalNILs:              totalNILs,
		TotalNOLs:              totalNOLs,
		TotalNROs:              totalNROs,
		ActivationDistribution: activationDistribution,
	}
}

// GeneralizeEvoHistory mirrors genotype:generalize_EvoHist/2 intent by
// replacing element IDs with generalized layer/type descriptors.
func GeneralizeEvoHistory(history []EvoHistoryEvent) []GeneralizedEvoHistoryEvent {
	if len(history) == 0 {
		return nil
	}
	out := make([]GeneralizedEvoHistoryEvent, 0, len(history))
	for _, event := range history {
		generalized := GeneralizedEvoHistoryEvent{
			Mutation: event.Mutation,
		}
		if len(event.IDs) > 0 {
			generalized.Elements = make([]GeneralizedElementRef, 0, len(event.IDs))
			for _, id := range event.IDs {
				generalized.Elements = append(generalized.Elements, generalizeElementID(id))
			}
		}
		out = append(out, generalized)
	}
	return out
}

func generalizeElementID(id string) GeneralizedElementRef {
	layer, hasLayer := parseLayerIndex(id)
	kind := inferElementKind(id, hasLayer)
	if hasLayer {
		layerCopy := layer
		return GeneralizedElementRef{
			Layer: &layerCopy,
			Kind:  kind,
		}
	}
	return GeneralizedElementRef{
		Kind: kind,
	}
}

func inferElementKind(id string, hasLayer bool) string {
	lower := strings.ToLower(strings.TrimSpace(id))
	switch {
	case strings.Contains(lower, "synapse"):
		return "synapse"
	case strings.Contains(lower, "sensor"):
		return "sensor"
	case strings.Contains(lower, "actuator"):
		return "actuator"
	case strings.Contains(lower, "cortex"):
		return "cortex"
	case strings.Contains(lower, "substrate"):
		return "substrate"
	case strings.Contains(lower, "strategy"):
		return "strategy"
	case strings.Contains(lower, "plasticity"):
		return "plasticity"
	case strings.Contains(lower, "neuron"):
		return "neuron"
	case hasLayer:
		return "neuron"
	default:
		return "element"
	}
}
