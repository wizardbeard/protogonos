package genotype

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"

	"protogonos/internal/model"
)

// ReferenceFingerprint captures a reference-closer generalized fingerprint
// structure akin to genotype:update_fingerprint/1 composition.
type ReferenceFingerprint struct {
	Pattern    []PatternLayer               `json:"pattern"`
	EvoHistory []GeneralizedEvoHistoryEvent `json:"evo_history,omitempty"`
	Sensors    []string                     `json:"sensors"`
	Actuators  []string                     `json:"actuators"`
	Topology   TopologySummary              `json:"topology"`
}

// BuildReferenceFingerprint builds generalized fingerprint parts mirroring
// reference update_fingerprint/1 intent.
func BuildReferenceFingerprint(genome model.Genome, history []EvoHistoryEvent) ReferenceFingerprint {
	pattern := buildReferencePattern(genome)
	return ReferenceFingerprint{
		Pattern:    pattern,
		EvoHistory: GeneralizeEvoHistory(history),
		Sensors:    sortedUniqueStrings(genome.SensorIDs),
		Actuators:  sortedUniqueStrings(genome.ActuatorIDs),
		Topology:   UpdateNNTopologySummary(genome),
	}
}

func buildReferencePattern(genome model.Genome) []PatternLayer {
	neuronIDs := uniqueGenomeNeuronIDs(genome)
	if len(neuronIDs) == 0 {
		return nil
	}

	pattern := CreateInitPattern(neuronIDs)
	if len(pattern) == 0 {
		return inferPatternFromTopology(genome)
	}

	covered := make(map[string]struct{}, len(neuronIDs))
	for _, layer := range pattern {
		for _, neuronID := range layer.NeuronIDs {
			covered[neuronID] = struct{}{}
		}
	}
	if len(covered) == len(neuronIDs) {
		return pattern
	}

	// If only a subset of IDs carried parseable layer tags, use a topology-
	// inferred fallback so pattern contributes all neurons consistently.
	inferred := inferPatternFromTopology(genome)
	if len(inferred) > 0 {
		return inferred
	}
	return pattern
}

func uniqueGenomeNeuronIDs(genome model.Genome) []string {
	seen := make(map[string]struct{}, len(genome.Neurons))
	out := make([]string, 0, len(genome.Neurons))
	for _, neuron := range genome.Neurons {
		id := strings.TrimSpace(neuron.ID)
		if id == "" {
			continue
		}
		if _, exists := seen[id]; exists {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

func inferPatternFromTopology(genome model.Genome) []PatternLayer {
	neuronIDs := uniqueGenomeNeuronIDs(genome)
	if len(neuronIDs) == 0 {
		return nil
	}

	neuronSet := make(map[string]struct{}, len(neuronIDs))
	for _, neuronID := range neuronIDs {
		neuronSet[neuronID] = struct{}{}
	}

	adjacency := make(map[string][]string, len(neuronIDs))
	inDegree := make(map[string]int, len(neuronIDs))
	edgeSeen := make(map[string]struct{}, len(genome.Synapses))
	for _, neuronID := range neuronIDs {
		inDegree[neuronID] = 0
	}
	for _, synapse := range genome.Synapses {
		if !synapse.Enabled || synapse.Recurrent {
			continue
		}
		from := strings.TrimSpace(synapse.From)
		to := strings.TrimSpace(synapse.To)
		if from == "" || to == "" || from == to {
			continue
		}
		if _, ok := neuronSet[from]; !ok {
			continue
		}
		if _, ok := neuronSet[to]; !ok {
			continue
		}
		edgeKey := from + "->" + to
		if _, exists := edgeSeen[edgeKey]; exists {
			continue
		}
		edgeSeen[edgeKey] = struct{}{}
		adjacency[from] = append(adjacency[from], to)
		inDegree[to]++
	}
	for from := range adjacency {
		sort.Strings(adjacency[from])
	}

	depth := make(map[string]int, len(neuronIDs))
	queue := make([]string, 0, len(neuronIDs))
	queued := make(map[string]struct{}, len(neuronIDs))
	for _, link := range genome.SensorNeuronLinks {
		neuronID := strings.TrimSpace(link.NeuronID)
		if _, ok := neuronSet[neuronID]; !ok {
			continue
		}
		if _, exists := queued[neuronID]; exists {
			continue
		}
		queue = append(queue, neuronID)
		queued[neuronID] = struct{}{}
	}
	for _, neuronID := range neuronIDs {
		if inDegree[neuronID] != 0 {
			continue
		}
		if _, exists := queued[neuronID]; exists {
			continue
		}
		queue = append(queue, neuronID)
		queued[neuronID] = struct{}{}
	}
	if len(queue) == 0 {
		queue = append(queue, neuronIDs...)
	}
	sort.Strings(queue)

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, next := range adjacency[current] {
			if depth[next] < depth[current]+1 {
				depth[next] = depth[current] + 1
			}
			inDegree[next]--
			if inDegree[next] == 0 {
				queue = append(queue, next)
			}
		}
	}

	byLayer := make(map[int][]string, len(neuronIDs))
	layerOrder := make([]int, 0, len(neuronIDs))
	layerSeen := make(map[int]struct{}, len(neuronIDs))
	for _, neuronID := range neuronIDs {
		layer := depth[neuronID]
		if _, exists := layerSeen[layer]; !exists {
			layerSeen[layer] = struct{}{}
			layerOrder = append(layerOrder, layer)
		}
		byLayer[layer] = append(byLayer[layer], neuronID)
	}
	sort.Ints(layerOrder)

	pattern := make([]PatternLayer, 0, len(layerOrder))
	for _, layer := range layerOrder {
		ids := append([]string(nil), byLayer[layer]...)
		sort.Strings(ids)
		pattern = append(pattern, PatternLayer{
			Layer:     float64(layer),
			NeuronIDs: ids,
		})
	}
	return pattern
}

// ComputeReferenceFingerprint hashes BuildReferenceFingerprint output in a
// deterministic text encoding.
func ComputeReferenceFingerprint(genome model.Genome, history []EvoHistoryEvent) string {
	fingerprint := BuildReferenceFingerprint(genome, history)
	parts := make([]string, 0, 32)

	for _, layer := range fingerprint.Pattern {
		parts = append(parts, fmt.Sprintf("p:%.6f:%d", layer.Layer, len(layer.NeuronIDs)))
	}
	for _, event := range fingerprint.EvoHistory {
		parts = append(parts, "m:"+strings.TrimSpace(event.Mutation))
		for _, element := range event.Elements {
			if element.Layer != nil {
				parts = append(parts, fmt.Sprintf("e:%s:%.6f", element.Kind, *element.Layer))
				continue
			}
			parts = append(parts, "e:"+element.Kind)
		}
	}
	for _, sensor := range fingerprint.Sensors {
		parts = append(parts, "s:"+sensor)
	}
	for _, actuator := range fingerprint.Actuators {
		parts = append(parts, "a:"+actuator)
	}
	parts = append(parts,
		fmt.Sprintf("t:%s", fingerprint.Topology.Type),
		fmt.Sprintf("n:%d", fingerprint.Topology.TotalNeurons),
		fmt.Sprintf("nils:%d", fingerprint.Topology.TotalNILs),
		fmt.Sprintf("nols:%d", fingerprint.Topology.TotalNOLs),
		fmt.Sprintf("nros:%d", fingerprint.Topology.TotalNROs),
	)
	appendDistribution := func(prefix string, values map[string]int) {
		keys := make([]string, 0, len(values))
		for key := range values {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			parts = append(parts, fmt.Sprintf("%s:%s=%d", prefix, key, values[key]))
		}
	}
	appendDistribution("af", fingerprint.Topology.ActivationDistribution)
	appendDistribution("ag", fingerprint.Topology.AggregatorDistribution)

	digest := sha1.Sum([]byte(strings.Join(parts, "|")))
	return hex.EncodeToString(digest[:8])
}

func sortedUniqueStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}
