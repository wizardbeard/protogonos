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
	neurons := make([]string, 0, len(genome.Neurons))
	for _, neuron := range genome.Neurons {
		neurons = append(neurons, neuron.ID)
	}
	return ReferenceFingerprint{
		Pattern:    CreateInitPattern(neurons),
		EvoHistory: GeneralizeEvoHistory(history),
		Sensors:    sortedUniqueStrings(genome.SensorIDs),
		Actuators:  sortedUniqueStrings(genome.ActuatorIDs),
		Topology:   UpdateNNTopologySummary(genome),
	}
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
