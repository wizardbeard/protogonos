package genotype

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"

	"protogonos/internal/model"
)

type TopologySummary struct {
	Type                   string         `json:"type,omitempty"`
	TotalNeurons           int            `json:"total_neurons"`
	TotalSynapses          int            `json:"total_synapses"`
	TotalRecurrentSynapses int            `json:"total_recurrent_synapses"`
	TotalNILs              int            `json:"total_n_ils"`
	TotalNOLs              int            `json:"total_n_ols"`
	TotalNROs              int            `json:"total_n_ros"`
	TotalSensors           int            `json:"total_sensors"`
	TotalActuators         int            `json:"total_actuators"`
	ActivationDistribution map[string]int `json:"activation_distribution"`
	AggregatorDistribution map[string]int `json:"aggregator_distribution"`
}

type GenomeSignature struct {
	Fingerprint string          `json:"fingerprint"`
	Summary     TopologySummary `json:"summary"`
}

// UpdateFingerprint is an explicit helper analog to genotype:update_fingerprint/1.
func UpdateFingerprint(genome model.Genome) string {
	return ComputeGenomeSignature(genome).Fingerprint
}

// UpdateNNTopologySummary is an explicit helper analog to
// genotype:update_NNTopologySummary/1.
func UpdateNNTopologySummary(genome model.Genome) TopologySummary {
	return ComputeGenomeSignature(genome).Summary
}

func ComputeGenomeSignature(genome model.Genome) GenomeSignature {
	actDist := make(map[string]int)
	aggrDist := make(map[string]int)
	recurrent := 0
	totalNILs := 0
	totalNOLs := 0
	totalNROs := 0
	neuronIDs := make(map[string]struct{}, len(genome.Neurons))

	for _, n := range genome.Neurons {
		neuronIDs[n.ID] = struct{}{}
		actDist[n.Activation]++
		aggr := n.Aggregator
		if aggr == "" {
			aggr = "dot_product"
		}
		aggrDist[aggr]++
	}
	for _, s := range genome.Synapses {
		if s.Recurrent {
			recurrent++
		}
		if _, ok := neuronIDs[s.To]; ok {
			totalNILs++
		}
		if _, ok := neuronIDs[s.From]; ok {
			totalNOLs++
			if s.Recurrent {
				totalNROs++
			}
		}
	}
	totalNILs += len(genome.SensorNeuronLinks)
	totalNOLs += len(genome.NeuronActuatorLinks)

	encodingType := "neural"
	if genome.Substrate != nil {
		encodingType = "substrate"
	}

	summary := TopologySummary{
		Type:                   encodingType,
		TotalNeurons:           len(genome.Neurons),
		TotalSynapses:          len(genome.Synapses),
		TotalRecurrentSynapses: recurrent,
		TotalNILs:              totalNILs,
		TotalNOLs:              totalNOLs,
		TotalNROs:              totalNROs,
		TotalSensors:           len(genome.SensorIDs),
		TotalActuators:         len(genome.ActuatorIDs),
		ActivationDistribution: actDist,
		AggregatorDistribution: aggrDist,
	}

	parts := []string{
		fmt.Sprintf("t=%s", summary.Type),
		fmt.Sprintf("n=%d", summary.TotalNeurons),
		fmt.Sprintf("s=%d", summary.TotalSynapses),
		fmt.Sprintf("r=%d", summary.TotalRecurrentSynapses),
		fmt.Sprintf("nils=%d", summary.TotalNILs),
		fmt.Sprintf("nols=%d", summary.TotalNOLs),
		fmt.Sprintf("nros=%d", summary.TotalNROs),
		fmt.Sprintf("si=%d", summary.TotalSensors),
		fmt.Sprintf("ao=%d", summary.TotalActuators),
	}
	appendDist := func(prefix string, m map[string]int) {
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%s:%s=%d", prefix, k, m[k]))
		}
	}
	appendDist("af", actDist)
	appendDist("aggr", aggrDist)

	digest := sha1.Sum([]byte(strings.Join(parts, "|")))
	fingerprint := hex.EncodeToString(digest[:8])
	return GenomeSignature{
		Fingerprint: fingerprint,
		Summary:     summary,
	}
}
