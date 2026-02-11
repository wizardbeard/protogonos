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
	TotalNeurons           int            `json:"total_neurons"`
	TotalSynapses          int            `json:"total_synapses"`
	TotalRecurrentSynapses int            `json:"total_recurrent_synapses"`
	TotalSensors           int            `json:"total_sensors"`
	TotalActuators         int            `json:"total_actuators"`
	ActivationDistribution map[string]int `json:"activation_distribution"`
	AggregatorDistribution map[string]int `json:"aggregator_distribution"`
}

type GenomeSignature struct {
	Fingerprint string          `json:"fingerprint"`
	Summary     TopologySummary `json:"summary"`
}

func ComputeGenomeSignature(genome model.Genome) GenomeSignature {
	actDist := make(map[string]int)
	aggrDist := make(map[string]int)
	recurrent := 0

	for _, n := range genome.Neurons {
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
	}

	summary := TopologySummary{
		TotalNeurons:           len(genome.Neurons),
		TotalSynapses:          len(genome.Synapses),
		TotalRecurrentSynapses: recurrent,
		TotalSensors:           len(genome.SensorIDs),
		TotalActuators:         len(genome.ActuatorIDs),
		ActivationDistribution: actDist,
		AggregatorDistribution: aggrDist,
	}

	parts := []string{
		fmt.Sprintf("n=%d", summary.TotalNeurons),
		fmt.Sprintf("s=%d", summary.TotalSynapses),
		fmt.Sprintf("r=%d", summary.TotalRecurrentSynapses),
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
