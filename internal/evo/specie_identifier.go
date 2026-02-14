package evo

import (
	"fmt"
	"strings"

	"protogonos/internal/model"
)

// SpecieIdentifier assigns a stable species key to a genome.
type SpecieIdentifier interface {
	Name() string
	Identify(genome model.Genome) string
}

// TopologySpecieIdentifier groups genomes by coarse topology shape.
type TopologySpecieIdentifier struct{}

func (TopologySpecieIdentifier) Name() string {
	return "topology"
}

func (TopologySpecieIdentifier) Identify(genome model.Genome) string {
	return fmt.Sprintf("n:%d-s:%d-si:%d-ai:%d",
		len(genome.Neurons),
		len(genome.Synapses),
		len(genome.SensorIDs),
		len(genome.ActuatorIDs),
	)
}

// TotNSpecieIdentifier groups genomes by total neuron count.
// This mirrors the reference specie_identifier:tot_n/1 distinguisher intent.
type TotNSpecieIdentifier struct{}

func (TotNSpecieIdentifier) Name() string {
	return "tot_n"
}

func (TotNSpecieIdentifier) Identify(genome model.Genome) string {
	return fmt.Sprintf("tot_n:%d", len(genome.Neurons))
}

func SpecieIdentifierFromName(name string) (SpecieIdentifier, error) {
	switch strings.TrimSpace(strings.ToLower(name)) {
	case "", "topology", "pattern":
		return TopologySpecieIdentifier{}, nil
	case "tot_n":
		return TotNSpecieIdentifier{}, nil
	default:
		return nil, fmt.Errorf("unsupported specie identifier: %s", name)
	}
}

func SpecieIdentifierNameFromDistinguishers(distinguishers []string) string {
	for _, raw := range distinguishers {
		name := strings.TrimSpace(strings.ToLower(raw))
		switch name {
		case "tot_n":
			return "tot_n"
		case "pattern", "topology":
			return "topology"
		}
	}
	return ""
}
