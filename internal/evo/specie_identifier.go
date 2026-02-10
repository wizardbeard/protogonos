package evo

import (
	"fmt"

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
