package evo

import (
	"protogonos/internal/genotype"
	"protogonos/internal/model"
)

type TopologySummary = genotype.TopologySummary

type GenomeSignature = genotype.GenomeSignature

func ComputeGenomeSignature(genome model.Genome) GenomeSignature {
	return genotype.ComputeGenomeSignature(genome)
}
