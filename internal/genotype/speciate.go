package genotype

import (
	"strings"

	"protogonos/internal/model"
)

// SpeciateByFingerprint groups genomes by exact topology fingerprint.
func SpeciateByFingerprint(genomes []model.Genome) map[string][]model.Genome {
	species := make(map[string][]model.Genome, len(genomes))
	for _, genome := range genomes {
		key := fingerprintSpeciesKey(genome)
		species[key] = append(species[key], CloneGenome(genome))
	}
	return species
}

// AssignToFingerprintSpecies appends one genome into its exact-fingerprint
// species bucket and returns the selected species key.
func AssignToFingerprintSpecies(genome model.Genome, species map[string][]model.Genome) (string, map[string][]model.Genome) {
	if species == nil {
		species = map[string][]model.Genome{}
	}
	key := fingerprintSpeciesKey(genome)
	species[key] = append(species[key], CloneGenome(genome))
	return key, species
}

// Speciate is an explicit entrypoint analog to genotype:speciate/1.
// Test genomes are not assigned to species buckets.
func Speciate(genome model.Genome, species map[string][]model.Genome) (string, map[string][]model.Genome) {
	if species == nil {
		species = map[string][]model.Genome{}
	}
	if strings.EqualFold(strings.TrimSpace(genome.ID), "test") {
		return "", species
	}
	return AssignToFingerprintSpecies(genome, species)
}

func fingerprintSpeciesKey(genome model.Genome) string {
	return "fp:" + ComputeGenomeSignature(genome).Fingerprint
}
