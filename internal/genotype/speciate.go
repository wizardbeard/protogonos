package genotype

import (
	"fmt"
	"strings"

	"protogonos/internal/model"
)

// FingerprintSpecies is a Go-native species record analog for
// genotype:speciate/1 side effects.
type FingerprintSpecies struct {
	ID          string
	Fingerprint string
	GenomeIDs   []string
}

// SpeciateByFingerprint groups genomes by exact generalized reference
// fingerprint (pattern/evo-history/io/topology summary analog).
func SpeciateByFingerprint(genomes []model.Genome) map[string][]model.Genome {
	species := make(map[string][]model.Genome, len(genomes))
	for _, genome := range genomes {
		key := fingerprintSpeciesKey(genome)
		species[key] = append(species[key], CloneGenome(genome))
	}
	return species
}

// AssignToFingerprintSpecies appends one genome into its exact generalized
// reference-fingerprint species bucket and returns the selected species key.
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

// SpeciateInPopulation mirrors genotype:speciate/1 side effects by reusing
// existing fingerprint species in a population view or creating a new species
// ID when no fingerprint match exists.
func SpeciateInPopulation(
	genome model.Genome,
	speciesByID map[string]FingerprintSpecies,
	nextSpeciesID func(fingerprint string) string,
) (string, map[string]FingerprintSpecies) {
	if speciesByID == nil {
		speciesByID = map[string]FingerprintSpecies{}
	}
	genomeID := strings.TrimSpace(genome.ID)
	if strings.EqualFold(genomeID, "test") || genomeID == "" {
		return "", speciesByID
	}
	fingerprint := fingerprintSpeciesKey(genome)

	for speciesID, species := range speciesByID {
		if species.Fingerprint != fingerprint {
			continue
		}
		species.GenomeIDs = appendUniqueString(species.GenomeIDs, genomeID)
		speciesByID[speciesID] = species
		return speciesID, speciesByID
	}

	speciesID := ""
	if nextSpeciesID != nil {
		speciesID = strings.TrimSpace(nextSpeciesID(fingerprint))
	}
	if speciesID == "" {
		speciesID = defaultSpeciesID(fingerprint, speciesByID)
	}
	speciesByID[speciesID] = FingerprintSpecies{
		ID:          speciesID,
		Fingerprint: fingerprint,
		GenomeIDs:   []string{genomeID},
	}
	return speciesID, speciesByID
}

// ComputeSpeciationFingerprintKey returns the stable species-key fingerprint
// used by Speciate* helpers. History is optional and defaults to empty in
// callsites that do not track evo-history.
func ComputeSpeciationFingerprintKey(genome model.Genome, history []EvoHistoryEvent) string {
	return "fp:" + ComputeReferenceFingerprint(genome, history)
}

func fingerprintSpeciesKey(genome model.Genome) string {
	return ComputeSpeciationFingerprintKey(genome, nil)
}

func defaultSpeciesID(fingerprint string, speciesByID map[string]FingerprintSpecies) string {
	base := strings.TrimPrefix(strings.TrimSpace(fingerprint), "fp:")
	if base == "" {
		base = "unknown"
	}
	candidate := fmt.Sprintf("species:%s", base)
	if _, exists := speciesByID[candidate]; !exists {
		return candidate
	}
	for i := 1; ; i++ {
		candidate = fmt.Sprintf("species:%s:%d", base, i)
		if _, exists := speciesByID[candidate]; exists {
			continue
		}
		return candidate
	}
}

func appendUniqueString(values []string, value string) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}
