package genotype

import "protogonos/internal/model"

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

func fingerprintSpeciesKey(genome model.Genome) string {
	return "fp:" + ComputeGenomeSignature(genome).Fingerprint
}
