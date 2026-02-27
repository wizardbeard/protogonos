package genotype

import "protogonos/internal/model"

// SubstrateCEPFaninPIDs derives ordered CEP fan-in neuron IDs from the
// genome's actuator-link topology. Only links targeting substrate CEP
// endpoints are considered, and duplicates are removed while preserving first
// occurrence order.
func SubstrateCEPFaninPIDs(genome model.Genome) []string {
	if genome.Substrate == nil || len(genome.Substrate.CEPIDs) == 0 || len(genome.NeuronActuatorLinks) == 0 {
		return nil
	}

	cepEndpointSet := make(map[string]struct{}, len(genome.Substrate.CEPIDs))
	for _, cepID := range genome.Substrate.CEPIDs {
		if cepID == "" {
			continue
		}
		cepEndpointSet[cepID] = struct{}{}
	}
	if len(cepEndpointSet) == 0 {
		return nil
	}

	seen := map[string]struct{}{}
	fanin := make([]string, 0, len(genome.NeuronActuatorLinks))
	for _, link := range genome.NeuronActuatorLinks {
		if link.NeuronID == "" {
			continue
		}
		if _, ok := cepEndpointSet[link.ActuatorID]; !ok {
			continue
		}
		if _, exists := seen[link.NeuronID]; exists {
			continue
		}
		seen[link.NeuronID] = struct{}{}
		fanin = append(fanin, link.NeuronID)
	}
	if len(fanin) == 0 {
		return nil
	}
	return fanin
}

// ResolveSubstrateCEPFaninPIDs returns CEP fan-in IDs derived from explicit
// substrate CEP endpoint links when available, otherwise falls back to ordered
// output-neuron IDs.
func ResolveSubstrateCEPFaninPIDs(genome model.Genome, fallbackOutputNeuronIDs []string) []string {
	if fanin := SubstrateCEPFaninPIDs(genome); len(fanin) > 0 {
		return fanin
	}
	return uniqueOrderedNonEmptyStrings(fallbackOutputNeuronIDs)
}

func uniqueOrderedNonEmptyStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
