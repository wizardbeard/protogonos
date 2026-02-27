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

// SubstrateCEPFaninPIDsByEndpoint derives ordered fan-in neuron IDs for each
// substrate CEP endpoint ID.
func SubstrateCEPFaninPIDsByEndpoint(genome model.Genome) map[string][]string {
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

	seenByEndpoint := make(map[string]map[string]struct{}, len(cepEndpointSet))
	faninByEndpoint := make(map[string][]string, len(cepEndpointSet))
	for _, link := range genome.NeuronActuatorLinks {
		neuronID := link.NeuronID
		endpointID := link.ActuatorID
		if neuronID == "" {
			continue
		}
		if _, ok := cepEndpointSet[endpointID]; !ok {
			continue
		}
		seen, ok := seenByEndpoint[endpointID]
		if !ok {
			seen = map[string]struct{}{}
			seenByEndpoint[endpointID] = seen
		}
		if _, exists := seen[neuronID]; exists {
			continue
		}
		seen[neuronID] = struct{}{}
		faninByEndpoint[endpointID] = append(faninByEndpoint[endpointID], neuronID)
	}
	if len(faninByEndpoint) == 0 {
		return nil
	}
	return faninByEndpoint
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

// ResolveSubstrateCEPFaninPIDsByCEP resolves per-CEP fan-in IDs in substrate
// endpoint order with output-neuron fallback for missing endpoints.
func ResolveSubstrateCEPFaninPIDsByCEP(genome model.Genome, fallbackOutputNeuronIDs []string) [][]string {
	if genome.Substrate == nil {
		return nil
	}
	fallback := uniqueOrderedNonEmptyStrings(fallbackOutputNeuronIDs)
	if len(genome.Substrate.CEPIDs) == 0 {
		if len(fallback) == 0 {
			return nil
		}
		return [][]string{fallback}
	}

	byEndpoint := SubstrateCEPFaninPIDsByEndpoint(genome)
	out := make([][]string, 0, len(genome.Substrate.CEPIDs))
	for _, cepID := range genome.Substrate.CEPIDs {
		if fanin := byEndpoint[cepID]; len(fanin) > 0 {
			out = append(out, append([]string(nil), fanin...))
			continue
		}
		if len(fallback) > 0 {
			out = append(out, append([]string(nil), fallback...))
		}
	}
	if len(out) == 0 && len(fallback) > 0 {
		return [][]string{fallback}
	}
	return out
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
