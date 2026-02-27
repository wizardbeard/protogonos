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
