package genotype

import "protogonos/internal/model"

func CloneGenome(g model.Genome) model.Genome {
	out := g
	out.Neurons = append([]model.Neuron(nil), g.Neurons...)
	out.Synapses = append([]model.Synapse(nil), g.Synapses...)
	out.SensorIDs = append([]string(nil), g.SensorIDs...)
	out.ActuatorIDs = append([]string(nil), g.ActuatorIDs...)

	if g.Substrate != nil {
		sub := *g.Substrate
		sub.Dimensions = append([]int(nil), g.Substrate.Dimensions...)
		if g.Substrate.Parameters != nil {
			sub.Parameters = make(map[string]float64, len(g.Substrate.Parameters))
			for k, v := range g.Substrate.Parameters {
				sub.Parameters[k] = v
			}
		}
		out.Substrate = &sub
	}
	if g.Plasticity != nil {
		p := *g.Plasticity
		out.Plasticity = &p
	}
	if g.Strategy != nil {
		s := *g.Strategy
		out.Strategy = &s
	}
	return out
}
