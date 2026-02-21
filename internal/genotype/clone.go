package genotype

import "protogonos/internal/model"

func CloneGenome(g model.Genome) model.Genome {
	out := g
	out.Neurons = append([]model.Neuron(nil), g.Neurons...)
	out.Synapses = append([]model.Synapse(nil), g.Synapses...)
	for i := range out.Synapses {
		if len(out.Synapses[i].PlasticityParams) == 0 {
			continue
		}
		out.Synapses[i].PlasticityParams = append([]float64(nil), out.Synapses[i].PlasticityParams...)
	}
	out.SensorIDs = append([]string(nil), g.SensorIDs...)
	out.ActuatorIDs = append([]string(nil), g.ActuatorIDs...)
	if g.ActuatorTunables != nil {
		out.ActuatorTunables = make(map[string]float64, len(g.ActuatorTunables))
		for k, v := range g.ActuatorTunables {
			out.ActuatorTunables[k] = v
		}
	}
	if g.ActuatorGenerations != nil {
		out.ActuatorGenerations = make(map[string]int, len(g.ActuatorGenerations))
		for k, v := range g.ActuatorGenerations {
			out.ActuatorGenerations[k] = v
		}
	}
	out.SensorNeuronLinks = append([]model.SensorNeuronLink(nil), g.SensorNeuronLinks...)
	out.NeuronActuatorLinks = append([]model.NeuronActuatorLink(nil), g.NeuronActuatorLinks...)

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
