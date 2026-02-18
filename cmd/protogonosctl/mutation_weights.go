package main

func normalizeMutationOperatorName(name string) string {
	switch name {
	case "add_CircuitNode":
		return "add_circuit_node"
	case "delete_CircuitNode":
		return "delete_circuit_node"
	case "add_CircuitLayer":
		return "add_circuit_layer"
	default:
		return name
	}
}

func mutationWeightBucket(name string) string {
	switch normalizeMutationOperatorName(name) {
	case "mutate_weights":
		return "perturb"
	case "add_bias":
		return "bias"
	case "remove_bias":
		return "remove_bias"
	case "mutate_af":
		return "activation"
	case "mutate_aggrf":
		return "aggregator"
	case "add_outlink", "add_inlink", "link_FromElementToElement", "link_FromNeuronToNeuron":
		return "add_synapse"
	case "remove_outlink", "remove_inlink", "cutlink_FromNeuronToNeuron", "cutlink_FromElementToElement":
		return "remove_synapse"
	case "add_neuron", "outsplice", "insplice":
		return "add_neuron"
	case "remove_neuron":
		return "remove_neuron"
	case "mutate_plasticity_parameters":
		return "plasticity"
	case "mutate_pf":
		return "plasticity_rule"
	case "add_sensor", "add_sensorlink", "add_actuator", "add_actuatorlink", "add_cpp", "remove_cpp", "add_cep", "remove_cep", "add_circuit_node", "delete_circuit_node", "add_circuit_layer", "remove_sensor", "remove_actuator", "cutlink_FromSensorToNeuron", "cutlink_FromNeuronToActuator", "link_FromSensorToNeuron", "link_FromNeuronToActuator", "mutate_tuning_selection", "mutate_tuning_annealing", "mutate_tot_topological_mutations", "mutate_heredity_type":
		return "substrate"
	default:
		return ""
	}
}
