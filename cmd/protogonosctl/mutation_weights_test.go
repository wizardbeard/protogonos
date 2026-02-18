package main

import "testing"

func TestMutationWeightBucketCoversGenomeMutatorSurface(t *testing.T) {
	cases := map[string]string{
		"mutate_weights":                   "perturb",
		"add_bias":                         "bias",
		"remove_bias":                      "remove_bias",
		"mutate_af":                        "activation",
		"mutate_aggrf":                     "aggregator",
		"add_inlink":                       "add_synapse",
		"add_outlink":                      "add_synapse",
		"remove_inlink":                    "remove_synapse",
		"remove_outlink":                   "remove_synapse",
		"cutlink_FromElementToElement":     "remove_synapse",
		"cutlink_FromNeuronToNeuron":       "remove_synapse",
		"link_FromElementToElement":        "add_synapse",
		"link_FromNeuronToNeuron":          "add_synapse",
		"add_neuron":                       "add_neuron",
		"outsplice":                        "add_neuron",
		"insplice":                         "add_neuron",
		"remove_neuron":                    "remove_neuron",
		"mutate_pf":                        "plasticity_rule",
		"mutate_plasticity_parameters":     "plasticity",
		"add_sensor":                       "substrate",
		"add_sensorlink":                   "substrate",
		"add_actuator":                     "substrate",
		"add_actuatorlink":                 "substrate",
		"remove_sensor":                    "substrate",
		"remove_actuator":                  "substrate",
		"cutlink_FromSensorToNeuron":       "substrate",
		"cutlink_FromNeuronToActuator":     "substrate",
		"link_FromSensorToNeuron":          "substrate",
		"link_FromNeuronToActuator":        "substrate",
		"add_cpp":                          "substrate",
		"remove_cpp":                       "substrate",
		"add_cep":                          "substrate",
		"remove_cep":                       "substrate",
		"add_circuit_node":                 "substrate",
		"delete_circuit_node":              "substrate",
		"add_circuit_layer":                "substrate",
		"add_CircuitNode":                  "substrate",
		"delete_CircuitNode":               "substrate",
		"add_CircuitLayer":                 "substrate",
		"mutate_tuning_selection":          "substrate",
		"mutate_tuning_annealing":          "substrate",
		"mutate_tot_topological_mutations": "substrate",
		"mutate_heredity_type":             "substrate",
	}
	for name, want := range cases {
		if got := mutationWeightBucket(name); got != want {
			t.Fatalf("mutationWeightBucket(%q)=%q want=%q", name, got, want)
		}
	}
}
