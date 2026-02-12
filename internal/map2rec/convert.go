package map2rec

import "math"

func Convert(kind string, in map[string]any) (any, error) {
	switch kind {
	case "constraint":
		return ConvertConstraint(in), nil
	case "pmp":
		return ConvertPMP(in), nil
	default:
		return nil, ErrUnsupportedKind
	}
}

func ConvertConstraint(in map[string]any) ConstraintRecord {
	out := defaultConstraintRecord()
	for key, val := range in {
		switch key {
		case "morphology":
			if s, ok := asString(val); ok {
				out.Morphology = s
			}
		case "connection_architecture":
			if s, ok := asString(val); ok {
				out.ConnectionArchitecture = s
			}
		case "neural_afs":
			if xs, ok := asStrings(val); ok {
				out.NeuralAFs = xs
			}
		case "neural_pfns":
			if xs, ok := asStrings(val); ok {
				out.NeuralPFNs = xs
			}
		case "substrate_plasticities":
			if xs, ok := asStrings(val); ok {
				out.SubstratePlasticities = xs
			}
		case "substrate_linkforms":
			if xs, ok := asStrings(val); ok {
				out.SubstrateLinkforms = xs
			}
		case "neural_aggr_fs":
			if xs, ok := asStrings(val); ok {
				out.NeuralAggrFs = xs
			}
		case "tuning_selection_fs":
			if xs, ok := asStrings(val); ok {
				out.TuningSelectionFs = xs
			}
		case "tuning_duration_f":
			if spec, ok := asDurationSpec(val); ok {
				out.TuningDurationF = spec
			}
		case "annealing_parameters":
			if xs, ok := asFloat64s(val); ok {
				out.AnnealingParameters = xs
			}
		case "perturbation_ranges":
			if xs, ok := asFloat64s(val); ok {
				out.PerturbationRanges = xs
			}
		case "agent_encoding_types":
			if xs, ok := asStrings(val); ok {
				out.AgentEncodingTypes = xs
			}
		case "heredity_types":
			if xs, ok := asStrings(val); ok {
				out.HeredityTypes = xs
			}
		case "mutation_operators":
			if xs, ok := asWeightedOperators(val); ok {
				out.MutationOperators = xs
			}
		case "tot_topological_mutations_fs":
			if xs, ok := asMutationCountPolicies(val); ok {
				out.TotTopologicalMutationsFs = xs
			}
		case "population_evo_alg_f":
			if s, ok := asString(val); ok {
				out.PopulationEvoAlgF = s
			}
		case "population_fitness_postprocessor_f":
			if s, ok := asString(val); ok {
				out.PopulationFitnessProcessorF = s
			}
		case "population_selection_f":
			if s, ok := asString(val); ok {
				out.PopulationSelectionF = s
			}
		case "specie_distinguishers":
			if xs, ok := asStrings(val); ok {
				out.SpecieDistinguishers = xs
			}
		case "hof_distinguishers":
			if xs, ok := asStrings(val); ok {
				out.HOFDistinguishers = xs
			}
		case "objectives":
			if xs, ok := asStrings(val); ok {
				out.Objectives = xs
			}
		}
	}
	return out
}

func ConvertPMP(in map[string]any) PMPRecord {
	out := defaultPMPRecord()
	for key, val := range in {
		switch key {
		case "op_mode":
			if s, ok := asString(val); ok {
				out.OpMode = s
			}
		case "population_id":
			if s, ok := asString(val); ok {
				out.PopulationID = s
			}
		case "survival_percentage":
			if f, ok := asFloat64(val); ok {
				out.SurvivalPercentage = f
			}
		case "specie_size_limit":
			if n, ok := asInt(val); ok {
				out.SpecieSizeLimit = n
			}
		case "init_specie_size":
			if n, ok := asInt(val); ok {
				out.InitSpecieSize = n
			}
		case "polis_id":
			if s, ok := asString(val); ok {
				out.PolisID = s
			}
		case "generation_limit":
			if n, ok := asInt(val); ok {
				out.GenerationLimit = n
			}
		case "evaluations_limit":
			if n, ok := asInt(val); ok {
				out.EvaluationsLimit = n
			}
		case "fitness_goal":
			if f, ok := asFloat64(val); ok {
				out.FitnessGoal = f
			}
			if s, ok := asString(val); ok && s == "inf" {
				out.FitnessGoal = math.Inf(1)
			}
		case "benchmarker_pid":
			if s, ok := asString(val); ok {
				out.BenchmarkerPID = s
			}
		case "committee_pid":
			if s, ok := asString(val); ok {
				out.CommitteePID = s
			}
		}
	}
	return out
}
