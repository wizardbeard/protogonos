package map2rec

import "math"

func Convert(kind string, in map[string]any) (any, error) {
	switch kind {
	case "constraint":
		return ConvertConstraint(in), nil
	case "pmp":
		return ConvertPMP(in), nil
	case "sensor":
		return ConvertSensor(in), nil
	case "actuator":
		return ConvertActuator(in), nil
	case "neuron":
		return ConvertNeuron(in), nil
	case "agent":
		return ConvertAgent(in), nil
	case "cortex":
		return ConvertCortex(in), nil
	case "substrate":
		return ConvertSubstrate(in), nil
	case "polis":
		return ConvertPolis(in), nil
	case "scape":
		return ConvertScape(in), nil
	case "sector":
		return ConvertSector(in), nil
	case "avatar":
		return ConvertAvatar(in), nil
	case "object":
		return ConvertObject(in), nil
	case "circle":
		return ConvertCircle(in), nil
	case "square":
		return ConvertSquare(in), nil
	case "specie":
		return ConvertSpecie(in), nil
	case "population":
		return ConvertPopulation(in), nil
	case "trace":
		return ConvertTrace(in), nil
	case "stat":
		return ConvertStat(in), nil
	case "topology_summary":
		return ConvertTopologySummary(in), nil
	case "signature":
		return ConvertSignature(in), nil
	case "champion":
		return ConvertChampion(in), nil
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

func ConvertSensor(in map[string]any) SensorRecord {
	out := defaultSensorRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "name":
			if s, ok := asString(val); ok {
				out.Name = s
			}
		case "type":
			if s, ok := asString(val); ok {
				out.Type = s
			}
		case "cx_id":
			out.CortexID = val
		case "scape":
			out.Scape = val
		case "vl":
			if n, ok := asInt(val); ok {
				out.VL = n
			}
		case "fanout_ids":
			if xs, ok := asAnySlice(val); ok {
				out.FanoutIDs = xs
			}
		case "generation":
			if n, ok := asInt(val); ok {
				out.Generation = n
			}
		case "format":
			out.Format = val
		case "parameters":
			out.Parameters = val
		case "gt_parameters":
			out.GTParameters = val
		case "phys_rep":
			out.PhysRep = val
		case "vis_rep":
			out.VisRep = val
		case "pre_f":
			if s, ok := asString(val); ok {
				out.PreF = s
			}
		case "post_f":
			if s, ok := asString(val); ok {
				out.PostF = s
			}
		}
	}
	return out
}

func ConvertActuator(in map[string]any) ActuatorRecord {
	out := defaultActuatorRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "name":
			if s, ok := asString(val); ok {
				out.Name = s
			}
		case "type":
			if s, ok := asString(val); ok {
				out.Type = s
			}
		case "cx_id":
			out.CortexID = val
		case "scape":
			out.Scape = val
		case "vl":
			if n, ok := asInt(val); ok {
				out.VL = n
			}
		case "fanin_ids":
			if xs, ok := asAnySlice(val); ok {
				out.FaninIDs = xs
			}
		case "generation":
			if n, ok := asInt(val); ok {
				out.Generation = n
			}
		case "format":
			out.Format = val
		case "parameters":
			out.Parameters = val
		case "gt_parameters":
			out.GTParameters = val
		case "phys_rep":
			out.PhysRep = val
		case "vis_rep":
			out.VisRep = val
		case "pre_f":
			if s, ok := asString(val); ok {
				out.PreF = s
			}
		case "post_f":
			if s, ok := asString(val); ok {
				out.PostF = s
			}
		}
	}
	return out
}

func ConvertNeuron(in map[string]any) NeuronRecord {
	out := defaultNeuronRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "generation":
			if n, ok := asInt(val); ok {
				out.Generation = n
			}
		case "cx_id":
			out.CortexID = val
		case "pre_processor":
			if s, ok := asString(val); ok {
				out.PreProcessor = s
			}
		case "signal_integrator":
			if s, ok := asString(val); ok {
				out.SignalIntegrator = s
			}
		case "af":
			if s, ok := asString(val); ok {
				out.ActivationFunction = s
			}
		case "post_processor":
			if s, ok := asString(val); ok {
				out.PostProcessor = s
			}
		case "pf":
			out.PlasticityFunction = val
		case "aggr_f":
			if s, ok := asString(val); ok {
				out.AggregatorFunction = s
			}
		case "input_idps":
			if xs, ok := asAnySlice(val); ok {
				out.InputIDPs = xs
			}
		case "input_idps_modulation":
			if xs, ok := asAnySlice(val); ok {
				out.InputIDPsMod = xs
			}
		case "output_ids":
			if xs, ok := asAnySlice(val); ok {
				out.OutputIDs = xs
			}
		case "ro_ids":
			if xs, ok := asAnySlice(val); ok {
				out.RecurrentOutputIDs = xs
			}
		}
	}
	return out
}

func ConvertAgent(in map[string]any) AgentRecord {
	out := defaultAgentRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "encoding_type":
			if s, ok := asString(val); ok {
				out.EncodingType = s
			}
		case "generation":
			if n, ok := asInt(val); ok {
				out.Generation = n
			}
		case "population_id":
			out.PopulationID = val
		case "specie_id":
			out.SpecieID = val
		case "cx_id":
			out.CortexID = val
		case "fingerprint":
			out.Fingerprint = val
		case "constraint":
			out.Constraint = val
		case "evo_hist":
			if xs, ok := asAnySlice(val); ok {
				out.EvoHist = xs
			}
		case "fitness":
			if f, ok := asFloat64(val); ok {
				out.Fitness = f
			}
		case "innovation_factor":
			out.InnovationFactor = val
		case "pattern":
			if xs, ok := asAnySlice(val); ok {
				out.Pattern = xs
			}
		case "tuning_selection_f":
			if s, ok := asString(val); ok {
				out.TuningSelectionF = s
			}
		case "annealing_parameter":
			out.AnnealingParameter = val
		case "tuning_duration_f":
			out.TuningDurationF = val
		case "perturbation_range":
			out.PerturbationRange = val
		case "mutation_operators":
			if xs, ok := asAnySlice(val); ok {
				out.MutationOperators = xs
			}
		case "tot_topological_mutations_f":
			out.TotTopologicalMutF = val
		case "heredity_type":
			if s, ok := asString(val); ok {
				out.HeredityType = s
			}
		case "substrate_id":
			out.SubstrateID = val
		case "offspring_ids":
			if xs, ok := asAnySlice(val); ok {
				out.OffspringIDs = xs
			}
		case "parent_ids":
			if xs, ok := asAnySlice(val); ok {
				out.ParentIDs = xs
			}
		case "champion_flag":
			if xs, ok := asAnySlice(val); ok {
				out.ChampionFlag = xs
			}
		case "evolvability":
			if f, ok := asFloat64(val); ok {
				out.Evolvability = f
			}
		case "brittleness":
			if f, ok := asFloat64(val); ok {
				out.Brittleness = f
			}
		case "robustness":
			if f, ok := asFloat64(val); ok {
				out.Robustness = f
			}
		case "evolutionary_capacitance":
			if f, ok := asFloat64(val); ok {
				out.EvolutionaryCap = f
			}
		case "behavioral_trace":
			out.BehavioralTrace = val
		case "fs":
			if f, ok := asFloat64(val); ok {
				out.FS = f
			}
		case "main_fitness":
			if f, ok := asFloat64(val); ok {
				out.MainFitness = f
			}
		}
	}
	return out
}

func ConvertCortex(in map[string]any) CortexRecord {
	out := defaultCortexRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "agent_id":
			out.AgentID = val
		case "neuron_ids":
			if xs, ok := asAnySlice(val); ok {
				out.NeuronIDs = xs
			}
		case "sensor_ids":
			if xs, ok := asAnySlice(val); ok {
				out.SensorIDs = xs
			}
		case "actuator_ids":
			if xs, ok := asAnySlice(val); ok {
				out.ActuatorIDs = xs
			}
		}
	}
	return out
}

func ConvertSubstrate(in map[string]any) SubstrateRecord {
	out := defaultSubstrateRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "agent_id":
			out.AgentID = val
		case "densities":
			out.Densities = val
		case "linkform":
			out.Linkform = val
		case "plasticity":
			out.Plasticity = val
		case "cpp_ids":
			if xs, ok := asAnySlice(val); ok {
				out.CPPIDs = xs
			}
		case "cep_ids":
			if xs, ok := asAnySlice(val); ok {
				out.CEPIDs = xs
			}
		}
	}
	return out
}

func ConvertPolis(in map[string]any) PolisRecord {
	out := defaultPolisRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "scape_ids":
			if xs, ok := asAnySlice(val); ok {
				out.ScapeIDs = xs
			}
		case "population_ids":
			if xs, ok := asAnySlice(val); ok {
				out.PopulationIDs = xs
			}
		case "specie_ids":
			if xs, ok := asAnySlice(val); ok {
				out.SpecieIDs = xs
			}
		case "dx_ids":
			if xs, ok := asAnySlice(val); ok {
				out.DXIDs = xs
			}
		case "parameters":
			if xs, ok := asAnySlice(val); ok {
				out.Parameters = xs
			}
		}
	}
	return out
}

func ConvertScape(in map[string]any) ScapeRecord {
	out := defaultScapeRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "type":
			out.Type = val
		case "physics":
			out.Physics = val
		case "metabolics":
			out.Metabolics = val
		case "sector2avatars":
			out.Sector2Avatars = val
		case "avatars":
			if xs, ok := asAnySlice(val); ok {
				out.Avatars = xs
			}
		case "plants":
			if xs, ok := asAnySlice(val); ok {
				out.Plants = xs
			}
		case "walls":
			if xs, ok := asAnySlice(val); ok {
				out.Walls = xs
			}
		case "pillars":
			if xs, ok := asAnySlice(val); ok {
				out.Pillars = xs
			}
		case "laws":
			if xs, ok := asAnySlice(val); ok {
				out.Laws = xs
			}
		case "anomolies":
			if xs, ok := asAnySlice(val); ok {
				out.Anomolies = xs
			}
		case "artifacts":
			if xs, ok := asAnySlice(val); ok {
				out.Artifacts = xs
			}
		case "objects":
			if xs, ok := asAnySlice(val); ok {
				out.Objects = xs
			}
		case "elements":
			if xs, ok := asAnySlice(val); ok {
				out.Elements = xs
			}
		case "atoms":
			if xs, ok := asAnySlice(val); ok {
				out.Atoms = xs
			}
		case "scheduler":
			if n, ok := asInt(val); ok {
				out.Scheduler = n
			}
		}
	}
	return out
}

func ConvertSector(in map[string]any) SectorRecord {
	out := defaultSectorRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "type":
			out.Type = val
		case "scape_pid":
			out.ScapePID = val
		case "sector_size":
			out.SectorSize = val
		case "physics":
			out.Physics = val
		case "metabolics":
			out.Metabolics = val
		case "sector2avatars":
			out.Sector2Avatars = val
		case "avatars":
			if xs, ok := asAnySlice(val); ok {
				out.Avatars = xs
			}
		case "plants":
			if xs, ok := asAnySlice(val); ok {
				out.Plants = xs
			}
		case "walls":
			if xs, ok := asAnySlice(val); ok {
				out.Walls = xs
			}
		case "pillars":
			if xs, ok := asAnySlice(val); ok {
				out.Pillars = xs
			}
		case "laws":
			if xs, ok := asAnySlice(val); ok {
				out.Laws = xs
			}
		case "anomolies":
			if xs, ok := asAnySlice(val); ok {
				out.Anomolies = xs
			}
		case "artifacts":
			if xs, ok := asAnySlice(val); ok {
				out.Artifacts = xs
			}
		case "objects":
			if xs, ok := asAnySlice(val); ok {
				out.Objects = xs
			}
		case "elements":
			if xs, ok := asAnySlice(val); ok {
				out.Elements = xs
			}
		case "atoms":
			if xs, ok := asAnySlice(val); ok {
				out.Atoms = xs
			}
		}
	}
	return out
}

func ConvertAvatar(in map[string]any) AvatarRecord {
	out := defaultAvatarRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "sector":
			out.Sector = val
		case "morphology":
			out.Morphology = val
		case "type":
			out.Type = val
		case "specie":
			out.Specie = val
		case "energy":
			if f, ok := asFloat64(val); ok {
				out.Energy = f
			}
		case "health":
			if f, ok := asFloat64(val); ok {
				out.Health = f
			}
		case "food":
			if f, ok := asFloat64(val); ok {
				out.Food = f
			}
		case "age":
			if f, ok := asFloat64(val); ok {
				out.Age = f
			}
		case "kills":
			if f, ok := asFloat64(val); ok {
				out.Kills = f
			}
		case "loc":
			out.Loc = val
		case "direction":
			out.Direction = val
		case "r":
			out.R = val
		case "mass":
			out.Mass = val
		case "objects":
			out.Objects = val
		case "vis":
			if xs, ok := asAnySlice(val); ok {
				out.Vis = xs
			}
		case "state":
			out.State = val
		case "stats":
			out.Stats = val
		case "actuators":
			out.Actuators = val
		case "sensors":
			out.Sensors = val
		case "sound":
			out.Sound = val
		case "gestalt":
			out.Gestalt = val
		case "spear":
			out.Spear = val
		}
	}
	return out
}

func ConvertObject(in map[string]any) ObjectRecord {
	out := defaultObjectRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "sector":
			out.Sector = val
		case "type":
			out.Type = val
		case "color":
			out.Color = val
		case "loc":
			out.Loc = val
		case "pivot":
			out.Pivot = val
		case "elements":
			if xs, ok := asAnySlice(val); ok {
				out.Elements = xs
			}
		case "parameters":
			if xs, ok := asAnySlice(val); ok {
				out.Parameters = xs
			}
		}
	}
	return out
}

func ConvertCircle(in map[string]any) CircleRecord {
	out := defaultCircleRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "sector":
			out.Sector = val
		case "color":
			out.Color = val
		case "loc":
			out.Loc = val
		case "pivot":
			out.Pivot = val
		case "r":
			out.R = val
		}
	}
	return out
}

func ConvertSquare(in map[string]any) SquareRecord {
	out := defaultSquareRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "sector":
			out.Sector = val
		case "color":
			out.Color = val
		case "loc":
			out.Loc = val
		case "pivot":
			out.Pivot = val
		case "r":
			out.R = val
		}
	}
	return out
}

func ConvertSpecie(in map[string]any) SpecieRecord {
	out := defaultSpecieRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "population_id":
			out.PopulationID = val
		case "fingerprint":
			out.Fingerprint = val
		case "constraint":
			out.Constraint = val
		case "all_agent_ids":
			if xs, ok := asAnySlice(val); ok {
				out.AllAgentIDs = xs
			}
		case "agent_ids":
			if xs, ok := asAnySlice(val); ok {
				out.AgentIDs = xs
			}
		case "dead_pool":
			if xs, ok := asAnySlice(val); ok {
				out.DeadPool = xs
			}
		case "champion_ids":
			if xs, ok := asAnySlice(val); ok {
				out.ChampionIDs = xs
			}
		case "fitness":
			out.Fitness = val
		case "innovation_factor":
			out.InnovationFactor = val
		case "stats":
			if xs, ok := asAnySlice(val); ok {
				out.Stats = xs
			}
		case "seed_agent_ids":
			if xs, ok := asAnySlice(val); ok {
				out.SeedAgentIDs = xs
			}
		case "hof_distinguishers":
			if xs, ok := asAnySlice(val); ok {
				out.HOFDistinguishers = xs
			}
		case "specie_distinguishers":
			if xs, ok := asAnySlice(val); ok {
				out.SpecieDistinguish = xs
			}
		case "hall_of_fame":
			if xs, ok := asAnySlice(val); ok {
				out.HallOfFame = xs
			}
		}
	}
	return out
}

func ConvertPopulation(in map[string]any) PopulationRecord {
	out := defaultPopulationRecord()
	for key, val := range in {
		switch key {
		case "id":
			out.ID = val
		case "polis_id":
			out.PolisID = val
		case "specie_ids":
			if xs, ok := asAnySlice(val); ok {
				out.SpecieIDs = xs
			}
		case "morphologies":
			if xs, ok := asAnySlice(val); ok {
				out.Morphologies = xs
			}
		case "innovation_factor":
			out.InnovationFactor = val
		case "evo_alg_f":
			if s, ok := asString(val); ok {
				out.EvoAlgF = s
			}
		case "fitness_postprocessor_f":
			if s, ok := asString(val); ok {
				out.FitnessPostprocF = s
			}
		case "selection_f":
			if s, ok := asString(val); ok {
				out.SelectionF = s
			}
		case "trace":
			out.Trace = val
		case "seed_agent_ids":
			if xs, ok := asAnySlice(val); ok {
				out.SeedAgentIDs = xs
			}
		case "seed_specie_ids":
			if xs, ok := asAnySlice(val); ok {
				out.SeedSpecieIDs = xs
			}
		}
	}
	return out
}

func ConvertTrace(in map[string]any) TraceRecord {
	out := defaultTraceRecord()
	for key, val := range in {
		switch key {
		case "stats":
			if xs, ok := asAnySlice(val); ok {
				out.Stats = xs
			}
		case "tot_evaluations":
			if n, ok := asInt(val); ok {
				out.TotalEvaluations = n
			}
		case "step_size":
			if n, ok := asInt(val); ok {
				out.StepSize = n
			}
		}
	}
	return out
}

func ConvertStat(in map[string]any) StatRecord {
	out := defaultStatRecord()
	for key, val := range in {
		switch key {
		case "morphology":
			out.Morphology = val
		case "specie_id":
			out.SpecieID = val
		case "avg_neurons":
			if f, ok := asFloat64(val); ok {
				out.AvgNeurons = f
			}
		case "std_neurons":
			if f, ok := asFloat64(val); ok {
				out.StdNeurons = f
			}
		case "avg_fitness":
			if f, ok := asFloat64(val); ok {
				out.AvgFitness = f
			}
		case "std_fitness":
			if f, ok := asFloat64(val); ok {
				out.StdFitness = f
			}
		case "max_fitness":
			if f, ok := asFloat64(val); ok {
				out.MaxFitness = f
			}
		case "min_fitness":
			if f, ok := asFloat64(val); ok {
				out.MinFitness = f
			}
		case "validation_fitness":
			if f, ok := asFloat64(val); ok {
				out.ValidationFitness = f
			}
		case "test_fitness":
			if f, ok := asFloat64(val); ok {
				out.TestFitness = f
			}
		case "avg_diversity":
			if f, ok := asFloat64(val); ok {
				out.AvgDiversity = f
			}
		case "evaluations":
			if n, ok := asInt(val); ok {
				out.Evaluations = n
			}
		case "time_stamp":
			out.TimeStamp = val
		}
	}
	return out
}

func ConvertTopologySummary(in map[string]any) TopologySummaryRecord {
	out := defaultTopologySummaryRecord()
	for key, val := range in {
		switch key {
		case "type":
			out.Type = val
		case "tot_neurons":
			if n, ok := asInt(val); ok {
				out.TotalNeurons = n
			}
		case "tot_n_ils":
			if n, ok := asInt(val); ok {
				out.TotalNILs = n
			}
		case "tot_n_ols":
			if n, ok := asInt(val); ok {
				out.TotalNOLs = n
			}
		case "tot_n_ros":
			if n, ok := asInt(val); ok {
				out.TotalNROs = n
			}
		case "af_distribution":
			out.AFDistribution = val
		}
	}
	return out
}

func ConvertSignature(in map[string]any) SignatureRecord {
	out := defaultSignatureRecord()
	for key, val := range in {
		switch key {
		case "generalized_Pattern":
			out.GeneralizedPattern = val
		case "generalized_EvoHist":
			out.GeneralizedEvoHist = val
		case "generalized_Sensors":
			out.GeneralizedSensors = val
		case "generalized_Actuators":
			out.GeneralizedActuators = val
		case "topology_summary":
			out.TopologySummary = val
		}
	}
	return out
}

func ConvertChampion(in map[string]any) ChampionRecord {
	out := defaultChampionRecord()
	for key, val := range in {
		switch key {
		case "hof_fingerprint":
			out.HOFFingerprint = val
		case "id":
			out.ID = val
		case "fitness":
			if f, ok := asFloat64(val); ok {
				out.Fitness = f
			}
		case "validation_fitness":
			if f, ok := asFloat64(val); ok {
				out.ValidationFitness = f
			}
		case "test_fitness":
			if f, ok := asFloat64(val); ok {
				out.TestFitness = f
			}
		case "main_fitness":
			if f, ok := asFloat64(val); ok {
				out.MainFitness = f
			}
		case "tot_n":
			if n, ok := asInt(val); ok {
				out.TotalNeurons = n
			}
		case "evolvability":
			if f, ok := asFloat64(val); ok {
				out.Evolvability = f
			}
		case "robustness":
			if f, ok := asFloat64(val); ok {
				out.Robustness = f
			}
		case "brittleness":
			if f, ok := asFloat64(val); ok {
				out.Brittleness = f
			}
		case "generation":
			if n, ok := asInt(val); ok {
				out.Generation = n
			}
		case "behavioral_differences":
			out.BehavioralDifferences = val
		case "fs":
			if f, ok := asFloat64(val); ok {
				out.FS = f
			}
		}
	}
	return out
}
