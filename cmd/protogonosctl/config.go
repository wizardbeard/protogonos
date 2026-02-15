package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"protogonos/internal/map2rec"
	protoapi "protogonos/pkg/protogonos"
)

func loadRunRequestFromConfig(path string) (protoapi.RunRequest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return protoapi.RunRequest{}, err
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return protoapi.RunRequest{}, err
	}

	var req protoapi.RunRequest
	if v, ok := asString(raw["run_id"]); ok {
		req.RunID = v
	}
	if v, ok := asString(raw["continue_population_id"]); ok {
		req.ContinuePopulationID = v
	}
	if v, ok := asString(raw["specie_identifier"]); ok {
		req.SpecieIdentifier = v
	}
	if v, ok := asString(raw["scape"]); ok {
		req.Scape = v
	}
	if v, ok := asInt(raw["population"]); ok {
		req.Population = v
	}
	if v, ok := asInt(raw["generations"]); ok {
		req.Generations = v
	}
	if v, ok := asFloat64(raw["survival_percentage"]); ok {
		req.SurvivalPercentage = v
	}
	if v, ok := asInt(raw["specie_size_limit"]); ok {
		req.SpecieSizeLimit = v
	}
	if v, ok := asFloat64(raw["fitness_goal"]); ok {
		req.FitnessGoal = v
	}
	if v, ok := asInt(raw["evaluations_limit"]); ok {
		req.EvaluationsLimit = v
	}
	if v, ok := asBool(raw["start_paused"]); ok {
		req.StartPaused = v
	}
	if v, ok := asInt(raw["auto_continue_ms"]); ok {
		req.AutoContinueAfter = time.Duration(v) * time.Millisecond
	}
	if v, ok := asInt64(raw["seed"]); ok {
		req.Seed = v
	}
	if v, ok := asInt(raw["workers"]); ok {
		req.Workers = v
	}
	if v, ok := asBool(raw["enable_tuning"]); ok {
		req.EnableTuning = v
	}
	if v, ok := asBool(raw["compare_tuning"]); ok {
		req.CompareTuning = v
	}
	if v, ok := asInt(raw["tune_attempts"]); ok {
		req.TuneAttempts = v
	}
	if v, ok := asInt(raw["tune_steps"]); ok {
		req.TuneSteps = v
	}
	if v, ok := asFloat64(raw["tune_step_size"]); ok {
		req.TuneStepSize = v
	}
	if v, ok := asFloat64(raw["tune_perturbation_range"]); ok {
		req.TunePerturbationRange = v
	}
	if v, ok := asFloat64(raw["tune_annealing_factor"]); ok {
		req.TuneAnnealingFactor = v
	}
	if v, ok := asFloat64(raw["tune_min_improvement"]); ok {
		req.TuneMinImprovement = v
	}
	if v, ok := asString(raw["tune_duration_policy"]); ok {
		req.TuneDurationPolicy = v
	}
	if v, ok := asFloat64(raw["tune_duration_param"]); ok {
		req.TuneDurationParam = v
	}
	if v, ok := asString(raw["fitness_postprocessor"]); ok {
		req.FitnessPostprocessor = v
	}
	if v, ok := asString(raw["topological_policy"]); ok {
		req.TopologicalPolicy = v
	}
	if v, ok := asInt(raw["topological_count"]); ok {
		req.TopologicalCount = v
	}
	if v, ok := asFloat64(raw["topological_param"]); ok {
		req.TopologicalParam = v
	}
	if v, ok := asInt(raw["topological_max"]); ok {
		req.TopologicalMax = v
	}

	if constraintMap, ok := raw["constraint"].(map[string]any); ok {
		constraint := map2rec.ConvertConstraint(constraintMap)
		req.Selection = mapPopulationSelection(constraint.PopulationSelectionF)
		if req.SpecieIdentifier == "" {
			req.SpecieIdentifier = mapSpecieIdentifier(constraint.SpecieDistinguishers)
		}
		req.FitnessPostprocessor = mapFitnessPostprocessor(constraint.PopulationFitnessProcessorF)
		req.TuneSelection = mapTuningSelection(firstOrEmpty(constraint.TuningSelectionFs))
		if constraint.TuningDurationF.Name != "" {
			req.TuneDurationPolicy = constraint.TuningDurationF.Name
			req.TuneDurationParam = constraint.TuningDurationF.Param
		}
		if len(constraint.TotTopologicalMutationsFs) > 0 {
			policy := constraint.TotTopologicalMutationsFs[0]
			req.TopologicalPolicy = policy.Name
			switch policy.Name {
			case "const":
				if req.TopologicalCount == 0 {
					req.TopologicalCount = int(policy.Param)
				}
			default:
				if req.TopologicalParam == 0 {
					req.TopologicalParam = policy.Param
				}
			}
		}
		for _, op := range constraint.MutationOperators {
			switch op.Name {
			case "mutate_weights":
				req.WeightPerturb += op.Weight
			case "add_bias":
				req.WeightBias += op.Weight
			case "remove_bias":
				req.WeightRemoveBias += op.Weight
			case "mutate_af":
				req.WeightActivation += op.Weight
			case "mutate_aggrf":
				req.WeightAggregator += op.Weight
			case "add_outlink", "add_inlink":
				req.WeightAddSynapse += op.Weight
			case "remove_outlink", "remove_inlink":
				req.WeightRemoveSynapse += op.Weight
			case "add_neuron", "outsplice", "insplice":
				req.WeightAddNeuron += op.Weight
			case "remove_neuron":
				req.WeightRemoveNeuron += op.Weight
			case "mutate_plasticity_parameters":
				req.WeightPlasticity += op.Weight
			case "mutate_pf":
				req.WeightPlasticityRule += op.Weight
			case "add_sensor", "add_sensorlink", "add_actuator", "add_cpp", "add_cep", "add_CircuitNode", "add_CircuitLayer", "add_circuit_node", "add_circuit_layer":
				req.WeightSubstrate += op.Weight
			}
		}
	}

	if pmpMap, ok := raw["pmp"].(map[string]any); ok {
		pmp := map2rec.ConvertPMP(pmpMap)
		if _, hasPopulationID := pmpMap["population_id"]; hasPopulationID {
			if req.ContinuePopulationID == "" && pmp.PopulationID != "" {
				req.ContinuePopulationID = pmp.PopulationID
			}
			if req.RunID == "" && pmp.PopulationID != "" {
				req.RunID = pmp.PopulationID
			}
		}
		if req.SurvivalPercentage == 0 {
			req.SurvivalPercentage = pmp.SurvivalPercentage
		}
		if req.Population == 0 {
			req.Population = pmp.InitSpecieSize
		}
		if req.SpecieSizeLimit == 0 {
			req.SpecieSizeLimit = pmp.SpecieSizeLimit
		}
		if req.Generations == 0 {
			req.Generations = pmp.GenerationLimit
		}
		if req.EvaluationsLimit == 0 {
			req.EvaluationsLimit = pmp.EvaluationsLimit
		}
		if req.FitnessGoal == 0 && !math.IsInf(pmp.FitnessGoal, 1) {
			req.FitnessGoal = pmp.FitnessGoal
		}
	}

	return req, nil
}

func asString(v any) (string, bool) {
	s, ok := v.(string)
	return s, ok
}

func asBool(v any) (bool, bool) {
	b, ok := v.(bool)
	return b, ok
}

func asInt(v any) (int, bool) {
	switch x := v.(type) {
	case int:
		return x, true
	case float64:
		return int(x), true
	default:
		return 0, false
	}
}

func asInt64(v any) (int64, bool) {
	switch x := v.(type) {
	case int64:
		return x, true
	case int:
		return int64(x), true
	case float64:
		return int64(x), true
	default:
		return 0, false
	}
}

func asFloat64(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	default:
		return 0, false
	}
}

func overrideFromFlags(req *protoapi.RunRequest, set map[string]bool, flagValue map[string]any) error {
	for name := range set {
		v, ok := flagValue[name]
		if !ok {
			continue
		}
		switch name {
		case "run-id":
			req.RunID = v.(string)
		case "continue-pop-id":
			req.ContinuePopulationID = v.(string)
		case "specie-identifier":
			req.SpecieIdentifier = v.(string)
		case "scape":
			req.Scape = v.(string)
		case "pop":
			req.Population = v.(int)
		case "gens":
			req.Generations = v.(int)
		case "survival-percentage":
			req.SurvivalPercentage = v.(float64)
		case "specie-size-limit":
			req.SpecieSizeLimit = v.(int)
		case "fitness-goal":
			req.FitnessGoal = v.(float64)
		case "evaluations-limit":
			req.EvaluationsLimit = v.(int)
		case "start-paused":
			req.StartPaused = v.(bool)
		case "auto-continue-ms":
			req.AutoContinueAfter = time.Duration(v.(int)) * time.Millisecond
		case "seed":
			req.Seed = v.(int64)
		case "workers":
			req.Workers = v.(int)
		case "tuning":
			req.EnableTuning = v.(bool)
		case "compare-tuning":
			req.CompareTuning = v.(bool)
		case "selection":
			req.Selection = v.(string)
		case "fitness-postprocessor":
			req.FitnessPostprocessor = v.(string)
		case "topo-policy":
			req.TopologicalPolicy = v.(string)
		case "topo-count":
			req.TopologicalCount = v.(int)
		case "topo-param":
			req.TopologicalParam = v.(float64)
		case "topo-max":
			req.TopologicalMax = v.(int)
		case "attempts":
			req.TuneAttempts = v.(int)
		case "tune-steps":
			req.TuneSteps = v.(int)
		case "tune-step-size":
			req.TuneStepSize = v.(float64)
		case "tune-perturbation-range":
			req.TunePerturbationRange = v.(float64)
		case "tune-annealing-factor":
			req.TuneAnnealingFactor = v.(float64)
		case "tune-min-improvement":
			req.TuneMinImprovement = v.(float64)
		case "tune-selection":
			req.TuneSelection = v.(string)
		case "tune-duration-policy":
			req.TuneDurationPolicy = v.(string)
		case "tune-duration-param":
			req.TuneDurationParam = v.(float64)
		case "w-perturb":
			req.WeightPerturb = v.(float64)
		case "w-bias":
			req.WeightBias = v.(float64)
		case "w-remove-bias":
			req.WeightRemoveBias = v.(float64)
		case "w-activation":
			req.WeightActivation = v.(float64)
		case "w-aggregator":
			req.WeightAggregator = v.(float64)
		case "w-add-synapse":
			req.WeightAddSynapse = v.(float64)
		case "w-remove-synapse":
			req.WeightRemoveSynapse = v.(float64)
		case "w-add-neuron":
			req.WeightAddNeuron = v.(float64)
		case "w-remove-neuron":
			req.WeightRemoveNeuron = v.(float64)
		case "w-plasticity":
			req.WeightPlasticity = v.(float64)
		case "w-plasticity-rule":
			req.WeightPlasticityRule = v.(float64)
		case "w-substrate":
			req.WeightSubstrate = v.(float64)
		}
	}
	if req.Scape == "" {
		req.Scape = "xor"
	}
	if req.TuneSelection != "" {
		req.TuneSelection = normalizeTuneSelection(req.TuneSelection)
	}
	return nil
}

func loadOrDefaultRunRequest(configPath string) (protoapi.RunRequest, error) {
	if configPath == "" {
		return protoapi.RunRequest{}, nil
	}
	req, err := loadRunRequestFromConfig(configPath)
	if err != nil {
		return protoapi.RunRequest{}, fmt.Errorf("load config: %w", err)
	}
	return req, nil
}

func hasAnyWeightOverrideFlag(set map[string]bool) bool {
	return set["w-perturb"] ||
		set["w-bias"] ||
		set["w-remove-bias"] ||
		set["w-activation"] ||
		set["w-aggregator"] ||
		set["w-add-synapse"] ||
		set["w-remove-synapse"] ||
		set["w-add-neuron"] ||
		set["w-remove-neuron"] ||
		set["w-plasticity-rule"] ||
		set["w-plasticity"] ||
		set["w-substrate"]
}

func mapFitnessPostprocessor(name string) string {
	switch name {
	case "nsize_proportional":
		return "nsize_proportional"
	case "size_proportional", "novelty_proportional", "none":
		return name
	default:
		return name
	}
}

func mapSpecieIdentifier(distinguishers []string) string {
	name := map2recFirstNonEmpty(distinguishers)
	switch name {
	case "tot_n":
		return "tot_n"
	case "pattern", "topology":
		return "topology"
	default:
		return ""
	}
}

func map2recFirstNonEmpty(xs []string) string {
	for _, item := range xs {
		if item != "" {
			return item
		}
	}
	return ""
}
