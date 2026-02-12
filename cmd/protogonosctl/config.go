package main

import (
	"encoding/json"
	"fmt"
	"os"

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
	if v, ok := asString(raw["scape"]); ok {
		req.Scape = v
	}
	if v, ok := asInt(raw["population"]); ok {
		req.Population = v
	}
	if v, ok := asInt(raw["generations"]); ok {
		req.Generations = v
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
		req.TuneSelection = mapTuningSelection(firstOrEmpty(constraint.TuningSelectionFs))
		for _, op := range constraint.MutationOperators {
			switch op.Name {
			case "mutate_weights", "add_bias":
				req.WeightPerturb += op.Weight
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
			case "add_sensor", "add_sensorlink", "add_actuator", "add_cpp", "add_cep":
				req.WeightSubstrate += op.Weight
			}
		}
	}

	if pmpMap, ok := raw["pmp"].(map[string]any); ok {
		pmp := map2rec.ConvertPMP(pmpMap)
		if req.Population == 0 {
			req.Population = pmp.InitSpecieSize
		}
		if req.Generations == 0 {
			req.Generations = pmp.GenerationLimit
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
		case "scape":
			req.Scape = v.(string)
		case "pop":
			req.Population = v.(int)
		case "gens":
			req.Generations = v.(int)
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
		case "tune-selection":
			req.TuneSelection = v.(string)
		case "tune-duration-policy":
			req.TuneDurationPolicy = v.(string)
		case "tune-duration-param":
			req.TuneDurationParam = v.(float64)
		case "w-perturb":
			req.WeightPerturb = v.(float64)
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
		set["w-add-synapse"] ||
		set["w-remove-synapse"] ||
		set["w-add-neuron"] ||
		set["w-remove-neuron"] ||
		set["w-plasticity"] ||
		set["w-substrate"]
}
