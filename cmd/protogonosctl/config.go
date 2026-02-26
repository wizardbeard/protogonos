package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
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
	if v, ok := asString(raw["gtsa_csv_path"]); ok {
		req.GTSACSVPath = v
	}
	if v, ok := asInt(raw["gtsa_train_end"]); ok {
		req.GTSATrainEnd = v
	}
	if v, ok := asInt(raw["gtsa_validation_end"]); ok {
		req.GTSAValidationEnd = v
	}
	if v, ok := asInt(raw["gtsa_test_end"]); ok {
		req.GTSATestEnd = v
	}
	if v, ok := asString(raw["fx_csv_path"]); ok {
		req.FXCSVPath = v
	}
	if v, ok := asString(raw["epitopes_csv_path"]); ok {
		req.EpitopesCSVPath = v
	}
	if v, ok := asString(raw["llvm_workflow_json"]); ok {
		req.LLVMWorkflowJSONPath = v
	}
	if v, ok := asString(raw["llvm_workflow_json_path"]); ok {
		req.LLVMWorkflowJSONPath = v
	}
	if v, ok := asInt(raw["epitopes_gt_start"]); ok {
		req.EpitopesGTStart = v
	}
	if v, ok := asInt(raw["epitopes_gt_end"]); ok {
		req.EpitopesGTEnd = v
	}
	if v, ok := asInt(raw["epitopes_validation_start"]); ok {
		req.EpitopesValidationStart = v
	}
	if v, ok := asInt(raw["epitopes_validation_end"]); ok {
		req.EpitopesValidationEnd = v
	}
	if v, ok := asInt(raw["epitopes_test_start"]); ok {
		req.EpitopesTestStart = v
	}
	if v, ok := asInt(raw["epitopes_test_end"]); ok {
		req.EpitopesTestEnd = v
	}
	if v, ok := asInt(raw["epitopes_benchmark_start"]); ok {
		req.EpitopesBenchmarkStart = v
	}
	if v, ok := asInt(raw["epitopes_benchmark_end"]); ok {
		req.EpitopesBenchmarkEnd = v
	}
	if v, ok := asString(raw["flatland_scanner_profile"]); ok {
		req.FlatlandScannerProfile = v
	}
	if v, ok := asFloat64(raw["flatland_scanner_spread"]); ok {
		req.FlatlandScannerSpread = float64Ptr(v)
	}
	if v, ok := asFloat64(raw["flatland_scanner_offset"]); ok {
		req.FlatlandScannerOffset = float64Ptr(v)
	}
	if v, ok := asBool(raw["flatland_layout_randomize"]); ok {
		req.FlatlandLayoutRandomize = boolPtr(v)
	}
	if v, ok := asInt(raw["flatland_layout_variants"]); ok {
		req.FlatlandLayoutVariants = intPtr(v)
	}
	if v, ok := asInt(raw["flatland_force_layout_variant"]); ok {
		req.FlatlandForceLayout = intPtr(v)
	}
	if v, ok := asInt(raw["flatland_benchmark_trials"]); ok {
		req.FlatlandBenchmarkTrials = intPtr(v)
	}
	if v, ok := asInt(raw["flatland_max_age"]); ok {
		req.FlatlandMaxAge = intPtr(v)
	}
	if v, ok := asInt(raw["flatland_forage_goal"]); ok {
		req.FlatlandForageGoal = intPtr(v)
	}
	if scapeData, ok := raw["scape_data"].(map[string]any); ok {
		applyScapeDataConfigFallbacks(&req, scapeData)
	}
	if v, ok := asString(raw["op_mode"]); ok {
		req.OpMode = v
	}
	if xs, ok := asAnySlice(raw["op_mode"]); ok {
		if joined, ok := joinStringSlice(xs); ok {
			req.OpMode = joined
		}
	}
	if v, ok := asString(raw["evolution_type"]); ok {
		req.EvolutionType = mapPopulationEvolutionType(v)
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
	if v, ok := asInt(raw["trace_step_size"]); ok {
		req.TraceStepSize = v
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
	if v, ok := asBool(raw["validation_probe"]); ok {
		req.ValidationProbe = v
	}
	if v, ok := asBool(raw["test_probe"]); ok {
		req.TestProbe = v
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
		req.EvolutionType = mapPopulationEvolutionType(constraint.PopulationEvoAlgF)
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
		applyMutationOperatorWeights(&req, constraint.MutationOperators)
	}

	if pmpMap, ok := raw["pmp"].(map[string]any); ok {
		pmp := map2rec.ConvertPMP(pmpMap)
		if req.OpMode == "" && pmp.OpMode != "" {
			req.OpMode = pmp.OpMode
		}
		if req.EvolutionType == "" && pmp.EvolutionType != "" {
			req.EvolutionType = mapPopulationEvolutionType(pmp.EvolutionType)
		}
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
	if populationMap, ok := raw["population"].(map[string]any); ok {
		population := map2rec.ConvertPopulation(populationMap)
		if req.EvolutionType == "" && population.EvoAlgF != "" {
			req.EvolutionType = mapPopulationEvolutionType(population.EvoAlgF)
		}
		if req.Selection == "" && population.SelectionF != "" {
			req.Selection = mapPopulationSelection(population.SelectionF)
		}
		if req.FitnessPostprocessor == "" && population.FitnessPostprocF != "" {
			req.FitnessPostprocessor = mapFitnessPostprocessor(population.FitnessPostprocF)
		}
		if req.TraceStepSize == 0 {
			if traceMap, ok := population.Trace.(map[string]any); ok {
				trace := map2rec.ConvertTrace(traceMap)
				req.TraceStepSize = trace.StepSize
			}
		}
	}
	if specieMap, ok := raw["specie"].(map[string]any); ok {
		specie := map2rec.ConvertSpecie(specieMap)
		if req.SpecieIdentifier == "" {
			req.SpecieIdentifier = mapSpecieIdentifier(specie.SpecieDistinguish)
		}
	}
	if agentMap, ok := raw["agent"].(map[string]any); ok {
		agent := map2rec.ConvertAgent(agentMap)
		if req.TuneSelection == "" && agent.TuningSelectionF != "" {
			req.TuneSelection = mapTuningSelection(agent.TuningSelectionF)
		}
		if req.TuneDurationPolicy == "" && agent.TuningDurationF.Name != "" {
			req.TuneDurationPolicy = agent.TuningDurationF.Name
			req.TuneDurationParam = agent.TuningDurationF.Param
		}
		if req.TopologicalPolicy == "" && agent.TotTopologicalMutF.Name != "" {
			req.TopologicalPolicy = agent.TotTopologicalMutF.Name
			switch agent.TotTopologicalMutF.Name {
			case "const":
				if req.TopologicalCount == 0 {
					req.TopologicalCount = int(agent.TotTopologicalMutF.Param)
				}
			default:
				if req.TopologicalParam == 0 {
					req.TopologicalParam = agent.TotTopologicalMutF.Param
				}
			}
		}
		if !hasAnyMutationWeightValue(req) {
			applyMutationOperatorWeights(&req, agent.MutationOperators)
		}
	}
	if traceMap, ok := raw["trace"].(map[string]any); ok {
		trace := map2rec.ConvertTrace(traceMap)
		if req.TraceStepSize == 0 {
			req.TraceStepSize = trace.StepSize
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

func asAnySlice(v any) ([]any, bool) {
	switch xs := v.(type) {
	case []any:
		return append([]any(nil), xs...), true
	case []string:
		out := make([]any, len(xs))
		for i, item := range xs {
			out[i] = item
		}
		return out, true
	default:
		return nil, false
	}
}

func joinStringSlice(values []any) (string, bool) {
	parts := make([]string, 0, len(values))
	for _, item := range values {
		s, ok := asString(item)
		if !ok {
			return "", false
		}
		parts = append(parts, s)
	}
	return strings.Join(parts, ","), true
}

func applyScapeDataConfigFallbacks(req *protoapi.RunRequest, scapeData map[string]any) {
	if gtsaData, ok := scapeData["gtsa"].(map[string]any); ok {
		if req.GTSACSVPath == "" {
			if v, ok := asString(gtsaData["csv_path"]); ok {
				req.GTSACSVPath = v
			}
		}
		if req.GTSATrainEnd == 0 {
			if v, ok := asInt(gtsaData["train_end"]); ok {
				req.GTSATrainEnd = v
			}
		}
		if req.GTSAValidationEnd == 0 {
			if v, ok := asInt(gtsaData["validation_end"]); ok {
				req.GTSAValidationEnd = v
			}
		}
		if req.GTSATestEnd == 0 {
			if v, ok := asInt(gtsaData["test_end"]); ok {
				req.GTSATestEnd = v
			}
		}
	}

	if fxData, ok := scapeData["fx"].(map[string]any); ok {
		if req.FXCSVPath == "" {
			if v, ok := asString(fxData["csv_path"]); ok {
				req.FXCSVPath = v
			}
		}
	}

	if epitopesData, ok := scapeData["epitopes"].(map[string]any); ok {
		if req.EpitopesCSVPath == "" {
			if v, ok := asString(epitopesData["csv_path"]); ok {
				req.EpitopesCSVPath = v
			}
		}
		if req.EpitopesGTStart == 0 {
			if v, ok := asInt(epitopesData["gt_start"]); ok {
				req.EpitopesGTStart = v
			}
		}
		if req.EpitopesGTEnd == 0 {
			if v, ok := asInt(epitopesData["gt_end"]); ok {
				req.EpitopesGTEnd = v
			}
		}
		if req.EpitopesValidationStart == 0 {
			if v, ok := asInt(epitopesData["validation_start"]); ok {
				req.EpitopesValidationStart = v
			}
		}
		if req.EpitopesValidationEnd == 0 {
			if v, ok := asInt(epitopesData["validation_end"]); ok {
				req.EpitopesValidationEnd = v
			}
		}
		if req.EpitopesTestStart == 0 {
			if v, ok := asInt(epitopesData["test_start"]); ok {
				req.EpitopesTestStart = v
			}
		}
		if req.EpitopesTestEnd == 0 {
			if v, ok := asInt(epitopesData["test_end"]); ok {
				req.EpitopesTestEnd = v
			}
		}
		if req.EpitopesBenchmarkStart == 0 {
			if v, ok := asInt(epitopesData["benchmark_start"]); ok {
				req.EpitopesBenchmarkStart = v
			}
		}
		if req.EpitopesBenchmarkEnd == 0 {
			if v, ok := asInt(epitopesData["benchmark_end"]); ok {
				req.EpitopesBenchmarkEnd = v
			}
		}
	}

	if llvmData, ok := scapeData["llvm"].(map[string]any); ok {
		if req.LLVMWorkflowJSONPath == "" {
			if v, ok := asString(llvmData["workflow_json_path"]); ok {
				req.LLVMWorkflowJSONPath = v
			}
		}
		if req.LLVMWorkflowJSONPath == "" {
			if v, ok := asString(llvmData["workflow_json"]); ok {
				req.LLVMWorkflowJSONPath = v
			}
		}
	}

	if flatlandData, ok := scapeData["flatland"].(map[string]any); ok {
		if req.FlatlandScannerProfile == "" {
			if v, ok := asString(flatlandData["scanner_profile"]); ok {
				req.FlatlandScannerProfile = v
			}
		}
		if req.FlatlandScannerSpread == nil {
			if v, ok := asFloat64(flatlandData["scanner_spread"]); ok {
				req.FlatlandScannerSpread = float64Ptr(v)
			}
		}
		if req.FlatlandScannerOffset == nil {
			if v, ok := asFloat64(flatlandData["scanner_offset"]); ok {
				req.FlatlandScannerOffset = float64Ptr(v)
			}
		}
		if req.FlatlandLayoutRandomize == nil {
			if v, ok := asBool(flatlandData["layout_randomize"]); ok {
				req.FlatlandLayoutRandomize = boolPtr(v)
			}
		}
		if req.FlatlandLayoutVariants == nil {
			if v, ok := asInt(flatlandData["layout_variants"]); ok {
				req.FlatlandLayoutVariants = intPtr(v)
			}
		}
		if req.FlatlandForceLayout == nil {
			if v, ok := asInt(flatlandData["force_layout_variant"]); ok {
				req.FlatlandForceLayout = intPtr(v)
			}
		}
		if req.FlatlandBenchmarkTrials == nil {
			if v, ok := asInt(flatlandData["benchmark_trials"]); ok {
				req.FlatlandBenchmarkTrials = intPtr(v)
			}
		}
		if req.FlatlandMaxAge == nil {
			if v, ok := asInt(flatlandData["max_age"]); ok {
				req.FlatlandMaxAge = intPtr(v)
			}
		}
		if req.FlatlandForageGoal == nil {
			if v, ok := asInt(flatlandData["forage_goal"]); ok {
				req.FlatlandForageGoal = intPtr(v)
			}
		}
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
		case "gtsa-csv":
			req.GTSACSVPath = v.(string)
		case "gtsa-train-end":
			req.GTSATrainEnd = v.(int)
		case "gtsa-validation-end":
			req.GTSAValidationEnd = v.(int)
		case "gtsa-test-end":
			req.GTSATestEnd = v.(int)
		case "fx-csv":
			req.FXCSVPath = v.(string)
		case "epitopes-csv":
			req.EpitopesCSVPath = v.(string)
		case "llvm-workflow-json":
			req.LLVMWorkflowJSONPath = v.(string)
		case "epitopes-gt-start":
			req.EpitopesGTStart = v.(int)
		case "epitopes-gt-end":
			req.EpitopesGTEnd = v.(int)
		case "epitopes-validation-start":
			req.EpitopesValidationStart = v.(int)
		case "epitopes-validation-end":
			req.EpitopesValidationEnd = v.(int)
		case "epitopes-test-start":
			req.EpitopesTestStart = v.(int)
		case "epitopes-test-end":
			req.EpitopesTestEnd = v.(int)
		case "epitopes-benchmark-start":
			req.EpitopesBenchmarkStart = v.(int)
		case "epitopes-benchmark-end":
			req.EpitopesBenchmarkEnd = v.(int)
		case "op-mode":
			req.OpMode = v.(string)
		case "evolution-type":
			req.EvolutionType = mapPopulationEvolutionType(v.(string))
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
		case "trace-step-size":
			req.TraceStepSize = v.(int)
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
		case "validation-probe":
			req.ValidationProbe = v.(bool)
		case "test-probe":
			req.TestProbe = v.(bool)
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
	if req.EvolutionType != "" {
		req.EvolutionType = mapPopulationEvolutionType(req.EvolutionType)
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
	case "fingerprint", "exact_fingerprint":
		return "fingerprint"
	case "tot_n":
		return "tot_n"
	case "pattern", "topology":
		return "topology"
	default:
		return ""
	}
}

func mapPopulationEvolutionType(name string) string {
	switch name {
	case "generational", "steady_state":
		return name
	default:
		return name
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

func float64Ptr(v float64) *float64 {
	return &v
}

func boolPtr(v bool) *bool {
	return &v
}

func intPtr(v int) *int {
	return &v
}

func applyMutationOperatorWeights(req *protoapi.RunRequest, operators []map2rec.WeightedOperator) {
	for _, op := range operators {
		switch mutationWeightBucket(op.Name) {
		case "perturb":
			req.WeightPerturb += op.Weight
		case "bias":
			req.WeightBias += op.Weight
		case "remove_bias":
			req.WeightRemoveBias += op.Weight
		case "activation":
			req.WeightActivation += op.Weight
		case "aggregator":
			req.WeightAggregator += op.Weight
		case "add_synapse":
			req.WeightAddSynapse += op.Weight
		case "remove_synapse":
			req.WeightRemoveSynapse += op.Weight
		case "add_neuron":
			req.WeightAddNeuron += op.Weight
		case "remove_neuron":
			req.WeightRemoveNeuron += op.Weight
		case "plasticity":
			req.WeightPlasticity += op.Weight
		case "plasticity_rule":
			req.WeightPlasticityRule += op.Weight
		case "substrate":
			req.WeightSubstrate += op.Weight
		}
	}
}

func hasAnyMutationWeightValue(req protoapi.RunRequest) bool {
	return req.WeightPerturb > 0 ||
		req.WeightBias > 0 ||
		req.WeightRemoveBias > 0 ||
		req.WeightActivation > 0 ||
		req.WeightAggregator > 0 ||
		req.WeightAddSynapse > 0 ||
		req.WeightRemoveSynapse > 0 ||
		req.WeightAddNeuron > 0 ||
		req.WeightRemoveNeuron > 0 ||
		req.WeightPlasticityRule > 0 ||
		req.WeightPlasticity > 0 ||
		req.WeightSubstrate > 0
}
