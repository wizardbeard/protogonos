package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadRunRequestFromConfigUsesConstraintAndPMP(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config.json")
	payload := map[string]any{
		"run_id":                  "cfg-run-1",
		"continue_population_id":  "pop-prev",
		"scape":                   "gtsa",
		"seed":                    77,
		"workers":                 3,
		"start_paused":            true,
		"auto_continue_ms":        25,
		"tune_perturbation_range": 1.8,
		"tune_annealing_factor":   0.95,
		"tune_min_improvement":    0.015,
		"validation_probe":        true,
		"test_probe":              true,
		"trace": map[string]any{
			"step_size": 333,
		},
		"pmp": map[string]any{
			"op_mode":             "validation",
			"survival_percentage": 0.6,
			"specie_size_limit":   3,
			"init_specie_size":    12,
			"generation_limit":    9,
			"evaluations_limit":   111,
			"fitness_goal":        0.88,
			"population_id":       "pmp-pop",
		},
		"constraint": map[string]any{
			"population_evo_alg_f":               "steady_state",
			"population_selection_f":             "hof_competition",
			"population_fitness_postprocessor_f": "nsize_proportional",
			"tuning_selection_fs":                []any{"dynamic_random"},
			"tuning_duration_f":                  []any{"const", 7},
			"tot_topological_mutations_fs": []any{
				[]any{"ncount_exponential", 0.8},
			},
			"mutation_operators": []any{
				[]any{"add_bias", 5},
				[]any{"remove_bias", 1},
				[]any{"mutate_af", 6},
				[]any{"mutate_aggrf", 7},
				[]any{"add_outlink", 4},
				[]any{"add_neuron", 3},
				[]any{"mutate_pf", 8},
				[]any{"mutate_plasticity_parameters", 2},
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.RunID != "cfg-run-1" || req.ContinuePopulationID != "pop-prev" || req.Scape != "gtsa" || req.Seed != 77 || req.Workers != 3 {
		t.Fatalf("unexpected base fields: %+v", req)
	}
	if !req.StartPaused || req.AutoContinueAfter != 25*time.Millisecond {
		t.Fatalf("expected pause controls from top-level config, got start=%t after=%s", req.StartPaused, req.AutoContinueAfter)
	}
	if req.Population != 12 || req.Generations != 9 {
		t.Fatalf("expected pmp derived population/generations, got pop=%d gens=%d", req.Population, req.Generations)
	}
	if req.OpMode != "validation" {
		t.Fatalf("expected pmp-derived op mode validation, got %s", req.OpMode)
	}
	if req.EvolutionType != "steady_state" {
		t.Fatalf("expected constraint-derived evolution type steady_state, got %s", req.EvolutionType)
	}
	if req.SurvivalPercentage != 0.6 || req.EvaluationsLimit != 111 || req.FitnessGoal != 0.88 {
		t.Fatalf("expected pmp-derived monitor controls, got survival=%f eval_limit=%d fitness_goal=%f", req.SurvivalPercentage, req.EvaluationsLimit, req.FitnessGoal)
	}
	if req.SpecieSizeLimit != 3 {
		t.Fatalf("expected pmp-derived specie size limit 3, got %d", req.SpecieSizeLimit)
	}
	if req.Selection != "species_shared_tournament" {
		t.Fatalf("unexpected selection mapping: %s", req.Selection)
	}
	if req.SpecieIdentifier != "tot_n" {
		t.Fatalf("unexpected specie identifier mapping: %s", req.SpecieIdentifier)
	}
	if req.TuneSelection != "dynamic_random" {
		t.Fatalf("unexpected tune selection mapping: %s", req.TuneSelection)
	}
	if req.TuneDurationPolicy != "const" || req.TuneDurationParam != 7 {
		t.Fatalf("unexpected tune duration mapping: policy=%s param=%f", req.TuneDurationPolicy, req.TuneDurationParam)
	}
	if req.TunePerturbationRange != 1.8 || req.TuneAnnealingFactor != 0.95 {
		t.Fatalf("unexpected tuning spread params: range=%f annealing=%f", req.TunePerturbationRange, req.TuneAnnealingFactor)
	}
	if req.TuneMinImprovement != 0.015 {
		t.Fatalf("unexpected tune min improvement mapping: %f", req.TuneMinImprovement)
	}
	if !req.ValidationProbe || !req.TestProbe {
		t.Fatalf("expected validation/test probes from config, got validation=%t test=%t", req.ValidationProbe, req.TestProbe)
	}
	if req.TraceStepSize != 333 {
		t.Fatalf("expected trace step size from trace record, got %d", req.TraceStepSize)
	}
	if req.TopologicalPolicy != "ncount_exponential" || req.TopologicalParam != 0.8 {
		t.Fatalf("unexpected topological policy mapping: policy=%s param=%f", req.TopologicalPolicy, req.TopologicalParam)
	}
	if req.FitnessPostprocessor != "nsize_proportional" {
		t.Fatalf("unexpected fitness postprocessor mapping: %s", req.FitnessPostprocessor)
	}
	if req.WeightBias != 5 || req.WeightRemoveBias != 1 || req.WeightActivation != 6 || req.WeightAggregator != 7 || req.WeightAddSynapse != 4 || req.WeightAddNeuron != 3 || req.WeightPlasticityRule != 8 || req.WeightPlasticity != 2 {
		t.Fatalf("unexpected mapped mutation weights: %+v", req)
	}
}

func TestLoadRunRequestFromConfigUsesTopLevelTraceStepSizeOverride(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_trace_step.json")
	payload := map[string]any{
		"trace_step_size": 777,
		"trace": map[string]any{
			"step_size": 250,
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.TraceStepSize != 777 {
		t.Fatalf("expected top-level trace_step_size override, got %d", req.TraceStepSize)
	}
}

func TestLoadRunRequestFromConfigParsesScapeDataSources(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_scape_data.json")
	payload := map[string]any{
		"gtsa_csv_path":       "top-gtsa.csv",
		"gtsa_train_end":      120,
		"gtsa_validation_end": 180,
		"gtsa_test_end":       240,
		"fx_csv_path":         "top-fx.csv",
		"epitopes_csv_path":   "top-ep.csv",
		"llvm_workflow_json":  "top-llvm.json",
		"epitopes_gt_start":   5,
		"epitopes_gt_end":     25,
		"epitopes_test_start": 50,
		"epitopes_test_end":   70,
		"scape_data": map[string]any{
			"gtsa": map[string]any{
				"csv_path":       "nested-gtsa.csv",
				"train_end":      90,
				"validation_end": 140,
				"test_end":       200,
			},
			"fx": map[string]any{
				"csv_path": "nested-fx.csv",
			},
			"epitopes": map[string]any{
				"csv_path":         "nested-ep.csv",
				"validation_start": 30,
				"validation_end":   40,
				"benchmark_start":  80,
				"benchmark_end":    110,
			},
			"llvm": map[string]any{
				"workflow_json_path": "nested-llvm.json",
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.GTSACSVPath != "top-gtsa.csv" || req.GTSATrainEnd != 120 || req.GTSAValidationEnd != 180 || req.GTSATestEnd != 240 {
		t.Fatalf("unexpected gtsa source fields: %+v", req)
	}
	if req.FXCSVPath != "top-fx.csv" {
		t.Fatalf("unexpected fx csv path: %+v", req)
	}
	if req.EpitopesCSVPath != "top-ep.csv" || req.EpitopesGTStart != 5 || req.EpitopesGTEnd != 25 || req.EpitopesTestStart != 50 || req.EpitopesTestEnd != 70 {
		t.Fatalf("unexpected epitopes top-level fields: %+v", req)
	}
	if req.LLVMWorkflowJSONPath != "top-llvm.json" {
		t.Fatalf("unexpected llvm workflow source: %+v", req)
	}
	// Nested scape_data fills remaining unset fields.
	if req.EpitopesValidationStart != 30 || req.EpitopesValidationEnd != 40 || req.EpitopesBenchmarkStart != 80 || req.EpitopesBenchmarkEnd != 110 {
		t.Fatalf("unexpected nested epitopes fallback fields: %+v", req)
	}
}

func TestLoadRunRequestFromConfigParsesNestedLLVMWorkflowSourceFallback(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_scape_data_llvm.json")
	payload := map[string]any{
		"scape_data": map[string]any{
			"llvm": map[string]any{
				"workflow_json": "nested-llvm.json",
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.LLVMWorkflowJSONPath != "nested-llvm.json" {
		t.Fatalf("expected nested llvm workflow fallback, got %+v", req)
	}
}

func TestLoadRunRequestFromConfigParsesFlatlandOverrides(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_flatland_overrides.json")
	payload := map[string]any{
		"flatland_scanner_profile": "core3",
		"flatland_scanner_spread":  0.21,
		"flatland_max_age":         95,
		"scape_data": map[string]any{
			"flatland": map[string]any{
				"scanner_profile":      "forward5",
				"scanner_offset":       -0.25,
				"layout_randomize":     false,
				"layout_variants":      6,
				"force_layout_variant": 3,
				"benchmark_trials":     4,
				"forage_goal":          7,
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	// Top-level value wins when both top-level and scape_data are provided.
	if req.FlatlandScannerProfile != "core3" {
		t.Fatalf("expected top-level flatland scanner profile core3, got %q", req.FlatlandScannerProfile)
	}
	if req.FlatlandScannerSpread == nil || *req.FlatlandScannerSpread != 0.21 {
		t.Fatalf("expected flatland scanner spread 0.21, got %+v", req.FlatlandScannerSpread)
	}
	// Nested scape_data fills remaining unset flatland fields.
	if req.FlatlandScannerOffset == nil || *req.FlatlandScannerOffset != -0.25 {
		t.Fatalf("expected nested flatland scanner offset -0.25, got %+v", req.FlatlandScannerOffset)
	}
	if req.FlatlandLayoutRandomize == nil || *req.FlatlandLayoutRandomize {
		t.Fatalf("expected nested flatland layout_randomize=false, got %+v", req.FlatlandLayoutRandomize)
	}
	if req.FlatlandLayoutVariants == nil || *req.FlatlandLayoutVariants != 6 {
		t.Fatalf("expected nested flatland layout variants=6, got %+v", req.FlatlandLayoutVariants)
	}
	if req.FlatlandForceLayout == nil || *req.FlatlandForceLayout != 3 {
		t.Fatalf("expected nested flatland force layout variant=3, got %+v", req.FlatlandForceLayout)
	}
	if req.FlatlandBenchmarkTrials == nil || *req.FlatlandBenchmarkTrials != 4 {
		t.Fatalf("expected nested flatland benchmark trials=4, got %+v", req.FlatlandBenchmarkTrials)
	}
	if req.FlatlandMaxAge == nil || *req.FlatlandMaxAge != 95 {
		t.Fatalf("expected top-level flatland max_age=95, got %+v", req.FlatlandMaxAge)
	}
	if req.FlatlandForageGoal == nil || *req.FlatlandForageGoal != 7 {
		t.Fatalf("expected nested flatland forage_goal=7, got %+v", req.FlatlandForageGoal)
	}
}

func TestLoadRunRequestFromConfigMapsOpModeList(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_op_mode_list.json")
	payload := map[string]any{
		"op_mode": []any{"gt", "validation"},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.OpMode != "gt,validation" {
		t.Fatalf("expected op_mode list to map as joined string, got %q", req.OpMode)
	}
}

func TestLoadRunRequestFromConfigUsesPMPEvolutionTypeFallback(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_pmp_evolution.json")
	payload := map[string]any{
		"pmp": map[string]any{
			"population_id":    "pmp-evo-pop",
			"init_specie_size": 7,
			"evolution_type":   "steady_state",
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.EvolutionType != "steady_state" {
		t.Fatalf("expected pmp evolution type fallback, got %s", req.EvolutionType)
	}
}

func TestMapSpecieIdentifierFingerprint(t *testing.T) {
	if got := mapSpecieIdentifier([]string{"fingerprint"}); got != "fingerprint" {
		t.Fatalf("unexpected fingerprint mapping: %q", got)
	}
	if got := mapSpecieIdentifier([]string{"exact_fingerprint"}); got != "fingerprint" {
		t.Fatalf("unexpected exact_fingerprint mapping: %q", got)
	}
}

func TestLoadRunRequestFromConfigUsesPMPPopulationIDForContinuationDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_pmp_pop.json")
	payload := map[string]any{
		"scape": "xor",
		"pmp": map[string]any{
			"population_id":    "pmp-cont-pop",
			"init_specie_size": 7,
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.ContinuePopulationID != "pmp-cont-pop" {
		t.Fatalf("expected continue population id from pmp population_id, got %s", req.ContinuePopulationID)
	}
	if req.RunID != "pmp-cont-pop" {
		t.Fatalf("expected run id default from pmp population_id, got %s", req.RunID)
	}
}

func TestLoadRunRequestFromConfigUsesPopulationSpecieAndAgentRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_population_specie_agent.json")
	payload := map[string]any{
		"population": map[string]any{
			"evo_alg_f":               "steady_state",
			"fitness_postprocessor_f": "size_proportional",
			"selection_f":             "competition",
			"trace": map[string]any{
				"step_size": 444,
			},
		},
		"specie": map[string]any{
			"specie_distinguishers": []any{"exact_fingerprint"},
		},
		"agent": map[string]any{
			"tuning_selection_f":          "active_random",
			"tuning_duration_f":           []any{"nsize_proportional", 0.25},
			"tot_topological_mutations_f": []any{"const", 3},
			"mutation_operators": []any{
				[]any{"add_inlink", 2.5},
				[]any{"mutate_pf", 1.5},
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.EvolutionType != "steady_state" || req.Selection != "competition" || req.FitnessPostprocessor != "size_proportional" {
		t.Fatalf("unexpected population record mapping: %+v", req)
	}
	if req.TraceStepSize != 444 {
		t.Fatalf("expected trace step size from population.trace, got %d", req.TraceStepSize)
	}
	if req.SpecieIdentifier != "fingerprint" {
		t.Fatalf("expected specie distinguisher mapping to fingerprint, got %s", req.SpecieIdentifier)
	}
	if req.TuneSelection != "active_random" || req.TuneDurationPolicy != "nsize_proportional" || req.TuneDurationParam != 0.25 {
		t.Fatalf("unexpected agent tuning mapping: %+v", req)
	}
	if req.TopologicalPolicy != "const" || req.TopologicalCount != 3 {
		t.Fatalf("unexpected agent topological mapping: policy=%s count=%d", req.TopologicalPolicy, req.TopologicalCount)
	}
	if req.WeightAddSynapse != 2.5 || req.WeightPlasticityRule != 1.5 {
		t.Fatalf("unexpected agent mutation weight mapping: add_synapse=%f plasticity_rule=%f", req.WeightAddSynapse, req.WeightPlasticityRule)
	}
}

func TestHasAnyWeightOverrideFlag(t *testing.T) {
	if hasAnyWeightOverrideFlag(map[string]bool{}) {
		t.Fatal("expected false for empty set")
	}
	if !hasAnyWeightOverrideFlag(map[string]bool{"w-add-neuron": true}) {
		t.Fatal("expected true when one weight flag is set")
	}
	if !hasAnyWeightOverrideFlag(map[string]bool{"w-bias": true}) {
		t.Fatal("expected true when bias weight flag is set")
	}
	if !hasAnyWeightOverrideFlag(map[string]bool{"w-remove-bias": true}) {
		t.Fatal("expected true when remove-bias weight flag is set")
	}
	if !hasAnyWeightOverrideFlag(map[string]bool{"w-plasticity-rule": true}) {
		t.Fatal("expected true when plasticity-rule weight flag is set")
	}
}

func TestLoadRunRequestFromConfigPreservesSelectionAliases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_aliases.json")
	payload := map[string]any{
		"constraint": map[string]any{
			"population_selection_f": "top3",
			"mutation_operators": []any{
				[]any{"add_bias", 1},
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.Selection != "top3" {
		t.Fatalf("expected top3 alias preserved, got %s", req.Selection)
	}
}

func TestLoadRunRequestFromConfigTreatsPMPInfiniteFitnessGoalAsDisabled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_inf_goal.json")
	payload := map[string]any{
		"pmp": map[string]any{
			"init_specie_size":  10,
			"generation_limit":  4,
			"evaluations_limit": 200,
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.FitnessGoal != 0 {
		t.Fatalf("expected disabled fitness goal for pmp inf default, got %f", req.FitnessGoal)
	}
	if req.EvaluationsLimit != 200 {
		t.Fatalf("expected evaluations_limit to map from pmp, got %d", req.EvaluationsLimit)
	}
}

func TestLoadRunRequestFromConfigMapsSubstrateMutationAliases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_substrate_aliases.json")
	payload := map[string]any{
		"constraint": map[string]any{
			"mutation_operators": []any{
				[]any{"cutlink_FromSensorToNeuron", 2.5},
				[]any{"cutlink_FromNeuronToActuator", 3.5},
				[]any{"remove_cpp", 1.25},
				[]any{"remove_cep", 0.75},
				[]any{"delete_CircuitNode", 4.0},
				[]any{"link_FromSensorToNeuron", 0.5},
				[]any{"link_FromNeuronToActuator", 1.5},
				[]any{"mutate_tuning_selection", 1.0},
				[]any{"mutate_tuning_annealing", 1.0},
				[]any{"mutate_tot_topological_mutations", 1.0},
				[]any{"mutate_heredity_type", 1.0},
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.WeightSubstrate != 18.0 {
		t.Fatalf("unexpected substrate alias weight total: got=%f want=18", req.WeightSubstrate)
	}
}

func TestLoadRunRequestFromConfigMapsCutlinkNeuronToNeuronAlias(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config_cutlink_n2n.json")
	payload := map[string]any{
		"constraint": map[string]any{
			"mutation_operators": []any{
				[]any{"cutlink_FromNeuronToNeuron", 2.25},
				[]any{"cutlink_FromElementToElement", 1.75},
				[]any{"link_FromElementToElement", 0.5},
				[]any{"link_FromNeuronToNeuron", 0.75},
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req, err := loadRunRequestFromConfig(path)
	if err != nil {
		t.Fatalf("load run request: %v", err)
	}
	if req.WeightRemoveSynapse != 4.0 {
		t.Fatalf("unexpected remove synapse alias weight total: got=%f want=4.0", req.WeightRemoveSynapse)
	}
	if req.WeightAddSynapse != 1.25 {
		t.Fatalf("unexpected add synapse alias weight total: got=%f want=1.25", req.WeightAddSynapse)
	}
}

func TestNormalizeMutationOperatorNameLegacyCircuitAliases(t *testing.T) {
	cases := map[string]string{
		"add_CircuitNode":    "add_circuit_node",
		"delete_CircuitNode": "delete_circuit_node",
		"add_CircuitLayer":   "add_circuit_layer",
		"remove_outLink":     "remove_outlink",
		"mutate_weights":     "mutate_weights",
	}
	for in, want := range cases {
		if got := normalizeMutationOperatorName(in); got != want {
			t.Fatalf("normalizeMutationOperatorName(%q)=%q want=%q", in, got, want)
		}
	}
}
