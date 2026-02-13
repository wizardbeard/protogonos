package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadRunRequestFromConfigUsesConstraintAndPMP(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run_config.json")
	payload := map[string]any{
		"scape":   "gtsa",
		"seed":    77,
		"workers": 3,
		"pmp": map[string]any{
			"init_specie_size": 12,
			"generation_limit": 9,
		},
		"constraint": map[string]any{
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
	if req.Scape != "gtsa" || req.Seed != 77 || req.Workers != 3 {
		t.Fatalf("unexpected base fields: %+v", req)
	}
	if req.Population != 12 || req.Generations != 9 {
		t.Fatalf("expected pmp derived population/generations, got pop=%d gens=%d", req.Population, req.Generations)
	}
	if req.Selection != "species_shared_tournament" {
		t.Fatalf("unexpected selection mapping: %s", req.Selection)
	}
	if req.TuneSelection != "dynamic_random" {
		t.Fatalf("unexpected tune selection mapping: %s", req.TuneSelection)
	}
	if req.TuneDurationPolicy != "const" || req.TuneDurationParam != 7 {
		t.Fatalf("unexpected tune duration mapping: policy=%s param=%f", req.TuneDurationPolicy, req.TuneDurationParam)
	}
	if req.TopologicalPolicy != "ncount_exponential" || req.TopologicalParam != 0.8 {
		t.Fatalf("unexpected topological policy mapping: policy=%s param=%f", req.TopologicalPolicy, req.TopologicalParam)
	}
	if req.FitnessPostprocessor != "nsize_proportional" {
		t.Fatalf("unexpected fitness postprocessor mapping: %s", req.FitnessPostprocessor)
	}
	if req.WeightBias != 5 || req.WeightRemoveBias != 1 || req.WeightActivation != 6 || req.WeightAggregator != 7 || req.WeightAddSynapse != 4 || req.WeightAddNeuron != 3 || req.WeightPlasticity != 2 {
		t.Fatalf("unexpected mapped mutation weights: %+v", req)
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
