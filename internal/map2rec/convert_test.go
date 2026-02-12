package map2rec

import (
	"errors"
	"math"
	"testing"
)

func TestConvertUnsupportedKind(t *testing.T) {
	_, err := Convert("unknown", map[string]any{})
	if !errors.Is(err, ErrUnsupportedKind) {
		t.Fatalf("expected ErrUnsupportedKind, got %v", err)
	}
}

func TestConvertConstraintOverridesKnownFieldsAndIgnoresUnknown(t *testing.T) {
	in := map[string]any{
		"morphology":             "gtsa_v1",
		"population_selection_f": "hof_competition",
		"tuning_duration_f":      []any{"nsize_proportional", 0.25},
		"mutation_operators":     []any{[]any{"add_neuron", 10.0}, map[string]any{"name": "add_inlink", "weight": 6.0}},
		"tot_topological_mutations_fs": []any{
			[]any{"ncount_linear", 1.0},
		},
		"unknown_field": 123,
	}

	out := ConvertConstraint(in)
	if out.Morphology != "gtsa_v1" {
		t.Fatalf("unexpected morphology: %s", out.Morphology)
	}
	if out.PopulationSelectionF != "hof_competition" {
		t.Fatalf("unexpected selection: %s", out.PopulationSelectionF)
	}
	if out.TuningDurationF.Name != "nsize_proportional" || out.TuningDurationF.Param != 0.25 {
		t.Fatalf("unexpected tuning duration: %+v", out.TuningDurationF)
	}
	if len(out.MutationOperators) != 2 {
		t.Fatalf("unexpected mutation operators: %+v", out.MutationOperators)
	}
	if len(out.TotTopologicalMutationsFs) != 1 || out.TotTopologicalMutationsFs[0].Name != "ncount_linear" {
		t.Fatalf("unexpected topological policies: %+v", out.TotTopologicalMutationsFs)
	}
}

func TestConvertConstraintKeepsDefaultsOnInvalidFieldShape(t *testing.T) {
	in := map[string]any{
		"neural_afs": []any{"tanh", 7},
	}
	out := ConvertConstraint(in)
	if len(out.NeuralAFs) != 3 {
		t.Fatalf("expected default neural_afs to be retained, got %+v", out.NeuralAFs)
	}
}

func TestConvertPMPMapsFields(t *testing.T) {
	in := map[string]any{
		"op_mode":             "gt",
		"population_id":       "pop-1",
		"survival_percentage": 0.75,
		"specie_size_limit":   12,
		"init_specie_size":    24,
		"polis_id":            "polis-a",
		"generation_limit":    300,
		"evaluations_limit":   9000,
		"fitness_goal":        "inf",
		"benchmarker_pid":     "bench-1",
		"committee_pid":       "committee-1",
	}
	out := ConvertPMP(in)
	if out.PopulationID != "pop-1" || out.GenerationLimit != 300 {
		t.Fatalf("unexpected pmp conversion: %+v", out)
	}
	if !math.IsInf(out.FitnessGoal, 1) {
		t.Fatalf("expected infinite fitness goal, got %f", out.FitnessGoal)
	}
}
