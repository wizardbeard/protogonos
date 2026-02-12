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

func TestConvertDispatchesSensorAndActuatorKinds(t *testing.T) {
	gotSensor, err := Convert("sensor", map[string]any{"name": "s"})
	if err != nil {
		t.Fatalf("convert sensor: %v", err)
	}
	sensor, ok := gotSensor.(SensorRecord)
	if !ok || sensor.Name != "s" {
		t.Fatalf("unexpected sensor dispatch result: %#v", gotSensor)
	}

	gotActuator, err := Convert("actuator", map[string]any{"name": "a"})
	if err != nil {
		t.Fatalf("convert actuator: %v", err)
	}
	actuator, ok := gotActuator.(ActuatorRecord)
	if !ok || actuator.Name != "a" {
		t.Fatalf("unexpected actuator dispatch result: %#v", gotActuator)
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

func TestConvertSensorMapsFields(t *testing.T) {
	in := map[string]any{
		"id":            "sensor-1",
		"name":          "rangefinder",
		"type":          "distance",
		"cx_id":         "cx-1",
		"scape":         []any{"public", "flatland"},
		"vl":            9,
		"fanout_ids":    []any{"n-1", "n-2"},
		"generation":    3,
		"format":        map[string]any{"kind": "geo"},
		"parameters":    []any{1, 2},
		"gt_parameters": map[string]any{"noise": 0.1},
		"phys_rep":      map[string]any{"radius": 1.2},
		"vis_rep":       map[string]any{"color": "green"},
		"pre_f":         "identity",
		"post_f":        "clip",
	}
	out := ConvertSensor(in)
	if out.Name != "rangefinder" || out.Type != "distance" || out.VL != 9 {
		t.Fatalf("unexpected sensor map2rec output: %+v", out)
	}
	if len(out.FanoutIDs) != 2 || out.PreF != "identity" || out.PostF != "clip" {
		t.Fatalf("unexpected sensor io fields: %+v", out)
	}
}

func TestConvertSensorMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertSensor(map[string]any{"fanout_ids": "bad"})
	if len(out.FanoutIDs) != 0 {
		t.Fatalf("expected default empty fanout IDs, got %+v", out.FanoutIDs)
	}
}

func TestConvertActuatorMapsFields(t *testing.T) {
	in := map[string]any{
		"id":            "act-1",
		"name":          "thrust",
		"type":          "scalar",
		"cx_id":         "cx-2",
		"scape":         []any{"private", "cart-pole-lite"},
		"vl":            1,
		"fanin_ids":     []any{"n-3"},
		"generation":    4,
		"format":        map[string]any{"kind": "raw"},
		"parameters":    map[string]any{"limit": 1.0},
		"gt_parameters": []any{0.2, 0.4},
		"phys_rep":      "none",
		"vis_rep":       map[string]any{"shape": "bar"},
		"pre_f":         "scale",
		"post_f":        "identity",
	}
	out := ConvertActuator(in)
	if out.Name != "thrust" || out.Type != "scalar" || out.VL != 1 {
		t.Fatalf("unexpected actuator map2rec output: %+v", out)
	}
	if len(out.FaninIDs) != 1 || out.PreF != "scale" || out.PostF != "identity" {
		t.Fatalf("unexpected actuator io fields: %+v", out)
	}
}

func TestConvertActuatorMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertActuator(map[string]any{"fanin_ids": map[string]any{"x": 1}})
	if len(out.FaninIDs) != 0 {
		t.Fatalf("expected default empty fanin IDs, got %+v", out.FaninIDs)
	}
}
