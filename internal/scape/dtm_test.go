package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestDTMScapeRewardsJunctionTurningPolicy(t *testing.T) {
	scape := DTMScape{}
	forward := scriptedStepAgent{
		id: "forward",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	junctionTurn := scriptedStepAgent{
		id: "junction-turn",
		fn: func(in []float64) []float64 {
			if len(in) >= 3 && in[0] > 0.5 && in[2] > 0.5 {
				return []float64{1}
			}
			return []float64{0}
		},
	}

	forwardFitness, _, err := scape.Evaluate(context.Background(), forward)
	if err != nil {
		t.Fatalf("evaluate forward: %v", err)
	}
	turnFitness, _, err := scape.Evaluate(context.Background(), junctionTurn)
	if err != nil {
		t.Fatalf("evaluate junction turn: %v", err)
	}
	if turnFitness <= forwardFitness {
		t.Fatalf("expected junction-turn policy to outperform forward policy, got turn=%f forward=%f", turnFitness, forwardFitness)
	}
}

func TestDTMScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.DTMRangeLeftSensorName,
			protoio.DTMRangeFrontSensorName,
			protoio.DTMRangeRightSensorName,
			protoio.DTMRewardSensorName,
		},
		ActuatorIDs: []string{protoio.DTMMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "rl", Activation: "identity"},
			{ID: "rf", Activation: "identity"},
			{ID: "rr", Activation: "identity"},
			{ID: "r", Activation: "identity"},
			{ID: "m", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "rl", To: "m", Weight: 1, Enabled: true},
			{From: "rr", To: "m", Weight: 1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.DTMRangeLeftSensorName:  protoio.NewScalarInputSensor(0),
		protoio.DTMRangeFrontSensorName: protoio.NewScalarInputSensor(0),
		protoio.DTMRangeRightSensorName: protoio.NewScalarInputSensor(0),
		protoio.DTMRewardSensorName:     protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.DTMMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"dtm-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"rl", "rf", "rr", "r"},
		[]string{"m"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := DTMScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["terminal_runs"].(int); !ok {
		t.Fatalf("trace missing terminal_runs: %+v", trace)
	}
}

func TestDTMScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := DTMScape{}
	junctionTurn := scriptedStepAgent{
		id: "junction-turn",
		fn: func(in []float64) []float64 {
			if len(in) >= 3 && in[0] > 0.5 && in[2] > 0.5 {
				return []float64{1}
			}
			return []float64{0}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), junctionTurn, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	validationRuns, ok := validationTrace["total_runs"].(int)
	if !ok || validationRuns <= 0 {
		t.Fatalf("expected positive validation total_runs, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), junctionTurn, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	testRuns, ok := testTrace["total_runs"].(int)
	if !ok || testRuns != validationRuns {
		t.Fatalf("expected matching total_runs between validation/test windows, got validation=%d test=%+v", validationRuns, testTrace["total_runs"])
	}
}

func TestDTMScapeTraceIncludesRunDiagnostics(t *testing.T) {
	scape := DTMScape{}
	junctionTurn := scriptedStepAgent{
		id: "junction-turn",
		fn: func(in []float64) []float64 {
			if len(in) >= 3 && in[0] > 0.5 && in[2] > 0.5 {
				return []float64{1}
			}
			return []float64{0}
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), junctionTurn)
	if err != nil {
		t.Fatalf("evaluate junction turn: %v", err)
	}
	if _, ok := trace["terminal_reward_total"].(float64); !ok {
		t.Fatalf("trace missing terminal_reward_total: %+v", trace)
	}
	if _, ok := trace["avg_steps_per_run"].(float64); !ok {
		t.Fatalf("trace missing avg_steps_per_run: %+v", trace)
	}
	if _, ok := trace["fitness_delta"].(float64); !ok {
		t.Fatalf("trace missing fitness_delta: %+v", trace)
	}
	if _, ok := trace["switch_triggered_at"].(int); !ok {
		t.Fatalf("trace missing switch_triggered_at: %+v", trace)
	}
	leftRuns, lok := trace["left_terminal_runs"].(int)
	rightRuns, rok := trace["right_terminal_runs"].(int)
	terminalRuns, tok := trace["terminal_runs"].(int)
	if !lok || !rok || !tok {
		t.Fatalf("trace missing terminal run side diagnostics: %+v", trace)
	}
	if leftRuns+rightRuns != terminalRuns {
		t.Fatalf("expected side terminal runs to sum to terminal_runs, got left=%d right=%d total=%d", leftRuns, rightRuns, terminalRuns)
	}
	timeoutRuns, ok := trace["timeout_runs"].(int)
	if !ok {
		t.Fatalf("trace missing timeout_runs: %+v", trace)
	}
	if timeoutRuns != 0 {
		t.Fatalf("expected zero timeout_runs for reference-style dtm run flow, got %+v", trace)
	}
	maxRunStepIndex, ok := trace["max_run_step_index"].(int)
	if !ok || maxRunStepIndex <= 0 {
		t.Fatalf("expected positive max_run_step_index, got %+v", trace)
	}
	if maxRunStepIndex > 3 {
		t.Fatalf("expected bounded dtm run-step depth <= 3, got %+v", trace)
	}
	if stepIndex, ok := trace["last_step_index"].(int); !ok || stepIndex != 0 {
		t.Fatalf("expected reset last_step_index=0 after terminal episode completion, got %+v", trace)
	}
}
