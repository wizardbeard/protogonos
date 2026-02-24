package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestPole2BalancingScapeEvaluatesStepPolicies(t *testing.T) {
	scape := Pole2BalancingScape{}
	thrash := scriptedStepAgent{
		id: "thrash",
		fn: func(_ []float64) []float64 { return []float64{1} },
	}
	stabilize := scriptedStepAgent{
		id: "stabilize",
		fn: func(in []float64) []float64 {
			if len(in) < 6 {
				return []float64{0}
			}
			force := -(0.9*in[0] + 0.6*in[1] + 8.0*in[2] + 1.4*in[3] + 10.0*in[4] + 1.8*in[5])
			return []float64{force}
		},
	}

	thrashFitness, thrashTrace, err := scape.Evaluate(context.Background(), thrash)
	if err != nil {
		t.Fatalf("evaluate thrash: %v", err)
	}
	stabilizeFitness, stabilizeTrace, err := scape.Evaluate(context.Background(), stabilize)
	if err != nil {
		t.Fatalf("evaluate stabilize: %v", err)
	}
	if thrashFitness <= 0 || stabilizeFitness <= 0 {
		t.Fatalf("expected positive fitness for evaluated policies, got stabilize=%f thrash=%f", stabilizeFitness, thrashFitness)
	}
	if _, ok := thrashTrace["steps_survived"].(int); !ok {
		t.Fatalf("thrash trace missing steps_survived: %+v", thrashTrace)
	}
	if _, ok := stabilizeTrace["steps_survived"].(int); !ok {
		t.Fatalf("stabilize trace missing steps_survived: %+v", stabilizeTrace)
	}
}

func TestPole2BalancingScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2CartVelocitySensorName,
			protoio.Pole2Angle1SensorName,
			protoio.Pole2Velocity1SensorName,
			protoio.Pole2Angle2SensorName,
			protoio.Pole2Velocity2SensorName,
		},
		ActuatorIDs: []string{protoio.Pole2PushActuatorName},
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "v", Activation: "identity"},
			{ID: "a1", Activation: "identity"},
			{ID: "w1", Activation: "identity"},
			{ID: "a2", Activation: "identity"},
			{ID: "w2", Activation: "identity"},
			{ID: "f", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -0.8, Enabled: true},
			{From: "v", To: "f", Weight: -0.2, Enabled: true},
			{From: "a1", To: "f", Weight: -4.5, Enabled: true},
			{From: "w1", To: "f", Weight: -0.9, Enabled: true},
			{From: "a2", To: "f", Weight: -6.0, Enabled: true},
			{From: "w2", To: "f", Weight: -1.1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.Pole2CartPositionSensorName: protoio.NewScalarInputSensor(0),
		protoio.Pole2CartVelocitySensorName: protoio.NewScalarInputSensor(0),
		protoio.Pole2Angle1SensorName:       protoio.NewScalarInputSensor(0),
		protoio.Pole2Velocity1SensorName:    protoio.NewScalarInputSensor(0),
		protoio.Pole2Angle2SensorName:       protoio.NewScalarInputSensor(0),
		protoio.Pole2Velocity2SensorName:    protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.Pole2PushActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"pole2-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"x", "v", "a1", "w1", "a2", "w2"},
		[]string{"f"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := Pole2BalancingScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["steps_survived"].(int); !ok {
		t.Fatalf("trace missing steps_survived: %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "6" {
		t.Fatalf("expected full pole2 sensor_surface=6, got %+v", trace)
	}
}

func TestPole2BalancingScapeEvaluateWithReducedIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2Angle1SensorName,
			protoio.Pole2Angle2SensorName,
		},
		ActuatorIDs: []string{protoio.Pole2PushActuatorName},
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "a1", Activation: "identity"},
			{ID: "a2", Activation: "identity"},
			{ID: "f", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -0.7, Enabled: true},
			{From: "a1", To: "f", Weight: -3.8, Enabled: true},
			{From: "a2", To: "f", Weight: -5.4, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.Pole2CartPositionSensorName: protoio.NewScalarInputSensor(0),
		protoio.Pole2Angle1SensorName:       protoio.NewScalarInputSensor(0),
		protoio.Pole2Angle2SensorName:       protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.Pole2PushActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"pole2-agent-io-reduced",
		genome,
		sensors,
		actuators,
		[]string{"x", "a1", "a2"},
		[]string{"f"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := Pole2BalancingScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "3" {
		t.Fatalf("expected reduced pole2 sensor_surface=3, got %+v", trace)
	}
}

func TestPole2BalancingScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := Pole2BalancingScape{}
	stabilize := scriptedStepAgent{
		id: "stabilize",
		fn: func(in []float64) []float64 {
			if len(in) < 6 {
				return []float64{0}
			}
			force := -(0.9*in[0] + 0.6*in[1] + 8.0*in[2] + 1.4*in[3] + 10.0*in[4] + 1.8*in[5])
			return []float64{force}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), stabilize, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	if maxSteps, ok := validationTrace["max_steps"].(int); !ok || maxSteps <= 0 {
		t.Fatalf("expected positive max_steps in validation trace, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), stabilize, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	if validationInit, vok := validationTrace["init_angle2"].(float64); vok {
		if testInit, tok := testTrace["init_angle2"].(float64); tok && validationInit == testInit {
			t.Fatalf("expected distinct mode initialization for pole2 windows, got validation=%f test=%f", validationInit, testInit)
		}
	}
}

func TestPole2BalancingScapeTraceIncludesTerminationAccounting(t *testing.T) {
	scape := Pole2BalancingScape{}
	thrash := scriptedStepAgent{
		id: "thrash",
		fn: func(_ []float64) []float64 { return []float64{1} },
	}

	_, trace, err := scape.Evaluate(context.Background(), thrash)
	if err != nil {
		t.Fatalf("evaluate thrash: %v", err)
	}
	if _, ok := trace["termination_reason"].(string); !ok {
		t.Fatalf("trace missing termination_reason: %+v", trace)
	}
	if _, ok := trace["fitness_acc"].(float64); !ok {
		t.Fatalf("trace missing fitness_acc: %+v", trace)
	}
	if _, ok := trace["avg_step_fitness"].(float64); !ok {
		t.Fatalf("trace missing avg_step_fitness: %+v", trace)
	}
	if _, ok := trace["goal_steps"].(int); !ok {
		t.Fatalf("trace missing goal_steps: %+v", trace)
	}
	if goalSteps, ok := trace["goal_steps"].(int); !ok || goalSteps != 100000 {
		t.Fatalf("expected reference goal_steps=100000 in gt mode, got %+v", trace)
	}
	if _, ok := trace["terminated_by_bounds"].(bool); !ok {
		t.Fatalf("trace missing terminated_by_bounds: %+v", trace)
	}
	if _, ok := trace["default_damping"].(bool); !ok {
		t.Fatalf("trace missing default_damping: %+v", trace)
	}
	if _, ok := trace["default_double_pole"].(bool); !ok {
		t.Fatalf("trace missing default_double_pole: %+v", trace)
	}
	if vectorSteps, ok := trace["vector_control_steps"].(int); !ok || vectorSteps != 0 {
		t.Fatalf("expected scalar thrash policy to report zero vector_control_steps, got %+v", trace)
	}
	if _, ok := trace["damping_off_steps"].(int); !ok {
		t.Fatalf("trace missing damping_off_steps: %+v", trace)
	}
	if _, ok := trace["single_pole_steps"].(int); !ok {
		t.Fatalf("trace missing single_pole_steps: %+v", trace)
	}
}

func TestPole2BalancingScapeSupportsVectorPushControls(t *testing.T) {
	scape := Pole2BalancingScape{}
	vectorControl := scriptedStepAgent{
		id: "vector-control",
		fn: func(_ []float64) []float64 {
			// Force + damping flag + double-pole flag.
			// Negative damping flag disables damping; negative double-pole flag emulates single-pole mode.
			return []float64{0.25, -1.0, -1.0}
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), vectorControl)
	if err != nil {
		t.Fatalf("evaluate vector-control policy: %v", err)
	}
	vectorSteps, ok := trace["vector_control_steps"].(int)
	if !ok || vectorSteps <= 0 {
		t.Fatalf("expected positive vector_control_steps, got %+v", trace)
	}
	dampingOffSteps, ok := trace["damping_off_steps"].(int)
	if !ok || dampingOffSteps <= 0 {
		t.Fatalf("expected positive damping_off_steps, got %+v", trace)
	}
	singlePoleSteps, ok := trace["single_pole_steps"].(int)
	if !ok || singlePoleSteps <= 0 {
		t.Fatalf("expected positive single_pole_steps, got %+v", trace)
	}
}
