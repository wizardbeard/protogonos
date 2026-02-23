package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestCartPoleLiteScapeEvaluateWithHandBuiltAgent(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "v", Activation: "identity"},
			{ID: "f", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -1.2, Enabled: true},
			{From: "v", To: "f", Weight: -0.6, Enabled: true},
		},
	}

	cortex, err := agent.NewCortex("cp-agent", genome, nil, nil, []string{"x", "v"}, []string{"f"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := CartPoleLiteScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	avgReward, ok := trace["avg_reward"].(float64)
	if !ok {
		t.Fatalf("trace missing avg_reward: %+v", trace)
	}
	if avgReward <= 0.5 {
		t.Fatalf("expected avg_reward > 0.5, got %f", avgReward)
	}
	if fitness <= 0.5 {
		t.Fatalf("expected fitness > 0.5, got %f", fitness)
	}
}

func TestCartPoleLiteScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.CartPolePositionSensorName, protoio.CartPoleVelocitySensorName},
		ActuatorIDs: []string{protoio.CartPoleForceActuatorName},
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "v", Activation: "identity"},
			{ID: "f", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -1.2, Enabled: true},
			{From: "v", To: "f", Weight: -0.6, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.CartPolePositionSensorName: protoio.NewScalarInputSensor(0),
		protoio.CartPoleVelocitySensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.CartPoleForceActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("cp-agent-io", genome, sensors, actuators, []string{"x", "v"}, []string{"f"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := CartPoleLiteScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	avgReward, ok := trace["avg_reward"].(float64)
	if !ok {
		t.Fatalf("trace missing avg_reward: %+v", trace)
	}
	if avgReward <= 0.5 {
		t.Fatalf("expected avg_reward > 0.5, got %f", avgReward)
	}
	if fitness <= 0.5 {
		t.Fatalf("expected fitness > 0.5, got %f", fitness)
	}
}

func TestCartPoleLiteScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := CartPoleLiteScape{}
	stabilizer := scriptedStepAgent{
		id: "stabilizer",
		fn: func(input []float64) []float64 {
			if len(input) < 2 {
				return []float64{0}
			}
			return []float64{-1.2*input[0] - 0.6*input[1]}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), stabilizer, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), stabilizer, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
}
