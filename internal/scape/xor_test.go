package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestXORScapeEvaluateWithHandBuiltAgent(t *testing.T) {
	// Hidden-layer sigmoid network that approximates XOR.
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "h1", Activation: "sigmoid", Bias: -10},
			{ID: "h2", Activation: "sigmoid", Bias: 30},
			{ID: "o", Activation: "sigmoid", Bias: -30},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "h1", Weight: 20, Enabled: true},
			{From: "i2", To: "h1", Weight: 20, Enabled: true},
			{From: "i1", To: "h2", Weight: -20, Enabled: true},
			{From: "i2", To: "h2", Weight: -20, Enabled: true},
			{From: "h1", To: "o", Weight: 20, Enabled: true},
			{From: "h2", To: "o", Weight: 20, Enabled: true},
		},
	}

	cortex, err := agent.NewCortex("xor-agent", genome, nil, nil, []string{"i1", "i2"}, []string{"o"})
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	xor := XORScape{}
	fitness, trace, err := xor.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}

	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	if mse > 0.05 {
		t.Fatalf("expected mse <= 0.05, got %f", mse)
	}
	if fitness < 0.95 {
		t.Fatalf("expected fitness >= 0.95, got %f", fitness)
	}
}

func TestXORScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "h1", Activation: "sigmoid", Bias: -10},
			{ID: "h2", Activation: "sigmoid", Bias: 30},
			{ID: "o", Activation: "sigmoid", Bias: -30},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "h1", Weight: 20, Enabled: true},
			{From: "i2", To: "h1", Weight: 20, Enabled: true},
			{From: "i1", To: "h2", Weight: -20, Enabled: true},
			{From: "i2", To: "h2", Weight: -20, Enabled: true},
			{From: "h1", To: "o", Weight: 20, Enabled: true},
			{From: "h2", To: "o", Weight: 20, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.XORInputLeftSensorName:  protoio.NewScalarInputSensor(0),
		protoio.XORInputRightSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.XOROutputActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("xor-agent-io", genome, sensors, actuators, []string{"i1", "i2"}, []string{"o"})
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	xor := XORScape{}
	fitness, trace, err := xor.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}

	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	if mse > 0.05 {
		t.Fatalf("expected mse <= 0.05, got %f", mse)
	}
	if fitness < 0.95 {
		t.Fatalf("expected fitness >= 0.95, got %f", fitness)
	}
}
