package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestRegressionMimicScapeEvaluateWithIdentityAgent(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0},
		},
		Synapses: []model.Synapse{
			{From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}

	cortex, err := agent.NewCortex("reg-agent", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := RegressionMimicScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}

	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	if mse > 1e-9 {
		t.Fatalf("expected mse ~0, got %f", mse)
	}
	if fitness < 0.999999 {
		t.Fatalf("expected near-perfect fitness, got %f", fitness)
	}
}

func TestRegressionMimicScapeEvaluateWithScalarIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.ScalarInputSensorName},
		ActuatorIDs: []string{protoio.ScalarOutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0},
		},
		Synapses: []model.Synapse{
			{From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.ScalarInputSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.ScalarOutputActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("reg-agent-io", genome, sensors, actuators, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := RegressionMimicScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}

	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	if mse > 1e-9 {
		t.Fatalf("expected mse ~0, got %f", mse)
	}
	if fitness < 0.999999 {
		t.Fatalf("expected near-perfect fitness, got %f", fitness)
	}
}
