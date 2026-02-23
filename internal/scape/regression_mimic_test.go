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

func TestRegressionMimicScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := RegressionMimicScape{}
	identity := scriptedStepAgent{
		id: "identity",
		fn: func(input []float64) []float64 {
			if len(input) == 0 {
				return []float64{0}
			}
			return []float64{input[0]}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), identity, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), identity, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	if validationSamples, vok := validationTrace["samples"].(int); vok {
		if testSamples, tok := testTrace["samples"].(int); tok && testSamples == validationSamples {
			t.Fatalf("expected distinct regression mode sample windows, got validation=%d test=%d", validationSamples, testSamples)
		}
	}
}
