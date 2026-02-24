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

	cortex, err := agent.NewCortex("xor-agent", genome, nil, nil, []string{"i1", "i2"}, []string{"o"}, nil)
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
	sse, ok := trace["sse"].(float64)
	if !ok {
		t.Fatalf("trace missing sse: %+v", trace)
	}
	if mse > 0.05 {
		t.Fatalf("expected mse <= 0.05, got %f", mse)
	}
	wantFitness := Fitness(1.0 / (sse + 0.000001))
	if diff := float64(fitness - wantFitness); diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("expected reciprocal-sse fitness %f, got %f (trace=%+v)", wantFitness, fitness, trace)
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

	cortex, err := agent.NewCortex("xor-agent-io", genome, sensors, actuators, []string{"i1", "i2"}, []string{"o"}, nil)
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
	sse, ok := trace["sse"].(float64)
	if !ok {
		t.Fatalf("trace missing sse: %+v", trace)
	}
	if mse > 0.05 {
		t.Fatalf("expected mse <= 0.05, got %f", mse)
	}
	wantFitness := Fitness(1.0 / (sse + 0.000001))
	if diff := float64(fitness - wantFitness); diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("expected reciprocal-sse fitness %f, got %f (trace=%+v)", wantFitness, fitness, trace)
	}
}

func TestXORScapeEvaluateModeAnnotatesMode(t *testing.T) {
	xor := XORScape{}
	parity := scriptedStepAgent{
		id: "xor-parity",
		fn: func(input []float64) []float64 {
			if len(input) < 2 {
				return []float64{0}
			}
			if int(input[0])^int(input[1]) == 1 {
				return []float64{1}
			}
			return []float64{0}
		},
	}

	_, validationTrace, err := xor.EvaluateMode(context.Background(), parity, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := xor.EvaluateMode(context.Background(), parity, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	if validationCases, vok := validationTrace["cases"].(int); vok {
		if testCases, tok := testTrace["cases"].(int); tok && testCases == validationCases {
			t.Fatalf("expected distinct xor mode case windows, got validation=%d test=%d", validationCases, testCases)
		}
	}
}
