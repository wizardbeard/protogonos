package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestGTSAScapeScoresBetterForSignalAwarePolicy(t *testing.T) {
	scape := GTSAScape{}
	zero := scriptedStepAgent{
		id: "zero",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	zeroFitness, _, err := scape.Evaluate(context.Background(), zero)
	if err != nil {
		t.Fatalf("evaluate zero: %v", err)
	}
	copyFitness, _, err := scape.Evaluate(context.Background(), copyInput)
	if err != nil {
		t.Fatalf("evaluate copy: %v", err)
	}
	if copyFitness <= zeroFitness {
		t.Fatalf("expected signal-aware policy to outperform zero, got copy=%f zero=%f", copyFitness, zeroFitness)
	}
}

func TestGTSAScapeEvaluateModeUsesConfiguredWindow(t *testing.T) {
	scape := GTSAScape{}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	gtFitness, gtTrace, err := scape.EvaluateMode(context.Background(), copyInput, "gt")
	if err != nil {
		t.Fatalf("evaluate gt mode: %v", err)
	}
	validationFitness, validationTrace, err := scape.EvaluateMode(context.Background(), copyInput, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if gtMode, _ := gtTrace["mode"].(string); gtMode != "gt" {
		t.Fatalf("expected gt mode trace marker, got %+v", gtTrace)
	}
	if validationMode, _ := validationTrace["mode"].(string); validationMode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	if gtFitness == validationFitness {
		t.Fatalf("expected mode windows to produce distinct fitness values, got gt=%f validation=%f", gtFitness, validationFitness)
	}
}

func TestGTSAScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.GTSAInputSensorName},
		ActuatorIDs: []string{protoio.GTSAPredictActuatorName},
		Neurons: []model.Neuron{
			{ID: "input", Activation: "identity"},
			{ID: "predict", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "input", To: "predict", Weight: 1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.GTSAInputSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.GTSAPredictActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"gtsa-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"input"},
		[]string{"predict"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := GTSAScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["mse"].(float64); !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
}
