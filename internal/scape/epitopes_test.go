package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestEpitopesScapeRewardsMemoryAwarePolicy(t *testing.T) {
	scape := EpitopesScape{}
	signalOnly := scriptedStepAgent{
		id: "signal-only",
		fn: func(in []float64) []float64 {
			if len(in) == 0 {
				return []float64{0}
			}
			return []float64{in[0]}
		},
	}
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	signalFitness, _, err := scape.Evaluate(context.Background(), signalOnly)
	if err != nil {
		t.Fatalf("evaluate signal-only: %v", err)
	}
	memoryFitness, _, err := scape.Evaluate(context.Background(), memoryAware)
	if err != nil {
		t.Fatalf("evaluate memory-aware: %v", err)
	}
	if memoryFitness <= signalFitness {
		t.Fatalf("expected memory-aware policy to outperform signal-only, got memory=%f signal=%f", memoryFitness, signalFitness)
	}
}

func TestEpitopesScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.EpitopesSignalSensorName,
			protoio.EpitopesMemorySensorName,
		},
		ActuatorIDs: []string{protoio.EpitopesResponseActuatorName},
		Neurons: []model.Neuron{
			{ID: "s", Activation: "identity"},
			{ID: "m", Activation: "identity"},
			{ID: "r", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "s", To: "r", Weight: 1.0, Enabled: true},
			{From: "m", To: "r", Weight: 0.7, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.EpitopesSignalSensorName: protoio.NewScalarInputSensor(0),
		protoio.EpitopesMemorySensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.EpitopesResponseActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("epitopes-agent-io", genome, sensors, actuators, []string{"s", "m"}, []string{"r"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := EpitopesScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness < 0.75 {
		t.Fatalf("expected fitness >= 0.75, got %f", fitness)
	}
	if _, ok := trace["accuracy"].(float64); !ok {
		t.Fatalf("trace missing accuracy: %+v", trace)
	}
}

func TestEpitopesScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := EpitopesScape{}
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	_, trace, err := scape.EvaluateMode(context.Background(), memoryAware, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := trace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", trace)
	}
	if startIndex, ok := trace["start_index"].(int); !ok || startIndex <= 0 {
		t.Fatalf("expected positive start_index in validation mode, got %+v", trace)
	}
}

func TestEpitopesScapeStepPerceptIncludesSequenceFeatures(t *testing.T) {
	scape := EpitopesScape{}
	maxPerceptWidth := 0
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) > maxPerceptWidth {
				maxPerceptWidth = len(in)
			}
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), memoryAware)
	if err != nil {
		t.Fatalf("evaluate memory-aware: %v", err)
	}
	if maxPerceptWidth <= 2 {
		t.Fatalf("expected step percept to include sequence features, got width=%d", maxPerceptWidth)
	}
	if width, ok := trace["feature_width"].(int); !ok || width <= 2 {
		t.Fatalf("expected feature_width > 2 in trace, got %+v", trace)
	}
	if seqLen, ok := trace["sequence_length"].(int); !ok || seqLen <= 0 {
		t.Fatalf("expected positive sequence_length in trace, got %+v", trace)
	}
}
