package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

type scriptedStepAgent struct {
	id string
	fn func(input []float64) []float64
}

func (a scriptedStepAgent) ID() string { return a.id }

func (a scriptedStepAgent) RunStep(_ context.Context, input []float64) ([]float64, error) {
	return a.fn(input), nil
}

func TestFlatlandScapeForagingCollectsResources(t *testing.T) {
	scape := FlatlandScape{}
	stationary := scriptedStepAgent{
		id: "stationary",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	forager := scriptedStepAgent{
		id: "forager",
		fn: func(input []float64) []float64 {
			if len(input) == 0 {
				return []float64{0}
			}
			if input[0] > 0 {
				return []float64{1}
			}
			if input[0] < 0 {
				return []float64{-1}
			}
			return []float64{0}
		},
	}

	stationaryFitness, stationaryTrace, err := scape.Evaluate(context.Background(), stationary)
	if err != nil {
		t.Fatalf("evaluate stationary: %v", err)
	}
	foragerFitness, foragerTrace, err := scape.Evaluate(context.Background(), forager)
	if err != nil {
		t.Fatalf("evaluate forager: %v", err)
	}
	if foragerFitness <= 0 || stationaryFitness <= 0 {
		t.Fatalf("expected positive fitness signals, got forager=%f stationary=%f", foragerFitness, stationaryFitness)
	}
	stationaryFood, ok := stationaryTrace["food_collected"].(int)
	if !ok {
		t.Fatalf("stationary trace missing food_collected: %+v", stationaryTrace)
	}
	foragerFood, ok := foragerTrace["food_collected"].(int)
	if !ok {
		t.Fatalf("forager trace missing food_collected: %+v", foragerTrace)
	}
	if foragerFood <= stationaryFood {
		t.Fatalf(
			"expected forager to collect more food than stationary, got forager=%d stationary=%d forager_trace=%+v stationary_trace=%+v",
			foragerFood,
			stationaryFood,
			foragerTrace,
			stationaryTrace,
		)
	}
}

func TestFlatlandScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FlatlandDistanceSensorName,
			protoio.FlatlandEnergySensorName,
		},
		ActuatorIDs: []string{protoio.FlatlandMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "distance", Activation: "identity"},
			{ID: "energy", Activation: "identity"},
			{ID: "move", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "distance", To: "move", Weight: 1, Enabled: true},
			{From: "energy", To: "move", Weight: 0.2, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FlatlandDistanceSensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandEnergySensorName:   protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FlatlandMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"flatland-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"distance", "energy"},
		[]string{"move"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["energy"].(float64); !ok {
		t.Fatalf("trace missing energy: %+v", trace)
	}
}

func TestFlatlandScapeTraceCapturesMetabolicsAndCollisions(t *testing.T) {
	scape := FlatlandScape{}
	forager := scriptedStepAgent{
		id: "forager",
		fn: func(input []float64) []float64 {
			if len(input) < 2 {
				return []float64{0}
			}
			if input[0] > 0 {
				return []float64{1}
			}
			if input[0] < 0 {
				return []float64{-1}
			}
			return []float64{0}
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), forager)
	if err != nil {
		t.Fatalf("evaluate forager: %v", err)
	}
	if _, ok := trace["age"].(int); !ok {
		t.Fatalf("trace missing age: %+v", trace)
	}
	if _, ok := trace["food_collected"].(int); !ok {
		t.Fatalf("trace missing food_collected: %+v", trace)
	}
	if _, ok := trace["poison_hits"].(int); !ok {
		t.Fatalf("trace missing poison_hits: %+v", trace)
	}
	if reason, ok := trace["terminal_reason"].(string); !ok || reason == "" {
		t.Fatalf("trace missing terminal_reason: %+v", trace)
	}
}

func TestFlatlandScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := FlatlandScape{}
	forager := scriptedStepAgent{
		id: "forager",
		fn: func(input []float64) []float64 {
			if len(input) == 0 {
				return []float64{0}
			}
			if input[0] > 0 {
				return []float64{1}
			}
			if input[0] < 0 {
				return []float64{-1}
			}
			return []float64{0}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), forager, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), forager, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
}
