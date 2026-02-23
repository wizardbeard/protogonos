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

func TestFlatlandScapeRewardsForwardMotion(t *testing.T) {
	scape := FlatlandScape{}
	stationary := scriptedStepAgent{
		id: "stationary",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	forward := scriptedStepAgent{
		id: "forward",
		fn: func(input []float64) []float64 {
			if len(input) == 0 {
				return []float64{0}
			}
			if input[0] > 0 {
				return []float64{1}
			}
			return []float64{-1}
		},
	}

	stationaryFitness, _, err := scape.Evaluate(context.Background(), stationary)
	if err != nil {
		t.Fatalf("evaluate stationary: %v", err)
	}
	forwardFitness, _, err := scape.Evaluate(context.Background(), forward)
	if err != nil {
		t.Fatalf("evaluate forward: %v", err)
	}
	if forwardFitness <= stationaryFitness {
		t.Fatalf("expected forward policy to outperform stationary, got forward=%f stationary=%f", forwardFitness, stationaryFitness)
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
