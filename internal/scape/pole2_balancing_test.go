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
}
