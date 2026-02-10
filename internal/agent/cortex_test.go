package agent

import (
	"context"
	"testing"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/substrate"
)

type testSensor struct {
	values []float64
}

func (s testSensor) Name() string { return "sensor" }

func (s testSensor) Read(context.Context) ([]float64, error) {
	return s.values, nil
}

type testActuator struct {
	last []float64
}

func (a *testActuator) Name() string { return "actuator" }

func (a *testActuator) Write(_ context.Context, values []float64) error {
	a.last = append([]float64(nil), values...)
	return nil
}

func TestCortexTickSensorToActuator(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1", "s2"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0.2},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "o", Weight: 1.0, Enabled: true},
			{From: "i2", To: "o", Weight: 2.0, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.5}},
		"s2": testSensor{values: []float64{0.25}},
	}
	act := &testActuator{}
	actuators := map[string]protoio.Actuator{"a1": act}

	c, err := NewCortex("agent-1", genome, sensors, actuators, []string{"i1", "i2"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	out, err := c.Tick(context.Background())
	if err != nil {
		t.Fatalf("tick: %v", err)
	}

	want := 0.5 + (2.0 * 0.25) + 0.2
	if len(out) != 1 || out[0] != want {
		t.Fatalf("unexpected output: got=%v want=[%f]", out, want)
	}
	if len(act.last) != 1 || act.last[0] != want {
		t.Fatalf("unexpected actuator write: got=%v want=[%f]", act.last, want)
	}
}

func TestCortexSubstrateTransformsOutputs(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0.0},
		},
		Synapses: []model.Synapse{
			{From: "i", To: "o", Weight: 1.0, Enabled: true},
		},
	}

	rt, err := substrate.NewSimpleRuntime(substrate.Spec{
		CPPName: substrate.DefaultCPPName,
		CEPName: substrate.DefaultCEPName,
		Parameters: map[string]float64{
			"scale": 1.0,
		},
	}, 1)
	if err != nil {
		t.Fatalf("new substrate runtime: %v", err)
	}

	c, err := NewCortex("agent-sub", genome, nil, nil, []string{"i"}, []string{"o"}, rt)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	out1, err := c.RunStep(context.Background(), []float64{1.5})
	if err != nil {
		t.Fatalf("run step 1: %v", err)
	}
	if len(out1) != 1 || out1[0] != 1.5 {
		t.Fatalf("unexpected output 1: %+v", out1)
	}

	out2, err := c.RunStep(context.Background(), []float64{1.5})
	if err != nil {
		t.Fatalf("run step 2: %v", err)
	}
	// Substrate keeps state and applies delta each step: second output should be larger.
	if len(out2) != 1 || out2[0] <= out1[0] {
		t.Fatalf("expected substrate-transformed output to increase: out1=%v out2=%v", out1, out2)
	}
}
