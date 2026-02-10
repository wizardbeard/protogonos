package agent

import (
	"context"
	"testing"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
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

	c, err := NewCortex("agent-1", genome, sensors, actuators, []string{"i1", "i2"}, []string{"o"})
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
