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

func TestCortexHebbianPlasticityStatefulWeights(t *testing.T) {
	genome := model.Genome{
		Plasticity: &model.PlasticityConfig{
			Rule: "hebbian",
			Rate: 0.1,
		},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0.0},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Weight: 1.0, Enabled: true},
		},
	}

	c, err := NewCortex("agent-plastic", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	out1, err := c.RunStep(context.Background(), []float64{2})
	if err != nil {
		t.Fatalf("run step 1: %v", err)
	}
	out2, err := c.RunStep(context.Background(), []float64{2})
	if err != nil {
		t.Fatalf("run step 2: %v", err)
	}
	if out2[0] <= out1[0] {
		t.Fatalf("expected second output to increase after hebbian plasticity: out1=%v out2=%v", out1, out2)
	}
}

func TestCortexTickSingleActuatorReceivesOutputVector(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1", "s2"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "o1", Activation: "identity"},
			{ID: "o2", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "o1", Weight: 1.0, Enabled: true},
			{From: "i2", To: "o2", Weight: 1.0, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.2}},
		"s2": testSensor{values: []float64{0.8}},
	}
	act := &testActuator{}
	actuators := map[string]protoio.Actuator{"a1": act}

	c, err := NewCortex("agent-vector", genome, sensors, actuators, []string{"i1", "i2"}, []string{"o1", "o2"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	_, err = c.Tick(context.Background())
	if err != nil {
		t.Fatalf("tick: %v", err)
	}

	if len(act.last) != 2 || act.last[0] != 0.2 || act.last[1] != 0.8 {
		t.Fatalf("unexpected actuator vector output: %+v", act.last)
	}
}

func TestCortexTickMultipleActuatorsReceiveEvenChunks(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1", "s2", "s3", "s4"},
		ActuatorIDs: []string{"a1", "a2"},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "i3", Activation: "identity"},
			{ID: "i4", Activation: "identity"},
			{ID: "o1", Activation: "identity"},
			{ID: "o2", Activation: "identity"},
			{ID: "o3", Activation: "identity"},
			{ID: "o4", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "o1", Weight: 1.0, Enabled: true},
			{From: "i2", To: "o2", Weight: 1.0, Enabled: true},
			{From: "i3", To: "o3", Weight: 1.0, Enabled: true},
			{From: "i4", To: "o4", Weight: 1.0, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.1}},
		"s2": testSensor{values: []float64{0.2}},
		"s3": testSensor{values: []float64{0.3}},
		"s4": testSensor{values: []float64{0.4}},
	}
	act1 := &testActuator{}
	act2 := &testActuator{}
	actuators := map[string]protoio.Actuator{
		"a1": act1,
		"a2": act2,
	}

	c, err := NewCortex("agent-chunks", genome, sensors, actuators, []string{"i1", "i2", "i3", "i4"}, []string{"o1", "o2", "o3", "o4"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	_, err = c.Tick(context.Background())
	if err != nil {
		t.Fatalf("tick: %v", err)
	}

	if len(act1.last) != 2 || act1.last[0] != 0.1 || act1.last[1] != 0.2 {
		t.Fatalf("unexpected actuator a1 chunk: %+v", act1.last)
	}
	if len(act2.last) != 2 || act2.last[0] != 0.3 || act2.last[1] != 0.4 {
		t.Fatalf("unexpected actuator a2 chunk: %+v", act2.last)
	}
}

func TestCortexTickRejectsUnevenActuatorOutputShape(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1", "s2", "s3"},
		ActuatorIDs: []string{"a1", "a2"},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "i3", Activation: "identity"},
			{ID: "o1", Activation: "identity"},
			{ID: "o2", Activation: "identity"},
			{ID: "o3", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "o1", Weight: 1.0, Enabled: true},
			{From: "i2", To: "o2", Weight: 1.0, Enabled: true},
			{From: "i3", To: "o3", Weight: 1.0, Enabled: true},
		},
	}
	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.1}},
		"s2": testSensor{values: []float64{0.2}},
		"s3": testSensor{values: []float64{0.3}},
	}
	actuators := map[string]protoio.Actuator{
		"a1": &testActuator{},
		"a2": &testActuator{},
	}

	c, err := NewCortex("agent-bad-shape", genome, sensors, actuators, []string{"i1", "i2", "i3"}, []string{"o1", "o2", "o3"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	if _, err := c.Tick(context.Background()); err == nil {
		t.Fatal("expected uneven actuator/output shape error")
	}
}
