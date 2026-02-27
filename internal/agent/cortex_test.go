package agent

import (
	"context"
	"errors"
	"math"
	"math/rand"
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

type scriptedFeedbackActuator struct {
	last      []float64
	feedbacks []ActuatorSyncFeedback
}

func (a *scriptedFeedbackActuator) Name() string { return "scripted-feedback" }

func (a *scriptedFeedbackActuator) Write(_ context.Context, values []float64) error {
	a.last = append([]float64(nil), values...)
	return nil
}

func (a *scriptedFeedbackActuator) ConsumeSyncFeedback() (ActuatorSyncFeedback, bool) {
	if len(a.feedbacks) == 0 {
		return ActuatorSyncFeedback{}, false
	}
	feedback := a.feedbacks[0]
	a.feedbacks = a.feedbacks[1:]
	return feedback, true
}

type terminableRuntimeStub struct {
	terminated bool
}

func (s *terminableRuntimeStub) Step(_ context.Context, inputs []float64) ([]float64, error) {
	out := append([]float64(nil), inputs...)
	return out, nil
}

func (s *terminableRuntimeStub) Weights() []float64 {
	return nil
}

func (s *terminableRuntimeStub) Terminate() {
	s.terminated = true
}

type faninRuntimeStub struct {
	stepCalled     bool
	lastInputs     []float64
	lastFaninByPID map[string]float64
}

func (s *faninRuntimeStub) Step(_ context.Context, inputs []float64) ([]float64, error) {
	s.stepCalled = true
	out := append([]float64(nil), inputs...)
	return out, nil
}

func (s *faninRuntimeStub) StepWithFanin(_ context.Context, inputs []float64, faninSignals map[string]float64) ([]float64, error) {
	s.lastInputs = append([]float64(nil), inputs...)
	s.lastFaninByPID = make(map[string]float64, len(faninSignals))
	for pid, signal := range faninSignals {
		s.lastFaninByPID[pid] = signal
	}
	out := append([]float64(nil), inputs...)
	return out, nil
}

func (s *faninRuntimeStub) Weights() []float64 {
	return nil
}

func TestCortexRegisteredActuatorResolvesCanonicalAlias(t *testing.T) {
	genome := model.Genome{
		ActuatorIDs: []string{protoio.XORSendOutputActuatorAliasName},
	}
	aliasActuator := &testActuator{}
	c, err := NewCortex(
		"agent-actuator-alias",
		genome,
		nil,
		map[string]protoio.Actuator{
			protoio.XORSendOutputActuatorAliasName: aliasActuator,
		},
		[]string{"i"},
		[]string{"o"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	gotAlias, ok := c.RegisteredActuator(protoio.XORSendOutputActuatorAliasName)
	if !ok || gotAlias != aliasActuator {
		t.Fatalf("expected alias actuator lookup to succeed: ok=%t actuator=%v", ok, gotAlias)
	}
	gotCanonical, ok := c.RegisteredActuator(protoio.XOROutputActuatorName)
	if !ok || gotCanonical != aliasActuator {
		t.Fatalf("expected canonical actuator lookup to resolve alias registration: ok=%t actuator=%v", ok, gotCanonical)
	}
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

	want := 1.0
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
	if len(out1) != 1 || out1[0] != 1.0 {
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

func TestCortexSubstrateFaninRuntimeReceivesNamedSignals(t *testing.T) {
	genome := model.Genome{
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

	rt := &faninRuntimeStub{}
	c, err := NewCortex("agent-sub-fanin", genome, nil, nil, []string{"i1", "i2"}, []string{"o1", "o2"}, rt)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	out, err := c.RunStep(context.Background(), []float64{0.25, 0.75})
	if err != nil {
		t.Fatalf("run step: %v", err)
	}
	if len(out) != 2 || out[0] != 0.25 || out[1] != 0.75 {
		t.Fatalf("unexpected output vector: %v", out)
	}
	if rt.stepCalled {
		t.Fatal("expected fanin runtime path to bypass Step")
	}
	if len(rt.lastInputs) != 2 || rt.lastInputs[0] != 0.25 || rt.lastInputs[1] != 0.75 {
		t.Fatalf("unexpected substrate input forwarding: %v", rt.lastInputs)
	}
	if len(rt.lastFaninByPID) != 2 || rt.lastFaninByPID["o1"] != 0.25 || rt.lastFaninByPID["o2"] != 0.75 {
		t.Fatalf("unexpected fan-in signals: %+v", rt.lastFaninByPID)
	}
}

func TestCortexBackupRestoreWeightsIncludesSubstrateState(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
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

	c, err := NewCortex("agent-sub-backup", genome, nil, nil, []string{"i"}, []string{"o"}, rt)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	out1, err := c.RunStep(context.Background(), []float64{1.5})
	if err != nil {
		t.Fatalf("run step 1: %v", err)
	}
	c.BackupWeights()

	out2, err := c.RunStep(context.Background(), []float64{1.5})
	if err != nil {
		t.Fatalf("run step 2: %v", err)
	}
	if out2[0] <= out1[0] {
		t.Fatalf("expected accumulated substrate output, out1=%v out2=%v", out1, out2)
	}

	if err := c.RestoreWeights(); err != nil {
		t.Fatalf("restore weights: %v", err)
	}
	out3, err := c.RunStep(context.Background(), []float64{1.5})
	if err != nil {
		t.Fatalf("run step 3: %v", err)
	}
	if out3[0] != out2[0] {
		t.Fatalf("expected restored substrate state to reproduce output, out2=%v out3=%v", out2, out3)
	}
}

func TestCortexReactivateResetsSubstrateState(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
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

	c, err := NewCortex("agent-sub-reactivate", genome, nil, nil, []string{"i"}, []string{"o"}, rt)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	out1, err := c.RunStep(context.Background(), []float64{1.5})
	if err != nil {
		t.Fatalf("run step 1: %v", err)
	}
	out2, err := c.RunStep(context.Background(), []float64{1.5})
	if err != nil {
		t.Fatalf("run step 2: %v", err)
	}
	if out2[0] <= out1[0] {
		t.Fatalf("expected substrate accumulation before reactivation, out1=%v out2=%v", out1, out2)
	}

	if err := c.Reactivate(); err != nil {
		t.Fatalf("reactivate: %v", err)
	}
	out3, err := c.RunStep(context.Background(), []float64{1.5})
	if err != nil {
		t.Fatalf("run step 3: %v", err)
	}
	if out3[0] != out1[0] {
		t.Fatalf("expected substrate reset on reactivation, out1=%v out3=%v", out1, out3)
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

	out1, err := c.RunStep(context.Background(), []float64{0.4})
	if err != nil {
		t.Fatalf("run step 1: %v", err)
	}
	out2, err := c.RunStep(context.Background(), []float64{0.4})
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

func TestCortexTickAppliesActuatorTunablesBeforeWrite(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		ActuatorTunables: map[string]float64{
			"a1": 0.25,
		},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "o1", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "o1", Weight: 1.0, Enabled: true},
		},
	}
	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.5}},
	}
	act := &testActuator{}
	actuators := map[string]protoio.Actuator{"a1": act}

	c, err := NewCortex("agent-act-tunable", genome, sensors, actuators, []string{"i1"}, []string{"o1"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	out, err := c.Tick(context.Background())
	if err != nil {
		t.Fatalf("tick: %v", err)
	}
	if len(out) != 1 || out[0] != 0.5 {
		t.Fatalf("unexpected raw output vector: %v", out)
	}
	if len(act.last) != 1 || act.last[0] != 0.75 {
		t.Fatalf("expected actuator-local offset to be applied, got=%v", act.last)
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

func TestCortexTickRoutesSensorInputsByExplicitLinks(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1", "s2", "s3"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "i3", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "o", Weight: 1.0, Enabled: true},
			{From: "i2", To: "o", Weight: 1.0, Enabled: true},
			{From: "i3", To: "o", Weight: 1.0, Enabled: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: "s1", NeuronID: "i1"},
			{SensorID: "s2", NeuronID: "i2"},
			{SensorID: "s3", NeuronID: "i3"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "o", ActuatorID: "a1"},
		},
	}
	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.1}},
		"s2": testSensor{values: []float64{0.2}},
		"s3": testSensor{values: []float64{0.3}},
	}
	act := &testActuator{}
	actuators := map[string]protoio.Actuator{"a1": act}

	// Simulate monitor-built cortex input IDs from an older/scarcer seed surface.
	c, err := NewCortex("agent-linked-inputs", genome, sensors, actuators, []string{"i1", "i2"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	out, err := c.Tick(context.Background())
	if err != nil {
		t.Fatalf("tick: %v", err)
	}
	if len(out) != 1 || math.Abs(out[0]-0.6) > 1e-12 {
		t.Fatalf("unexpected output: got=%v want=[0.6]", out)
	}
	if len(act.last) != 1 || math.Abs(act.last[0]-0.6) > 1e-12 {
		t.Fatalf("unexpected actuator write: got=%v want=[0.6]", act.last)
	}
}

func TestCortexTickRoutesActuatorsByExplicitLinks(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1", "a2"},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "o1", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "o1", Weight: 1.0, Enabled: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: "s1", NeuronID: "i1"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "o1", ActuatorID: "a1"},
			{NeuronID: "o1", ActuatorID: "a2"},
		},
	}
	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.4}},
	}
	act1 := &testActuator{}
	act2 := &testActuator{}
	actuators := map[string]protoio.Actuator{
		"a1": act1,
		"a2": act2,
	}

	// Only one output neuron is available; explicit links should still route
	// both actuators without chunk-shape errors.
	c, err := NewCortex("agent-linked-outputs", genome, sensors, actuators, []string{"i1"}, []string{"o1"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	if _, err := c.Tick(context.Background()); err != nil {
		t.Fatalf("tick: %v", err)
	}
	if len(act1.last) != 1 || math.Abs(act1.last[0]-0.4) > 1e-12 {
		t.Fatalf("unexpected actuator a1 output: %v", act1.last)
	}
	if len(act2.last) != 1 || math.Abs(act2.last[0]-0.4) > 1e-12 {
		t.Fatalf("unexpected actuator a2 output: %v", act2.last)
	}
}

func TestCortexDiffProductUsesStepInputDeltas(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "o", Activation: "identity", Aggregator: "diff_product"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i1", To: "o", Weight: 1.0, Enabled: true},
			{ID: "s2", From: "i2", To: "o", Weight: 1.0, Enabled: true},
		},
	}

	c, err := NewCortex("agent-diff", genome, nil, nil, []string{"i1", "i2"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	out1, err := c.RunStep(context.Background(), []float64{0.6, 0.2})
	if err != nil {
		t.Fatalf("run step 1: %v", err)
	}
	if len(out1) != 1 || math.Abs(out1[0]-0.8) > 1e-9 {
		t.Fatalf("unexpected output 1: %+v", out1)
	}

	out2, err := c.RunStep(context.Background(), []float64{0.7, 0.4})
	if err != nil {
		t.Fatalf("run step 2: %v", err)
	}
	if len(out2) != 1 || math.Abs(out2[0]-0.3) > 1e-9 {
		t.Fatalf("unexpected output 2: %+v", out2)
	}
}

func TestCortexRunUntilEvaluationCompleteAggregatesActuatorFeedback(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}
	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.5}},
	}
	actuator := &scriptedFeedbackActuator{
		feedbacks: []ActuatorSyncFeedback{
			{Fitness: []float64{1.0, 2.0}},
			{Fitness: []float64{3.0, 4.0}, EndFlag: 1},
		},
	}
	actuators := map[string]protoio.Actuator{"a1": actuator}

	c, err := NewCortex("agent-episode", genome, sensors, actuators, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	report, err := c.RunUntilEvaluationComplete(context.Background(), 10)
	if err != nil {
		t.Fatalf("run episode: %v", err)
	}
	if !report.Completed {
		t.Fatal("expected episode to complete on actuator end-flag")
	}
	if report.Cycles != 2 {
		t.Fatalf("expected 2 cycles before completion, got=%d", report.Cycles)
	}
	if len(report.Fitness) != 2 || report.Fitness[0] != 4.0 || report.Fitness[1] != 6.0 {
		t.Fatalf("unexpected aggregated fitness vector: %+v", report.Fitness)
	}
	if c.Status() != CortexStatusInactive {
		t.Fatalf("expected cortex inactive after completed episode, got=%s", c.Status())
	}
}

func TestCortexRunUntilEvaluationCompleteRequiresReactivateAfterCompletion(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}
	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.5}},
	}
	actuator := &scriptedFeedbackActuator{
		feedbacks: []ActuatorSyncFeedback{
			{Fitness: []float64{1.0}, EndFlag: 1},
			{Fitness: []float64{2.0}, EndFlag: 1},
		},
	}
	actuators := map[string]protoio.Actuator{"a1": actuator}

	c, err := NewCortex("agent-reactivate", genome, sensors, actuators, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	if _, err := c.RunUntilEvaluationComplete(context.Background(), 5); err != nil {
		t.Fatalf("first run episode: %v", err)
	}
	if _, err := c.RunUntilEvaluationComplete(context.Background(), 5); !errors.Is(err, ErrCortexInactive) {
		t.Fatalf("expected ErrCortexInactive before reactivation, got %v", err)
	}
	if err := c.Reactivate(); err != nil {
		t.Fatalf("reactivate: %v", err)
	}
	report, err := c.RunUntilEvaluationComplete(context.Background(), 5)
	if err != nil {
		t.Fatalf("second run episode: %v", err)
	}
	if !report.Completed || report.Cycles != 1 {
		t.Fatalf("expected one-cycle completion after reactivation, report=%+v", report)
	}
}

func TestCortexRunUntilEvaluationCompleteGoalReachedTerminatesEpisode(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}
	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.5}},
	}
	actuator := &scriptedFeedbackActuator{
		feedbacks: []ActuatorSyncFeedback{
			{Fitness: []float64{0.75}, GoalReached: true},
		},
	}
	actuators := map[string]protoio.Actuator{"a1": actuator}

	c, err := NewCortex("agent-goal", genome, sensors, actuators, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	report, err := c.RunUntilEvaluationComplete(context.Background(), 5)
	if err != nil {
		t.Fatalf("run episode: %v", err)
	}
	if !report.Completed || !report.GoalReached {
		t.Fatalf("expected goal-driven completion, report=%+v", report)
	}
	if report.EndFlagTotal <= 0 {
		t.Fatalf("expected positive end-flag total from goal-reached feedback, report=%+v", report)
	}
}

func TestCortexTerminateBlocksFurtherExecution(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{"s1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}
	sensors := map[string]protoio.Sensor{
		"s1": testSensor{values: []float64{0.5}},
	}
	c, err := NewCortex("agent-terminated", genome, sensors, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	c.Terminate()
	if _, err := c.Tick(context.Background()); !errors.Is(err, ErrCortexTerminated) {
		t.Fatalf("expected ErrCortexTerminated on Tick, got %v", err)
	}
	if err := c.Reactivate(); !errors.Is(err, ErrCortexTerminated) {
		t.Fatalf("expected ErrCortexTerminated on Reactivate, got %v", err)
	}
}

func TestCortexTerminateSignalsSubstrateRuntime(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}

	rt := &terminableRuntimeStub{}
	c, err := NewCortex("agent-terminated-substrate", genome, nil, nil, []string{"i"}, []string{"o"}, rt)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	c.Terminate()
	if !rt.terminated {
		t.Fatal("expected cortex terminate to propagate substrate terminate")
	}
}

func TestCortexBackupRestoreWeights(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 0.2, Enabled: true},
		},
	}
	c, err := NewCortex("agent-backup", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	c.BackupWeights()
	if err := c.PerturbWeights(rand.New(rand.NewSource(151)), 2.0); err != nil {
		t.Fatalf("perturb weights: %v", err)
	}
	if c.genome.Synapses[0].Weight == 0.2 {
		t.Fatal("expected perturbed synapse weight to differ from backup")
	}
	if err := c.RestoreWeights(); err != nil {
		t.Fatalf("restore weights: %v", err)
	}
	if c.genome.Synapses[0].Weight != 0.2 {
		t.Fatalf("expected restored synapse weight 0.2, got=%f", c.genome.Synapses[0].Weight)
	}
}

func TestCortexSnapshotGenomeReturnsClone(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 0.2, Enabled: true},
		},
	}
	c, err := NewCortex("agent-snapshot", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	snap := c.SnapshotGenome()
	snap.Synapses[0].Weight = 7.5
	if c.genome.Synapses[0].Weight == 7.5 {
		t.Fatal("expected snapshot mutation not to affect runtime genome")
	}
}

func TestCortexApplyGenomeReplacesRuntimeWeights(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 0.2, Enabled: true},
		},
	}
	c, err := NewCortex("agent-apply", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	applied := genome
	applied.Synapses = append([]model.Synapse(nil), genome.Synapses...)
	applied.Synapses[0].Weight = -3.7
	if err := c.ApplyGenome(applied); err != nil {
		t.Fatalf("apply genome: %v", err)
	}
	if got := c.SnapshotGenome().Synapses[0].Weight; got != -3.7 {
		t.Fatalf("expected applied synapse weight -3.7, got=%f", got)
	}
}

func TestCortexApplyGenomeTerminatedError(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 0.2, Enabled: true},
		},
	}
	c, err := NewCortex("agent-apply-term", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	c.Terminate()
	if err := c.ApplyGenome(genome); !errors.Is(err, ErrCortexTerminated) {
		t.Fatalf("expected ErrCortexTerminated, got %v", err)
	}
}

func TestCortexRestoreWeightsWithoutBackupErrors(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: 0.2, Enabled: true},
		},
	}
	c, err := NewCortex("agent-no-backup", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	if err := c.RestoreWeights(); !errors.Is(err, ErrNoWeightBackup) {
		t.Fatalf("expected ErrNoWeightBackup, got %v", err)
	}
}

func TestCortexPerturbWeightsSaturatesToReferenceLimit(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: math.Pi*10 - 0.01, Enabled: true},
		},
	}
	c, err := NewCortex("agent-sat", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	if err := c.PerturbWeights(rand.New(rand.NewSource(157)), 1000); err != nil {
		t.Fatalf("perturb weights: %v", err)
	}
	if c.genome.Synapses[0].Weight > math.Pi*10 || c.genome.Synapses[0].Weight < -math.Pi*10 {
		t.Fatalf("expected saturated synapse within +/-pi*10, got=%f", c.genome.Synapses[0].Weight)
	}
}

func TestCortexPerturbWeightsNoSynapses(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
		},
	}
	c, err := NewCortex("agent-nosyn", genome, nil, nil, []string{"i"}, []string{"i"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	if err := c.PerturbWeights(rand.New(rand.NewSource(163)), 1.0); !errors.Is(err, ErrNoSynapses) {
		t.Fatalf("expected ErrNoSynapses, got %v", err)
	}
}
