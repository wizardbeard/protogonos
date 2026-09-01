package substrate

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type testSubstrateActorRuntime struct {
	inputs     [][]float64
	weights    []float64
	backup     []float64
	resetCount int
	terminated bool
}

func (r *testSubstrateActorRuntime) Step(_ context.Context, inputs []float64) ([]float64, error) {
	if r.terminated {
		return nil, ErrSubstrateRuntimeTerminated
	}
	r.inputs = append(r.inputs, append([]float64(nil), inputs...))
	if len(inputs) >= 3 {
		return []float64{inputs[0], inputs[1], inputs[2]}, nil
	}
	out := append([]float64(nil), inputs...)
	for len(out) < 3 {
		out = append(out, 0)
	}
	return out, nil
}

func (r *testSubstrateActorRuntime) Weights() []float64 {
	return append([]float64(nil), r.weights...)
}

func (r *testSubstrateActorRuntime) Backup() {
	r.backup = append([]float64(nil), r.weights...)
}

func (r *testSubstrateActorRuntime) Restore() error {
	if r.backup == nil {
		return ErrNoSubstrateBackup
	}
	r.weights = append([]float64(nil), r.backup...)
	r.terminated = false
	return nil
}

func (r *testSubstrateActorRuntime) Reset() {
	r.resetCount++
	r.weights = nil
	r.terminated = false
}

func (r *testSubstrateActorRuntime) Terminate() {
	r.terminated = true
}

type testSubstrateActuatorTarget struct {
	pid     string
	fromPID string
	outputs [][]float64
}

func (t *testSubstrateActuatorTarget) SubstrateActuatorPID() string {
	return t.pid
}

func (t *testSubstrateActuatorTarget) ForwardFromSubstrate(fromPID string, output []float64) error {
	t.fromPID = fromPID
	t.outputs = append(t.outputs, append([]float64(nil), output...))
	return nil
}

func TestSubstrateActorRequiresInit(t *testing.T) {
	actor := NewSubstrateActorWithOwner("substrate_test", "exoself_test")
	if _, err := actor.Forward(context.Background(), "s1", []float64{1}); !errors.Is(err, ErrSubstrateActorUninitialized) {
		t.Fatalf("expected ErrSubstrateActorUninitialized, got %v", err)
	}
	ready, err := actor.Control(context.Background(), "exoself_test", SubstrateControlTerminate)
	if err != nil {
		t.Fatalf("terminate uninitialized actor: %v", err)
	}
	if ready {
		t.Fatalf("terminate should not send ready")
	}
}

func TestSubstrateActorInitValidatesOwnerAndRuntime(t *testing.T) {
	actor := NewSubstrateActorWithOwner("substrate_test", "exoself_test")
	if err := actor.Init(context.Background(), "other_exoself", SubstrateActorInitState{
		Runtime:    &testSubstrateActorRuntime{},
		SensorPIDs: []string{"s1"},
	}); !errors.Is(err, ErrUnexpectedSubstrateInitPID) {
		t.Fatalf("expected ErrUnexpectedSubstrateInitPID, got %v", err)
	}
	if err := actor.Init(context.Background(), "exoself_test", SubstrateActorInitState{
		SensorPIDs: []string{"s1"},
	}); !errors.Is(err, ErrSubstrateActorInitRuntime) {
		t.Fatalf("expected ErrSubstrateActorInitRuntime, got %v", err)
	}
	_, _ = actor.Control(context.Background(), "exoself_test", SubstrateControlTerminate)
}

func TestSubstrateActorOrderedSensorFaninAndActuatorFanout(t *testing.T) {
	runtime := &testSubstrateActorRuntime{}
	actor := NewSubstrateActorWithOwner("substrate_test", "exoself_test")
	act1 := &testSubstrateActuatorTarget{pid: "a1"}
	act2 := &testSubstrateActuatorTarget{pid: "a2"}
	actor.RegisterActuatorTargets(act1, act2)
	if err := actor.Init(context.Background(), "exoself_test", SubstrateActorInitState{
		Runtime:      runtime,
		SensorPIDs:   []string{"s1", "s2"},
		ActuatorPIDs: []string{"a1", "a2"},
		ActuatorVLs:  []int{2, 1},
	}); err != nil {
		t.Fatalf("init actor: %v", err)
	}

	outputs, err := actor.Forward(context.Background(), "s2", []float64{2})
	if err != nil {
		t.Fatalf("forward out-of-order sensor: %v", err)
	}
	if outputs != nil {
		t.Fatalf("expected no outputs until ordered fan-in completes, got=%v", outputs)
	}
	outputs, err = actor.Forward(context.Background(), "s1", []float64{1})
	if err != nil {
		t.Fatalf("forward expected sensor: %v", err)
	}
	if !reflect.DeepEqual(outputs, []float64{2, 1, 0}) {
		t.Fatalf("unexpected actor outputs: %v", outputs)
	}
	if !reflect.DeepEqual(runtime.inputs, [][]float64{{2, 1}}) {
		t.Fatalf("expected reference SAcc order into runtime, got=%v", runtime.inputs)
	}
	if act1.fromPID != "substrate_test" || !reflect.DeepEqual(act1.outputs, [][]float64{{2, 1}}) {
		t.Fatalf("unexpected first actuator fanout: from=%q outputs=%v", act1.fromPID, act1.outputs)
	}
	if act2.fromPID != "substrate_test" || !reflect.DeepEqual(act2.outputs, [][]float64{{0}}) {
		t.Fatalf("unexpected second actuator fanout: from=%q outputs=%v", act2.fromPID, act2.outputs)
	}
	_, _ = actor.Control(context.Background(), "exoself_test", SubstrateControlTerminate)
}

func TestSubstrateActorControlMessagesReturnReadyAndManageRuntime(t *testing.T) {
	runtime := &testSubstrateActorRuntime{weights: []float64{1, 2}}
	actor := NewSubstrateActorWithOwner("substrate_test", "exoself_test")
	if err := actor.Init(context.Background(), "exoself_test", SubstrateActorInitState{
		Runtime:    runtime,
		SensorPIDs: []string{"s1"},
	}); err != nil {
		t.Fatalf("init actor: %v", err)
	}

	ready, err := actor.Control(context.Background(), "exoself_test", SubstrateControlBackup)
	if err != nil || !ready {
		t.Fatalf("backup control: ready=%v err=%v", ready, err)
	}
	runtime.weights[0] = 9
	ready, err = actor.Control(context.Background(), "exoself_test", SubstrateControlRevert)
	if err != nil || !ready {
		t.Fatalf("revert control: ready=%v err=%v", ready, err)
	}
	if !reflect.DeepEqual(runtime.weights, []float64{1, 2}) {
		t.Fatalf("expected restored runtime weights, got=%v", runtime.weights)
	}
	ready, err = actor.Control(context.Background(), "exoself_test", SubstrateControlReset)
	if err != nil || !ready {
		t.Fatalf("reset control: ready=%v err=%v", ready, err)
	}
	if runtime.resetCount != 1 {
		t.Fatalf("expected runtime reset, got=%d", runtime.resetCount)
	}
	if _, err := actor.Control(context.Background(), "other_exoself", SubstrateControlReset); !errors.Is(err, ErrUnexpectedSubstrateControlPID) {
		t.Fatalf("expected ErrUnexpectedSubstrateControlPID, got %v", err)
	}
	ready, err = actor.Control(context.Background(), "exoself_test", SubstrateControlTerminate)
	if err != nil || ready {
		t.Fatalf("terminate control: ready=%v err=%v", ready, err)
	}
	if !runtime.terminated {
		t.Fatalf("expected runtime terminate")
	}
}

func TestSubstrateActorFlushDropsBufferedSensorMessages(t *testing.T) {
	runtime := &testSubstrateActorRuntime{}
	actor := NewSubstrateActorWithOwner("substrate_test", "exoself_test")
	if err := actor.Init(context.Background(), "exoself_test", SubstrateActorInitState{
		Runtime:    runtime,
		SensorPIDs: []string{"s1", "s2"},
	}); err != nil {
		t.Fatalf("init actor: %v", err)
	}
	if _, err := actor.Forward(context.Background(), "s2", []float64{2}); err != nil {
		t.Fatalf("forward pending sensor: %v", err)
	}
	ready, err := actor.Control(context.Background(), "exoself_test", SubstrateControlFlush)
	if err != nil || !ready {
		t.Fatalf("flush control: ready=%v err=%v", ready, err)
	}
	if _, err := actor.Forward(context.Background(), "s1", []float64{1}); err != nil {
		t.Fatalf("forward after flush: %v", err)
	}
	if len(runtime.inputs) != 0 {
		t.Fatalf("expected flushed pending message to prevent cycle, got=%v", runtime.inputs)
	}
	if _, err := actor.Forward(context.Background(), "s2", []float64{3}); err != nil {
		t.Fatalf("forward completing cycle after flush: %v", err)
	}
	if !reflect.DeepEqual(runtime.inputs, [][]float64{{3, 1}}) {
		t.Fatalf("unexpected inputs after flush cycle: %v", runtime.inputs)
	}
	_, _ = actor.Control(context.Background(), "exoself_test", SubstrateControlTerminate)
}

func TestSubstrateActorReportsMissingActuatorTarget(t *testing.T) {
	actor := NewSubstrateActorWithOwner("substrate_test", "exoself_test")
	if err := actor.Init(context.Background(), "exoself_test", SubstrateActorInitState{
		Runtime:      &testSubstrateActorRuntime{},
		SensorPIDs:   []string{"s1"},
		ActuatorPIDs: []string{"missing"},
		ActuatorVLs:  []int{1},
	}); err != nil {
		t.Fatalf("init actor: %v", err)
	}
	if _, err := actor.Forward(context.Background(), "s1", []float64{1}); !errors.Is(err, ErrMissingSubstrateActuatorTarget) {
		t.Fatalf("expected ErrMissingSubstrateActuatorTarget, got %v", err)
	}
	_, _ = actor.Control(context.Background(), "exoself_test", SubstrateControlTerminate)
}
