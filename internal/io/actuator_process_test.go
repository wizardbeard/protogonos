package io

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type captureActuator struct {
	values []float64
	err    error
}

func (a *captureActuator) Name() string {
	return "capture"
}

func (a *captureActuator) Write(_ context.Context, values []float64) error {
	a.values = append([]float64(nil), values...)
	return a.err
}

type processAwareActuator struct {
	call ActuatorProcessCall
}

func (a *processAwareActuator) Name() string {
	return "process-aware"
}

func (a *processAwareActuator) Write(context.Context, []float64) error {
	return errors.New("direct write should not be used")
}

func (a *processAwareActuator) WriteForActuatorProcess(_ context.Context, call ActuatorProcessCall) (ActuatorSyncMessage, error) {
	a.call = call
	return ActuatorSyncMessage{Fitness: []float64{7}, EndFlag: 1, GoalReached: true}, nil
}

type feedbackActuator struct {
	captureActuator
	feedback ActuatorSyncMessage
	ok       bool
}

func (a *feedbackActuator) ConsumeActuatorFeedback() (ActuatorSyncMessage, bool) {
	if !a.ok {
		return ActuatorSyncMessage{}, false
	}
	a.ok = false
	return a.feedback, true
}

func TestActuatorProcessAccumulatesOrderedFanin(t *testing.T) {
	actuator := &captureActuator{}
	process, err := NewActuatorProcess("actuator_pid", "exo_pid", "cx_pid", actuator, 3, []string{"n1", "n2"})
	if err != nil {
		t.Fatalf("NewActuatorProcess: %v", err)
	}

	sync, err := process.ForwardFrom(context.Background(), "n1", []float64{1})
	if err != nil {
		t.Fatalf("ForwardFrom first: %v", err)
	}
	if sync != nil {
		t.Fatalf("first sync = %+v, want nil until all fanin arrives", sync)
	}
	sync, err = process.ForwardFrom(context.Background(), "n2", []float64{2, 3})
	if err != nil {
		t.Fatalf("ForwardFrom second: %v", err)
	}
	if sync == nil {
		t.Fatalf("sync nil, want completion sync")
	}
	if sync.FromPID != "actuator_pid" {
		t.Fatalf("sync FromPID = %q, want actuator_pid", sync.FromPID)
	}
	if !reflect.DeepEqual(actuator.values, []float64{1, 2, 3}) {
		t.Fatalf("actuator values = %v, want [1 2 3]", actuator.values)
	}
}

func TestActuatorProcessRejectsUnexpectedFaninOrder(t *testing.T) {
	process, err := NewActuatorProcess("actuator_pid", "exo_pid", "cx_pid", &captureActuator{}, 2, []string{"n1", "n2"})
	if err != nil {
		t.Fatalf("NewActuatorProcess: %v", err)
	}

	_, err = process.ForwardFrom(context.Background(), "n2", []float64{2})
	if !errors.Is(err, ErrUnexpectedActuatorForwardPID) {
		t.Fatalf("ForwardFrom err = %v, want ErrUnexpectedActuatorForwardPID", err)
	}
}

func TestActuatorProcessNormalizesOutputWidth(t *testing.T) {
	actuator := &captureActuator{}
	process, err := NewActuatorProcess("actuator_pid", "exo_pid", "cx_pid", actuator, 3, nil)
	if err != nil {
		t.Fatalf("NewActuatorProcess: %v", err)
	}
	if _, err := process.ForwardFrom(context.Background(), "n1", []float64{1}); err != nil {
		t.Fatalf("ForwardFrom short: %v", err)
	}
	if !reflect.DeepEqual(actuator.values, []float64{1, 0, 0}) {
		t.Fatalf("short values = %v, want padded output", actuator.values)
	}

	if _, err := process.ForwardFrom(context.Background(), "n1", []float64{1, 2, 3, 4}); err != nil {
		t.Fatalf("ForwardFrom long: %v", err)
	}
	if !reflect.DeepEqual(actuator.values, []float64{1, 2, 3}) {
		t.Fatalf("long values = %v, want truncated output", actuator.values)
	}
}

func TestActuatorProcessUsesProcessAwareWriter(t *testing.T) {
	actuator := &processAwareActuator{}
	process, err := NewActuatorProcessWithState(ActuatorInitMessage{
		ID:           "actuator_pid",
		FromPID:      "exo_pid",
		CxPID:        "cx_pid",
		Scape:        "xor",
		ActuatorName: "xor_SendOutput",
		VL:           2,
		Parameters:   map[string]float64{"gain": 0.25},
		OpMode:       "test",
		Actuator:     actuator,
	})
	if err != nil {
		t.Fatalf("NewActuatorProcessWithState: %v", err)
	}

	sync, err := process.ForwardFrom(context.Background(), "n1", []float64{0.5})
	if err != nil {
		t.Fatalf("ForwardFrom: %v", err)
	}
	if sync == nil || sync.FromPID != "actuator_pid" || !reflect.DeepEqual(sync.Fitness, []float64{7}) || sync.EndFlag != 1 || !sync.GoalReached {
		t.Fatalf("sync = %+v, want process-aware feedback", sync)
	}
	if actuator.call.ProcessID != "actuator_pid" || actuator.call.ExoSelfPID != "exo_pid" || actuator.call.CortexPID != "cx_pid" {
		t.Fatalf("unexpected process ids in call: %+v", actuator.call)
	}
	if actuator.call.Scape != "xor" || actuator.call.ActuatorName != "xor_SendOutput" || actuator.call.OpMode != "test" || actuator.call.VL != 2 {
		t.Fatalf("unexpected call metadata: %+v", actuator.call)
	}
	if !reflect.DeepEqual(actuator.call.Output, []float64{0.5, 0}) {
		t.Fatalf("call output = %v, want padded output [0.5 0]", actuator.call.Output)
	}
	if actuator.call.Parameters["gain"] != 0.25 {
		t.Fatalf("unexpected call parameters: %+v", actuator.call.Parameters)
	}
	actuator.call.Parameters["gain"] = 9
	if process.parameters["gain"] != 0.25 {
		t.Fatalf("process parameters were not isolated: %+v", process.parameters)
	}
}

func TestActuatorProcessCapturesFeedback(t *testing.T) {
	actuator := &feedbackActuator{
		feedback: ActuatorSyncMessage{Fitness: []float64{4.5}, EndFlag: 1, GoalReached: true},
		ok:       true,
	}
	process, err := NewActuatorProcess("actuator_pid", "exo_pid", "cx_pid", actuator, 1, nil)
	if err != nil {
		t.Fatalf("NewActuatorProcess: %v", err)
	}

	sync, err := process.ForwardFrom(context.Background(), "n1", []float64{0.5})
	if err != nil {
		t.Fatalf("ForwardFrom: %v", err)
	}
	if sync == nil || !reflect.DeepEqual(sync.Fitness, []float64{4.5}) || sync.EndFlag != 1 || !sync.GoalReached {
		t.Fatalf("sync = %+v, want feedback", sync)
	}
	consumed, ok := process.ConsumeActuatorFeedback()
	if !ok {
		t.Fatalf("ConsumeActuatorFeedback ok = false, want true")
	}
	if !reflect.DeepEqual(consumed.Fitness, []float64{4.5}) || consumed.EndFlag != 1 || !consumed.GoalReached {
		t.Fatalf("consumed = %+v, want feedback", consumed)
	}
	_, ok = process.ConsumeActuatorFeedback()
	if ok {
		t.Fatalf("second ConsumeActuatorFeedback ok = true, want false")
	}
}

func TestActuatorProcessTerminateRequiresExoSelfPID(t *testing.T) {
	process, err := NewActuatorProcess("actuator_pid", "exo_pid", "cx_pid", &captureActuator{}, 1, nil)
	if err != nil {
		t.Fatalf("NewActuatorProcess: %v", err)
	}
	if err := process.TerminateFrom("other"); !errors.Is(err, ErrUnexpectedActuatorTerminatePID) {
		t.Fatalf("TerminateFrom err = %v, want ErrUnexpectedActuatorTerminatePID", err)
	}
	if err := process.TerminateFrom("exo_pid"); err != nil {
		t.Fatalf("TerminateFrom owner: %v", err)
	}
	_, err = process.ForwardFrom(context.Background(), "n1", []float64{1})
	if !errors.Is(err, ErrActuatorProcessTerminated) {
		t.Fatalf("ForwardFrom after terminate err = %v, want ErrActuatorProcessTerminated", err)
	}
}

func TestActuatorActorInitForwardAndTerminate(t *testing.T) {
	actor, err := NewActuatorActorWithOwner("exo_pid", nil)
	if err != nil {
		t.Fatalf("NewActuatorActorWithOwner: %v", err)
	}
	defer func() {
		_ = actor.TerminateFrom("exo_pid")
	}()

	actuator := &captureActuator{}
	err = actor.InitFrom(context.Background(), ActuatorInitMessage{
		FromPID:   "exo_pid",
		ID:        "actuator_pid",
		CxPID:     "cx_pid",
		VL:        2,
		FaninPIDs: []string{"n1", "n2"},
		Actuator:  actuator,
	})
	if err != nil {
		t.Fatalf("InitFrom: %v", err)
	}
	sync, err := actor.ForwardFrom(context.Background(), "n1", []float64{0.25})
	if err != nil {
		t.Fatalf("ForwardFrom first: %v", err)
	}
	if sync != nil {
		t.Fatalf("first sync = %+v, want nil", sync)
	}
	sync, err = actor.ForwardFrom(context.Background(), "n2", []float64{0.75})
	if err != nil {
		t.Fatalf("ForwardFrom second: %v", err)
	}
	if sync == nil || sync.FromPID != "actuator_pid" {
		t.Fatalf("sync = %+v, want actuator sync", sync)
	}
	if !reflect.DeepEqual(actuator.values, []float64{0.25, 0.75}) {
		t.Fatalf("actuator values = %v, want [0.25 0.75]", actuator.values)
	}
	if err := actor.TerminateFrom("exo_pid"); err != nil {
		t.Fatalf("TerminateFrom: %v", err)
	}
	_, err = actor.ForwardFrom(context.Background(), "n1", []float64{1})
	if !errors.Is(err, ErrActuatorActorTerminated) {
		t.Fatalf("ForwardFrom after terminate err = %v, want ErrActuatorActorTerminated", err)
	}
}

func TestActuatorActorRejectsUnexpectedInitPID(t *testing.T) {
	actor, err := NewActuatorActorWithOwner("exo_pid", nil)
	if err != nil {
		t.Fatalf("NewActuatorActorWithOwner: %v", err)
	}
	defer func() {
		_ = actor.TerminateFrom("exo_pid")
	}()

	err = actor.InitFrom(context.Background(), ActuatorInitMessage{
		FromPID:  "other",
		ID:       "actuator_pid",
		CxPID:    "cx_pid",
		VL:       1,
		Actuator: &captureActuator{},
	})
	if !errors.Is(err, ErrUnexpectedActuatorInitPID) {
		t.Fatalf("InitFrom err = %v, want ErrUnexpectedActuatorInitPID", err)
	}
}
