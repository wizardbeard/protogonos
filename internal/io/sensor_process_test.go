package io

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type scriptedSensor struct {
	values []float64
	err    error
}

func (s scriptedSensor) Name() string {
	return "scripted"
}

func (s scriptedSensor) Read(context.Context) ([]float64, error) {
	return append([]float64(nil), s.values...), s.err
}

type captureSensorFanout struct {
	pid     string
	fromPID string
	values  []float64
}

func (c *captureSensorFanout) SensorFanoutPID() string {
	return c.pid
}

func (c *captureSensorFanout) ForwardFromSensor(fromPID string, values []float64) error {
	c.fromPID = fromPID
	c.values = append([]float64(nil), values...)
	return nil
}

func TestSensorProcessSyncForwardsVector(t *testing.T) {
	process, err := NewSensorProcess("sensor_pid", "exo_pid", "cx_pid", scriptedSensor{values: []float64{0.25, 0.75}}, 2, []string{"n1"})
	if err != nil {
		t.Fatalf("NewSensorProcess: %v", err)
	}
	target := &captureSensorFanout{pid: "n1"}
	if err := process.AddFanoutTarget(target); err != nil {
		t.Fatalf("AddFanoutTarget: %v", err)
	}

	values, err := process.SyncFrom(context.Background(), "cx_pid")
	if err != nil {
		t.Fatalf("SyncFrom: %v", err)
	}
	if !reflect.DeepEqual(values, []float64{0.25, 0.75}) {
		t.Fatalf("values = %v, want [0.25 0.75]", values)
	}
	if target.fromPID != "sensor_pid" {
		t.Fatalf("fanout fromPID = %q, want sensor_pid", target.fromPID)
	}
	if !reflect.DeepEqual(target.values, []float64{0.25, 0.75}) {
		t.Fatalf("fanout values = %v, want [0.25 0.75]", target.values)
	}
}

func TestSensorProcessWrongVectorLengthFallsBackToZeros(t *testing.T) {
	process, err := NewSensorProcess("sensor_pid", "exo_pid", "cx_pid", scriptedSensor{values: []float64{1, 2, 3}}, 2, nil)
	if err != nil {
		t.Fatalf("NewSensorProcess: %v", err)
	}

	values, err := process.SyncFrom(context.Background(), "cx_pid")
	if err != nil {
		t.Fatalf("SyncFrom: %v", err)
	}
	if !reflect.DeepEqual(values, []float64{0, 0}) {
		t.Fatalf("values = %v, want zero fallback", values)
	}
}

func TestSensorProcessRejectsUnexpectedSyncPID(t *testing.T) {
	process, err := NewSensorProcess("sensor_pid", "exo_pid", "cx_pid", scriptedSensor{values: []float64{1}}, 1, nil)
	if err != nil {
		t.Fatalf("NewSensorProcess: %v", err)
	}

	_, err = process.SyncFrom(context.Background(), "other")
	if !errors.Is(err, ErrUnexpectedSensorSyncPID) {
		t.Fatalf("SyncFrom err = %v, want ErrUnexpectedSensorSyncPID", err)
	}
}

func TestSensorProcessTerminateRequiresExoSelfPID(t *testing.T) {
	process, err := NewSensorProcess("sensor_pid", "exo_pid", "cx_pid", scriptedSensor{values: []float64{1}}, 1, nil)
	if err != nil {
		t.Fatalf("NewSensorProcess: %v", err)
	}
	if err := process.TerminateFrom("other"); !errors.Is(err, ErrUnexpectedSensorTerminatePID) {
		t.Fatalf("TerminateFrom err = %v, want ErrUnexpectedSensorTerminatePID", err)
	}
	if err := process.TerminateFrom("exo_pid"); err != nil {
		t.Fatalf("TerminateFrom owner: %v", err)
	}
	_, err = process.SyncFrom(context.Background(), "cx_pid")
	if !errors.Is(err, ErrSensorProcessTerminated) {
		t.Fatalf("SyncFrom after terminate err = %v, want ErrSensorProcessTerminated", err)
	}
}

func TestSensorActorInitSyncAndTerminate(t *testing.T) {
	actor, err := NewSensorActorWithOwner("exo_pid", nil)
	if err != nil {
		t.Fatalf("NewSensorActorWithOwner: %v", err)
	}
	defer func() {
		_ = actor.TerminateFrom("exo_pid")
	}()

	err = actor.InitFrom(context.Background(), SensorInitMessage{
		FromPID: "exo_pid",
		ID:      "sensor_pid",
		CxPID:   "cx_pid",
		VL:      2,
		Sensor:  scriptedSensor{values: []float64{0.5, -0.5}},
	})
	if err != nil {
		t.Fatalf("InitFrom: %v", err)
	}

	values, err := actor.SyncFrom(context.Background(), "cx_pid")
	if err != nil {
		t.Fatalf("SyncFrom: %v", err)
	}
	if !reflect.DeepEqual(values, []float64{0.5, -0.5}) {
		t.Fatalf("values = %v, want [0.5 -0.5]", values)
	}
	if err := actor.TerminateFrom("exo_pid"); err != nil {
		t.Fatalf("TerminateFrom: %v", err)
	}
	_, err = actor.SyncFrom(context.Background(), "cx_pid")
	if !errors.Is(err, ErrSensorActorTerminated) {
		t.Fatalf("SyncFrom after terminate err = %v, want ErrSensorActorTerminated", err)
	}
}

func TestSensorActorRejectsUnexpectedInitPID(t *testing.T) {
	actor, err := NewSensorActorWithOwner("exo_pid", nil)
	if err != nil {
		t.Fatalf("NewSensorActorWithOwner: %v", err)
	}
	defer func() {
		_ = actor.TerminateFrom("exo_pid")
	}()

	err = actor.InitFrom(context.Background(), SensorInitMessage{
		FromPID: "other",
		ID:      "sensor_pid",
		CxPID:   "cx_pid",
		VL:      1,
		Sensor:  scriptedSensor{values: []float64{1}},
	})
	if !errors.Is(err, ErrUnexpectedSensorInitPID) {
		t.Fatalf("InitFrom err = %v, want ErrUnexpectedSensorInitPID", err)
	}
}
