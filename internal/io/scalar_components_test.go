package io

import (
	"context"
	"testing"
)

func TestScalarInputSensor(t *testing.T) {
	s := NewScalarInputSensor(0.25)
	values, err := s.Read(context.Background())
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(values) != 1 || values[0] != 0.25 {
		t.Fatalf("unexpected sensor values: %+v", values)
	}

	s.Set(0.75)
	values, err = s.Read(context.Background())
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if values[0] != 0.75 {
		t.Fatalf("unexpected updated value: %f", values[0])
	}
}

func TestScalarOutputActuator(t *testing.T) {
	a := NewScalarOutputActuator()
	if err := a.Write(context.Background(), []float64{0.9}); err != nil {
		t.Fatalf("write: %v", err)
	}
	last := a.Last()
	if len(last) != 1 || last[0] != 0.9 {
		t.Fatalf("unexpected actuator last output: %+v", last)
	}
}

func TestScalarComponentsRegistered(t *testing.T) {
	sensor, err := ResolveSensor(ScalarInputSensorName, "regression-mimic")
	if err != nil {
		t.Fatalf("resolve sensor: %v", err)
	}
	if sensor.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected sensor name: %s", sensor.Name())
	}

	actuator, err := ResolveActuator(ScalarOutputActuatorName, "regression-mimic")
	if err != nil {
		t.Fatalf("resolve actuator: %v", err)
	}
	if actuator.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected actuator name: %s", actuator.Name())
	}

	xorLeft, err := ResolveSensor(XORInputLeftSensorName, "xor")
	if err != nil {
		t.Fatalf("resolve xor left sensor: %v", err)
	}
	if xorLeft.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected xor left sensor name: %s", xorLeft.Name())
	}

	xorRight, err := ResolveSensor(XORInputRightSensorName, "xor")
	if err != nil {
		t.Fatalf("resolve xor right sensor: %v", err)
	}
	if xorRight.Name() != ScalarInputSensorName {
		t.Fatalf("unexpected xor right sensor name: %s", xorRight.Name())
	}

	xorActuator, err := ResolveActuator(XOROutputActuatorName, "xor")
	if err != nil {
		t.Fatalf("resolve xor actuator: %v", err)
	}
	if xorActuator.Name() != ScalarOutputActuatorName {
		t.Fatalf("unexpected xor actuator name: %s", xorActuator.Name())
	}
}
