package io

import (
	"context"
	"errors"
	"testing"
)

type testSensor struct{}

func (testSensor) Name() string                            { return "test-sensor" }
func (testSensor) Read(context.Context) ([]float64, error) { return []float64{1}, nil }

type testActuator struct{}

func (testActuator) Name() string                           { return "test-actuator" }
func (testActuator) Write(context.Context, []float64) error { return nil }

func TestRegisterAndResolveSensor(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterSensor("s", func() Sensor { return testSensor{} }); err != nil {
		t.Fatalf("register sensor: %v", err)
	}
	s, err := ResolveSensor("s", "xor")
	if err != nil {
		t.Fatalf("resolve sensor: %v", err)
	}
	if s.Name() != "test-sensor" {
		t.Fatalf("unexpected sensor: %s", s.Name())
	}
}

func TestRegisterAndResolveActuator(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterActuator("a", func() Actuator { return testActuator{} }); err != nil {
		t.Fatalf("register actuator: %v", err)
	}
	a, err := ResolveActuator("a", "xor")
	if err != nil {
		t.Fatalf("resolve actuator: %v", err)
	}
	if a.Name() != "test-actuator" {
		t.Fatalf("unexpected actuator: %s", a.Name())
	}
}

func TestRegistryValidationAndDuplicates(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterSensor("", func() Sensor { return testSensor{} }); err == nil {
		t.Fatal("expected sensor name validation")
	}
	if err := RegisterActuator("", func() Actuator { return testActuator{} }); err == nil {
		t.Fatal("expected actuator name validation")
	}
	if err := RegisterSensor("dup", func() Sensor { return testSensor{} }); err != nil {
		t.Fatalf("register sensor: %v", err)
	}
	if err := RegisterSensor("dup", func() Sensor { return testSensor{} }); !errors.Is(err, ErrSensorExists) {
		t.Fatalf("expected ErrSensorExists, got: %v", err)
	}
	if err := RegisterActuator("dup", func() Actuator { return testActuator{} }); err != nil {
		t.Fatalf("register actuator: %v", err)
	}
	if err := RegisterActuator("dup", func() Actuator { return testActuator{} }); !errors.Is(err, ErrActuatorExists) {
		t.Fatalf("expected ErrActuatorExists, got: %v", err)
	}
}

func TestRegistryCompatibilityChecks(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	compatErr := errors.New("not allowed")
	if err := RegisterSensorWithSpec(SensorSpec{
		Name:          "restricted-s",
		Factory:       func() Sensor { return testSensor{} },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return compatErr
			}
			return nil
		},
	}); err != nil {
		t.Fatalf("register sensor with compatibility: %v", err)
	}

	if _, err := ResolveSensor("restricted-s", "cartpole"); !errors.Is(err, ErrIncompatible) {
		t.Fatalf("expected ErrIncompatible for sensor, got: %v", err)
	}

	if err := RegisterActuatorWithSpec(ActuatorSpec{
		Name:          "restricted-a",
		Factory:       func() Actuator { return testActuator{} },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return compatErr
			}
			return nil
		},
	}); err != nil {
		t.Fatalf("register actuator with compatibility: %v", err)
	}

	if _, err := ResolveActuator("restricted-a", "cartpole"); !errors.Is(err, ErrIncompatible) {
		t.Fatalf("expected ErrIncompatible for actuator, got: %v", err)
	}
}

func TestRegistryListsSorted(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterSensor("b", func() Sensor { return testSensor{} }); err != nil {
		t.Fatalf("register sensor b: %v", err)
	}
	if err := RegisterSensor("a", func() Sensor { return testSensor{} }); err != nil {
		t.Fatalf("register sensor a: %v", err)
	}
	if err := RegisterActuator("b", func() Actuator { return testActuator{} }); err != nil {
		t.Fatalf("register actuator b: %v", err)
	}
	if err := RegisterActuator("a", func() Actuator { return testActuator{} }); err != nil {
		t.Fatalf("register actuator a: %v", err)
	}

	sensors := ListSensors()
	if len(sensors) < 3 || sensors[0] != "a" || sensors[1] != "b" {
		t.Fatalf("unexpected sensor list: %+v", sensors)
	}

	actuators := ListActuators()
	if len(actuators) < 3 || actuators[0] != "a" || actuators[1] != "b" {
		t.Fatalf("unexpected actuator list: %+v", actuators)
	}
}
