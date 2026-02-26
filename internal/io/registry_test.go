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

func TestSensorCompatibilityHelpers(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if !SensorCompatibleWithScape(Pole2CartPositionSensorName, "pb_sim") {
		t.Fatal("expected pb_sim alias to be compatible with pole2 cart position sensor")
	}
	if SensorCompatibleWithScape(Pole2CartPositionSensorName, "xor") {
		t.Fatal("expected pole2 cart position sensor to be incompatible with xor")
	}
	if SensorCompatibleWithScape("unknown_sensor", "xor") {
		t.Fatal("expected unknown sensor to be incompatible")
	}

	if err := RegisterSensorWithSpec(SensorSpec{
		Name:          "custom_xor_sensor",
		Factory:       func() Sensor { return testSensor{} },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return errors.New("not xor")
			}
			return nil
		},
	}); err != nil {
		t.Fatalf("register custom sensor: %v", err)
	}

	xorSensors := ListSensorsForScape("xor")
	if !containsString(xorSensors, "custom_xor_sensor") {
		t.Fatalf("expected custom sensor in xor filtered list, got=%v", xorSensors)
	}
	dtmSensors := ListSensorsForScape("dtm")
	if containsString(dtmSensors, "custom_xor_sensor") {
		t.Fatalf("expected custom sensor excluded from dtm filtered list, got=%v", dtmSensors)
	}
}

func TestActuatorCompatibilityHelpers(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if !ActuatorCompatibleWithScape(FlatlandTwoWheelsActuatorName, "scape_flatland") {
		t.Fatal("expected scape_flatland alias to be compatible with flatland actuator")
	}
	if ActuatorCompatibleWithScape(FlatlandTwoWheelsActuatorName, "xor") {
		t.Fatal("expected flatland actuator to be incompatible with xor")
	}
	if ActuatorCompatibleWithScape("unknown_actuator", "xor") {
		t.Fatal("expected unknown actuator to be incompatible")
	}

	if err := RegisterActuatorWithSpec(ActuatorSpec{
		Name:          "custom_xor_actuator",
		Factory:       func() Actuator { return testActuator{} },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return errors.New("not xor")
			}
			return nil
		},
	}); err != nil {
		t.Fatalf("register custom actuator: %v", err)
	}

	xorActuators := ListActuatorsForScape("xor")
	if !containsString(xorActuators, "custom_xor_actuator") {
		t.Fatalf("expected custom actuator in xor filtered list, got=%v", xorActuators)
	}
	dtmActuators := ListActuatorsForScape("dtm")
	if containsString(dtmActuators, "custom_xor_actuator") {
		t.Fatalf("expected custom actuator excluded from dtm filtered list, got=%v", dtmActuators)
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
