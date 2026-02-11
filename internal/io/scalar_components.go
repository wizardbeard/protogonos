package io

import (
	"context"
	"fmt"
	"sync"
)

const (
	ScalarInputSensorName      = "scalar_input"
	ScalarOutputActuatorName   = "scalar_output"
	XORInputLeftSensorName     = "xor_input_left"
	XORInputRightSensorName    = "xor_input_right"
	XOROutputActuatorName      = "xor_output"
	CartPolePositionSensorName = "cart_pole_position"
	CartPoleVelocitySensorName = "cart_pole_velocity"
	CartPoleForceActuatorName  = "cart_pole_force"
)

type ScalarInputSensor struct {
	mu    sync.RWMutex
	value float64
}

func NewScalarInputSensor(initial float64) *ScalarInputSensor {
	return &ScalarInputSensor{value: initial}
}

func (s *ScalarInputSensor) Name() string {
	return ScalarInputSensorName
}

func (s *ScalarInputSensor) Read(_ context.Context) ([]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return []float64{s.value}, nil
}

func (s *ScalarInputSensor) Set(value float64) {
	s.mu.Lock()
	s.value = value
	s.mu.Unlock()
}

type ScalarOutputActuator struct {
	mu   sync.RWMutex
	last []float64
}

func NewScalarOutputActuator() *ScalarOutputActuator {
	return &ScalarOutputActuator{}
}

func (a *ScalarOutputActuator) Name() string {
	return ScalarOutputActuatorName
}

func (a *ScalarOutputActuator) Write(_ context.Context, values []float64) error {
	a.mu.Lock()
	a.last = append([]float64(nil), values...)
	a.mu.Unlock()
	return nil
}

func (a *ScalarOutputActuator) Last() []float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return append([]float64(nil), a.last...)
}

func init() {
	initializeDefaultComponents()
}

func initializeDefaultComponents() {
	err := RegisterSensorWithSpec(SensorSpec{
		Name:          ScalarInputSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "regression-mimic" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          CartPolePositionSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "cart-pole-lite" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          CartPoleVelocitySensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "cart-pole-lite" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          XORInputLeftSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterSensorWithSpec(SensorSpec{
		Name:          XORInputRightSensorName,
		Factory:       func() Sensor { return NewScalarInputSensor(0) },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}

	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          ScalarOutputActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "regression-mimic" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          XOROutputActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "xor" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
	err = RegisterActuatorWithSpec(ActuatorSpec{
		Name:          CartPoleForceActuatorName,
		Factory:       func() Actuator { return NewScalarOutputActuator() },
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Compatible: func(scape string) error {
			if scape != "cart-pole-lite" {
				return fmt.Errorf("unsupported scape: %s", scape)
			}
			return nil
		},
	})
	if err != nil {
		panic(err)
	}
}
