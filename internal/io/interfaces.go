package io

import "context"

type Sensor interface {
	Name() string
	Read(ctx context.Context) ([]float64, error)
}

// ScalarSensorSetter is an optional sensor capability used by scapes that
// drive scalar inputs through concrete morphology components.
type ScalarSensorSetter interface {
	Set(value float64)
}

// VectorSensorSetter is an optional sensor capability used by scapes that
// provide variable-width feature vectors.
type VectorSensorSetter interface {
	Set(values []float64)
}

type Actuator interface {
	Name() string
	Write(ctx context.Context, values []float64) error
}

// SnapshotActuator is an optional actuator capability used by scapes that
// inspect the most recent actuator output.
type SnapshotActuator interface {
	Last() []float64
}
