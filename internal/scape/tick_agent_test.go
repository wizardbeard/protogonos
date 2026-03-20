package scape

import (
	"context"

	protoio "protogonos/internal/io"
)

type scriptedTickAgent struct {
	id        string
	sensors   map[string]protoio.Sensor
	actuators map[string]protoio.Actuator
	fn        func(context.Context, map[string]protoio.Sensor) ([]float64, error)
}

func (a scriptedTickAgent) ID() string { return a.id }

func (a scriptedTickAgent) Tick(ctx context.Context) ([]float64, error) {
	return a.fn(ctx, a.sensors)
}

func (a scriptedTickAgent) RegisteredSensor(id string) (protoio.Sensor, bool) {
	sensor, ok := a.sensors[id]
	return sensor, ok
}

func (a scriptedTickAgent) RegisteredActuator(id string) (protoio.Actuator, bool) {
	actuator, ok := a.actuators[id]
	return actuator, ok
}

type writeOnlyActuator struct {
	name string
	last []float64
}

func (a *writeOnlyActuator) Name() string {
	return a.name
}

func (a *writeOnlyActuator) Write(_ context.Context, values []float64) error {
	a.last = append(a.last[:0], values...)
	return nil
}
