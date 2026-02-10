package scape

import (
	"context"
	"fmt"

	protoio "protogonos/internal/io"
)

// RegressionMimicScape evaluates a one-dimensional regression target y=x.
type RegressionMimicScape struct{}

func (RegressionMimicScape) Name() string {
	return "regression-mimic"
}

func (RegressionMimicScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateRegressionMimicWithTick(ctx, ticker)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	inputs := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	predictions := make([]float64, 0, len(inputs))

	var squaredErr float64
	for _, x := range inputs {
		out, err := runner.RunStep(ctx, []float64{x})
		if err != nil {
			return 0, nil, err
		}
		if len(out) != 1 {
			return 0, nil, fmt.Errorf("regression-mimic requires one output, got %d", len(out))
		}

		predictions = append(predictions, out[0])
		delta := out[0] - x
		squaredErr += delta * delta
	}

	mse := squaredErr / float64(len(inputs))
	fitness := Fitness(1.0 - mse)
	return fitness, Trace{"mse": mse, "predictions": predictions}, nil
}

func evaluateRegressionMimicWithTick(ctx context.Context, ticker TickAgent) (Fitness, Trace, error) {
	inputs := []float64{0.0, 0.25, 0.5, 0.75, 1.0}
	predictions := make([]float64, 0, len(inputs))

	var squaredErr float64
	for _, x := range inputs {
		setter, output, err := regressionMimicIO(ticker)
		if err != nil {
			return 0, nil, err
		}
		setter.Set(x)

		out, err := ticker.Tick(ctx)
		if err != nil {
			return 0, nil, err
		}

		predicted := 0.0
		if len(output.Last()) > 0 {
			predicted = output.Last()[0]
		} else if len(out) > 0 {
			predicted = out[0]
		} else {
			return 0, nil, fmt.Errorf("regression-mimic requires one output, got 0")
		}

		predictions = append(predictions, predicted)
		delta := predicted - x
		squaredErr += delta * delta
	}

	mse := squaredErr / float64(len(inputs))
	fitness := Fitness(1.0 - mse)
	return fitness, Trace{"mse": mse, "predictions": predictions}, nil
}

func regressionMimicIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	sensor, ok := typed.RegisteredSensor(protoio.ScalarInputSensorName)
	if !ok {
		return nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.ScalarInputSensorName)
	}
	setter, ok := sensor.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.ScalarInputSensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.ScalarOutputActuatorName)
	if !ok {
		return nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.ScalarOutputActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.ScalarOutputActuatorName)
	}
	return setter, output, nil
}
