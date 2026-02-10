package scape

import (
	"context"
	"fmt"

	protoio "protogonos/internal/io"
)

type StepAgent interface {
	Agent
	RunStep(ctx context.Context, input []float64) ([]float64, error)
}

type XORScape struct{}

func (XORScape) Name() string {
	return "xor"
}

func (XORScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateXORWithTick(ctx, ticker)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	cases := []struct {
		in   []float64
		want float64
	}{
		{in: []float64{0, 0}, want: 0},
		{in: []float64{0, 1}, want: 1},
		{in: []float64{1, 0}, want: 1},
		{in: []float64{1, 1}, want: 0},
	}

	var squaredErr float64
	predictions := make([]float64, 0, len(cases))
	for _, c := range cases {
		out, err := runner.RunStep(ctx, c.in)
		if err != nil {
			return 0, nil, err
		}
		if len(out) != 1 {
			return 0, nil, fmt.Errorf("xor requires one output, got %d", len(out))
		}
		predictions = append(predictions, out[0])
		delta := out[0] - c.want
		squaredErr += delta * delta
	}

	mse := squaredErr / float64(len(cases))
	fitness := Fitness(1.0 - mse)
	return fitness, Trace{"mse": mse, "predictions": predictions}, nil
}

func evaluateXORWithTick(ctx context.Context, ticker TickAgent) (Fitness, Trace, error) {
	cases := []struct {
		left  float64
		right float64
		want  float64
	}{
		{left: 0, right: 0, want: 0},
		{left: 0, right: 1, want: 1},
		{left: 1, right: 0, want: 1},
		{left: 1, right: 1, want: 0},
	}

	leftSetter, rightSetter, output, err := xorIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	var squaredErr float64
	predictions := make([]float64, 0, len(cases))
	for _, c := range cases {
		leftSetter.Set(c.left)
		rightSetter.Set(c.right)

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
			return 0, nil, fmt.Errorf("xor requires one output, got 0")
		}

		predictions = append(predictions, predicted)
		delta := predicted - c.want
		squaredErr += delta * delta
	}

	mse := squaredErr / float64(len(cases))
	fitness := Fitness(1.0 - mse)
	return fitness, Trace{"mse": mse, "predictions": predictions}, nil
}

func xorIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	left, ok := typed.RegisteredSensor(protoio.XORInputLeftSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.XORInputLeftSensorName)
	}
	leftSetter, ok := left.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.XORInputLeftSensorName)
	}

	right, ok := typed.RegisteredSensor(protoio.XORInputRightSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.XORInputRightSensorName)
	}
	rightSetter, ok := right.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.XORInputRightSensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.XOROutputActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.XOROutputActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.XOROutputActuatorName)
	}
	return leftSetter, rightSetter, output, nil
}
