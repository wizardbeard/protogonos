package scape

import (
	"context"
	"fmt"
	"math"

	protoio "protogonos/internal/io"
)

type FlatlandScape struct{}

func (FlatlandScape) Name() string {
	return "flatland"
}

func (FlatlandScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateFlatlandWithTick(ctx, ticker)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateFlatlandWithStep(ctx, runner)
}

func evaluateFlatlandWithStep(ctx context.Context, runner StepAgent) (Fitness, Trace, error) {
	return evaluateFlatland(ctx, func(ctx context.Context, distance, energy float64) (float64, error) {
		out, err := runner.RunStep(ctx, []float64{distance, energy})
		if err != nil {
			return 0, err
		}
		if len(out) != 1 {
			return 0, fmt.Errorf("flatland requires one output, got %d", len(out))
		}
		return out[0], nil
	})
}

func evaluateFlatlandWithTick(ctx context.Context, ticker TickAgent) (Fitness, Trace, error) {
	distanceSetter, energySetter, moveOutput, err := flatlandIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateFlatland(ctx, func(ctx context.Context, distance, energy float64) (float64, error) {
		distanceSetter.Set(distance)
		energySetter.Set(energy)
		out, err := ticker.Tick(ctx)
		if err != nil {
			return 0, err
		}
		last := moveOutput.Last()
		if len(last) > 0 {
			return last[0], nil
		}
		if len(out) > 0 {
			return out[0], nil
		}
		return 0, nil
	})
}

func evaluateFlatland(
	ctx context.Context,
	chooseMove func(context.Context, float64, float64) (float64, error),
) (Fitness, Trace, error) {
	const steps = 24
	position := 0.0
	target := 1.0
	energy := 1.0
	reward := 0.0

	for i := 0; i < steps; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}
		distance := target - position
		move, err := chooseMove(ctx, distance, energy)
		if err != nil {
			return 0, nil, err
		}
		move = clamp(move, -1, 1)
		position += move * 0.08
		energy -= 0.02 + 0.03*math.Abs(move)
		if energy < 0 {
			energy = 0
		}
		reward += 1.0 - math.Abs(target-position)
		if energy <= 0 {
			break
		}
	}

	normalizedReward := reward / float64(steps)
	fitness := clamp(normalizedReward, -1, 1)*0.8 + energy*0.2
	if math.Abs(target-position) < 0.15 {
		fitness += 0.2
	}
	return Fitness(clamp(fitness, 0, 1.2)), Trace{
		"position": position,
		"energy":   energy,
		"reward":   normalizedReward,
	}, nil
}

func flatlandIO(agent TickAgent) (
	protoio.ScalarSensorSetter,
	protoio.ScalarSensorSetter,
	protoio.SnapshotActuator,
	error,
) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	distance, ok := typed.RegisteredSensor(protoio.FlatlandDistanceSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.FlatlandDistanceSensorName)
	}
	distanceSetter, ok := distance.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.FlatlandDistanceSensorName)
	}

	energy, ok := typed.RegisteredSensor(protoio.FlatlandEnergySensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.FlatlandEnergySensorName)
	}
	energySetter, ok := energy.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.FlatlandEnergySensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.FlatlandMoveActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.FlatlandMoveActuatorName)
	}
	moveOutput, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.FlatlandMoveActuatorName)
	}
	return distanceSetter, energySetter, moveOutput, nil
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
