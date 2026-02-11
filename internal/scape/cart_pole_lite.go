package scape

import (
	"context"
	"fmt"
	"math"

	protoio "protogonos/internal/io"
)

// CartPoleLiteScape is a simplified 1D balancing control task.
type CartPoleLiteScape struct{}

func (CartPoleLiteScape) Name() string {
	return "cart-pole-lite"
}

func (CartPoleLiteScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateCartPoleLiteWithTick(ctx, ticker)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateCartPoleLiteWithStep(ctx, runner)
}

func evaluateCartPoleLiteWithStep(ctx context.Context, runner StepAgent) (Fitness, Trace, error) {
	totalReward := 0.0
	stepsSurvived := 0

	for _, start := range []float64{-0.8, -0.4, 0.0, 0.4, 0.8} {
		x := start
		v := 0.0

		for step := 0; step < 60; step++ {
			if err := ctx.Err(); err != nil {
				return 0, nil, err
			}

			out, err := runner.RunStep(ctx, []float64{x, v})
			if err != nil {
				return 0, nil, err
			}
			if len(out) != 1 {
				return 0, nil, fmt.Errorf("cart-pole-lite requires one output, got %d", len(out))
			}
			var reward float64
			x, v, reward = cartPoleLiteStep(x, v, out[0])
			totalReward += reward
			stepsSurvived++
			if math.Abs(x) > 2.0 {
				break
			}
		}
	}

	if stepsSurvived == 0 {
		return 0, Trace{"avg_reward": 0.0, "steps_survived": 0}, nil
	}
	avgReward := totalReward / float64(stepsSurvived)
	return Fitness(avgReward), Trace{"avg_reward": avgReward, "steps_survived": stepsSurvived}, nil
}

func evaluateCartPoleLiteWithTick(ctx context.Context, ticker TickAgent) (Fitness, Trace, error) {
	positionSetter, velocitySetter, forceOutput, err := cartPoleLiteIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	totalReward := 0.0
	stepsSurvived := 0
	for _, start := range []float64{-0.8, -0.4, 0.0, 0.4, 0.8} {
		x := start
		v := 0.0
		for step := 0; step < 60; step++ {
			if err := ctx.Err(); err != nil {
				return 0, nil, err
			}

			positionSetter.Set(x)
			velocitySetter.Set(v)
			out, err := ticker.Tick(ctx)
			if err != nil {
				return 0, nil, err
			}

			force := 0.0
			last := forceOutput.Last()
			if len(last) > 0 {
				force = last[0]
			} else if len(out) > 0 {
				force = out[0]
			}
			var reward float64
			x, v, reward = cartPoleLiteStep(x, v, force)
			totalReward += reward
			stepsSurvived++
			if math.Abs(x) > 2.0 {
				break
			}
		}
	}

	if stepsSurvived == 0 {
		return 0, Trace{"avg_reward": 0.0, "steps_survived": 0}, nil
	}
	avgReward := totalReward / float64(stepsSurvived)
	return Fitness(avgReward), Trace{"avg_reward": avgReward, "steps_survived": stepsSurvived}, nil
}

func cartPoleLiteStep(x, v, force float64) (nextX, nextV, reward float64) {
	const (
		dt       = 0.1
		kPos     = 0.45
		kVel     = 0.15
		forceK   = 1.25
		maxForce = 1.0
	)
	if force > maxForce {
		force = maxForce
	}
	if force < -maxForce {
		force = -maxForce
	}

	acc := forceK*force - kPos*x - kVel*v
	v = v + acc*dt
	x = x + v*dt
	reward = 1.0 - math.Min(1.0, math.Abs(x)/2.0)
	return x, v, reward
}

func cartPoleLiteIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	position, ok := typed.RegisteredSensor(protoio.CartPolePositionSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.CartPolePositionSensorName)
	}
	positionSetter, ok := position.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.CartPolePositionSensorName)
	}

	velocity, ok := typed.RegisteredSensor(protoio.CartPoleVelocitySensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.CartPoleVelocitySensorName)
	}
	velocitySetter, ok := velocity.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.CartPoleVelocitySensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.CartPoleForceActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.CartPoleForceActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.CartPoleForceActuatorName)
	}
	return positionSetter, velocitySetter, output, nil
}
