package scape

import (
	"context"
	"fmt"
	"math"
	"strings"

	protoio "protogonos/internal/io"
)

// CartPoleLiteScape is a simplified 1D balancing control task.
type CartPoleLiteScape struct{}

func (CartPoleLiteScape) Name() string {
	return "cart-pole-lite"
}

func (CartPoleLiteScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return CartPoleLiteScape{}.EvaluateMode(ctx, agent, "gt")
}

func (CartPoleLiteScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := cartPoleLiteConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateCartPoleLiteWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateCartPoleLiteWithStep(ctx, runner, cfg)
}

type cartPoleLiteModeConfig struct {
	mode            string
	startPositions  []float64
	stepsPerEpisode int
}

func cartPoleLiteConfigForMode(mode string) (cartPoleLiteModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return cartPoleLiteModeConfig{
			mode:            "gt",
			startPositions:  []float64{-0.8, -0.4, 0.0, 0.4, 0.8},
			stepsPerEpisode: 60,
		}, nil
	case "validation":
		return cartPoleLiteModeConfig{
			mode:            "validation",
			startPositions:  []float64{-1.0, -0.5, 0.5, 1.0},
			stepsPerEpisode: 48,
		}, nil
	case "test":
		return cartPoleLiteModeConfig{
			mode:            "test",
			startPositions:  []float64{-1.2, -0.6, 0.0, 0.6, 1.2},
			stepsPerEpisode: 48,
		}, nil
	case "benchmark":
		return cartPoleLiteModeConfig{
			mode:            "benchmark",
			startPositions:  []float64{-1.2, -0.6, 0.0, 0.6, 1.2},
			stepsPerEpisode: 48,
		}, nil
	default:
		return cartPoleLiteModeConfig{}, fmt.Errorf("unsupported cart-pole-lite mode: %s", mode)
	}
}

func evaluateCartPoleLiteWithStep(ctx context.Context, runner StepAgent, cfg cartPoleLiteModeConfig) (Fitness, Trace, error) {
	return evaluateCartPoleLite(
		ctx,
		cfg,
		func(ctx context.Context, x, v float64) (float64, error) {
			out, err := runner.RunStep(ctx, []float64{x, v})
			if err != nil {
				return 0, err
			}
			if len(out) != 1 {
				return 0, fmt.Errorf("cart-pole-lite requires one output, got %d", len(out))
			}
			return out[0], nil
		},
	)
}

func evaluateCartPoleLiteWithTick(ctx context.Context, ticker TickAgent, cfg cartPoleLiteModeConfig) (Fitness, Trace, error) {
	positionSetter, velocitySetter, forceOutput, err := cartPoleLiteIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateCartPoleLite(
		ctx,
		cfg,
		func(ctx context.Context, x, v float64) (float64, error) {
			positionSetter.Set(x)
			velocitySetter.Set(v)
			out, err := ticker.Tick(ctx)
			if err != nil {
				return 0, err
			}
			last := forceOutput.Last()
			if len(last) > 0 {
				return last[0], nil
			}
			if len(out) > 0 {
				return out[0], nil
			}
			return 0, nil
		},
	)
}

func evaluateCartPoleLite(
	ctx context.Context,
	cfg cartPoleLiteModeConfig,
	chooseForce func(context.Context, float64, float64) (float64, error),
) (Fitness, Trace, error) {
	totalReward := 0.0
	stepsSurvived := 0

	for _, start := range cfg.startPositions {
		x := start
		v := 0.0

		for step := 0; step < cfg.stepsPerEpisode; step++ {
			if err := ctx.Err(); err != nil {
				return 0, nil, err
			}

			force, err := chooseForce(ctx, x, v)
			if err != nil {
				return 0, nil, err
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
		return 0, Trace{
			"avg_reward":        0.0,
			"steps_survived":    0,
			"mode":              cfg.mode,
			"episodes":          len(cfg.startPositions),
			"steps_per_episode": cfg.stepsPerEpisode,
		}, nil
	}
	avgReward := totalReward / float64(stepsSurvived)
	return Fitness(avgReward), Trace{
		"avg_reward":        avgReward,
		"steps_survived":    stepsSurvived,
		"mode":              cfg.mode,
		"episodes":          len(cfg.startPositions),
		"steps_per_episode": cfg.stepsPerEpisode,
	}, nil
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
