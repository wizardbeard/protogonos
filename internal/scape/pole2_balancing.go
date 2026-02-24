package scape

import (
	"context"
	"fmt"
	"math"
	"strings"

	protoio "protogonos/internal/io"
)

// Pole2BalancingScape mirrors the reference pole2 double-pole control task.
type Pole2BalancingScape struct{}

func (Pole2BalancingScape) Name() string {
	return "pole2-balancing"
}

func (Pole2BalancingScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return Pole2BalancingScape{}.EvaluateMode(ctx, agent, "gt")
}

func (Pole2BalancingScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := pole2ConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluatePole2BalancingWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluatePole2BalancingWithStep(ctx, runner, cfg)
}

type pole2State struct {
	cartPosition float64
	cartVelocity float64
	angle1       float64
	velocity1    float64
	angle2       float64
	velocity2    float64
}

type pole2ModeConfig struct {
	mode       string
	maxSteps   int
	goalSteps  int
	angleLimit float64
	initAngle1 float64
	initAngle2 float64
	damping    bool
	doublePole bool
}

func pole2ConfigForMode(mode string) (pole2ModeConfig, error) {
	rad := 2 * math.Pi / 360
	angleLimit := 36.0 * rad

	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return pole2ModeConfig{
			mode:       "gt",
			maxSteps:   100000,
			goalSteps:  100000,
			angleLimit: angleLimit,
			initAngle1: 3.6 * rad,
			initAngle2: 0,
			damping:    true,
			doublePole: true,
		}, nil
	case "validation":
		return pole2ModeConfig{
			mode:       "validation",
			maxSteps:   1200,
			goalSteps:  1200,
			angleLimit: angleLimit,
			initAngle1: 2.4 * rad,
			initAngle2: 1.2 * rad,
			damping:    true,
			doublePole: true,
		}, nil
	case "test":
		return pole2ModeConfig{
			mode:       "test",
			maxSteps:   1200,
			goalSteps:  1200,
			angleLimit: angleLimit,
			initAngle1: 4.8 * rad,
			initAngle2: -1.8 * rad,
			damping:    true,
			doublePole: true,
		}, nil
	case "benchmark":
		return pole2ModeConfig{
			mode:       "benchmark",
			maxSteps:   1200,
			goalSteps:  1200,
			angleLimit: angleLimit,
			initAngle1: 4.8 * rad,
			initAngle2: -1.8 * rad,
			damping:    true,
			doublePole: true,
		}, nil
	default:
		return pole2ModeConfig{}, fmt.Errorf("unsupported pole2-balancing mode: %s", mode)
	}
}

func initialPole2State(cfg pole2ModeConfig) pole2State {
	return pole2State{
		angle1: cfg.initAngle1,
		angle2: cfg.initAngle2,
	}
}

type pole2EpisodeResult struct {
	finalState         pole2State
	stepsSurvived      int
	fitnessAcc         float64
	avgStepFitness     float64
	goalReached        bool
	terminationReason  string
	terminatedByBounds bool
}

func evaluatePole2BalancingWithStep(ctx context.Context, runner StepAgent, cfg pole2ModeConfig) (Fitness, Trace, error) {
	return evaluatePole2Balancing(
		ctx,
		cfg,
		func(ctx context.Context, state pole2State) (float64, error) {
			in := pole2Observation(state, cfg.angleLimit)
			out, err := runner.RunStep(ctx, in)
			if err != nil {
				return 0, err
			}
			if len(out) != 1 {
				return 0, fmt.Errorf("pole2-balancing requires one output, got %d", len(out))
			}
			return out[0], nil
		},
	)
}

func evaluatePole2BalancingWithTick(ctx context.Context, ticker TickAgent, cfg pole2ModeConfig) (Fitness, Trace, error) {
	positionSetter, velocitySetter, angle1Setter, velocity1Setter, angle2Setter, velocity2Setter, forceOutput, err := pole2BalancingIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluatePole2Balancing(
		ctx,
		cfg,
		func(ctx context.Context, state pole2State) (float64, error) {
			positionSetter.Set(scaleToUnit(state.cartPosition, 2.4, -2.4))
			velocitySetter.Set(scaleToUnit(state.cartVelocity, 10, -10))
			angle1Setter.Set(scaleToUnit(state.angle1, cfg.angleLimit, -cfg.angleLimit))
			velocity1Setter.Set(state.velocity1)
			angle2Setter.Set(scaleToUnit(state.angle2, cfg.angleLimit, -cfg.angleLimit))
			velocity2Setter.Set(state.velocity2)

			out, err := ticker.Tick(ctx)
			if err != nil {
				return 0, err
			}

			force := 0.0
			last := forceOutput.Last()
			if len(last) > 0 {
				force = last[0]
			} else if len(out) > 0 {
				force = out[0]
			}
			return force, nil
		},
	)
}

func evaluatePole2Balancing(
	ctx context.Context,
	cfg pole2ModeConfig,
	chooseForce func(context.Context, pole2State) (float64, error),
) (Fitness, Trace, error) {
	state := initialPole2State(cfg)
	stepsSurvived := 0
	fitnessAcc := 0.0
	terminationReason := "max_steps"
	goalReached := false
	terminatedByBounds := false

	for step := 0; step < cfg.maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		force, err := chooseForce(ctx, state)
		if err != nil {
			return 0, nil, err
		}
		force = clampPole2(force, -1, 1)

		state = simulateDoublePole(force*10, state, 2)
		stepsSurvived++

		terminated, reason, reachedGoal := pole2Termination(state, cfg, stepsSurvived)
		if terminated {
			terminationReason = reason
			goalReached = reachedGoal
			terminatedByBounds = !reachedGoal
			break
		}

		// Mirror the reference damping-oriented fitness accumulator while the run is active.
		fitnessAcc += pole2StepFitness(stepsSurvived, state, cfg.damping)
	}

	avgStepFitness := 0.0
	if stepsSurvived > 0 {
		avgStepFitness = fitnessAcc / float64(stepsSurvived)
	}
	result := pole2EpisodeResult{
		finalState:         state,
		stepsSurvived:      stepsSurvived,
		fitnessAcc:         fitnessAcc,
		avgStepFitness:     avgStepFitness,
		goalReached:        goalReached,
		terminationReason:  terminationReason,
		terminatedByBounds: terminatedByBounds,
	}

	return summarizePole2Outcome(result, cfg), Trace{
		"steps_survived":       stepsSurvived,
		"max_steps":            cfg.maxSteps,
		"goal_steps":           cfg.goalSteps,
		"goal_reached":         goalReached,
		"termination_reason":   terminationReason,
		"terminated_by_bounds": terminatedByBounds,
		"fitness_acc":          fitnessAcc,
		"avg_step_fitness":     avgStepFitness,
		"cart_position":        state.cartPosition,
		"cart_velocity":        state.cartVelocity,
		"angle1":               state.angle1,
		"velocity1":            state.velocity1,
		"angle2":               state.angle2,
		"velocity2":            state.velocity2,
		"mode":                 cfg.mode,
		"init_angle1":          cfg.initAngle1,
		"init_angle2":          cfg.initAngle2,
		"damping":              cfg.damping,
		"double_pole":          cfg.doublePole,
	}, nil
}

func pole2Observation(state pole2State, angleLimit float64) []float64 {
	return []float64{
		scaleToUnit(state.cartPosition, 2.4, -2.4),
		scaleToUnit(state.cartVelocity, 10, -10),
		scaleToUnit(state.angle1, angleLimit, -angleLimit),
		state.velocity1,
		scaleToUnit(state.angle2, angleLimit, -angleLimit),
		state.velocity2,
	}
}

func pole2Termination(state pole2State, cfg pole2ModeConfig, stepsSurvived int) (terminated bool, reason string, goalReached bool) {
	angle1Out := math.Abs(state.angle1) > cfg.angleLimit
	angle2Out := cfg.doublePole && math.Abs(state.angle2) > cfg.angleLimit
	cartOut := math.Abs(state.cartPosition) > 2.4
	stepOut := stepsSurvived >= cfg.maxSteps
	terminated = angle1Out || angle2Out || cartOut || stepOut
	if !terminated {
		return false, "", false
	}

	// Preserve reference-style ordering where reaching goal-step budget dominates
	// terminal signaling once a terminal state has been observed.
	if stepsSurvived >= cfg.goalSteps {
		return true, "goal_reached", true
	}
	if angle1Out {
		return true, "angle1_limit", false
	}
	if angle2Out {
		return true, "angle2_limit", false
	}
	if cartOut {
		return true, "cart_limit", false
	}
	return true, "max_steps", false
}

func pole2StepFitness(step int, state pole2State, damping bool) float64 {
	if !damping {
		return 1
	}
	fitness1 := float64(step) / 1000.0
	if step < 100 {
		return fitness1 * 0.1
	}
	denom := math.Abs(state.cartPosition) + math.Abs(state.cartVelocity) + math.Abs(state.angle1) + math.Abs(state.velocity1)
	if denom < 1e-9 {
		denom = 1e-9
	}
	fitness2 := 0.75 / denom
	return fitness1*0.1 + fitness2*0.9
}

func summarizePole2Outcome(result pole2EpisodeResult, cfg pole2ModeConfig) Fitness {
	if cfg.maxSteps <= 0 || result.stepsSurvived <= 0 {
		return 0
	}
	survival := float64(result.stepsSurvived) / float64(cfg.maxSteps)
	fitness := survival + 0.08*result.avgStepFitness
	if result.goalReached {
		fitness += 0.2
	}
	if math.IsNaN(fitness) || math.IsInf(fitness, 0) {
		return 0
	}
	if fitness < 0 {
		return 0
	}
	return Fitness(fitness)
}

func simulateDoublePole(force float64, state pole2State, steps int) pole2State {
	const (
		halfLength1 = 0.5
		halfLength2 = 0.05
		cartMass    = 1.0
		poleMass1   = 0.1
		poleMass2   = 0.01
		muC         = 0.0005
		muP         = 0.000002
		gravity     = -9.81
		delta       = 0.01
	)

	if steps <= 0 {
		return state
	}

	next := state
	for i := 0; i < steps; i++ {
		cur := next

		em1 := poleMass1 * (1 - (3.0/4.0)*math.Pow(math.Cos(cur.angle1), 2))
		em2 := poleMass2 * (1 - (3.0/4.0)*math.Pow(math.Cos(cur.angle2), 2))

		ef1 := poleMass1*halfLength1*math.Pow(cur.velocity1, 2)*math.Sin(cur.angle1) +
			(3.0/4.0)*poleMass1*math.Cos(cur.angle1)*(((muP*cur.velocity1)/(poleMass1*halfLength1))+gravity*math.Sin(cur.angle1))
		ef2 := poleMass2*halfLength2*math.Pow(cur.velocity2, 2)*math.Sin(cur.angle2) +
			(3.0/4.0)*poleMass2*math.Cos(cur.angle2)*(((muP*cur.velocity2)/(poleMass2*halfLength2))+gravity*math.Sin(cur.angle2))

		nextCartAccel := (force - muC*sgn(cur.cartVelocity) + ef1 + ef2) / (cartMass + em1 + em2)
		nextPoleAccel1 := -(3.0 / (4.0 * halfLength1)) * ((nextCartAccel * math.Cos(cur.angle1)) + (gravity * math.Sin(cur.angle1)) + ((muP * cur.velocity1) / (poleMass1 * halfLength1)))
		nextPoleAccel2 := -(3.0 / (4.0 * halfLength2)) * ((nextCartAccel * math.Cos(cur.angle2)) + (gravity * math.Sin(cur.angle2)) + ((muP * cur.velocity2) / (poleMass2 * halfLength2)))

		nextCartVelocity := cur.cartVelocity + delta*nextCartAccel
		nextCartPosition := cur.cartPosition + delta*cur.cartVelocity
		nextVelocity1 := cur.velocity1 + delta*nextPoleAccel1
		nextAngle1 := cur.angle1 + delta*nextVelocity1
		nextVelocity2 := cur.velocity2 + delta*nextPoleAccel2
		nextAngle2 := cur.angle2 + delta*nextVelocity2

		next = pole2State{
			cartPosition: nextCartPosition,
			cartVelocity: nextCartVelocity,
			angle1:       nextAngle1,
			velocity1:    nextVelocity1,
			angle2:       nextAngle2,
			velocity2:    nextVelocity2,
		}
	}

	return next
}

func pole2BalancingIO(agent TickAgent) (
	protoio.ScalarSensorSetter,
	protoio.ScalarSensorSetter,
	protoio.ScalarSensorSetter,
	protoio.ScalarSensorSetter,
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
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	position, ok := typed.RegisteredSensor(protoio.Pole2CartPositionSensorName)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.Pole2CartPositionSensorName)
	}
	positionSetter, ok := position.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.Pole2CartPositionSensorName)
	}

	velocity, ok := typed.RegisteredSensor(protoio.Pole2CartVelocitySensorName)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.Pole2CartVelocitySensorName)
	}
	velocitySetter, ok := velocity.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.Pole2CartVelocitySensorName)
	}

	angle1, ok := typed.RegisteredSensor(protoio.Pole2Angle1SensorName)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.Pole2Angle1SensorName)
	}
	angle1Setter, ok := angle1.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.Pole2Angle1SensorName)
	}

	velocity1, ok := typed.RegisteredSensor(protoio.Pole2Velocity1SensorName)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.Pole2Velocity1SensorName)
	}
	velocity1Setter, ok := velocity1.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.Pole2Velocity1SensorName)
	}

	angle2, ok := typed.RegisteredSensor(protoio.Pole2Angle2SensorName)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.Pole2Angle2SensorName)
	}
	angle2Setter, ok := angle2.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.Pole2Angle2SensorName)
	}

	velocity2, ok := typed.RegisteredSensor(protoio.Pole2Velocity2SensorName)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.Pole2Velocity2SensorName)
	}
	velocity2Setter, ok := velocity2.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.Pole2Velocity2SensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.Pole2PushActuatorName)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.Pole2PushActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.Pole2PushActuatorName)
	}

	return positionSetter, velocitySetter, angle1Setter, velocity1Setter, angle2Setter, velocity2Setter, output, nil
}

func sgn(v float64) float64 {
	if v > 0 {
		return 1
	}
	if v < 0 {
		return -1
	}
	return 0
}

func scaleToUnit(v, max, min float64) float64 {
	if max == min {
		return 0
	}
	scaled := ((v-min)/(max-min))*2 - 1
	if scaled > 1 {
		return 1
	}
	if scaled < -1 {
		return -1
	}
	return scaled
}

func clampPole2(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
