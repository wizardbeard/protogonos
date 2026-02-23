package scape

import (
	"context"
	"fmt"
	"math"

	protoio "protogonos/internal/io"
)

// LLVMPhaseOrderingScape is a deterministic surrogate for the reference
// phase-ordering workflow, preserving a phase-indexed optimize/unroll loop.
type LLVMPhaseOrderingScape struct{}

func (LLVMPhaseOrderingScape) Name() string {
	return "llvm-phase-ordering"
}

func (LLVMPhaseOrderingScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateLLVMPhaseOrderingWithTick(ctx, ticker)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateLLVMPhaseOrderingWithStep(ctx, runner)
}

func evaluateLLVMPhaseOrderingWithStep(ctx context.Context, runner StepAgent) (Fitness, Trace, error) {
	return evaluateLLVMPhaseOrdering(
		ctx,
		func(ctx context.Context, in []float64) (float64, error) {
			out, err := runner.RunStep(ctx, in)
			if err != nil {
				return 0, err
			}
			if len(out) != 1 {
				return 0, fmt.Errorf("llvm-phase-ordering requires one output, got %d", len(out))
			}
			return out[0], nil
		},
	)
}

func evaluateLLVMPhaseOrderingWithTick(ctx context.Context, ticker TickAgent) (Fitness, Trace, error) {
	complexitySetter, passSetter, phaseOutput, err := llvmPhaseOrderingIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateLLVMPhaseOrdering(
		ctx,
		func(ctx context.Context, in []float64) (float64, error) {
			complexitySetter.Set(in[0])
			passSetter.Set(in[1])

			out, err := ticker.Tick(ctx)
			if err != nil {
				return 0, err
			}

			phase := 0.0
			last := phaseOutput.Last()
			if len(last) > 0 {
				phase = last[0]
			} else if len(out) > 0 {
				phase = out[0]
			}
			return phase, nil
		},
	)
}

func evaluateLLVMPhaseOrdering(ctx context.Context, choosePhase func(context.Context, []float64) (float64, error)) (Fitness, Trace, error) {
	const maxPhases = 24
	complexity := 1.2
	alignmentAcc := 0.0
	phasesUsed := 0
	done := false

	for phase := 0; phase < maxPhases; phase++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		passNorm := 0.0
		if maxPhases > 1 {
			passNorm = float64(phase) / float64(maxPhases-1)
		}
		in := []float64{complexity, passNorm}
		output, err := choosePhase(ctx, in)
		if err != nil {
			return 0, nil, err
		}

		action := clampLLVM(output, -1, 1)
		target := llvmTargetPhase(passNorm)
		alignment := 1.0 - 0.5*math.Abs(action-target)
		alignmentAcc += alignment
		phasesUsed++

		improvement := 0.045*alignment - 0.015*math.Abs(action)
		complexity = clampLLVM(complexity-improvement, 0.05, 2.0)

		if action < -0.97 {
			done = true
			break
		}
	}

	if phasesUsed == 0 {
		return 0, Trace{"phases": 0, "final_complexity": complexity}, nil
	}

	alignmentAvg := alignmentAcc / float64(phasesUsed)
	runtimeScore := 1.0 / (1.0 + complexity)
	fitness := 0.7*runtimeScore + 0.3*alignmentAvg
	if done && phasesUsed < maxPhases {
		fitness -= 0.05 * float64(maxPhases-phasesUsed) / float64(maxPhases)
	}
	fitness = clampLLVM(fitness, 0, 1.5)

	return Fitness(fitness), Trace{
		"fitness":          fitness,
		"phases":           phasesUsed,
		"max_phases":       maxPhases,
		"done":             done,
		"alignment":        alignmentAvg,
		"final_complexity": complexity,
	}, nil
}

func llvmTargetPhase(passNorm float64) float64 {
	return clampLLVM(1.0-2.0*passNorm, -1, 1)
}

func llvmPhaseOrderingIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	complexity, ok := typed.RegisteredSensor(protoio.LLVMComplexitySensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.LLVMComplexitySensorName)
	}
	complexitySetter, ok := complexity.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.LLVMComplexitySensorName)
	}

	passIndex, ok := typed.RegisteredSensor(protoio.LLVMPassIndexSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.LLVMPassIndexSensorName)
	}
	passSetter, ok := passIndex.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.LLVMPassIndexSensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.LLVMPhaseActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.LLVMPhaseActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.LLVMPhaseActuatorName)
	}

	return complexitySetter, passSetter, output, nil
}

func clampLLVM(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
