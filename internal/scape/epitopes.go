package scape

import (
	"context"
	"fmt"
	"math"

	protoio "protogonos/internal/io"
)

// EpitopesScape is a deterministic sequence classification proxy for epitopes:sim.
type EpitopesScape struct{}

func (EpitopesScape) Name() string {
	return "epitopes"
}

func (EpitopesScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateEpitopesWithTick(ctx, ticker)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateEpitopesWithStep(ctx, runner)
}

func evaluateEpitopesWithStep(ctx context.Context, runner StepAgent) (Fitness, Trace, error) {
	const samples = 64
	correct := 0
	predictions := make([]float64, 0, samples)
	prev := 0.0

	for i := 0; i < samples; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		signal := epitopesSignal(i)
		memory := prev
		out, err := runner.RunStep(ctx, []float64{signal, memory})
		if err != nil {
			return 0, nil, err
		}
		if len(out) != 1 {
			return 0, nil, fmt.Errorf("epitopes requires one output, got %d", len(out))
		}

		pred := out[0]
		target := epitopesTarget(signal, memory)
		if binarySign(pred) == target {
			correct++
		}
		predictions = append(predictions, pred)
		prev = signal
	}

	accuracy := float64(correct) / samples
	return Fitness(accuracy), Trace{
		"accuracy":    accuracy,
		"correct":     correct,
		"total":       samples,
		"predictions": predictions,
	}, nil
}

func evaluateEpitopesWithTick(ctx context.Context, ticker TickAgent) (Fitness, Trace, error) {
	signalSetter, memorySetter, responseOutput, err := epitopesIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	const samples = 64
	correct := 0
	predictions := make([]float64, 0, samples)
	prev := 0.0

	for i := 0; i < samples; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		signal := epitopesSignal(i)
		memory := prev
		signalSetter.Set(signal)
		memorySetter.Set(memory)

		out, err := ticker.Tick(ctx)
		if err != nil {
			return 0, nil, err
		}

		pred := 0.0
		last := responseOutput.Last()
		if len(last) > 0 {
			pred = last[0]
		} else if len(out) > 0 {
			pred = out[0]
		}

		target := epitopesTarget(signal, memory)
		if binarySign(pred) == target {
			correct++
		}
		predictions = append(predictions, pred)
		prev = signal
	}

	accuracy := float64(correct) / samples
	return Fitness(accuracy), Trace{
		"accuracy":    accuracy,
		"correct":     correct,
		"total":       samples,
		"predictions": predictions,
	}, nil
}

func epitopesSignal(index int) float64 {
	i := float64(index + 1)
	return math.Sin(i*0.43) + 0.5*math.Sin(i*0.11)
}

func epitopesTarget(signal, memory float64) float64 {
	if signal+0.7*memory >= 0 {
		return 1
	}
	return -1
}

func binarySign(v float64) float64 {
	if v >= 0 {
		return 1
	}
	return -1
}

func epitopesIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	signal, ok := typed.RegisteredSensor(protoio.EpitopesSignalSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.EpitopesSignalSensorName)
	}
	signalSetter, ok := signal.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.EpitopesSignalSensorName)
	}

	memory, ok := typed.RegisteredSensor(protoio.EpitopesMemorySensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.EpitopesMemorySensorName)
	}
	memorySetter, ok := memory.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.EpitopesMemorySensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.EpitopesResponseActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.EpitopesResponseActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.EpitopesResponseActuatorName)
	}

	return signalSetter, memorySetter, output, nil
}
