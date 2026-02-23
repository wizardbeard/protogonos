package scape

import (
	"context"
	"fmt"
	"math"
	"strings"

	protoio "protogonos/internal/io"
)

// EpitopesScape is a deterministic sequence classification proxy for epitopes:sim.
type EpitopesScape struct{}

func (EpitopesScape) Name() string {
	return "epitopes"
}

func (EpitopesScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return EpitopesScape{}.EvaluateMode(ctx, agent, "gt")
}

func (EpitopesScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	cfg, err := epitopesConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateEpitopesWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateEpitopesWithStep(ctx, runner, cfg)
}

func evaluateEpitopesWithStep(ctx context.Context, runner StepAgent, cfg epitopesModeConfig) (Fitness, Trace, error) {
	return evaluateEpitopes(
		ctx,
		cfg,
		func(ctx context.Context, percept []float64) ([]float64, error) {
			out, err := runner.RunStep(ctx, percept)
			if err != nil {
				return nil, err
			}
			if len(out) == 0 {
				return nil, fmt.Errorf("epitopes requires at least one output")
			}
			return append([]float64(nil), out...), nil
		},
	)
}

func evaluateEpitopesWithTick(ctx context.Context, ticker TickAgent, cfg epitopesModeConfig) (Fitness, Trace, error) {
	signalSetter, memorySetter, responseOutput, err := epitopesIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateEpitopes(
		ctx,
		cfg,
		func(ctx context.Context, percept []float64) ([]float64, error) {
			signalSetter.Set(percept[0])
			memorySetter.Set(percept[1])
			out, err := ticker.Tick(ctx)
			if err != nil {
				return nil, err
			}
			last := responseOutput.Last()
			if len(last) > 0 {
				return append([]float64(nil), last...), nil
			}
			if len(out) > 0 {
				return append([]float64(nil), out...), nil
			}
			return []float64{0}, nil
		},
	)
}

func evaluateEpitopes(
	ctx context.Context,
	cfg epitopesModeConfig,
	chooseClassification func(context.Context, []float64) ([]float64, error),
) (Fitness, Trace, error) {
	correct := 0
	predictions := make([]float64, 0, cfg.samples)
	prevSignal := 0.0
	positiveTargets := 0
	negativeTargets := 0
	marginAcc := 0.0
	signalAcc := 0.0
	perceptWidth := 2 + cfg.sequenceLength*epitopesAlphabetSize

	for i := 0; i < cfg.samples; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		index := cfg.startIndex + i
		sequence := epitopesSequence(index, cfg.sequenceLength)
		signal := epitopesSignal(sequence, index)
		memory := prevSignal
		percept := epitopesPercept(signal, memory, sequence)
		out, err := chooseClassification(ctx, percept)
		if err != nil {
			return 0, nil, err
		}
		pred := epitopesOutputToScalar(out)
		target := epitopesTarget(signal, memory)
		if target > 0 {
			positiveTargets++
		} else {
			negativeTargets++
		}

		if binarySign(pred) == target {
			correct++
		}
		predictions = append(predictions, pred)
		marginAcc += pred * target
		signalAcc += signal
		prevSignal = signal
	}

	accuracy := float64(correct) / float64(maxIntEpitopes(1, cfg.samples))
	meanMargin := marginAcc / float64(maxIntEpitopes(1, cfg.samples))
	meanSignal := signalAcc / float64(maxIntEpitopes(1, cfg.samples))
	return Fitness(accuracy), Trace{
		"accuracy":            accuracy,
		"correct":             correct,
		"total":               cfg.samples,
		"predictions":         predictions,
		"mode":                cfg.mode,
		"dataset":             cfg.table,
		"start_index":         cfg.startIndex,
		"end_index":           cfg.startIndex + cfg.samples - 1,
		"sequence_length":     cfg.sequenceLength,
		"feature_width":       perceptWidth,
		"positive_targets":    positiveTargets,
		"negative_targets":    negativeTargets,
		"mean_signal":         meanSignal,
		"mean_margin":         meanMargin,
		"classification_skew": float64(positiveTargets-negativeTargets) / float64(maxIntEpitopes(1, cfg.samples)),
	}, nil
}

type epitopesModeConfig struct {
	mode           string
	table          string
	startIndex     int
	samples        int
	sequenceLength int
}

func epitopesConfigForMode(mode string) (epitopesModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return epitopesModeConfig{
			mode:           "gt",
			table:          "abc_pred16",
			startIndex:     0,
			samples:        64,
			sequenceLength: 16,
		}, nil
	case "validation":
		return epitopesModeConfig{
			mode:           "validation",
			table:          "abc_pred16",
			startIndex:     512,
			samples:        32,
			sequenceLength: 16,
		}, nil
	case "test":
		return epitopesModeConfig{
			mode:           "test",
			table:          "abc_pred16",
			startIndex:     1024,
			samples:        32,
			sequenceLength: 16,
		}, nil
	case "benchmark":
		return epitopesModeConfig{
			mode:           "benchmark",
			table:          "abc_pred16",
			startIndex:     840,
			samples:        280,
			sequenceLength: 16,
		}, nil
	default:
		return epitopesModeConfig{}, fmt.Errorf("unsupported epitopes mode: %s", mode)
	}
}

const epitopesAlphabetSize = 21

func epitopesSequence(index, sequenceLength int) []int {
	if sequenceLength <= 0 {
		return nil
	}
	seq := make([]int, sequenceLength)
	state := uint32(index*1103515245 + 12345)
	for i := 0; i < sequenceLength; i++ {
		state = state*1664525 + 1013904223 + uint32(i*97+31)
		seq[i] = int((state >> 16) % epitopesAlphabetSize)
	}
	// Inject deterministic motifs/regimes so the surrogate has non-trivial boundaries.
	if sequenceLength >= 6 && index%11 == 0 {
		seq[1], seq[2], seq[3] = 4, 10, 15
	}
	if sequenceLength >= 5 && index%17 == 0 {
		seq[sequenceLength-3], seq[sequenceLength-2] = 2, 2
	}
	return seq
}

func epitopesPercept(signal, memory float64, sequence []int) []float64 {
	out := make([]float64, 2+len(sequence)*epitopesAlphabetSize)
	out[0] = signal
	out[1] = memory
	offset := 2
	for _, residue := range sequence {
		index := offset + clampIntEpitopes(residue, 0, epitopesAlphabetSize-1)
		out[index] = 1
		offset += epitopesAlphabetSize
	}
	return out
}

func epitopesSignal(sequence []int, index int) float64 {
	if len(sequence) == 0 {
		return 0
	}
	acc := 0.0
	for i, residue := range sequence {
		weight := -1.0 + 2.0*float64(residue)/float64(epitopesAlphabetSize-1)
		acc += weight * (1.0 + 0.15*math.Sin(float64(i)*0.7))
	}
	if epitopesHasMotif(sequence, []int{4, 10, 15}) {
		acc += 1.1
	}
	if epitopesHasMotif(sequence, []int{2, 2}) {
		acc -= 0.7
	}
	regime := 0.20 * math.Sin(float64(index)*0.09)
	score := acc/float64(len(sequence)) + regime
	return clampEpitopes(score, -1, 1)
}

func epitopesHasMotif(sequence, motif []int) bool {
	if len(motif) == 0 || len(sequence) < len(motif) {
		return false
	}
	for i := 0; i <= len(sequence)-len(motif); i++ {
		match := true
		for j := 0; j < len(motif); j++ {
			if sequence[i+j] != motif[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func epitopesOutputToScalar(output []float64) float64 {
	if len(output) == 1 {
		return output[0]
	}
	if argmax(output) == 0 {
		return -1
	}
	return 1
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

func clampEpitopes(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampIntEpitopes(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxIntEpitopes(a, b int) int {
	if a > b {
		return a
	}
	return b
}
