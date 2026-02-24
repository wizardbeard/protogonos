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
	table := defaultEpitopesTable(cfg.table, cfg.sequenceLength)
	session, err := newEpitopesSession(cfg, table)
	if err != nil {
		return 0, nil, err
	}

	correct := 0
	predictions := make([]float64, 0, cfg.maxSamples)
	positiveTargets := 0
	negativeTargets := 0
	marginAcc := 0.0
	signalAcc := 0.0
	memoryAcc := 0.0
	evaluated := 0
	perceptWidth := 2 + cfg.sequenceLength*epitopesAlphabetSize

	for i := 0; i < cfg.maxSamples; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		percept, row, _, err := session.sense()
		if err != nil {
			return 0, nil, err
		}

		out, err := chooseClassification(ctx, percept)
		if err != nil {
			return 0, nil, err
		}
		pred := epitopesOutputToBinary(out)
		reward, halt, target, err := session.classify(pred)
		if err != nil {
			return 0, nil, err
		}

		if target == 1 {
			positiveTargets++
		} else {
			negativeTargets++
		}
		if reward == 1 {
			correct++
		}
		signedPred := epitopesBinaryToSigned(pred)
		signedTarget := epitopesBinaryToSigned(target)
		predictions = append(predictions, signedPred)
		marginAcc += signedPred * signedTarget
		signalAcc += row.signal
		memoryAcc += row.memory
		evaluated++
		if halt {
			break
		}
	}

	accuracy := float64(correct) / float64(maxIntEpitopes(1, evaluated))
	meanMargin := marginAcc / float64(maxIntEpitopes(1, evaluated))
	meanSignal := signalAcc / float64(maxIntEpitopes(1, evaluated))
	meanMemory := memoryAcc / float64(maxIntEpitopes(1, evaluated))
	return Fitness(accuracy), Trace{
		"accuracy":            accuracy,
		"correct":             correct,
		"total":               evaluated,
		"predictions":         predictions,
		"mode":                cfg.mode,
		"op_mode":             cfg.opMode,
		"dataset":             cfg.table,
		"table_name":          table.name,
		"start_index":         session.startIndex,
		"end_index":           session.endIndex,
		"index_current":       session.indexCurrent,
		"configured_max":      cfg.maxSamples,
		"sequence_length":     cfg.sequenceLength,
		"feature_width":       perceptWidth,
		"positive_targets":    positiveTargets,
		"negative_targets":    negativeTargets,
		"mean_signal":         meanSignal,
		"mean_memory":         meanMemory,
		"mean_margin":         meanMargin,
		"classification_skew": float64(positiveTargets-negativeTargets) / float64(maxIntEpitopes(1, evaluated)),
	}, nil
}

type epitopesModeConfig struct {
	mode           string
	opMode         string
	table          string
	startIndex     int
	endIndex       int
	startBench     int
	endBench       int
	maxSamples     int
	sequenceLength int
}

func epitopesConfigForMode(mode string) (epitopesModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return epitopesModeConfig{
			mode:           "gt",
			opMode:         "gt",
			table:          "abc_pred16",
			startIndex:     1,
			endIndex:       64,
			startBench:     841,
			endBench:       1120,
			maxSamples:     64,
			sequenceLength: 16,
		}, nil
	case "validation":
		return epitopesModeConfig{
			mode:           "validation",
			opMode:         "gt",
			table:          "abc_pred16",
			startIndex:     513,
			endIndex:       544,
			startBench:     841,
			endBench:       1120,
			maxSamples:     32,
			sequenceLength: 16,
		}, nil
	case "test":
		return epitopesModeConfig{
			mode:           "test",
			opMode:         "gt",
			table:          "abc_pred16",
			startIndex:     1025,
			endIndex:       1056,
			startBench:     841,
			endBench:       1120,
			maxSamples:     32,
			sequenceLength: 16,
		}, nil
	case "benchmark":
		return epitopesModeConfig{
			mode:           "benchmark",
			opMode:         "benchmark",
			table:          "abc_pred16",
			startIndex:     1,
			endIndex:       64,
			startBench:     841,
			endBench:       1120,
			maxSamples:     280,
			sequenceLength: 16,
		}, nil
	default:
		return epitopesModeConfig{}, fmt.Errorf("unsupported epitopes mode: %s", mode)
	}
}

type epitopesRow struct {
	sequence       []int
	signal         float64
	memory         float64
	classification int
}

type epitopesTable struct {
	name    string
	rows    []epitopesRow // 1-based
	modBase int
}

func defaultEpitopesTable(name string, sequenceLength int) epitopesTable {
	if sequenceLength <= 0 {
		sequenceLength = 16
	}
	const totalRows = 1400

	rows := make([]epitopesRow, totalRows+1)
	prevSignal := 0.0
	for index := 1; index <= totalRows; index++ {
		sequence := epitopesSequence(index-1, sequenceLength)
		signal := epitopesSignal(sequence, index-1)
		memory := prevSignal
		classification := 0
		if signal+0.7*memory >= 0 {
			classification = 1
		}
		rows[index] = epitopesRow{
			sequence:       sequence,
			signal:         signal,
			memory:         memory,
			classification: classification,
		}
		prevSignal = signal
	}

	return epitopesTable{
		name:    name,
		rows:    rows,
		modBase: totalRows + 1, // mirrors reference rem 1401 index wrap behavior.
	}
}

func (t epitopesTable) rowAt(index int) (epitopesRow, error) {
	if index <= 0 || index >= len(t.rows) {
		return epitopesRow{}, fmt.Errorf("epitopes row index out of bounds: %d", index)
	}
	return t.rows[index], nil
}

type epitopesSession struct {
	opMode       string
	table        epitopesTable
	startIndex   int
	endIndex     int
	indexCurrent int
	halted       bool
}

func newEpitopesSession(cfg epitopesModeConfig, table epitopesTable) (*epitopesSession, error) {
	start := cfg.startIndex
	end := cfg.endIndex
	if strings.EqualFold(cfg.opMode, "benchmark") {
		start = cfg.startBench
		end = cfg.endBench
	}
	if start <= 0 || end < start {
		return nil, fmt.Errorf("invalid epitopes window start=%d end=%d", start, end)
	}
	if _, err := table.rowAt(start); err != nil {
		return nil, err
	}
	if _, err := table.rowAt(end); err != nil {
		return nil, err
	}
	return &epitopesSession{
		opMode:       cfg.opMode,
		table:        table,
		startIndex:   start,
		endIndex:     end,
		indexCurrent: 0,
		halted:       false,
	}, nil
}

func (s *epitopesSession) sense() ([]float64, epitopesRow, int, error) {
	if s.halted {
		return nil, epitopesRow{}, 0, fmt.Errorf("epitopes session halted")
	}
	if s.indexCurrent == 0 {
		s.indexCurrent = s.startIndex
	}
	row, err := s.table.rowAt(s.indexCurrent)
	if err != nil {
		return nil, epitopesRow{}, 0, err
	}
	return epitopesPercept(row.signal, row.memory, row.sequence), row, s.indexCurrent, nil
}

func (s *epitopesSession) classify(prediction int) (reward int, halt bool, target int, err error) {
	if s.halted || s.indexCurrent == 0 {
		return 0, false, 0, fmt.Errorf("epitopes classify called on inactive session")
	}
	row, err := s.table.rowAt(s.indexCurrent)
	if err != nil {
		return 0, false, 0, err
	}
	target = row.classification
	if prediction == target {
		reward = 1
	}

	if s.indexCurrent == s.endIndex {
		s.halted = true
		s.indexCurrent = 0
		return reward, true, target, nil
	}

	nextIndex := (s.indexCurrent + 1) % s.table.modBase
	if nextIndex == 0 {
		nextIndex = 1
	}
	s.indexCurrent = nextIndex
	return reward, false, target, nil
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

func epitopesOutputToBinary(output []float64) int {
	if len(output) == 1 {
		if output[0] >= 0 {
			return 1
		}
		return 0
	}
	if argmax(output) == 0 {
		return 0
	}
	return 1
}

func epitopesBinaryToSigned(v int) float64 {
	if v == 0 {
		return -1
	}
	return 1
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
