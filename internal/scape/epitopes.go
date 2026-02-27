package scape

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

// EpitopesScape is a deterministic sequence classification proxy for epitopes:sim.
type EpitopesScape struct{}

type epitopesWindows struct {
	gtStart         int
	gtEnd           int
	validationStart int
	validationEnd   int
	testStart       int
	testEnd         int
	benchmarkStart  int
	benchmarkEnd    int
}

type epitopesSource struct {
	tableName string
	table     epitopesTable
	windows   epitopesWindows
}

// EpitopesTableBounds configures mode windows for a loaded table.
// Zero values use table-size-derived defaults.
type EpitopesTableBounds struct {
	GTStart         int
	GTEnd           int
	ValidationStart int
	ValidationEnd   int
	TestStart       int
	TestEnd         int
	BenchmarkStart  int
	BenchmarkEnd    int
}

var (
	epitopesSourceMu    sync.RWMutex
	epitopesSourceState = defaultEpitopesSource()

	defaultEpitopesTablesOnce sync.Once
	defaultEpitopesTablesByID map[string]epitopesTable
	defaultEpitopesTableNames []string
)

func (EpitopesScape) Name() string {
	return "epitopes"
}

var defaultEpitopesTableSpecs = []struct {
	name           string
	sequenceLength int
}{
	{name: "abc_pred10", sequenceLength: 10},
	{name: "abc_pred12", sequenceLength: 12},
	{name: "abc_pred14", sequenceLength: 14},
	{name: "abc_pred16", sequenceLength: 16},
	{name: "abc_pred18", sequenceLength: 18},
	{name: "abc_pred20", sequenceLength: 20},
}

// ResetEpitopesTableSource restores the deterministic built-in epitopes table.
func ResetEpitopesTableSource() {
	epitopesSourceMu.Lock()
	defer epitopesSourceMu.Unlock()
	epitopesSourceState = defaultEpitopesSource()
}

// AvailableEpitopesTableNames lists built-in table names that mirror reference ets:file2tab defaults.
func AvailableEpitopesTableNames() []string {
	ensureDefaultEpitopesTableCatalog()
	return append([]string(nil), defaultEpitopesTableNames...)
}

// SelectEpitopesTableSource selects a built-in table by name and makes it active globally.
func SelectEpitopesTableSource(tableName string, bounds EpitopesTableBounds) error {
	source, err := loadDefaultEpitopesSource(tableName, bounds)
	if err != nil {
		return err
	}
	epitopesSourceMu.Lock()
	defer epitopesSourceMu.Unlock()
	epitopesSourceState = source
	return nil
}

// LoadEpitopesTableCSV loads epitopes rows from CSV and makes the table active.
// Expected columns per row:
// signal,memory,classification[,residue0,residue1,...]
func LoadEpitopesTableCSV(path string, bounds EpitopesTableBounds) error {
	source, err := loadEpitopesSourceCSV(path, bounds)
	if err != nil {
		return err
	}

	epitopesSourceMu.Lock()
	defer epitopesSourceMu.Unlock()
	epitopesSourceState = source
	return nil
}

func (EpitopesScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return EpitopesScape{}.EvaluateMode(ctx, agent, "gt")
}

func (EpitopesScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	source := currentEpitopesSource(ctx)
	cfg, err := epitopesConfigForMode(mode, source)
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
		func(ctx context.Context, percept epitopesSenseInput) ([]float64, error) {
			out, err := runner.RunStep(ctx, percept.vector)
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
	io, err := epitopesIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateEpitopes(
		ctx,
		cfg,
		func(ctx context.Context, percept epitopesSenseInput) ([]float64, error) {
			io.signal.Set(percept.signal)
			io.memory.Set(percept.memory)
			if io.target != nil {
				io.target.Set(percept.target)
			}
			if io.progress != nil {
				io.progress.Set(percept.progress)
			}
			if io.margin != nil {
				io.margin.Set(percept.margin)
			}
			out, err := ticker.Tick(ctx)
			if err != nil {
				return nil, err
			}
			last := io.responseOutput.Last()
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
	chooseClassification func(context.Context, epitopesSenseInput) ([]float64, error),
) (Fitness, Trace, error) {
	table := cfg.table
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
	targetAcc := 0.0
	progressAcc := 0.0
	decisionMarginAcc := 0.0
	evaluated := 0
	perceptWidth := 2 + cfg.sequenceLength*epitopesAlphabetSize

	for i := 0; i < cfg.maxSamples; i++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		perceptVector, row, index, err := session.sense()
		if err != nil {
			return 0, nil, err
		}
		targetSigned := epitopesBinaryToSigned(row.classification)
		progress := epitopesSessionProgress(index, session.startIndex, session.endIndex)
		decisionMargin := row.signal + 0.7*row.memory

		out, err := chooseClassification(ctx, epitopesSenseInput{
			vector:   perceptVector,
			signal:   row.signal,
			memory:   row.memory,
			target:   targetSigned,
			progress: progress,
			margin:   decisionMargin,
		})
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
		targetAcc += targetSigned
		progressAcc += progress
		decisionMarginAcc += decisionMargin
		evaluated++
		if halt {
			break
		}
	}

	accuracy := float64(correct) / float64(maxIntEpitopes(1, evaluated))
	meanMargin := marginAcc / float64(maxIntEpitopes(1, evaluated))
	meanSignal := signalAcc / float64(maxIntEpitopes(1, evaluated))
	meanMemory := memoryAcc / float64(maxIntEpitopes(1, evaluated))
	meanTarget := targetAcc / float64(maxIntEpitopes(1, evaluated))
	meanProgress := progressAcc / float64(maxIntEpitopes(1, evaluated))
	meanDecisionMargin := decisionMarginAcc / float64(maxIntEpitopes(1, evaluated))
	return Fitness(accuracy), Trace{
		"accuracy":             accuracy,
		"correct":              correct,
		"total":                evaluated,
		"predictions":          predictions,
		"mode":                 cfg.mode,
		"op_mode":              cfg.opMode,
		"dataset":              cfg.tableName,
		"table_name":           table.name,
		"start_index":          session.startIndex,
		"end_index":            session.endIndex,
		"index_current":        session.indexCurrent,
		"configured_max":       cfg.maxSamples,
		"sequence_length":      cfg.sequenceLength,
		"feature_width":        perceptWidth,
		"positive_targets":     positiveTargets,
		"negative_targets":     negativeTargets,
		"mean_signal":          meanSignal,
		"mean_memory":          meanMemory,
		"mean_margin":          meanMargin,
		"mean_target":          meanTarget,
		"mean_progress":        meanProgress,
		"mean_decision_margin": meanDecisionMargin,
		"classification_skew":  float64(positiveTargets-negativeTargets) / float64(maxIntEpitopes(1, evaluated)),
	}, nil
}

type epitopesSenseInput struct {
	vector   []float64
	signal   float64
	memory   float64
	target   float64
	progress float64
	margin   float64
}

type epitopesModeConfig struct {
	mode           string
	opMode         string
	tableName      string
	table          epitopesTable
	startIndex     int
	endIndex       int
	startBench     int
	endBench       int
	maxSamples     int
	sequenceLength int
}

func epitopesConfigForMode(mode string, source epitopesSource) (epitopesModeConfig, error) {
	table := source.table
	tableName := strings.TrimSpace(source.tableName)
	if tableName == "" {
		tableName = strings.TrimSpace(table.name)
	}
	if tableName == "" {
		tableName = "epitopes.table"
	}
	if strings.TrimSpace(table.name) == "" {
		table.name = tableName
	}

	sequenceLength := table.sequenceLength
	if sequenceLength <= 0 {
		sequenceLength = 16
	}
	gtSamples := maxIntEpitopes(1, source.windows.gtEnd-source.windows.gtStart+1)
	validationSamples := maxIntEpitopes(1, source.windows.validationEnd-source.windows.validationStart+1)
	testSamples := maxIntEpitopes(1, source.windows.testEnd-source.windows.testStart+1)
	benchmarkSamples := maxIntEpitopes(1, source.windows.benchmarkEnd-source.windows.benchmarkStart+1)

	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return epitopesModeConfig{
			mode:           "gt",
			opMode:         "gt",
			tableName:      tableName,
			table:          table,
			startIndex:     source.windows.gtStart,
			endIndex:       source.windows.gtEnd,
			startBench:     source.windows.benchmarkStart,
			endBench:       source.windows.benchmarkEnd,
			maxSamples:     gtSamples,
			sequenceLength: sequenceLength,
		}, nil
	case "validation":
		return epitopesModeConfig{
			mode:           "validation",
			opMode:         "gt",
			tableName:      tableName,
			table:          table,
			startIndex:     source.windows.validationStart,
			endIndex:       source.windows.validationEnd,
			startBench:     source.windows.benchmarkStart,
			endBench:       source.windows.benchmarkEnd,
			maxSamples:     validationSamples,
			sequenceLength: sequenceLength,
		}, nil
	case "test":
		return epitopesModeConfig{
			mode:           "test",
			opMode:         "gt",
			tableName:      tableName,
			table:          table,
			startIndex:     source.windows.testStart,
			endIndex:       source.windows.testEnd,
			startBench:     source.windows.benchmarkStart,
			endBench:       source.windows.benchmarkEnd,
			maxSamples:     testSamples,
			sequenceLength: sequenceLength,
		}, nil
	case "benchmark":
		return epitopesModeConfig{
			mode:           "benchmark",
			opMode:         "benchmark",
			tableName:      tableName,
			table:          table,
			startIndex:     source.windows.gtStart,
			endIndex:       source.windows.gtEnd,
			startBench:     source.windows.benchmarkStart,
			endBench:       source.windows.benchmarkEnd,
			maxSamples:     benchmarkSamples,
			sequenceLength: sequenceLength,
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
	name           string
	rows           []epitopesRow // 1-based
	modBase        int
	sequenceLength int
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
		name:           name,
		rows:           rows,
		modBase:        totalRows + 1, // mirrors reference rem 1401 index wrap behavior.
		sequenceLength: sequenceLength,
	}
}

func ensureDefaultEpitopesTableCatalog() {
	defaultEpitopesTablesOnce.Do(func() {
		defaultEpitopesTablesByID = make(map[string]epitopesTable, len(defaultEpitopesTableSpecs))
		defaultEpitopesTableNames = make([]string, 0, len(defaultEpitopesTableSpecs))
		for _, spec := range defaultEpitopesTableSpecs {
			table := defaultEpitopesTable(spec.name, spec.sequenceLength)
			defaultEpitopesTablesByID[canonicalEpitopesTableName(spec.name)] = table
			defaultEpitopesTableNames = append(defaultEpitopesTableNames, table.name)
		}
	})
}

func canonicalEpitopesTableName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

func hasAnyEpitopesBounds(bounds EpitopesTableBounds) bool {
	return bounds.GTStart != 0 ||
		bounds.GTEnd != 0 ||
		bounds.ValidationStart != 0 ||
		bounds.ValidationEnd != 0 ||
		bounds.TestStart != 0 ||
		bounds.TestEnd != 0 ||
		bounds.BenchmarkStart != 0 ||
		bounds.BenchmarkEnd != 0
}

func defaultEpitopesWindows() epitopesWindows {
	return epitopesWindows{
		gtStart:         1,
		gtEnd:           64,
		validationStart: 513,
		validationEnd:   544,
		testStart:       1025,
		testEnd:         1056,
		benchmarkStart:  841,
		benchmarkEnd:    1120,
	}
}

func defaultEpitopesTableByName(name string) (epitopesTable, bool) {
	ensureDefaultEpitopesTableCatalog()
	table, ok := defaultEpitopesTablesByID[canonicalEpitopesTableName(name)]
	if !ok {
		return epitopesTable{}, false
	}
	return table, true
}

func loadDefaultEpitopesSource(tableName string, bounds EpitopesTableBounds) (epitopesSource, error) {
	name := strings.TrimSpace(tableName)
	if name == "" {
		name = "abc_pred16"
	}
	table, ok := defaultEpitopesTableByName(name)
	if !ok {
		return epitopesSource{}, fmt.Errorf("unknown epitopes table %q", tableName)
	}

	windows := defaultEpitopesWindows()
	if hasAnyEpitopesBounds(bounds) {
		var err error
		windows, err = buildEpitopesWindows(len(table.rows)-1, bounds)
		if err != nil {
			return epitopesSource{}, err
		}
	}

	return epitopesSource{
		tableName: table.name,
		table:     table,
		windows:   windows,
	}, nil
}

func defaultEpitopesSource() epitopesSource {
	source, err := loadDefaultEpitopesSource("abc_pred16", EpitopesTableBounds{})
	if err != nil {
		table := defaultEpitopesTable("abc_pred16", 16)
		return epitopesSource{
			tableName: table.name,
			table:     table,
			windows:   defaultEpitopesWindows(),
		}
	}
	return source
}

func currentEpitopesSource(ctx context.Context) epitopesSource {
	if source, ok := epitopesSourceFromContext(ctx); ok {
		return source
	}
	epitopesSourceMu.RLock()
	defer epitopesSourceMu.RUnlock()
	return epitopesSourceState
}

func loadEpitopesSourceCSV(path string, bounds EpitopesTableBounds) (epitopesSource, error) {
	return loadEpitopesSourceCSVWithName(path, bounds, "")
}

func loadEpitopesSourceCSVWithName(path string, bounds EpitopesTableBounds, tableName string) (epitopesSource, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return epitopesSource{}, fmt.Errorf("epitopes csv path is required")
	}

	f, err := os.Open(path)
	if err != nil {
		return epitopesSource{}, fmt.Errorf("open epitopes csv %s: %w", path, err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1

	rows := make([]epitopesRow, 0, 1024)
	sequenceLength := 0
	fileRow := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return epitopesSource{}, fmt.Errorf("read epitopes csv row %d: %w", fileRow+1, err)
		}
		fileRow++

		parsed, ok, err := parseEpitopesCSVRow(record, fileRow)
		if err != nil {
			if len(rows) == 0 && fileRow == 1 {
				// Allow a single header row.
				continue
			}
			return epitopesSource{}, err
		}
		if !ok {
			continue
		}

		if len(parsed.sequence) == 0 {
			generatedLength := sequenceLength
			if generatedLength <= 0 {
				generatedLength = 16
			}
			parsed.sequence = epitopesSequence(len(rows), generatedLength)
		}
		if sequenceLength == 0 {
			sequenceLength = len(parsed.sequence)
		}
		if len(parsed.sequence) != sequenceLength {
			return epitopesSource{}, fmt.Errorf(
				"inconsistent epitopes sequence length at file row %d: got=%d want=%d",
				fileRow,
				len(parsed.sequence),
				sequenceLength,
			)
		}
		rows = append(rows, parsed)
	}

	resolvedTableName := strings.TrimSpace(tableName)
	if resolvedTableName == "" {
		resolvedTableName = fmt.Sprintf("epitopes.csv.%s", filepath.Base(path))
	}
	table, err := buildEpitopesTable(resolvedTableName, rows)
	if err != nil {
		return epitopesSource{}, err
	}
	windows, err := buildEpitopesWindows(len(rows), bounds)
	if err != nil {
		return epitopesSource{}, err
	}
	return epitopesSource{tableName: table.name, table: table, windows: windows}, nil
}

func parseEpitopesCSVRow(record []string, fileRow int) (epitopesRow, bool, error) {
	fields := make([]string, 0, len(record))
	for _, field := range record {
		trimmed := strings.TrimSpace(field)
		if trimmed != "" {
			fields = append(fields, trimmed)
		}
	}
	if len(fields) == 0 {
		return epitopesRow{}, false, nil
	}
	if len(fields) < 3 {
		return epitopesRow{}, false, fmt.Errorf("epitopes csv row %d requires at least 3 columns", fileRow)
	}

	signal, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return epitopesRow{}, false, fmt.Errorf("parse epitopes signal row %d: %w", fileRow, err)
	}
	memory, err := strconv.ParseFloat(fields[1], 64)
	if err != nil {
		return epitopesRow{}, false, fmt.Errorf("parse epitopes memory row %d: %w", fileRow, err)
	}
	classification, err := parseEpitopesClassification(fields[2], fileRow)
	if err != nil {
		return epitopesRow{}, false, err
	}

	sequence := make([]int, 0, len(fields)-3)
	for i := 3; i < len(fields); i++ {
		residue, err := parseEpitopesResidue(fields[i], fileRow, i+1)
		if err != nil {
			return epitopesRow{}, false, err
		}
		sequence = append(sequence, residue)
	}

	return epitopesRow{
		sequence:       sequence,
		signal:         signal,
		memory:         memory,
		classification: classification,
	}, true, nil
}

func parseEpitopesClassification(raw string, fileRow int) (int, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "0", "false", "f", "neg", "negative", "n", "no":
		return 0, nil
	case "1", "true", "t", "pos", "positive", "y", "yes":
		return 1, nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("parse epitopes classification row %d: %w", fileRow, err)
	}
	if v > 0 {
		return 1, nil
	}
	return 0, nil
}

func parseEpitopesResidue(raw string, fileRow, column int) (int, error) {
	v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil {
		return 0, fmt.Errorf("parse epitopes residue row %d column %d: %w", fileRow, column, err)
	}
	residue := int(math.Round(v))
	if residue < 0 || residue >= epitopesAlphabetSize {
		return 0, fmt.Errorf(
			"epitopes residue out of range row %d column %d: %d",
			fileRow,
			column,
			residue,
		)
	}
	return residue, nil
}

func buildEpitopesTable(name string, rows []epitopesRow) (epitopesTable, error) {
	if len(rows) == 0 {
		return epitopesTable{}, fmt.Errorf("epitopes table %s has no data rows", name)
	}

	sequenceLength := len(rows[0].sequence)
	if sequenceLength <= 0 {
		return epitopesTable{}, fmt.Errorf("epitopes table %s has empty sequence rows", name)
	}

	total := len(rows)
	tableRows := make([]epitopesRow, total+1)
	for i, row := range rows {
		index := i + 1
		tableRows[index] = epitopesRow{
			sequence:       append([]int(nil), row.sequence...),
			signal:         row.signal,
			memory:         row.memory,
			classification: row.classification,
		}
	}

	tableName := strings.TrimSpace(name)
	if tableName == "" {
		tableName = "epitopes.custom"
	}
	return epitopesTable{
		name:           tableName,
		rows:           tableRows,
		modBase:        total + 1,
		sequenceLength: sequenceLength,
	}, nil
}

func buildEpitopesWindows(total int, bounds EpitopesTableBounds) (epitopesWindows, error) {
	if total <= 0 {
		return epitopesWindows{}, fmt.Errorf("epitopes table has no rows")
	}

	windows := epitopesWindows{}
	if bounds.GTStart == 0 && bounds.GTEnd == 0 {
		windows.gtStart = 1
		windows.gtEnd = minIntEpitopes(64, total)
	} else {
		windows.gtStart = maxIntEpitopes(1, bounds.GTStart)
		windows.gtEnd = bounds.GTEnd
		if windows.gtEnd == 0 {
			windows.gtEnd = total
		}
	}
	if err := validateEpitopesWindow("gt", windows.gtStart, windows.gtEnd, total); err != nil {
		return epitopesWindows{}, err
	}

	if bounds.ValidationStart == 0 && bounds.ValidationEnd == 0 {
		windows.validationStart = minIntEpitopes(total, windows.gtEnd+1)
		windows.validationEnd = minIntEpitopes(total, windows.validationStart+31)
	} else {
		windows.validationStart = bounds.ValidationStart
		if windows.validationStart == 0 {
			windows.validationStart = windows.gtEnd + 1
		}
		windows.validationEnd = bounds.ValidationEnd
		if windows.validationEnd == 0 {
			windows.validationEnd = total
		}
	}
	if windows.validationStart > total {
		windows.validationStart = total
	}
	if err := validateEpitopesWindow("validation", windows.validationStart, windows.validationEnd, total); err != nil {
		return epitopesWindows{}, err
	}

	if bounds.TestStart == 0 && bounds.TestEnd == 0 {
		windows.testStart = minIntEpitopes(total, windows.validationEnd+1)
		windows.testEnd = minIntEpitopes(total, windows.testStart+31)
	} else {
		windows.testStart = bounds.TestStart
		if windows.testStart == 0 {
			windows.testStart = windows.validationEnd + 1
		}
		windows.testEnd = bounds.TestEnd
		if windows.testEnd == 0 {
			windows.testEnd = total
		}
	}
	if windows.testStart > total {
		windows.testStart = total
	}
	if err := validateEpitopesWindow("test", windows.testStart, windows.testEnd, total); err != nil {
		return epitopesWindows{}, err
	}

	if bounds.BenchmarkStart == 0 && bounds.BenchmarkEnd == 0 {
		windows.benchmarkStart = maxIntEpitopes(1, total-279)
		windows.benchmarkEnd = total
	} else {
		windows.benchmarkStart = bounds.BenchmarkStart
		if windows.benchmarkStart == 0 {
			windows.benchmarkStart = 1
		}
		windows.benchmarkEnd = bounds.BenchmarkEnd
		if windows.benchmarkEnd == 0 {
			windows.benchmarkEnd = total
		}
	}
	if err := validateEpitopesWindow("benchmark", windows.benchmarkStart, windows.benchmarkEnd, total); err != nil {
		return epitopesWindows{}, err
	}

	return windows, nil
}

func validateEpitopesWindow(name string, start, end, total int) error {
	switch {
	case start <= 0:
		return fmt.Errorf("invalid epitopes %s start=%d", name, start)
	case start > total:
		return fmt.Errorf("invalid epitopes %s start=%d total=%d", name, start, total)
	case end < start:
		return fmt.Errorf("invalid epitopes %s range start=%d end=%d", name, start, end)
	case end > total:
		return fmt.Errorf("invalid epitopes %s end=%d total=%d", name, end, total)
	}
	return nil
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

type epitopesIOBindings struct {
	signal         protoio.ScalarSensorSetter
	memory         protoio.ScalarSensorSetter
	target         protoio.ScalarSensorSetter
	progress       protoio.ScalarSensorSetter
	margin         protoio.ScalarSensorSetter
	responseOutput protoio.SnapshotActuator
}

func epitopesIO(agent TickAgent) (epitopesIOBindings, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return epitopesIOBindings{}, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	signal, ok := typed.RegisteredSensor(protoio.EpitopesSignalSensorName)
	if !ok {
		return epitopesIOBindings{}, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.EpitopesSignalSensorName)
	}
	signalSetter, ok := signal.(protoio.ScalarSensorSetter)
	if !ok {
		return epitopesIOBindings{}, fmt.Errorf("sensor %s does not support scalar set", protoio.EpitopesSignalSensorName)
	}

	memory, ok := typed.RegisteredSensor(protoio.EpitopesMemorySensorName)
	if !ok {
		return epitopesIOBindings{}, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.EpitopesMemorySensorName)
	}
	memorySetter, ok := memory.(protoio.ScalarSensorSetter)
	if !ok {
		return epitopesIOBindings{}, fmt.Errorf("sensor %s does not support scalar set", protoio.EpitopesMemorySensorName)
	}
	targetSetter, err := optionalEpitopesSensorSetter(typed, protoio.EpitopesTargetSensorName)
	if err != nil {
		return epitopesIOBindings{}, err
	}
	progressSetter, err := optionalEpitopesSensorSetter(typed, protoio.EpitopesProgressSensorName)
	if err != nil {
		return epitopesIOBindings{}, err
	}
	marginSetter, err := optionalEpitopesSensorSetter(typed, protoio.EpitopesMarginSensorName)
	if err != nil {
		return epitopesIOBindings{}, err
	}

	actuator, ok := typed.RegisteredActuator(protoio.EpitopesResponseActuatorName)
	if !ok {
		return epitopesIOBindings{}, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.EpitopesResponseActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return epitopesIOBindings{}, fmt.Errorf("actuator %s does not support output snapshot", protoio.EpitopesResponseActuatorName)
	}

	return epitopesIOBindings{
		signal:         signalSetter,
		memory:         memorySetter,
		target:         targetSetter,
		progress:       progressSetter,
		margin:         marginSetter,
		responseOutput: output,
	}, nil
}

func optionalEpitopesSensorSetter(
	typed interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
	},
	sensorID string,
) (protoio.ScalarSensorSetter, error) {
	sensor, ok := typed.RegisteredSensor(sensorID)
	if !ok {
		return nil, nil
	}
	setter, ok := sensor.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, fmt.Errorf("sensor %s does not support scalar set", sensorID)
	}
	return setter, nil
}

func epitopesSessionProgress(index, start, end int) float64 {
	denom := maxIntEpitopes(1, end-start)
	progress := float64(index-start) / float64(denom)
	return clampEpitopes(progress, 0, 1)
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

func minIntEpitopes(a, b int) int {
	if a < b {
		return a
	}
	return b
}
