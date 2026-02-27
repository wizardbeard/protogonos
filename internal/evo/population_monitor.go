package evo

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"sync"

	"protogonos/internal/agent"
	"protogonos/internal/genotype"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/morphology"
	"protogonos/internal/scape"
	"protogonos/internal/substrate"
	"protogonos/internal/tuning"
)

type ScoredGenome struct {
	Genome  model.Genome
	Fitness float64
	Trace   scape.Trace
}

type RunResult struct {
	BestByGeneration      []float64
	GenerationDiagnostics []GenerationDiagnostics
	SpeciesHistory        []SpeciesGeneration
	FinalPopulation       []ScoredGenome
	Lineage               []LineageRecord
}

type SpeciesGeneration struct {
	Generation     int              `json:"generation"`
	Species        []SpeciesMetrics `json:"species"`
	NewSpecies     []string         `json:"new_species,omitempty"`
	ExtinctSpecies []string         `json:"extinct_species,omitempty"`
}

type SpeciesMetrics struct {
	Key         string  `json:"key"`
	Size        int     `json:"size"`
	MeanFitness float64 `json:"mean_fitness"`
	BestFitness float64 `json:"best_fitness"`
}

type GenerationDiagnostics struct {
	Generation            int     `json:"generation"`
	BestFitness           float64 `json:"best_fitness"`
	MeanFitness           float64 `json:"mean_fitness"`
	MinFitness            float64 `json:"min_fitness"`
	SpeciesCount          int     `json:"species_count"`
	FingerprintDiversity  int     `json:"fingerprint_diversity"`
	SpeciationThreshold   float64 `json:"speciation_threshold"`
	TargetSpeciesCount    int     `json:"target_species_count"`
	MeanSpeciesSize       float64 `json:"mean_species_size"`
	LargestSpeciesSize    int     `json:"largest_species_size"`
	TuningInvocations     int     `json:"tuning_invocations"`
	TuningAttempts        int     `json:"tuning_attempts"`
	TuningEvaluations     int     `json:"tuning_evaluations"`
	TuningAccepted        int     `json:"tuning_accepted"`
	TuningRejected        int     `json:"tuning_rejected"`
	TuningGoalHits        int     `json:"tuning_goal_hits"`
	TuningAcceptRate      float64 `json:"tuning_accept_rate"`
	TuningEvalsPerAttempt float64 `json:"tuning_evals_per_attempt"`
}

type TraceUpdateReason string

const (
	TraceUpdateReasonStep      TraceUpdateReason = "step"
	TraceUpdateReasonPrint     TraceUpdateReason = "print_trace"
	TraceUpdateReasonCompleted TraceUpdateReason = "completed"
)

type TraceUpdate struct {
	Reason             TraceUpdateReason     `json:"reason"`
	TotalEvaluations   int                   `json:"total_evaluations"`
	GoalReached        bool                  `json:"goal_reached"`
	StepEvaluations    int                   `json:"step_evaluations,omitempty"`
	StepCycles         float64               `json:"step_cycles,omitempty"`
	StepTime           float64               `json:"step_time,omitempty"`
	SpeciesEvaluations map[string]int        `json:"species_evaluations,omitempty"`
	Species            []TraceSpeciesMetrics `json:"species,omitempty"`
	Diagnostics        GenerationDiagnostics `json:"diagnostics"`
}

type TraceSpeciesMetrics struct {
	Key               string   `json:"key"`
	Size              int      `json:"size"`
	MeanFitness       float64  `json:"mean_fitness"`
	StdFitness        float64  `json:"std_fitness,omitempty"`
	BestFitness       float64  `json:"best_fitness"`
	MinFitness        float64  `json:"min_fitness,omitempty"`
	AvgNeurons        float64  `json:"avg_neurons,omitempty"`
	StdNeurons        float64  `json:"std_neurons,omitempty"`
	Diversity         int      `json:"diversity,omitempty"`
	Evaluations       int      `json:"evaluations,omitempty"`
	ChampionGenomeID  string   `json:"champion_genome_id,omitempty"`
	ValidationFitness *float64 `json:"validation_fitness,omitempty"`
	TestFitness       *float64 `json:"test_fitness,omitempty"`
}

type LineageRecord struct {
	GenomeID    string                     `json:"genome_id"`
	ParentID    string                     `json:"parent_id"`
	Generation  int                        `json:"generation"`
	Operation   string                     `json:"operation"`
	Events      []genotype.EvoHistoryEvent `json:"events,omitempty"`
	Fingerprint string                     `json:"fingerprint,omitempty"`
	Summary     TopologySummary            `json:"summary,omitempty"`
}

type MonitorConfig struct {
	Scape                scape.Scape
	OpMode               string
	EvolutionType        string
	SpeciationMode       string
	Mutation             Operator
	MutationPolicy       []WeightedMutation
	Selector             Selector
	Postprocessor        FitnessPostprocessor
	TopologicalMutations TopologicalMutationPolicy
	PopulationSize       int
	EliteCount           int
	SurvivalPercentage   float64
	SpecieSizeLimit      int
	Generations          int
	GenerationOffset     int
	FitnessGoal          float64
	EvaluationsLimit     int
	Workers              int
	Seed                 int64
	InputNeuronIDs       []string
	OutputNeuronIDs      []string
	Tuner                tuning.Tuner
	TuneAttempts         int
	TuneAttemptPolicy    tuning.AttemptPolicy
	ValidationProbe      bool
	TestProbe            bool
	Control              <-chan MonitorCommand
	TraceStepSize        int
	TraceUpdateHook      func(TraceUpdate)
}

type PopulationMonitor struct {
	cfg                    MonitorConfig
	rng                    *rand.Rand
	speciation             *AdaptiveSpeciation
	paused                 bool
	stopRequested          bool
	goalReached            bool
	totalEvaluations       int
	nextTraceEvaluation    int
	stepEvaluations        int
	stepCycles             float64
	stepTime               float64
	stepSpeciesEvaluations map[string]int
	lastTraceSpecies       []TraceSpeciesMetrics
	lastDiagnostics        GenerationDiagnostics
	hasDiagnostics         bool
}

type goalAwareTuner interface {
	SetGoalFitness(goal float64)
}

type tuningGenerationStats struct {
	Invocations int
	Attempts    int
	Evaluations int
	Accepted    int
	Rejected    int
	GoalHits    int
}

type MonitorCommand string

const (
	CommandPause       MonitorCommand = "pause"
	CommandContinue    MonitorCommand = "continue"
	CommandStop        MonitorCommand = "stop"
	CommandGoalReached MonitorCommand = "goal_reached"
	CommandPrintTrace  MonitorCommand = "print_trace"
)

const (
	OpModeGT         = "gt"
	OpModeValidation = "validation"
	OpModeTest       = "test"
)

const (
	EvolutionTypeGenerational = "generational"
	EvolutionTypeSteadyState  = "steady_state"
)

const (
	SpeciationModeAdaptive    = "adaptive"
	SpeciationModeFingerprint = "fingerprint"
)

const defaultTraceStepSize = 500

type noOpMutation struct{}

func (noOpMutation) Name() string { return "noop" }

func (noOpMutation) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	return genome, nil
}

func NewPopulationMonitor(cfg MonitorConfig) (*PopulationMonitor, error) {
	if cfg.Scape == nil {
		return nil, fmt.Errorf("scape is required")
	}
	if cfg.OpMode == "" {
		cfg.OpMode = OpModeGT
	}
	switch cfg.OpMode {
	case OpModeGT, OpModeValidation, OpModeTest:
	default:
		return nil, fmt.Errorf("unsupported op mode: %s", cfg.OpMode)
	}
	if cfg.EvolutionType == "" {
		cfg.EvolutionType = EvolutionTypeGenerational
	}
	switch cfg.EvolutionType {
	case EvolutionTypeGenerational, EvolutionTypeSteadyState:
	default:
		return nil, fmt.Errorf("unsupported evolution type: %s", cfg.EvolutionType)
	}

	if cfg.OpMode == OpModeGT && cfg.Mutation == nil && len(cfg.MutationPolicy) == 0 {
		return nil, fmt.Errorf("mutation operator or policy is required")
	}
	if cfg.OpMode != OpModeGT && cfg.Mutation == nil {
		cfg.Mutation = noOpMutation{}
	}
	positivePolicyWeight := false
	for i, item := range cfg.MutationPolicy {
		if item.Operator == nil {
			return nil, fmt.Errorf("mutation policy operator is required at index %d", i)
		}
		if item.Weight < 0 {
			return nil, fmt.Errorf("mutation policy weight must be >= 0 at index %d", i)
		}
		if item.Weight > 0 {
			positivePolicyWeight = true
		}
	}
	if len(cfg.MutationPolicy) > 0 && !positivePolicyWeight {
		return nil, fmt.Errorf("mutation policy requires at least one positive weight")
	}
	if cfg.PopulationSize <= 0 {
		return nil, fmt.Errorf("population size must be > 0")
	}
	if cfg.SurvivalPercentage < 0 || cfg.SurvivalPercentage > 1 {
		return nil, fmt.Errorf("survival percentage must be in [0, 1]")
	}
	if cfg.EliteCount <= 0 {
		if cfg.SurvivalPercentage > 0 {
			cfg.EliteCount = int(math.Ceil(float64(cfg.PopulationSize) * cfg.SurvivalPercentage))
			if cfg.EliteCount < 1 {
				cfg.EliteCount = 1
			}
		}
	}
	if cfg.EliteCount <= 0 || cfg.EliteCount > cfg.PopulationSize {
		return nil, fmt.Errorf("elite count must be in [1, population size]")
	}
	if cfg.Generations <= 0 {
		return nil, fmt.Errorf("generations must be > 0")
	}
	if cfg.GenerationOffset < 0 {
		return nil, fmt.Errorf("generation offset must be >= 0")
	}
	if cfg.SpecieSizeLimit < 0 {
		return nil, fmt.Errorf("specie size limit must be >= 0")
	}
	if cfg.EvaluationsLimit < 0 {
		return nil, fmt.Errorf("evaluations limit must be >= 0")
	}
	if cfg.TraceStepSize < 0 {
		return nil, fmt.Errorf("trace step size must be >= 0")
	}
	if cfg.TraceStepSize == 0 {
		cfg.TraceStepSize = defaultTraceStepSize
	}
	if cfg.SpeciationMode == "" {
		cfg.SpeciationMode = SpeciationModeAdaptive
	}
	switch cfg.SpeciationMode {
	case SpeciationModeAdaptive, SpeciationModeFingerprint:
	default:
		return nil, fmt.Errorf("unsupported speciation mode: %s", cfg.SpeciationMode)
	}
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}
	if len(cfg.InputNeuronIDs) == 0 {
		return nil, fmt.Errorf("input neuron ids are required")
	}
	if len(cfg.OutputNeuronIDs) == 0 {
		return nil, fmt.Errorf("output neuron ids are required")
	}
	if cfg.Tuner != nil && cfg.TuneAttempts < 0 {
		return nil, fmt.Errorf("tune attempts must be >= 0")
	}
	if cfg.Tuner != nil && cfg.TuneAttemptPolicy == nil {
		cfg.TuneAttemptPolicy = tuning.FixedAttemptPolicy{}
	}
	if cfg.Tuner != nil && cfg.FitnessGoal > 0 {
		if tuner, ok := cfg.Tuner.(goalAwareTuner); ok {
			tuner.SetGoalFitness(cfg.FitnessGoal)
		}
	}
	if cfg.Selector == nil {
		cfg.Selector = EliteSelector{}
	}
	if cfg.Postprocessor == nil {
		cfg.Postprocessor = NoopFitnessPostprocessor{}
	}
	if cfg.TopologicalMutations == nil {
		cfg.TopologicalMutations = ConstTopologicalMutations{Count: 1}
	}

	var adaptiveSpeciation *AdaptiveSpeciation
	if cfg.SpeciationMode == SpeciationModeAdaptive {
		adaptiveSpeciation = NewAdaptiveSpeciation(cfg.PopulationSize)
	}

	return &PopulationMonitor{
		cfg:        cfg,
		rng:        rand.New(rand.NewSource(cfg.Seed)),
		speciation: adaptiveSpeciation,
	}, nil
}

func (m *PopulationMonitor) Run(ctx context.Context, initial []model.Genome) (RunResult, error) {
	if len(initial) != m.cfg.PopulationSize {
		return RunResult{}, fmt.Errorf("initial population mismatch: got=%d want=%d", len(initial), m.cfg.PopulationSize)
	}
	m.resetRunState()
	if m.cfg.EvolutionType == EvolutionTypeSteadyState {
		return m.runSteadyState(ctx, initial)
	}

	population := make([]model.Genome, len(initial))
	copy(population, initial)

	bestHistory := make([]float64, 0, m.cfg.Generations)
	diagnostics := make([]GenerationDiagnostics, 0, m.cfg.Generations)
	speciesHistory := make([]SpeciesGeneration, 0, m.cfg.Generations)
	lineage := make([]LineageRecord, 0, len(initial)*(m.cfg.Generations+1))
	prevSpeciesSet := map[string]struct{}{}
	evoHistoryByGenomeID := initializeEvoHistoryByGenomeID(population)
	for _, genome := range population {
		sig := ComputeGenomeSignature(genome)
		operation := "seed"
		if m.cfg.GenerationOffset > 0 {
			operation = "continue_seed"
		}
		lineage = append(lineage, LineageRecord{
			GenomeID:    genome.ID,
			ParentID:    "",
			Generation:  m.cfg.GenerationOffset,
			Operation:   operation,
			Fingerprint: sig.Fingerprint,
			Summary:     sig.Summary,
		})
	}
	var scored []ScoredGenome

	for gen := 0; gen < m.cfg.Generations; gen++ {
		if err := ctx.Err(); err != nil {
			return RunResult{}, err
		}
		if m.stopRequested {
			break
		}
		stop, err := m.applyControl(ctx, false)
		if err != nil {
			return RunResult{}, err
		}
		if stop {
			break
		}

		logicalGeneration := m.cfg.GenerationOffset + gen
		var tuningStats tuningGenerationStats
		var countedEvaluations []bool
		scored, tuningStats, countedEvaluations, err = m.evaluatePopulation(ctx, population, logicalGeneration)
		if err != nil {
			return RunResult{}, err
		}
		if m.cfg.OpMode == OpModeGT {
			scored = m.cfg.Postprocessor.Process(scored)
		}

		sort.Slice(scored, func(i, j int) bool {
			return scored[i].Fitness > scored[j].Fitness
		})
		m.totalEvaluations += countTrue(countedEvaluations)
		bestHistory = append(bestHistory, scored[0].Fitness)
		speciesByGenomeID, speciationStats := m.assignSpecies(scored, evoHistoryByGenomeID)
		generationDiagnostics := summarizeGeneration(scored, logicalGeneration+1, speciationStats, tuningStats)
		diagnostics = append(diagnostics, generationDiagnostics)
		m.recordGenerationDiagnostics(generationDiagnostics)
		m.accumulateStepWindow(scored, speciesByGenomeID, countedEvaluations)
		if err := m.captureTraceSpecies(ctx, scored, speciesByGenomeID); err != nil {
			return RunResult{}, err
		}
		m.emitStepTraceUpdates()
		history, currentSet := summarizeSpeciesGeneration(scored, speciesByGenomeID, logicalGeneration+1, prevSpeciesSet)
		speciesHistory = append(speciesHistory, history)
		prevSpeciesSet = currentSet
		if m.cfg.OpMode != OpModeGT {
			break
		}
		if m.stopRequested {
			break
		}
		if (m.cfg.FitnessGoal > 0 && scored[0].Fitness >= m.cfg.FitnessGoal) ||
			(m.cfg.EvaluationsLimit > 0 && m.totalEvaluations >= m.cfg.EvaluationsLimit) ||
			m.goalReached {
			break
		}
		stop, err = m.applyControl(ctx, true)
		if err != nil {
			return RunResult{}, err
		}
		if stop {
			break
		}

		var generationLineage []LineageRecord
		population, generationLineage, err = m.nextGeneration(ctx, scored, speciesByGenomeID, logicalGeneration)
		if err != nil {
			return RunResult{}, err
		}
		lineage = append(lineage, generationLineage...)
		evoHistoryByGenomeID = evolveHistoryByGenomeID(population, generationLineage, evoHistoryByGenomeID)
	}

	result := RunResult{
		BestByGeneration:      bestHistory,
		GenerationDiagnostics: diagnostics,
		SpeciesHistory:        speciesHistory,
		FinalPopulation:       scored,
		Lineage:               lineage,
	}
	m.emitTraceUpdate(TraceUpdateReasonCompleted, m.totalEvaluations)
	return result, nil
}

func (m *PopulationMonitor) runSteadyState(ctx context.Context, initial []model.Genome) (RunResult, error) {
	population := make([]model.Genome, len(initial))
	copy(population, initial)

	bestHistory := make([]float64, 0, m.cfg.Generations)
	diagnostics := make([]GenerationDiagnostics, 0, m.cfg.Generations)
	speciesHistory := make([]SpeciesGeneration, 0, m.cfg.Generations)
	lineage := make([]LineageRecord, 0, len(initial)*(m.cfg.Generations+1))
	prevSpeciesSet := map[string]struct{}{}
	evoHistoryByGenomeID := initializeEvoHistoryByGenomeID(population)
	for _, genome := range population {
		sig := ComputeGenomeSignature(genome)
		operation := "seed"
		if m.cfg.GenerationOffset > 0 {
			operation = "continue_seed"
		}
		lineage = append(lineage, LineageRecord{
			GenomeID:    genome.ID,
			ParentID:    "",
			Generation:  m.cfg.GenerationOffset,
			Operation:   operation,
			Fingerprint: sig.Fingerprint,
			Summary:     sig.Summary,
		})
	}

	var finalScored []ScoredGenome

	for gen := 0; gen < m.cfg.Generations; gen++ {
		if err := ctx.Err(); err != nil {
			return RunResult{}, err
		}
		if m.stopRequested {
			break
		}
		stop, err := m.applyControl(ctx, false)
		if err != nil {
			return RunResult{}, err
		}
		if stop {
			break
		}

		logicalGeneration := m.cfg.GenerationOffset + gen
		scored, tuningStats, countedEvaluations, err := m.evaluatePopulation(ctx, population, logicalGeneration)
		if err != nil {
			return RunResult{}, err
		}
		if m.cfg.OpMode == OpModeGT {
			scored = m.cfg.Postprocessor.Process(scored)
		}

		ranked := append([]ScoredGenome(nil), scored...)
		sort.Slice(ranked, func(i, j int) bool {
			return ranked[i].Fitness > ranked[j].Fitness
		})
		finalScored = ranked
		m.totalEvaluations += countTrue(countedEvaluations)
		bestHistory = append(bestHistory, ranked[0].Fitness)
		speciesByGenomeID, speciationStats := m.assignSpecies(ranked, evoHistoryByGenomeID)
		generationDiagnostics := summarizeGeneration(ranked, logicalGeneration+1, speciationStats, tuningStats)
		diagnostics = append(diagnostics, generationDiagnostics)
		m.recordGenerationDiagnostics(generationDiagnostics)
		m.accumulateStepWindow(ranked, speciesByGenomeID, countedEvaluations)
		if err := m.captureTraceSpecies(ctx, ranked, speciesByGenomeID); err != nil {
			return RunResult{}, err
		}
		m.emitStepTraceUpdates()
		history, currentSet := summarizeSpeciesGeneration(ranked, speciesByGenomeID, logicalGeneration+1, prevSpeciesSet)
		speciesHistory = append(speciesHistory, history)
		prevSpeciesSet = currentSet

		if m.cfg.OpMode != OpModeGT {
			break
		}
		if m.stopRequested {
			break
		}
		if (m.cfg.FitnessGoal > 0 && ranked[0].Fitness >= m.cfg.FitnessGoal) ||
			(m.cfg.EvaluationsLimit > 0 && m.totalEvaluations >= m.cfg.EvaluationsLimit) ||
			m.goalReached {
			break
		}
		stop, err = m.applyControl(ctx, true)
		if err != nil {
			return RunResult{}, err
		}
		if stop {
			break
		}

		nextPopulation, generationLineage, err := m.nextSteadyStatePopulation(ctx, ranked, speciesByGenomeID, logicalGeneration)
		if err != nil {
			return RunResult{}, err
		}
		population = nextPopulation
		lineage = append(lineage, generationLineage...)
		evoHistoryByGenomeID = evolveHistoryByGenomeID(population, generationLineage, evoHistoryByGenomeID)
	}

	result := RunResult{
		BestByGeneration:      bestHistory,
		GenerationDiagnostics: diagnostics,
		SpeciesHistory:        speciesHistory,
		FinalPopulation:       finalScored,
		Lineage:               lineage,
	}
	m.emitTraceUpdate(TraceUpdateReasonCompleted, m.totalEvaluations)
	return result, nil
}

func (m *PopulationMonitor) nextSteadyStatePopulation(
	ctx context.Context,
	ranked []ScoredGenome,
	speciesByGenomeID map[string]string,
	generation int,
) ([]model.Genome, []LineageRecord, error) {
	if len(ranked) == 0 {
		return nil, nil, fmt.Errorf("steady-state population is empty")
	}
	parentPool := ranked
	if m.cfg.SpecieSizeLimit > 0 {
		parentPool = limitSpeciesParentPool(ranked, speciesByGenomeID, m.cfg.SpecieSizeLimit)
		if len(parentPool) == 0 {
			parentPool = ranked
		}
	}

	next := make([]model.Genome, len(ranked))
	for i, item := range ranked {
		next[i] = item.Genome
	}

	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	// Reference steady-state semantics replace one terminated agent at a time.
	replacementIndex := m.rng.Intn(len(ranked))
	replaced := ranked[replacementIndex]
	speciesKey := speciesByGenomeID[replaced.Genome.ID]
	speciesRanked := filterRankedBySpecies(parentPool, speciesByGenomeID, speciesKey)
	if len(speciesRanked) == 0 {
		speciesRanked = parentPool
	}
	parent, err := m.pickParentForSpecies(parentPool, speciesRanked, speciesByGenomeID, generation)
	if err != nil {
		return nil, nil, err
	}
	child, record, err := m.mutateFromParent(ctx, parent, generation, replacementIndex)
	if err != nil {
		return nil, nil, err
	}
	next[replacementIndex] = child
	lineage := []LineageRecord{record}

	return next, lineage, nil
}

func (m *PopulationMonitor) applyControl(ctx context.Context, waitIfPaused bool) (bool, error) {
	if m.stopRequested {
		return true, nil
	}
	if m.cfg.Control == nil {
		return false, nil
	}
	for {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case cmd, ok := <-m.cfg.Control:
			if !ok {
				return false, nil
			}
			action := m.handleCommand(cmd)
			if action.printTrace {
				m.emitTraceUpdate(TraceUpdateReasonPrint, m.totalEvaluations)
			}
			if action.stop {
				return true, nil
			}
		default:
			if !m.paused || !waitIfPaused {
				return false, nil
			}
			select {
			case <-ctx.Done():
				return false, ctx.Err()
			case cmd, ok := <-m.cfg.Control:
				if !ok {
					return false, nil
				}
				action := m.handleCommand(cmd)
				if action.printTrace {
					m.emitTraceUpdate(TraceUpdateReasonPrint, m.totalEvaluations)
				}
				if action.stop {
					return true, nil
				}
				if !m.paused {
					return false, nil
				}
			}
		}
	}
}

type monitorCommandAction struct {
	stop       bool
	printTrace bool
}

func (m *PopulationMonitor) handleCommand(cmd MonitorCommand) monitorCommandAction {
	switch cmd {
	case CommandPause:
		m.paused = true
	case CommandContinue:
		m.paused = false
	case CommandStop:
		m.stopRequested = true
		return monitorCommandAction{stop: true}
	case CommandGoalReached:
		m.goalReached = true
		// Avoid deadlock when goal is reached while the monitor is paused.
		m.paused = false
	case CommandPrintTrace:
		return monitorCommandAction{printTrace: true}
	}
	return monitorCommandAction{}
}

func (m *PopulationMonitor) resetRunState() {
	m.paused = false
	m.stopRequested = false
	m.goalReached = false
	m.totalEvaluations = 0
	m.resetStepWindow()
	m.lastTraceSpecies = nil
	m.lastDiagnostics = GenerationDiagnostics{}
	m.hasDiagnostics = false
	m.nextTraceEvaluation = m.cfg.TraceStepSize
}

func (m *PopulationMonitor) recordGenerationDiagnostics(diag GenerationDiagnostics) {
	m.lastDiagnostics = diag
	m.hasDiagnostics = true
}

func (m *PopulationMonitor) emitStepTraceUpdates() {
	if m.cfg.TraceUpdateHook == nil || m.cfg.TraceStepSize <= 0 {
		return
	}
	if m.totalEvaluations < m.nextTraceEvaluation {
		return
	}
	m.emitTraceUpdate(TraceUpdateReasonStep, m.nextTraceEvaluation)
	m.nextTraceEvaluation += m.cfg.TraceStepSize
	m.resetStepWindow()
}

func (m *PopulationMonitor) resetStepWindow() {
	m.stepEvaluations = 0
	m.stepCycles = 0
	m.stepTime = 0
	m.stepSpeciesEvaluations = make(map[string]int)
}

func (m *PopulationMonitor) accumulateStepWindow(scored []ScoredGenome, speciesByGenomeID map[string]string, counted []bool) {
	for i, item := range scored {
		if counted != nil {
			if i >= len(counted) || !counted[i] {
				continue
			}
		}
		m.stepEvaluations++
		m.stepCycles += traceCycleMetric(item.Trace)
		m.stepTime += traceTimeMetric(item.Trace)
		speciesKey := speciesByGenomeID[item.Genome.ID]
		if speciesKey == "" {
			speciesKey = "species:unknown"
		}
		m.stepSpeciesEvaluations[speciesKey]++
	}
}

func (m *PopulationMonitor) captureTraceSpecies(ctx context.Context, scored []ScoredGenome, speciesByGenomeID map[string]string) error {
	type aggregate struct {
		size             int
		sum              float64
		sumSquares       float64
		best             float64
		min              float64
		neuronSum        float64
		neuronSumSquares float64
		fingerprints     map[string]struct{}
		champion         model.Genome
	}
	bySpecies := make(map[string]*aggregate)
	for _, item := range scored {
		key := speciesByGenomeID[item.Genome.ID]
		if key == "" {
			key = "species:unknown"
		}
		bucket := bySpecies[key]
		if bucket == nil {
			bucket = &aggregate{
				best:         item.Fitness,
				min:          item.Fitness,
				fingerprints: map[string]struct{}{},
				champion:     item.Genome,
			}
			bySpecies[key] = bucket
		}
		bucket.size++
		bucket.sum += item.Fitness
		bucket.sumSquares += item.Fitness * item.Fitness
		neurons := float64(len(item.Genome.Neurons))
		bucket.neuronSum += neurons
		bucket.neuronSumSquares += neurons * neurons
		fingerprint := ComputeGenomeSignature(item.Genome).Fingerprint
		bucket.fingerprints[fingerprint] = struct{}{}
		if item.Fitness > bucket.best {
			bucket.best = item.Fitness
			bucket.champion = item.Genome
		}
		if item.Fitness < bucket.min {
			bucket.min = item.Fitness
		}
	}

	keys := make([]string, 0, len(bySpecies))
	for key := range bySpecies {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	out := make([]TraceSpeciesMetrics, 0, len(keys))
	for _, key := range keys {
		bucket := bySpecies[key]
		meanFitness := bucket.sum / float64(bucket.size)
		entry := TraceSpeciesMetrics{
			Key:              key,
			Size:             bucket.size,
			MeanFitness:      meanFitness,
			StdFitness:       stdFromSums(bucket.sum, bucket.sumSquares, bucket.size),
			BestFitness:      bucket.best,
			MinFitness:       bucket.min,
			AvgNeurons:       bucket.neuronSum / float64(bucket.size),
			StdNeurons:       stdFromSums(bucket.neuronSum, bucket.neuronSumSquares, bucket.size),
			Diversity:        len(bucket.fingerprints),
			Evaluations:      m.stepSpeciesEvaluations[key],
			ChampionGenomeID: bucket.champion.ID,
		}
		if m.cfg.OpMode == OpModeGT && (m.cfg.ValidationProbe || m.cfg.TestProbe) {
			runValidation := m.cfg.ValidationProbe
			runTest := m.cfg.TestProbe || runValidation
			if m.cfg.ValidationProbe {
				fitness, _, err := m.evaluateGenome(ctx, bucket.champion, OpModeValidation)
				if err != nil {
					return fmt.Errorf("validation probe for species %s champion %s: %w", key, bucket.champion.ID, err)
				}
				val := fitness
				entry.ValidationFitness = &val
			}
			if runTest {
				fitness, _, err := m.evaluateGenome(ctx, bucket.champion, OpModeTest)
				if err != nil {
					return fmt.Errorf("test probe for species %s champion %s: %w", key, bucket.champion.ID, err)
				}
				val := fitness
				entry.TestFitness = &val
			}
			if !runValidation {
				entry.ValidationFitness = nil
			}
		}
		out = append(out, entry)
	}
	m.lastTraceSpecies = out
	return nil
}

func (m *PopulationMonitor) emitTraceUpdate(reason TraceUpdateReason, totalEvaluations int) {
	if m.cfg.TraceUpdateHook == nil {
		return
	}
	update := TraceUpdate{
		Reason:           reason,
		TotalEvaluations: totalEvaluations,
		GoalReached:      m.goalReached,
		StepEvaluations:  m.stepEvaluations,
		StepCycles:       m.stepCycles,
		StepTime:         m.stepTime,
	}
	if len(m.stepSpeciesEvaluations) > 0 {
		update.SpeciesEvaluations = cloneSpeciesEvaluationCounts(m.stepSpeciesEvaluations)
	}
	if len(m.lastTraceSpecies) > 0 {
		update.Species = cloneTraceSpeciesMetrics(m.lastTraceSpecies)
	}
	if m.hasDiagnostics {
		update.Diagnostics = m.lastDiagnostics
	}
	m.cfg.TraceUpdateHook(update)
}

func cloneSpeciesEvaluationCounts(in map[string]int) map[string]int {
	out := make(map[string]int, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func cloneTraceSpeciesMetrics(in []TraceSpeciesMetrics) []TraceSpeciesMetrics {
	out := make([]TraceSpeciesMetrics, len(in))
	for i, item := range in {
		out[i] = item
		if item.ValidationFitness != nil {
			val := *item.ValidationFitness
			out[i].ValidationFitness = &val
		}
		if item.TestFitness != nil {
			val := *item.TestFitness
			out[i].TestFitness = &val
		}
	}
	return out
}

func traceCycleMetric(trace scape.Trace) float64 {
	if v, ok := traceNumber(trace, "cycles", "cycle", "steps_survived", "steps"); ok {
		return v
	}
	return 0
}

func traceTimeMetric(trace scape.Trace) float64 {
	if v, ok := traceNumber(trace, "time", "duration", "elapsed", "time_ms", "duration_ms", "elapsed_ms"); ok {
		return v
	}
	return 0
}

func traceNumber(trace scape.Trace, keys ...string) (float64, bool) {
	if trace == nil {
		return 0, false
	}
	for _, key := range keys {
		raw, ok := trace[key]
		if !ok {
			continue
		}
		switch value := raw.(type) {
		case int:
			return float64(value), true
		case int8:
			return float64(value), true
		case int16:
			return float64(value), true
		case int32:
			return float64(value), true
		case int64:
			return float64(value), true
		case uint:
			return float64(value), true
		case uint8:
			return float64(value), true
		case uint16:
			return float64(value), true
		case uint32:
			return float64(value), true
		case uint64:
			return float64(value), true
		case float32:
			return float64(value), true
		case float64:
			return value, true
		}
	}
	return 0, false
}

func summarizeGeneration(scored []ScoredGenome, generation int, speciationStats SpeciationStats, tuningStats tuningGenerationStats) GenerationDiagnostics {
	acceptRate, evalsPerAttempt := tuningRatios(tuningStats)
	if len(scored) == 0 {
		return GenerationDiagnostics{
			Generation:            generation,
			TuningInvocations:     tuningStats.Invocations,
			TuningAttempts:        tuningStats.Attempts,
			TuningEvaluations:     tuningStats.Evaluations,
			TuningAccepted:        tuningStats.Accepted,
			TuningRejected:        tuningStats.Rejected,
			TuningGoalHits:        tuningStats.GoalHits,
			TuningAcceptRate:      acceptRate,
			TuningEvalsPerAttempt: evalsPerAttempt,
		}
	}

	total := 0.0
	minFitness := scored[0].Fitness
	fingerprints := make(map[string]struct{}, len(scored))
	for _, item := range scored {
		total += item.Fitness
		if item.Fitness < minFitness {
			minFitness = item.Fitness
		}
		fingerprint := ComputeGenomeSignature(item.Genome).Fingerprint
		fingerprints[fingerprint] = struct{}{}
	}

	return GenerationDiagnostics{
		Generation:            generation,
		BestFitness:           scored[0].Fitness,
		MeanFitness:           total / float64(len(scored)),
		MinFitness:            minFitness,
		SpeciesCount:          speciationStats.SpeciesCount,
		FingerprintDiversity:  len(fingerprints),
		SpeciationThreshold:   speciationStats.Threshold,
		TargetSpeciesCount:    speciationStats.TargetSpeciesCount,
		MeanSpeciesSize:       speciationStats.MeanSpeciesSize,
		LargestSpeciesSize:    speciationStats.LargestSpeciesSize,
		TuningInvocations:     tuningStats.Invocations,
		TuningAttempts:        tuningStats.Attempts,
		TuningEvaluations:     tuningStats.Evaluations,
		TuningAccepted:        tuningStats.Accepted,
		TuningRejected:        tuningStats.Rejected,
		TuningGoalHits:        tuningStats.GoalHits,
		TuningAcceptRate:      acceptRate,
		TuningEvalsPerAttempt: evalsPerAttempt,
	}
}

func tuningRatios(stats tuningGenerationStats) (float64, float64) {
	acceptRate := 0.0
	totalDecisions := stats.Accepted + stats.Rejected
	if totalDecisions > 0 {
		acceptRate = float64(stats.Accepted) / float64(totalDecisions)
	}
	evalsPerAttempt := 0.0
	if stats.Attempts > 0 {
		evalsPerAttempt = float64(stats.Evaluations) / float64(stats.Attempts)
	}
	return acceptRate, evalsPerAttempt
}

func stdFromSums(sum, sumSquares float64, count int) float64 {
	if count <= 1 {
		return 0
	}
	mean := sum / float64(count)
	variance := (sumSquares / float64(count)) - mean*mean
	if variance < 0 {
		variance = 0
	}
	return math.Sqrt(variance)
}

func countTrue(values []bool) int {
	total := 0
	for _, value := range values {
		if value {
			total++
		}
	}
	return total
}

func (m *PopulationMonitor) assignSpecies(scored []ScoredGenome, evoHistoryByGenomeID map[string][]genotype.EvoHistoryEvent) (map[string]string, SpeciationStats) {
	genomes := make([]model.Genome, 0, len(scored))
	for _, item := range scored {
		genomes = append(genomes, item.Genome)
	}
	var (
		bySpecies map[string][]model.Genome
		stats     SpeciationStats
	)
	switch m.cfg.SpeciationMode {
	case SpeciationModeFingerprint:
		bySpecies = genotype.SpeciateByFingerprintWithHistory(genomes, evoHistoryByGenomeID)
		stats = summarizeStaticSpeciation(bySpecies)
	default:
		if m.speciation == nil {
			m.speciation = NewAdaptiveSpeciation(m.cfg.PopulationSize)
		}
		bySpecies, stats = m.speciation.Assign(genomes)
	}
	speciesByGenomeID := make(map[string]string, len(scored))
	for key, members := range bySpecies {
		for _, genome := range members {
			speciesByGenomeID[genome.ID] = key
		}
	}
	return speciesByGenomeID, stats
}

func initializeEvoHistoryByGenomeID(population []model.Genome) map[string][]genotype.EvoHistoryEvent {
	out := make(map[string][]genotype.EvoHistoryEvent, len(population))
	for _, genome := range population {
		id := strings.TrimSpace(genome.ID)
		if id == "" {
			continue
		}
		out[id] = nil
	}
	return out
}

func evolveHistoryByGenomeID(
	nextPopulation []model.Genome,
	generationLineage []LineageRecord,
	previous map[string][]genotype.EvoHistoryEvent,
) map[string][]genotype.EvoHistoryEvent {
	next := make(map[string][]genotype.EvoHistoryEvent, len(nextPopulation))
	lineageByGenomeID := make(map[string]LineageRecord, len(generationLineage))
	for _, record := range generationLineage {
		genomeID := strings.TrimSpace(record.GenomeID)
		if genomeID == "" {
			continue
		}
		lineageByGenomeID[genomeID] = record
	}

	for _, genome := range nextPopulation {
		genomeID := strings.TrimSpace(genome.ID)
		if genomeID == "" {
			continue
		}
		record, ok := lineageByGenomeID[genomeID]
		if !ok {
			next[genomeID] = cloneEvoHistory(previous[genomeID])
			continue
		}
		parentID := strings.TrimSpace(record.ParentID)
		history := cloneEvoHistory(previous[parentID])
		events := cloneEvoHistory(record.Events)
		if len(events) == 0 {
			events = operationHistoryEvents(record.Operation)
		}
		history = append(history, events...)
		next[genomeID] = history
	}
	return next
}

func cloneEvoHistory(history []genotype.EvoHistoryEvent) []genotype.EvoHistoryEvent {
	if len(history) == 0 {
		return nil
	}
	out := make([]genotype.EvoHistoryEvent, 0, len(history))
	for _, event := range history {
		out = append(out, genotype.EvoHistoryEvent{
			Mutation: event.Mutation,
			IDs:      append([]string(nil), event.IDs...),
		})
	}
	return out
}

func operationHistoryEvents(operation string) []genotype.EvoHistoryEvent {
	operation = strings.TrimSpace(operation)
	switch operation {
	case "", "seed", "continue_seed", "elite_clone":
		return nil
	}
	parts := strings.Split(operation, "+")
	events := make([]genotype.EvoHistoryEvent, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		events = append(events, genotype.EvoHistoryEvent{Mutation: part})
	}
	if len(events) == 0 {
		return nil
	}
	return events
}

func deriveMutationEvent(before, after model.Genome, mutation string) genotype.EvoHistoryEvent {
	return genotype.EvoHistoryEvent{
		Mutation: strings.TrimSpace(mutation),
		IDs:      mutationChangedIDs(before, after),
	}
}

func mutationChangedIDs(before, after model.Genome) []string {
	ids := make([]string, 0, 8)
	seen := make(map[string]struct{})
	appendID := func(id string) {
		id = strings.TrimSpace(id)
		if id == "" {
			return
		}
		if _, exists := seen[id]; exists {
			return
		}
		seen[id] = struct{}{}
		ids = append(ids, id)
	}

	beforeSensors := setFromStrings(before.SensorIDs)
	afterSensors := setFromStrings(after.SensorIDs)
	beforeActuators := setFromStrings(before.ActuatorIDs)
	afterActuators := setFromStrings(after.ActuatorIDs)

	beforeNeurons := mapNeuronsByID(before.Neurons)
	afterNeurons := mapNeuronsByID(after.Neurons)
	knownNeuronIDs := keysFromNeuronMap(beforeNeurons)
	for id := range keysFromNeuronMap(afterNeurons) {
		knownNeuronIDs[id] = struct{}{}
	}
	knownSensorIDs := setFromStrings(before.SensorIDs)
	for id := range afterSensors {
		knownSensorIDs[id] = struct{}{}
	}
	knownActuatorIDs := setFromStrings(before.ActuatorIDs)
	for id := range afterActuators {
		knownActuatorIDs[id] = struct{}{}
	}
	classifyIDKind := func(id string) string {
		id = strings.TrimSpace(id)
		if id == "" {
			return "element"
		}
		if _, ok := knownNeuronIDs[id]; ok {
			return "neuron"
		}
		if _, ok := knownSensorIDs[id]; ok {
			return "sensor"
		}
		if _, ok := knownActuatorIDs[id]; ok {
			return "actuator"
		}
		lower := strings.ToLower(id)
		switch {
		case strings.Contains(lower, "sensor"):
			return "sensor"
		case strings.Contains(lower, "actuator"):
			return "actuator"
		case strings.Contains(lower, "neuron"):
			return "neuron"
		default:
			return "element"
		}
	}
	for _, id := range sortedSetDiff(keysFromNeuronMap(afterNeurons), keysFromNeuronMap(beforeNeurons)) {
		appendTypedElementIDs(appendID, "neuron", id)
	}
	for _, id := range sortedSetDiff(keysFromNeuronMap(beforeNeurons), keysFromNeuronMap(afterNeurons)) {
		appendTypedElementIDs(appendID, "neuron", id)
	}
	for _, id := range sortedIntersection(keysFromNeuronMap(beforeNeurons), keysFromNeuronMap(afterNeurons)) {
		if !reflect.DeepEqual(beforeNeurons[id], afterNeurons[id]) {
			appendTypedElementIDs(appendID, "neuron", id)
		}
	}

	beforeSynapses := mapSynapsesByID(before.Synapses)
	afterSynapses := mapSynapsesByID(after.Synapses)
	for _, id := range sortedSetDiff(keysFromSynapseMap(afterSynapses), keysFromSynapseMap(beforeSynapses)) {
		synapse := afterSynapses[id]
		appendTypedElementIDs(appendID, "synapse", id)
		appendTypedElementIDs(appendID, classifyIDKind(synapse.From), synapse.From)
		appendTypedElementIDs(appendID, classifyIDKind(synapse.To), synapse.To)
	}
	for _, id := range sortedSetDiff(keysFromSynapseMap(beforeSynapses), keysFromSynapseMap(afterSynapses)) {
		synapse := beforeSynapses[id]
		appendTypedElementIDs(appendID, "synapse", id)
		appendTypedElementIDs(appendID, classifyIDKind(synapse.From), synapse.From)
		appendTypedElementIDs(appendID, classifyIDKind(synapse.To), synapse.To)
	}
	for _, id := range sortedIntersection(keysFromSynapseMap(beforeSynapses), keysFromSynapseMap(afterSynapses)) {
		if !reflect.DeepEqual(beforeSynapses[id], afterSynapses[id]) {
			synapse := afterSynapses[id]
			appendTypedElementIDs(appendID, "synapse", id)
			appendTypedElementIDs(appendID, classifyIDKind(synapse.From), synapse.From)
			appendTypedElementIDs(appendID, classifyIDKind(synapse.To), synapse.To)
		}
	}

	for _, id := range sortedSetDiff(afterSensors, beforeSensors) {
		appendTypedElementIDs(appendID, "sensor", id)
	}
	for _, id := range sortedSetDiff(beforeSensors, afterSensors) {
		appendTypedElementIDs(appendID, "sensor", id)
	}

	for _, id := range sortedSetDiff(afterActuators, beforeActuators) {
		appendTypedElementIDs(appendID, "actuator", id)
	}
	for _, id := range sortedSetDiff(beforeActuators, afterActuators) {
		appendTypedElementIDs(appendID, "actuator", id)
	}

	beforeSensorLinks := mapSensorLinks(before.SensorNeuronLinks)
	afterSensorLinks := mapSensorLinks(after.SensorNeuronLinks)
	for _, key := range sortedSetDiff(keysFromSensorLinkMap(afterSensorLinks), keysFromSensorLinkMap(beforeSensorLinks)) {
		link := afterSensorLinks[key]
		appendTypedElementIDs(appendID, "sensor", link.SensorID)
		appendTypedElementIDs(appendID, "neuron", link.NeuronID)
	}
	for _, key := range sortedSetDiff(keysFromSensorLinkMap(beforeSensorLinks), keysFromSensorLinkMap(afterSensorLinks)) {
		link := beforeSensorLinks[key]
		appendTypedElementIDs(appendID, "sensor", link.SensorID)
		appendTypedElementIDs(appendID, "neuron", link.NeuronID)
	}

	beforeActuatorLinks := mapActuatorLinks(before.NeuronActuatorLinks)
	afterActuatorLinks := mapActuatorLinks(after.NeuronActuatorLinks)
	for _, key := range sortedSetDiff(keysFromActuatorLinkMap(afterActuatorLinks), keysFromActuatorLinkMap(beforeActuatorLinks)) {
		link := afterActuatorLinks[key]
		appendTypedElementIDs(appendID, "neuron", link.NeuronID)
		appendTypedElementIDs(appendID, "actuator", link.ActuatorID)
	}
	for _, key := range sortedSetDiff(keysFromActuatorLinkMap(beforeActuatorLinks), keysFromActuatorLinkMap(afterActuatorLinks)) {
		link := beforeActuatorLinks[key]
		appendTypedElementIDs(appendID, "neuron", link.NeuronID)
		appendTypedElementIDs(appendID, "actuator", link.ActuatorID)
	}

	appendSubstrateDifferences(before.Substrate, after.Substrate, appendID)
	appendStrategyDifferences(before.Strategy, after.Strategy, appendID)
	appendPlasticityDifferences(before.Plasticity, after.Plasticity, appendID)

	return ids
}

func appendSubstrateDifferences(
	before, after *model.SubstrateConfig,
	appendID func(id string),
) {
	if before == nil && after == nil {
		return
	}
	if before == nil || after == nil {
		appendID("substrate")
		if after != nil {
			for _, id := range after.CPPIDs {
				appendTypedElementIDs(appendID, "sensor", id)
			}
			for _, id := range after.CEPIDs {
				appendTypedElementIDs(appendID, "actuator", id)
			}
		}
		return
	}

	beforeCPPs := setFromStrings(before.CPPIDs)
	afterCPPs := setFromStrings(after.CPPIDs)
	for _, id := range sortedSetDiff(afterCPPs, beforeCPPs) {
		appendTypedElementIDs(appendID, "sensor", id)
	}
	for _, id := range sortedSetDiff(beforeCPPs, afterCPPs) {
		appendTypedElementIDs(appendID, "sensor", id)
	}

	beforeCEPs := setFromStrings(before.CEPIDs)
	afterCEPs := setFromStrings(after.CEPIDs)
	for _, id := range sortedSetDiff(afterCEPs, beforeCEPs) {
		appendTypedElementIDs(appendID, "actuator", id)
	}
	for _, id := range sortedSetDiff(beforeCEPs, afterCEPs) {
		appendTypedElementIDs(appendID, "actuator", id)
	}

	if before.CPPName != after.CPPName {
		appendID("substrate:cpp")
	}
	if before.CEPName != after.CEPName {
		appendID("substrate:cep")
	}
	if before.WeightCount != after.WeightCount ||
		!reflect.DeepEqual(before.Dimensions, after.Dimensions) ||
		!reflect.DeepEqual(before.Parameters, after.Parameters) {
		appendID("substrate")
	}
}

func appendStrategyDifferences(
	before, after *model.StrategyConfig,
	appendID func(id string),
) {
	if reflect.DeepEqual(before, after) {
		return
	}
	appendID("strategy")
}

func appendPlasticityDifferences(
	before, after *model.PlasticityConfig,
	appendID func(id string),
) {
	if reflect.DeepEqual(before, after) {
		return
	}
	appendID("plasticity")
}

func appendTypedElementIDs(appendID func(id string), kind string, ids ...string) {
	kind = strings.TrimSpace(strings.ToLower(kind))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		appendID(id)
		if kind == "" {
			continue
		}
		if strings.HasPrefix(strings.ToLower(id), kind+":") {
			continue
		}
		appendID(kind + ":" + id)
	}
}

func setFromStrings(values []string) map[string]struct{} {
	out := make(map[string]struct{}, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		out[value] = struct{}{}
	}
	return out
}

func sortedSetDiff(left, right map[string]struct{}) []string {
	if len(left) == 0 {
		return nil
	}
	out := make([]string, 0, len(left))
	for value := range left {
		if _, ok := right[value]; ok {
			continue
		}
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func sortedIntersection(left, right map[string]struct{}) []string {
	if len(left) == 0 || len(right) == 0 {
		return nil
	}
	out := make([]string, 0, len(left))
	for value := range left {
		if _, ok := right[value]; !ok {
			continue
		}
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func mapNeuronsByID(neurons []model.Neuron) map[string]model.Neuron {
	out := make(map[string]model.Neuron, len(neurons))
	for _, neuron := range neurons {
		if strings.TrimSpace(neuron.ID) == "" {
			continue
		}
		out[neuron.ID] = neuron
	}
	return out
}

func keysFromNeuronMap(values map[string]model.Neuron) map[string]struct{} {
	out := make(map[string]struct{}, len(values))
	for id := range values {
		out[id] = struct{}{}
	}
	return out
}

func mapSynapsesByID(synapses []model.Synapse) map[string]model.Synapse {
	out := make(map[string]model.Synapse, len(synapses))
	for _, synapse := range synapses {
		if strings.TrimSpace(synapse.ID) == "" {
			continue
		}
		out[synapse.ID] = synapse
	}
	return out
}

func keysFromSynapseMap(values map[string]model.Synapse) map[string]struct{} {
	out := make(map[string]struct{}, len(values))
	for id := range values {
		out[id] = struct{}{}
	}
	return out
}

func mapSensorLinks(links []model.SensorNeuronLink) map[string]model.SensorNeuronLink {
	out := make(map[string]model.SensorNeuronLink, len(links))
	for _, link := range links {
		key := sensorLinkKey(link)
		if key == "" {
			continue
		}
		out[key] = link
	}
	return out
}

func keysFromSensorLinkMap(values map[string]model.SensorNeuronLink) map[string]struct{} {
	out := make(map[string]struct{}, len(values))
	for key := range values {
		out[key] = struct{}{}
	}
	return out
}

func mapActuatorLinks(links []model.NeuronActuatorLink) map[string]model.NeuronActuatorLink {
	out := make(map[string]model.NeuronActuatorLink, len(links))
	for _, link := range links {
		key := actuatorLinkKey(link)
		if key == "" {
			continue
		}
		out[key] = link
	}
	return out
}

func keysFromActuatorLinkMap(values map[string]model.NeuronActuatorLink) map[string]struct{} {
	out := make(map[string]struct{}, len(values))
	for key := range values {
		out[key] = struct{}{}
	}
	return out
}

func sensorLinkKey(link model.SensorNeuronLink) string {
	sensorID := strings.TrimSpace(link.SensorID)
	neuronID := strings.TrimSpace(link.NeuronID)
	if sensorID == "" || neuronID == "" {
		return ""
	}
	return sensorID + "->" + neuronID
}

func actuatorLinkKey(link model.NeuronActuatorLink) string {
	neuronID := strings.TrimSpace(link.NeuronID)
	actuatorID := strings.TrimSpace(link.ActuatorID)
	if neuronID == "" || actuatorID == "" {
		return ""
	}
	return neuronID + "->" + actuatorID
}

func summarizeStaticSpeciation(bySpecies map[string][]model.Genome) SpeciationStats {
	if len(bySpecies) == 0 {
		return SpeciationStats{}
	}
	totalMembers := 0
	largest := 0
	for _, members := range bySpecies {
		size := len(members)
		totalMembers += size
		if size > largest {
			largest = size
		}
	}
	speciesCount := len(bySpecies)
	return SpeciationStats{
		SpeciesCount:       speciesCount,
		TargetSpeciesCount: speciesCount,
		Threshold:          0,
		MeanSpeciesSize:    float64(totalMembers) / float64(speciesCount),
		LargestSpeciesSize: largest,
	}
}

func (m *PopulationMonitor) evaluatePopulation(ctx context.Context, population []model.Genome, generation int) ([]ScoredGenome, tuningGenerationStats, []bool, error) {
	type job struct {
		idx    int
		genome model.Genome
	}
	type result struct {
		idx    int
		scored ScoredGenome
		tune   tuning.TuneReport
		err    error
	}

	jobs := make(chan job)
	results := make(chan result, len(population))

	workerCount := m.cfg.Workers
	if workerCount > len(population) {
		workerCount = len(population)
	}

	var wg sync.WaitGroup
	wg.Add(workerCount)
	for w := 0; w < workerCount; w++ {
		go func() {
			defer wg.Done()
			for j := range jobs {
				if err := ctx.Err(); err != nil {
					results <- result{idx: j.idx, err: err}
					continue
				}

				candidate := j.genome
				tuneReport := tuning.TuneReport{}
				attempts := m.cfg.TuneAttempts
				if m.cfg.TuneAttemptPolicy != nil {
					attempts = m.cfg.TuneAttemptPolicy.Attempts(m.cfg.TuneAttempts, generation, m.cfg.Generations, j.genome)
				}
				if m.cfg.OpMode == OpModeGT && m.cfg.Tuner != nil && attempts > 0 {
					if runtimeTuner, ok := m.cfg.Tuner.(tuning.RuntimeReportingTuner); ok && len(j.genome.Synapses) > 0 {
						scoredRuntime, runtimeReport, err := m.evaluateGenomeWithRuntimeTuning(ctx, j.genome, attempts, runtimeTuner)
						if err != nil {
							results <- result{idx: j.idx, err: err}
							continue
						}
						results <- result{idx: j.idx, scored: scoredRuntime, tune: runtimeReport}
						continue
					}
					if reporting, ok := m.cfg.Tuner.(tuning.ReportingTuner); ok {
						tuned, report, err := reporting.TuneWithReport(ctx, j.genome, attempts, func(ctx context.Context, g model.Genome) (float64, error) {
							fitness, _, err := m.evaluateGenome(ctx, g, OpModeGT)
							if err != nil {
								return 0, err
							}
							return fitness, nil
						})
						tuneReport = report
						if err != nil {
							results <- result{idx: j.idx, err: err}
							continue
						}
						candidate = tuned
					} else {
						tuned, err := m.cfg.Tuner.Tune(ctx, j.genome, attempts, func(ctx context.Context, g model.Genome) (float64, error) {
							fitness, _, err := m.evaluateGenome(ctx, g, OpModeGT)
							if err != nil {
								return 0, err
							}
							return fitness, nil
						})
						if err != nil {
							results <- result{idx: j.idx, err: err}
							continue
						}
						tuneReport.AttemptsPlanned = attempts
						tuneReport.AttemptsExecuted = attempts
						candidate = tuned
					}
				}

				fitness, trace, err := m.evaluateGenome(ctx, candidate, m.cfg.OpMode)
				if err != nil {
					results <- result{idx: j.idx, err: err}
					continue
				}
				results <- result{idx: j.idx, scored: ScoredGenome{Genome: candidate, Fitness: fitness, Trace: trace}, tune: tuneReport}
			}
		}()
	}

	for i := range population {
		jobs <- job{idx: i, genome: population[i]}
	}
	close(jobs)

	scored := make([]ScoredGenome, len(population))
	countedEvaluations := make([]bool, len(population))
	shouldCountEvaluations := !m.goalReached
	tuningStats := tuningGenerationStats{}
	control := m.cfg.Control
	for received := 0; received < len(population); received++ {
		if m.goalReached {
			shouldCountEvaluations = false
		}

		var res result
	waitResult:
		for {
			select {
			case <-ctx.Done():
				return nil, tuningGenerationStats{}, nil, ctx.Err()
			case res = <-results:
				break waitResult
			case cmd, ok := <-control:
				if !ok {
					control = nil
					continue
				}
				action := m.handleCommand(cmd)
				if action.printTrace {
					m.emitTraceUpdate(TraceUpdateReasonPrint, m.totalEvaluations)
				}
			}
		}
		if res.err != nil {
			return nil, tuningGenerationStats{}, nil, res.err
		}
		scored[res.idx] = res.scored
		if shouldCountEvaluations {
			countedEvaluations[res.idx] = true
		}
		if res.tune.AttemptsPlanned > 0 || res.tune.AttemptsExecuted > 0 || res.tune.CandidateEvaluations > 0 {
			tuningStats.Invocations++
		}
		tuningStats.Attempts += res.tune.AttemptsExecuted
		tuningStats.Evaluations += res.tune.CandidateEvaluations
		tuningStats.Accepted += res.tune.AcceptedCandidates
		tuningStats.Rejected += res.tune.RejectedCandidates
		if res.tune.GoalReached {
			tuningStats.GoalHits++
		}
	}
	wg.Wait()

	return scored, tuningStats, countedEvaluations, nil
}

func (m *PopulationMonitor) evaluateGenomeWithRuntimeTuning(
	ctx context.Context,
	genome model.Genome,
	attempts int,
	tuner tuning.RuntimeReportingTuner,
) (ScoredGenome, tuning.TuneReport, error) {
	cortex, err := m.buildCortex(genome)
	if err != nil {
		return ScoredGenome{}, tuning.TuneReport{}, err
	}

	runtimeResult, err := tuner.TuneRuntimeWithReport(
		ctx,
		cortex,
		attempts,
		OpModeGT,
		func(ctx context.Context, mode string) (float64, map[string]any, bool, error) {
			fitness, trace, err := m.evaluateCortex(ctx, cortex, mode)
			if err != nil {
				return 0, nil, false, err
			}
			return fitness, map[string]any(trace), traceGoalReached(trace), nil
		},
	)
	if err != nil {
		return ScoredGenome{}, tuning.TuneReport{}, err
	}

	trace := scape.Trace(runtimeResult.Trace)
	fitness := runtimeResult.Fitness
	if trace == nil {
		if err := cortex.Reactivate(); err != nil {
			return ScoredGenome{}, tuning.TuneReport{}, err
		}
		var evalErr error
		fitness, trace, evalErr = m.evaluateCortex(ctx, cortex, OpModeGT)
		if evalErr != nil {
			return ScoredGenome{}, tuning.TuneReport{}, evalErr
		}
	}

	return ScoredGenome{
		Genome:  runtimeResult.Genome,
		Fitness: fitness,
		Trace:   trace,
	}, runtimeResult.Report, nil
}

func (m *PopulationMonitor) applyQueuedControl(ctx context.Context) error {
	if m.cfg.Control == nil {
		return nil
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case cmd, ok := <-m.cfg.Control:
			if !ok {
				return nil
			}
			action := m.handleCommand(cmd)
			if action.printTrace {
				m.emitTraceUpdate(TraceUpdateReasonPrint, m.totalEvaluations)
			}
		default:
			return nil
		}
	}
}

func (m *PopulationMonitor) evaluateGenome(ctx context.Context, genome model.Genome, mode string) (float64, scape.Trace, error) {
	cortex, err := m.buildCortex(genome)
	if err != nil {
		return 0, nil, err
	}
	return m.evaluateCortex(ctx, cortex, mode)
}

func (m *PopulationMonitor) buildCortex(genome model.Genome) (*agent.Cortex, error) {
	sensors, actuators, err := m.buildIO(genome)
	if err != nil {
		return nil, err
	}
	substrateRuntime, err := m.buildSubstrate(genome)
	if err != nil {
		return nil, err
	}

	cortex, err := agent.NewCortex(
		genome.ID,
		genome,
		sensors,
		actuators,
		m.cfg.InputNeuronIDs,
		m.cfg.OutputNeuronIDs,
		substrateRuntime,
	)
	if err != nil {
		return nil, err
	}
	return cortex, nil
}

func (m *PopulationMonitor) evaluateCortex(ctx context.Context, cortex *agent.Cortex, mode string) (float64, scape.Trace, error) {
	if cortex == nil {
		return 0, nil, fmt.Errorf("cortex is required")
	}
	var (
		fitness scape.Fitness
		trace   scape.Trace
		err     error
	)
	if modeAware, ok := m.cfg.Scape.(scape.ModeAwareScape); ok {
		fitness, trace, err = modeAware.EvaluateMode(ctx, cortex, mode)
	} else {
		fitness, trace, err = m.cfg.Scape.Evaluate(ctx, cortex)
	}
	if err != nil {
		return 0, nil, err
	}
	return float64(fitness), trace, nil
}

func traceGoalReached(trace scape.Trace) bool {
	if trace == nil {
		return false
	}
	keys := []string{"goal_reached", "goalReached", "goal_reached_flag", "goal"}
	for _, key := range keys {
		raw, ok := trace[key]
		if !ok {
			continue
		}
		switch value := raw.(type) {
		case bool:
			return value
		case int:
			return value > 0
		case int8:
			return value > 0
		case int16:
			return value > 0
		case int32:
			return value > 0
		case int64:
			return value > 0
		case uint:
			return value > 0
		case uint8:
			return value > 0
		case uint16:
			return value > 0
		case uint32:
			return value > 0
		case uint64:
			return value > 0
		case float32:
			return value > 0
		case float64:
			return value > 0
		}
	}
	return false
}

func (m *PopulationMonitor) buildIO(genome model.Genome) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	scapeName := m.cfg.Scape.Name()

	var sensors map[string]protoio.Sensor
	if len(genome.SensorIDs) > 0 {
		sensors = make(map[string]protoio.Sensor, len(genome.SensorIDs))
		for _, sensorID := range genome.SensorIDs {
			sensor, err := protoio.ResolveSensor(sensorID, scapeName)
			if err != nil {
				return nil, nil, fmt.Errorf("resolve sensor %s for scape %s: %w", sensorID, scapeName, err)
			}
			sensors[sensorID] = sensor
		}
	}

	var actuators map[string]protoio.Actuator
	if len(genome.ActuatorIDs) > 0 {
		actuators = make(map[string]protoio.Actuator, len(genome.ActuatorIDs))
		for _, actuatorID := range genome.ActuatorIDs {
			actuator, err := protoio.ResolveActuator(actuatorID, scapeName)
			if err != nil {
				return nil, nil, fmt.Errorf("resolve actuator %s for scape %s: %w", actuatorID, scapeName, err)
			}
			actuators[actuatorID] = actuator
		}
	}

	return sensors, actuators, nil
}

func (m *PopulationMonitor) buildSubstrate(genome model.Genome) (substrate.Runtime, error) {
	if genome.Substrate == nil {
		return nil, nil
	}
	cfg := genome.Substrate
	spec := substrate.Spec{
		CPPName:    cfg.CPPName,
		CEPName:    cfg.CEPName,
		Dimensions: append([]int(nil), cfg.Dimensions...),
		Parameters: map[string]float64{},
	}
	for k, v := range cfg.Parameters {
		spec.Parameters[k] = v
	}
	weightCount := cfg.WeightCount
	if weightCount <= 0 {
		weightCount = len(m.cfg.OutputNeuronIDs)
	}
	rt, err := substrate.NewSimpleRuntime(spec, weightCount)
	if err != nil {
		return nil, fmt.Errorf("build substrate runtime for genome %s: %w", genome.ID, err)
	}
	return rt, nil
}

func (m *PopulationMonitor) nextGeneration(ctx context.Context, ranked []ScoredGenome, speciesByGenomeID map[string]string, generation int) ([]model.Genome, []LineageRecord, error) {
	next := make([]model.Genome, 0, m.cfg.PopulationSize)
	lineage := make([]LineageRecord, 0, m.cfg.PopulationSize)
	nextGeneration := generation + 1
	parentPool := ranked
	if m.cfg.SpecieSizeLimit > 0 {
		parentPool = limitSpeciesParentPool(ranked, speciesByGenomeID, m.cfg.SpecieSizeLimit)
		if len(parentPool) == 0 {
			parentPool = ranked
		}
	}

	for i := 0; i < m.cfg.EliteCount; i++ {
		elite := genotype.CloneAgent(ranked[i].Genome, ranked[i].Genome.ID)
		sig := ComputeGenomeSignature(elite)
		next = append(next, elite)
		lineage = append(lineage, LineageRecord{
			GenomeID:    elite.ID,
			ParentID:    ranked[i].Genome.ID,
			Generation:  nextGeneration,
			Operation:   "elite_clone",
			Fingerprint: sig.Fingerprint,
			Summary:     sig.Summary,
		})
	}

	remaining := m.cfg.PopulationSize - len(next)
	offspringPlan := buildSpeciesOffspringPlan(parentPool, speciesByGenomeID, remaining)
	for _, item := range offspringPlan {
		if len(next) >= m.cfg.PopulationSize {
			break
		}
		speciesRanked := filterRankedBySpecies(parentPool, speciesByGenomeID, item.SpeciesKey)
		if len(speciesRanked) == 0 {
			continue
		}
		for i := 0; i < item.Count; i++ {
			if len(next) >= m.cfg.PopulationSize {
				break
			}
			if err := ctx.Err(); err != nil {
				return nil, nil, err
			}

			parent, err := m.pickParentForSpecies(parentPool, speciesRanked, speciesByGenomeID, generation)
			if err != nil {
				return nil, nil, err
			}
			child, record, err := m.mutateFromParent(ctx, parent, generation, len(next))
			if err != nil {
				return nil, nil, err
			}
			next = append(next, child)
			lineage = append(lineage, record)
		}
	}

	for len(next) < m.cfg.PopulationSize {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}

		parent, err := m.pickParentForSpecies(parentPool, parentPool, speciesByGenomeID, generation)
		if err != nil {
			return nil, nil, err
		}
		child, record, err := m.mutateFromParent(ctx, parent, generation, len(next))
		if err != nil {
			return nil, nil, err
		}
		next = append(next, child)
		lineage = append(lineage, record)
	}

	return next, lineage, nil
}

func limitSpeciesParentPool(ranked []ScoredGenome, speciesByGenomeID map[string]string, perSpeciesLimit int) []ScoredGenome {
	if perSpeciesLimit <= 0 {
		return append([]ScoredGenome(nil), ranked...)
	}
	countBySpecies := make(map[string]int, len(speciesByGenomeID))
	out := make([]ScoredGenome, 0, len(ranked))
	for _, item := range ranked {
		key := speciesByGenomeID[item.Genome.ID]
		if key == "" {
			key = "species:unknown"
		}
		if countBySpecies[key] >= perSpeciesLimit {
			continue
		}
		countBySpecies[key]++
		out = append(out, item)
	}
	return out
}

func (m *PopulationMonitor) pickParentForSpecies(allRanked, speciesRanked []ScoredGenome, speciesByGenomeID map[string]string, generation int) (model.Genome, error) {
	eliteCount := m.cfg.EliteCount
	if eliteCount > len(speciesRanked) {
		eliteCount = len(speciesRanked)
	}
	if eliteCount <= 0 {
		eliteCount = 1
	}
	if speciesAware, ok := m.cfg.Selector.(SpeciesAwareGenerationSelector); ok {
		return speciesAware.PickParentForGenerationWithSpecies(m.rng, speciesRanked, eliteCount, generation, speciesByGenomeID)
	}
	if generationAware, ok := m.cfg.Selector.(GenerationAwareSelector); ok {
		return generationAware.PickParentForGeneration(m.rng, speciesRanked, eliteCount, generation)
	}
	return m.cfg.Selector.PickParent(m.rng, speciesRanked, eliteCount)
}

func (m *PopulationMonitor) mutateFromParent(ctx context.Context, parent model.Genome, generation, nextIndex int) (model.Genome, LineageRecord, error) {
	child := genotype.CloneAgent(parent, fmt.Sprintf("%s-g%d-i%d", parent.ID, generation+1, nextIndex))
	mutationCount, err := m.cfg.TopologicalMutations.MutationCount(parent, generation, m.rng)
	if err != nil {
		return model.Genome{}, LineageRecord{}, err
	}
	if mutationCount <= 0 {
		return model.Genome{}, LineageRecord{}, fmt.Errorf("invalid mutation count from policy: %d", mutationCount)
	}

	mutated := child
	operationNames := make([]string, 0, mutationCount)
	operationEvents := make([]genotype.EvoHistoryEvent, 0, mutationCount)
	successes := 0
	attempts := 0
	maxAttempts := mutationCount * m.maxMutationAttemptsPerStep()
	for successes < mutationCount {
		if err := ctx.Err(); err != nil {
			return model.Genome{}, LineageRecord{}, err
		}
		attempts++
		if attempts > maxAttempts {
			return model.Genome{}, LineageRecord{}, fmt.Errorf("failed to apply %d successful mutations after %d attempts", mutationCount, attempts-1)
		}
		beforeMutation := mutated
		operator := m.chooseMutation(mutated)
		next, opErr := operator.Apply(ctx, mutated)
		operationName := operator.Name()
		if opErr != nil {
			if m.cfg.Mutation != nil && operator != m.cfg.Mutation {
				next, opErr = m.cfg.Mutation.Apply(ctx, mutated)
				operationName = m.cfg.Mutation.Name() + "(fallback)"
			}
		}
		if opErr != nil {
			if errors.Is(opErr, ErrNoSynapses) || errors.Is(opErr, ErrNoNeurons) {
				continue
			}
			return model.Genome{}, LineageRecord{}, opErr
		}
		if err := morphology.EnsureGenomeIOCompatibility(m.cfg.Scape.Name(), next); err != nil {
			continue
		}
		mutated = next
		operationNames = append(operationNames, operationName)
		operationEvents = append(operationEvents, deriveMutationEvent(beforeMutation, next, operationName))
		successes++
	}

	sig := ComputeGenomeSignature(mutated)
	return mutated, LineageRecord{
		GenomeID:    mutated.ID,
		ParentID:    parent.ID,
		Generation:  generation + 1,
		Operation:   strings.Join(operationNames, "+"),
		Events:      operationEvents,
		Fingerprint: sig.Fingerprint,
		Summary:     sig.Summary,
	}, nil
}

func (m *PopulationMonitor) maxMutationAttemptsPerStep() int {
	// Keep retries finite when configured operators are systematically inapplicable.
	base := 4
	if len(m.cfg.MutationPolicy) > 0 {
		base += len(m.cfg.MutationPolicy) * 4
	}
	return base
}

type speciesQuota struct {
	SpeciesKey string
	Count      int
}

func buildSpeciesOffspringPlan(ranked []ScoredGenome, speciesByGenomeID map[string]string, totalOffspring int) []speciesQuota {
	if totalOffspring <= 0 || len(ranked) == 0 {
		return nil
	}
	type agg struct {
		key   string
		sum   float64
		size  int
		score float64
	}
	byKey := map[string]*agg{}
	for _, item := range ranked {
		key := speciesByGenomeID[item.Genome.ID]
		if key == "" {
			key = "species:unknown"
		}
		if byKey[key] == nil {
			byKey[key] = &agg{key: key}
		}
		byKey[key].sum += item.Fitness
		byKey[key].size++
	}
	keys := make([]string, 0, len(byKey))
	minMean := 0.0
	for key, bucket := range byKey {
		bucket.score = bucket.sum / float64(bucket.size)
		if len(keys) == 0 || bucket.score < minMean {
			minMean = bucket.score
		}
		keys = append(keys, key)
	}
	sort.Strings(keys)
	shift := 0.0
	if minMean <= 0 {
		shift = -minMean + 1e-9
	}
	totalScore := 0.0
	for _, key := range keys {
		byKey[key].score += shift
		totalScore += byKey[key].score
	}
	if totalScore <= 0 {
		for _, key := range keys {
			byKey[key].score = 1.0
		}
		totalScore = float64(len(keys))
	}

	type alloc struct {
		key       string
		count     int
		remainder float64
	}
	allocs := make([]alloc, 0, len(keys))
	assigned := 0
	for _, key := range keys {
		share := byKey[key].score / totalScore * float64(totalOffspring)
		base := int(math.Floor(share))
		allocs = append(allocs, alloc{
			key:       key,
			count:     base,
			remainder: share - float64(base),
		})
		assigned += base
	}
	left := totalOffspring - assigned
	sort.Slice(allocs, func(i, j int) bool {
		if allocs[i].remainder == allocs[j].remainder {
			return allocs[i].key < allocs[j].key
		}
		return allocs[i].remainder > allocs[j].remainder
	})
	for i := 0; i < left; i++ {
		allocs[i%len(allocs)].count++
	}
	sort.Slice(allocs, func(i, j int) bool { return allocs[i].key < allocs[j].key })

	out := make([]speciesQuota, 0, len(allocs))
	for _, item := range allocs {
		if item.count <= 0 {
			continue
		}
		out = append(out, speciesQuota{SpeciesKey: item.key, Count: item.count})
	}
	return out
}

func filterRankedBySpecies(ranked []ScoredGenome, speciesByGenomeID map[string]string, speciesKey string) []ScoredGenome {
	out := make([]ScoredGenome, 0, len(ranked))
	for _, item := range ranked {
		if speciesByGenomeID[item.Genome.ID] == speciesKey {
			out = append(out, item)
		}
	}
	return out
}

func summarizeSpeciesGeneration(ranked []ScoredGenome, speciesByGenomeID map[string]string, generation int, prevSpeciesSet map[string]struct{}) (SpeciesGeneration, map[string]struct{}) {
	type aggregate struct {
		size int
		sum  float64
		best float64
	}
	bySpecies := map[string]*aggregate{}
	currentSet := map[string]struct{}{}
	for _, item := range ranked {
		key := speciesByGenomeID[item.Genome.ID]
		if key == "" {
			key = "species:unknown"
		}
		currentSet[key] = struct{}{}
		bucket := bySpecies[key]
		if bucket == nil {
			bucket = &aggregate{best: item.Fitness}
			bySpecies[key] = bucket
		}
		bucket.size++
		bucket.sum += item.Fitness
		if item.Fitness > bucket.best {
			bucket.best = item.Fitness
		}
	}
	keys := make([]string, 0, len(bySpecies))
	for key := range bySpecies {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	metrics := make([]SpeciesMetrics, 0, len(keys))
	for _, key := range keys {
		item := bySpecies[key]
		metrics = append(metrics, SpeciesMetrics{
			Key:         key,
			Size:        item.size,
			MeanFitness: item.sum / float64(item.size),
			BestFitness: item.best,
		})
	}

	newSpecies := make([]string, 0)
	for _, key := range keys {
		if _, ok := prevSpeciesSet[key]; !ok {
			newSpecies = append(newSpecies, key)
		}
	}
	sort.Strings(newSpecies)

	extinctSpecies := make([]string, 0)
	for key := range prevSpeciesSet {
		if _, ok := currentSet[key]; !ok {
			extinctSpecies = append(extinctSpecies, key)
		}
	}
	sort.Strings(extinctSpecies)

	return SpeciesGeneration{
		Generation:     generation,
		Species:        metrics,
		NewSpecies:     newSpecies,
		ExtinctSpecies: extinctSpecies,
	}, currentSet
}

func (m *PopulationMonitor) chooseMutation(genome model.Genome) Operator {
	if len(m.cfg.MutationPolicy) == 0 {
		return m.cfg.Mutation
	}

	total := 0.0
	candidates := make([]WeightedMutation, 0, len(m.cfg.MutationPolicy))
	for _, item := range m.cfg.MutationPolicy {
		if !m.isOperatorApplicable(item.Operator, genome) {
			continue
		}
		candidates = append(candidates, item)
		total += item.Weight
	}
	if total <= 0 {
		if m.isOperatorApplicable(m.cfg.Mutation, genome) {
			return m.cfg.Mutation
		}
		// No compatible operator; fall back to legacy behavior.
		return m.cfg.MutationPolicy[len(m.cfg.MutationPolicy)-1].Operator
	}
	pick := m.rng.Float64() * total
	acc := 0.0
	for _, item := range candidates {
		acc += item.Weight
		if pick <= acc {
			return item.Operator
		}
	}
	return candidates[len(candidates)-1].Operator
}

func (m *PopulationMonitor) isOperatorApplicable(operator Operator, genome model.Genome) bool {
	if operator == nil {
		return false
	}
	if contextual, ok := operator.(ContextualOperator); ok {
		return contextual.Applicable(genome, m.cfg.Scape.Name())
	}
	return true
}
