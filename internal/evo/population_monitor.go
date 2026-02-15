package evo

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"

	"protogonos/internal/agent"
	"protogonos/internal/genotype"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
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

type LineageRecord struct {
	GenomeID    string          `json:"genome_id"`
	ParentID    string          `json:"parent_id"`
	Generation  int             `json:"generation"`
	Operation   string          `json:"operation"`
	Fingerprint string          `json:"fingerprint,omitempty"`
	Summary     TopologySummary `json:"summary,omitempty"`
}

type MonitorConfig struct {
	Scape                scape.Scape
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
	Control              <-chan MonitorCommand
}

type PopulationMonitor struct {
	cfg        MonitorConfig
	rng        *rand.Rand
	speciation *AdaptiveSpeciation
	paused     bool
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
	CommandPause    MonitorCommand = "pause"
	CommandContinue MonitorCommand = "continue"
	CommandStop     MonitorCommand = "stop"
)

func NewPopulationMonitor(cfg MonitorConfig) (*PopulationMonitor, error) {
	if cfg.Scape == nil {
		return nil, fmt.Errorf("scape is required")
	}
	if cfg.Mutation == nil && len(cfg.MutationPolicy) == 0 {
		return nil, fmt.Errorf("mutation operator or policy is required")
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

	return &PopulationMonitor{
		cfg:        cfg,
		rng:        rand.New(rand.NewSource(cfg.Seed)),
		speciation: NewAdaptiveSpeciation(cfg.PopulationSize),
	}, nil
}

func (m *PopulationMonitor) Run(ctx context.Context, initial []model.Genome) (RunResult, error) {
	if len(initial) != m.cfg.PopulationSize {
		return RunResult{}, fmt.Errorf("initial population mismatch: got=%d want=%d", len(initial), m.cfg.PopulationSize)
	}

	population := make([]model.Genome, len(initial))
	copy(population, initial)

	bestHistory := make([]float64, 0, m.cfg.Generations)
	diagnostics := make([]GenerationDiagnostics, 0, m.cfg.Generations)
	speciesHistory := make([]SpeciesGeneration, 0, m.cfg.Generations)
	lineage := make([]LineageRecord, 0, len(initial)*(m.cfg.Generations+1))
	prevSpeciesSet := map[string]struct{}{}
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
	evaluations := 0

	for gen := 0; gen < m.cfg.Generations; gen++ {
		if err := ctx.Err(); err != nil {
			return RunResult{}, err
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
		scored, tuningStats, err = m.evaluatePopulation(ctx, population, logicalGeneration)
		if err != nil {
			return RunResult{}, err
		}
		scored = m.cfg.Postprocessor.Process(scored)

		sort.Slice(scored, func(i, j int) bool {
			return scored[i].Fitness > scored[j].Fitness
		})
		evaluations += len(scored)
		bestHistory = append(bestHistory, scored[0].Fitness)
		speciesByGenomeID, speciationStats := m.assignSpecies(scored)
		diagnostics = append(diagnostics, summarizeGeneration(scored, logicalGeneration+1, speciationStats, tuningStats))
		history, currentSet := summarizeSpeciesGeneration(scored, speciesByGenomeID, logicalGeneration+1, prevSpeciesSet)
		speciesHistory = append(speciesHistory, history)
		prevSpeciesSet = currentSet
		if (m.cfg.FitnessGoal > 0 && scored[0].Fitness >= m.cfg.FitnessGoal) ||
			(m.cfg.EvaluationsLimit > 0 && evaluations >= m.cfg.EvaluationsLimit) {
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
	}

	return RunResult{
		BestByGeneration:      bestHistory,
		GenerationDiagnostics: diagnostics,
		SpeciesHistory:        speciesHistory,
		FinalPopulation:       scored,
		Lineage:               lineage,
	}, nil
}

func (m *PopulationMonitor) applyControl(ctx context.Context, waitIfPaused bool) (bool, error) {
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
			if m.handleCommand(cmd) {
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
				if m.handleCommand(cmd) {
					return true, nil
				}
				if !m.paused {
					return false, nil
				}
			}
		}
	}
}

func (m *PopulationMonitor) handleCommand(cmd MonitorCommand) bool {
	switch cmd {
	case CommandPause:
		m.paused = true
	case CommandContinue:
		m.paused = false
	case CommandStop:
		return true
	}
	return false
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

func (m *PopulationMonitor) assignSpecies(scored []ScoredGenome) (map[string]string, SpeciationStats) {
	genomes := make([]model.Genome, 0, len(scored))
	for _, item := range scored {
		genomes = append(genomes, item.Genome)
	}
	bySpecies, stats := m.speciation.Assign(genomes)
	speciesByGenomeID := make(map[string]string, len(scored))
	for key, members := range bySpecies {
		for _, genome := range members {
			speciesByGenomeID[genome.ID] = key
		}
	}
	return speciesByGenomeID, stats
}

func (m *PopulationMonitor) evaluatePopulation(ctx context.Context, population []model.Genome, generation int) ([]ScoredGenome, tuningGenerationStats, error) {
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
				if m.cfg.Tuner != nil && attempts > 0 {
					if reporting, ok := m.cfg.Tuner.(tuning.ReportingTuner); ok {
						tuned, report, err := reporting.TuneWithReport(ctx, j.genome, attempts, func(ctx context.Context, g model.Genome) (float64, error) {
							fitness, _, err := m.evaluateGenome(ctx, g)
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
							fitness, _, err := m.evaluateGenome(ctx, g)
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

				fitness, trace, err := m.evaluateGenome(ctx, candidate)
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

	wg.Wait()
	close(results)

	scored := make([]ScoredGenome, len(population))
	tuningStats := tuningGenerationStats{}
	for res := range results {
		if res.err != nil {
			return nil, tuningGenerationStats{}, res.err
		}
		scored[res.idx] = res.scored
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

	return scored, tuningStats, nil
}

func (m *PopulationMonitor) evaluateGenome(ctx context.Context, genome model.Genome) (float64, scape.Trace, error) {
	sensors, actuators, err := m.buildIO(genome)
	if err != nil {
		return 0, nil, err
	}
	substrateRuntime, err := m.buildSubstrate(genome)
	if err != nil {
		return 0, nil, err
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
		return 0, nil, err
	}
	fitness, trace, err := m.cfg.Scape.Evaluate(ctx, cortex)
	if err != nil {
		return 0, nil, err
	}
	return float64(fitness), trace, nil
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
		mutated = next
		operationNames = append(operationNames, operationName)
		successes++
	}

	sig := ComputeGenomeSignature(mutated)
	return mutated, LineageRecord{
		GenomeID:    mutated.ID,
		ParentID:    parent.ID,
		Generation:  generation + 1,
		Operation:   strings.Join(operationNames, "+"),
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
