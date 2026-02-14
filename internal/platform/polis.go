package platform

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"protogonos/internal/evo"
	"protogonos/internal/genotype"
	"protogonos/internal/model"
	"protogonos/internal/scape"
	"protogonos/internal/storage"
	"protogonos/internal/tuning"
)

type Config struct {
	Store storage.Store
}

type EvolutionConfig struct {
	RunID                string
	ScapeName            string
	PopulationSize       int
	Generations          int
	SurvivalPercentage   float64
	SpecieSizeLimit      int
	FitnessGoal          float64
	EvaluationsLimit     int
	EliteCount           int
	Workers              int
	Seed                 int64
	InputNeuronIDs       []string
	OutputNeuronIDs      []string
	Mutation             evo.Operator
	MutationPolicy       []evo.WeightedMutation
	Selector             evo.Selector
	Postprocessor        evo.FitnessPostprocessor
	TopologicalMutations evo.TopologicalMutationPolicy
	Tuner                tuning.Tuner
	TuneAttempts         int
	TuneAttemptPolicy    tuning.AttemptPolicy
	Control              chan evo.MonitorCommand
	Initial              []model.Genome
}

type EvolutionResult struct {
	BestByGeneration      []float64
	GenerationDiagnostics []model.GenerationDiagnostics
	SpeciesHistory        []model.SpeciesGeneration
	BestFinalFitness      float64
	TopFinal              []evo.ScoredGenome
	Lineage               []evo.LineageRecord
}

type Polis struct {
	store storage.Store

	mu      sync.RWMutex
	scapes  map[string]scape.Scape
	started bool
	runs    map[string]chan evo.MonitorCommand
}

func NewPolis(cfg Config) *Polis {
	return &Polis{
		store:  cfg.Store,
		scapes: make(map[string]scape.Scape),
		runs:   make(map[string]chan evo.MonitorCommand),
	}
}

func (p *Polis) Init(ctx context.Context) error {
	if p.store == nil {
		return fmt.Errorf("store is required")
	}
	p.mu.RLock()
	alreadyStarted := p.started
	p.mu.RUnlock()
	if alreadyStarted {
		return nil
	}
	if err := p.store.Init(ctx); err != nil {
		return err
	}

	p.mu.Lock()
	p.started = true
	p.mu.Unlock()
	return nil
}

func (p *Polis) RegisterScape(s scape.Scape) error {
	if s == nil {
		return fmt.Errorf("scape is nil")
	}

	name := s.Name()
	if name == "" {
		return fmt.Errorf("scape name is required")
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.started {
		return fmt.Errorf("polis is not initialized")
	}
	p.scapes[name] = s
	return nil
}

func (p *Polis) GetScape(name string) (scape.Scape, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	s, ok := p.scapes[name]
	return s, ok
}

func (p *Polis) Stop() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, control := range p.runs {
		select {
		case control <- evo.CommandStop:
		default:
		}
	}

	p.started = false
	p.scapes = make(map[string]scape.Scape)
	p.runs = make(map[string]chan evo.MonitorCommand)
}

func (p *Polis) RunEvolution(ctx context.Context, cfg EvolutionConfig) (EvolutionResult, error) {
	if len(cfg.Initial) != cfg.PopulationSize {
		return EvolutionResult{}, fmt.Errorf("initial population mismatch: got=%d want=%d", len(cfg.Initial), cfg.PopulationSize)
	}
	if cfg.Mutation == nil {
		return EvolutionResult{}, fmt.Errorf("mutation operator is required")
	}
	if cfg.ScapeName == "" {
		return EvolutionResult{}, fmt.Errorf("scape name is required")
	}
	if cfg.EliteCount <= 0 {
		cfg.EliteCount = 1
	}
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}

	p.mu.RLock()
	targetScape, ok := p.scapes[cfg.ScapeName]
	started := p.started
	p.mu.RUnlock()

	if !started {
		return EvolutionResult{}, fmt.Errorf("polis is not initialized")
	}
	if !ok {
		return EvolutionResult{}, fmt.Errorf("scape not registered: %s", cfg.ScapeName)
	}

	runID := cfg.RunID
	if runID == "" {
		runID = fmt.Sprintf("evo:%s:%d", cfg.ScapeName, cfg.Seed)
	}
	control := cfg.Control
	if control == nil {
		control = make(chan evo.MonitorCommand, 16)
	}
	if err := p.registerRunControl(runID, control); err != nil {
		return EvolutionResult{}, err
	}
	defer p.unregisterRunControl(runID)

	monitor, err := evo.NewPopulationMonitor(evo.MonitorConfig{
		Scape:                targetScape,
		Mutation:             cfg.Mutation,
		PopulationSize:       cfg.PopulationSize,
		EliteCount:           cfg.EliteCount,
		SurvivalPercentage:   cfg.SurvivalPercentage,
		SpecieSizeLimit:      cfg.SpecieSizeLimit,
		Generations:          cfg.Generations,
		FitnessGoal:          cfg.FitnessGoal,
		EvaluationsLimit:     cfg.EvaluationsLimit,
		Workers:              cfg.Workers,
		Seed:                 cfg.Seed,
		InputNeuronIDs:       cfg.InputNeuronIDs,
		OutputNeuronIDs:      cfg.OutputNeuronIDs,
		MutationPolicy:       cfg.MutationPolicy,
		Selector:             cfg.Selector,
		Postprocessor:        cfg.Postprocessor,
		TopologicalMutations: cfg.TopologicalMutations,
		Tuner:                cfg.Tuner,
		TuneAttempts:         cfg.TuneAttempts,
		TuneAttemptPolicy:    cfg.TuneAttemptPolicy,
		Control:              control,
	})
	if err != nil {
		return EvolutionResult{}, err
	}

	result, err := monitor.Run(ctx, cfg.Initial)
	if err != nil {
		return EvolutionResult{}, err
	}
	finalGenomes := make([]model.Genome, 0, len(result.FinalPopulation))
	for _, scored := range result.FinalPopulation {
		finalGenomes = append(finalGenomes, scored.Genome)
	}
	executedGenerations := len(result.BestByGeneration)
	persistenceRunID := runID
	populationID := persistenceRunID
	if err := genotype.SavePopulationSnapshot(ctx, p.store, populationID, executedGenerations, finalGenomes); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.store.SaveFitnessHistory(ctx, persistenceRunID, result.BestByGeneration); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.store.SaveGenerationDiagnostics(ctx, persistenceRunID, toModelDiagnostics(result.GenerationDiagnostics)); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.store.SaveSpeciesHistory(ctx, persistenceRunID, toModelSpeciesHistory(result.SpeciesHistory)); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.store.SaveLineage(ctx, persistenceRunID, toModelLineage(result.Lineage)); err != nil {
		return EvolutionResult{}, err
	}

	bestFinal := 0.0
	topFinal := []evo.ScoredGenome{}
	if len(result.FinalPopulation) > 0 {
		ranked := append([]evo.ScoredGenome(nil), result.FinalPopulation...)
		sort.Slice(ranked, func(i, j int) bool {
			return ranked[i].Fitness > ranked[j].Fitness
		})
		bestFinal = ranked[0].Fitness
		topCount := 5
		if len(ranked) < topCount {
			topCount = len(ranked)
		}
		topFinal = append(topFinal, ranked[:topCount]...)
	}
	if err := p.store.SaveTopGenomes(ctx, persistenceRunID, toModelTopGenomes(topFinal)); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.updateScapeSummary(ctx, cfg.ScapeName, bestFinal); err != nil {
		return EvolutionResult{}, err
	}

	return EvolutionResult{
		BestByGeneration:      result.BestByGeneration,
		GenerationDiagnostics: toModelDiagnostics(result.GenerationDiagnostics),
		SpeciesHistory:        toModelSpeciesHistory(result.SpeciesHistory),
		BestFinalFitness:      bestFinal,
		TopFinal:              topFinal,
		Lineage:               result.Lineage,
	}, nil
}

func toModelLineage(lineage []evo.LineageRecord) []model.LineageRecord {
	out := make([]model.LineageRecord, 0, len(lineage))
	for _, rec := range lineage {
		out = append(out, model.LineageRecord{
			VersionedRecord: model.VersionedRecord{
				SchemaVersion: storage.CurrentSchemaVersion,
				CodecVersion:  storage.CurrentCodecVersion,
			},
			GenomeID:    rec.GenomeID,
			ParentID:    rec.ParentID,
			Generation:  rec.Generation,
			Operation:   rec.Operation,
			Fingerprint: rec.Fingerprint,
			Summary: model.LineageSummary{
				TotalNeurons:           rec.Summary.TotalNeurons,
				TotalSynapses:          rec.Summary.TotalSynapses,
				TotalRecurrentSynapses: rec.Summary.TotalRecurrentSynapses,
				TotalSensors:           rec.Summary.TotalSensors,
				TotalActuators:         rec.Summary.TotalActuators,
				ActivationDistribution: rec.Summary.ActivationDistribution,
				AggregatorDistribution: rec.Summary.AggregatorDistribution,
			},
		})
	}
	return out
}

func toModelDiagnostics(diags []evo.GenerationDiagnostics) []model.GenerationDiagnostics {
	out := make([]model.GenerationDiagnostics, 0, len(diags))
	for _, d := range diags {
		out = append(out, model.GenerationDiagnostics{
			Generation:           d.Generation,
			BestFitness:          d.BestFitness,
			MeanFitness:          d.MeanFitness,
			MinFitness:           d.MinFitness,
			SpeciesCount:         d.SpeciesCount,
			FingerprintDiversity: d.FingerprintDiversity,
			SpeciationThreshold:  d.SpeciationThreshold,
			TargetSpeciesCount:   d.TargetSpeciesCount,
			MeanSpeciesSize:      d.MeanSpeciesSize,
			LargestSpeciesSize:   d.LargestSpeciesSize,
		})
	}
	return out
}

func toModelTopGenomes(top []evo.ScoredGenome) []model.TopGenomeRecord {
	out := make([]model.TopGenomeRecord, 0, len(top))
	for i, item := range top {
		out = append(out, model.TopGenomeRecord{
			Rank:    i + 1,
			Fitness: item.Fitness,
			Genome:  item.Genome,
		})
	}
	return out
}

func toModelSpeciesHistory(history []evo.SpeciesGeneration) []model.SpeciesGeneration {
	out := make([]model.SpeciesGeneration, 0, len(history))
	for _, generation := range history {
		species := make([]model.SpeciesMetrics, 0, len(generation.Species))
		for _, item := range generation.Species {
			species = append(species, model.SpeciesMetrics{
				Key:         item.Key,
				Size:        item.Size,
				MeanFitness: item.MeanFitness,
				BestFitness: item.BestFitness,
			})
		}
		out = append(out, model.SpeciesGeneration{
			Generation:     generation.Generation,
			Species:        species,
			NewSpecies:     append([]string(nil), generation.NewSpecies...),
			ExtinctSpecies: append([]string(nil), generation.ExtinctSpecies...),
		})
	}
	return out
}

func (p *Polis) updateScapeSummary(ctx context.Context, scapeName string, fitness float64) error {
	summary, ok, err := p.store.GetScapeSummary(ctx, scapeName)
	if err != nil {
		return err
	}
	if !ok {
		summary = model.ScapeSummary{
			VersionedRecord: model.VersionedRecord{
				SchemaVersion: storage.CurrentSchemaVersion,
				CodecVersion:  storage.CurrentCodecVersion,
			},
			Name:        scapeName,
			Description: fmt.Sprintf("best observed fitness for scape %s", scapeName),
		}
	}
	if fitness > summary.BestFitness {
		summary.BestFitness = fitness
	}
	return p.store.SaveScapeSummary(ctx, summary)
}

func (p *Polis) PauseRun(runID string) error {
	return p.sendRunCommand(runID, evo.CommandPause)
}

func (p *Polis) ContinueRun(runID string) error {
	return p.sendRunCommand(runID, evo.CommandContinue)
}

func (p *Polis) StopRun(runID string) error {
	return p.sendRunCommand(runID, evo.CommandStop)
}

func (p *Polis) registerRunControl(runID string, control chan evo.MonitorCommand) error {
	if runID == "" {
		return fmt.Errorf("run id is required")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.started {
		return fmt.Errorf("polis is not initialized")
	}
	if _, exists := p.runs[runID]; exists {
		return fmt.Errorf("run already active: %s", runID)
	}
	p.runs[runID] = control
	return nil
}

func (p *Polis) unregisterRunControl(runID string) {
	if runID == "" {
		return
	}
	p.mu.Lock()
	delete(p.runs, runID)
	p.mu.Unlock()
}

func (p *Polis) sendRunCommand(runID string, cmd evo.MonitorCommand) error {
	if runID == "" {
		return fmt.Errorf("run id is required")
	}
	p.mu.RLock()
	control, ok := p.runs[runID]
	p.mu.RUnlock()
	if !ok {
		return fmt.Errorf("run not active: %s", runID)
	}
	select {
	case control <- cmd:
		return nil
	default:
		return fmt.Errorf("run control channel is full: %s", runID)
	}
}

func (p *Polis) RegisteredScapes() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	names := make([]string, 0, len(p.scapes))
	for name := range p.scapes {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (p *Polis) Started() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.started
}
