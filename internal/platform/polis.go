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
	Store          storage.Store
	SupportModules []SupportModule
	PublicScapes   []PublicScapeSpec
}

type SupportModule interface {
	Name() string
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
}

type PublicScapeSpec struct {
	Scape      scape.Scape
	Type       string
	Parameters []any
	Metabolics any
	Physics    any
}

type PublicScapeSummary struct {
	Name       string `json:"name"`
	Type       string `json:"type,omitempty"`
	Parameters []any  `json:"parameters,omitempty"`
	Metabolics any    `json:"metabolics,omitempty"`
	Physics    any    `json:"physics,omitempty"`
}

type StopReason string

const (
	StopReasonNormal   StopReason = "normal"
	StopReasonShutdown StopReason = "shutdown"
)

type EvolutionConfig struct {
	RunID                string
	OpMode               string
	ScapeName            string
	PopulationSize       int
	Generations          int
	InitialGeneration    int
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

	mu sync.RWMutex

	scapes            map[string]scape.Scape
	supportModules    map[string]SupportModule
	publicScapes      map[string]PublicScapeSummary
	publicScapeByType map[string]string
	started           bool
	lastStopReason    StopReason
	runs              map[string]chan evo.MonitorCommand

	config Config
}

var (
	defaultPolisMu sync.Mutex
	defaultPolis   *Polis
)

func NewPolis(cfg Config) *Polis {
	return &Polis{
		store:             cfg.Store,
		scapes:            make(map[string]scape.Scape),
		supportModules:    make(map[string]SupportModule),
		publicScapes:      make(map[string]PublicScapeSummary),
		publicScapeByType: make(map[string]string),
		runs:              make(map[string]chan evo.MonitorCommand),
		config:            cfg,
		lastStopReason:    StopReasonNormal,
	}
}

func StartDefault(ctx context.Context, cfg Config) (*Polis, error) {
	defaultPolisMu.Lock()
	defer defaultPolisMu.Unlock()

	if defaultPolis != nil && defaultPolis.Started() {
		return defaultPolis, nil
	}

	p := NewPolis(cfg)
	if err := p.Init(ctx); err != nil {
		return nil, err
	}
	defaultPolis = p
	return defaultPolis, nil
}

func Default() (*Polis, bool) {
	defaultPolisMu.Lock()
	p := defaultPolis
	defaultPolisMu.Unlock()

	if p == nil || !p.Started() {
		return nil, false
	}
	return p, true
}

func StopDefault(reason StopReason) error {
	defaultPolisMu.Lock()
	p := defaultPolis
	defaultPolisMu.Unlock()
	if p == nil {
		return nil
	}
	if err := p.StopWithReason(reason); err != nil {
		return err
	}
	defaultPolisMu.Lock()
	if defaultPolis == p {
		defaultPolis = nil
	}
	defaultPolisMu.Unlock()
	return nil
}

func (p *Polis) Init(ctx context.Context) error {
	if p.store == nil {
		return fmt.Errorf("store is required")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.started {
		return nil
	}
	if err := p.store.Init(ctx); err != nil {
		return err
	}

	startedModules := make([]SupportModule, 0, len(p.config.SupportModules))
	for i, module := range p.config.SupportModules {
		if module == nil {
			stopSupportModules(ctx, startedModules)
			p.supportModules = make(map[string]SupportModule)
			p.publicScapes = make(map[string]PublicScapeSummary)
			p.publicScapeByType = make(map[string]string)
			p.scapes = make(map[string]scape.Scape)
			return fmt.Errorf("support module is nil at index %d", i)
		}
		name := module.Name()
		if name == "" {
			stopSupportModules(ctx, startedModules)
			p.supportModules = make(map[string]SupportModule)
			p.publicScapes = make(map[string]PublicScapeSummary)
			p.publicScapeByType = make(map[string]string)
			p.scapes = make(map[string]scape.Scape)
			return fmt.Errorf("support module name is required at index %d", i)
		}
		if _, exists := p.supportModules[name]; exists {
			stopSupportModules(ctx, startedModules)
			p.supportModules = make(map[string]SupportModule)
			p.publicScapes = make(map[string]PublicScapeSummary)
			p.publicScapeByType = make(map[string]string)
			p.scapes = make(map[string]scape.Scape)
			return fmt.Errorf("duplicate support module: %s", name)
		}
		if err := module.Start(ctx); err != nil {
			stopSupportModules(ctx, startedModules)
			p.supportModules = make(map[string]SupportModule)
			p.publicScapes = make(map[string]PublicScapeSummary)
			p.publicScapeByType = make(map[string]string)
			p.scapes = make(map[string]scape.Scape)
			return fmt.Errorf("start support module %s: %w", name, err)
		}
		p.supportModules[name] = module
		startedModules = append(startedModules, module)
	}

	startedScapes := make([]managedScape, 0, len(p.config.PublicScapes))
	for i, spec := range p.config.PublicScapes {
		if spec.Scape == nil {
			stopManagedScapes(ctx, startedScapes)
			stopSupportModules(ctx, startedModules)
			p.supportModules = make(map[string]SupportModule)
			p.publicScapes = make(map[string]PublicScapeSummary)
			p.publicScapeByType = make(map[string]string)
			p.scapes = make(map[string]scape.Scape)
			return fmt.Errorf("public scape is nil at index %d", i)
		}
		name := spec.Scape.Name()
		if name == "" {
			stopManagedScapes(ctx, startedScapes)
			stopSupportModules(ctx, startedModules)
			p.supportModules = make(map[string]SupportModule)
			p.publicScapes = make(map[string]PublicScapeSummary)
			p.publicScapeByType = make(map[string]string)
			p.scapes = make(map[string]scape.Scape)
			return fmt.Errorf("public scape name is required at index %d", i)
		}
		if _, exists := p.scapes[name]; exists {
			stopManagedScapes(ctx, startedScapes)
			stopSupportModules(ctx, startedModules)
			p.supportModules = make(map[string]SupportModule)
			p.publicScapes = make(map[string]PublicScapeSummary)
			p.publicScapeByType = make(map[string]string)
			p.scapes = make(map[string]scape.Scape)
			return fmt.Errorf("duplicate public scape: %s", name)
		}
		if managed, ok := spec.Scape.(managedScape); ok {
			if err := managed.Start(ctx); err != nil {
				stopManagedScapes(ctx, startedScapes)
				stopSupportModules(ctx, startedModules)
				p.supportModules = make(map[string]SupportModule)
				p.publicScapes = make(map[string]PublicScapeSummary)
				p.publicScapeByType = make(map[string]string)
				p.scapes = make(map[string]scape.Scape)
				return fmt.Errorf("start public scape %s: %w", name, err)
			}
			startedScapes = append(startedScapes, managed)
		}
		p.scapes[name] = spec.Scape
		summary := PublicScapeSummary{
			Name:       name,
			Type:       spec.Type,
			Parameters: append([]any(nil), spec.Parameters...),
			Metabolics: spec.Metabolics,
			Physics:    spec.Physics,
		}
		if summary.Type == "" {
			summary.Type = name
		}
		p.publicScapes[name] = summary
		if _, exists := p.publicScapeByType[summary.Type]; !exists {
			p.publicScapeByType[summary.Type] = name
		}
	}

	p.started = true
	return nil
}

func (p *Polis) Create(ctx context.Context) error {
	return p.Init(ctx)
}

func (p *Polis) Reset(ctx context.Context) error {
	_ = p.StopWithReason(StopReasonShutdown)
	if resetter, ok := p.store.(storage.Resetter); ok {
		if err := resetter.Reset(ctx); err != nil {
			return err
		}
	}
	return p.Init(ctx)
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

func (p *Polis) GetScapeByType(scapeType string) (scape.Scape, bool) {
	if scapeType == "" {
		return nil, false
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	name, ok := p.publicScapeByType[scapeType]
	if !ok {
		return nil, false
	}
	s, ok := p.scapes[name]
	return s, ok
}

func (p *Polis) Stop() {
	_ = p.StopWithReason(StopReasonNormal)
}

func (p *Polis) Shutdown() {
	_ = p.StopWithReason(StopReasonShutdown)
}

func (p *Polis) StopWithReason(reason StopReason) error {
	if reason == "" {
		reason = StopReasonNormal
	}
	if !isValidStopReason(reason) {
		return fmt.Errorf("unsupported stop reason: %s", reason)
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	for _, control := range p.runs {
		select {
		case control <- evo.CommandStop:
		default:
		}
	}
	for _, sc := range p.scapes {
		if managed, ok := sc.(managedScape); ok {
			if withReason, ok := sc.(reasonAwareManagedScape); ok {
				_ = withReason.StopWithReason(context.Background(), reason)
			} else {
				_ = managed.Stop(context.Background())
			}
		}
	}
	for _, module := range p.supportModules {
		if withReason, ok := module.(reasonAwareSupportModule); ok {
			_ = withReason.StopWithReason(context.Background(), reason)
		} else {
			_ = module.Stop(context.Background())
		}
	}

	p.started = false
	p.lastStopReason = reason
	p.scapes = make(map[string]scape.Scape)
	p.supportModules = make(map[string]SupportModule)
	p.publicScapes = make(map[string]PublicScapeSummary)
	p.publicScapeByType = make(map[string]string)
	p.runs = make(map[string]chan evo.MonitorCommand)
	return nil
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
		OpMode:               cfg.OpMode,
		Mutation:             cfg.Mutation,
		PopulationSize:       cfg.PopulationSize,
		EliteCount:           cfg.EliteCount,
		SurvivalPercentage:   cfg.SurvivalPercentage,
		SpecieSizeLimit:      cfg.SpecieSizeLimit,
		Generations:          cfg.Generations,
		GenerationOffset:     cfg.InitialGeneration,
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
	if cfg.InitialGeneration > 0 {
		result, err = p.mergeExistingRunHistory(ctx, persistenceRunID(cfg, runID), result)
		if err != nil {
			return EvolutionResult{}, err
		}
	}
	finalGenomes := make([]model.Genome, 0, len(result.FinalPopulation))
	for _, scored := range result.FinalPopulation {
		finalGenomes = append(finalGenomes, scored.Genome)
	}
	executedGenerations := len(result.BestByGeneration) + cfg.InitialGeneration
	persistenceRunID := persistenceRunID(cfg, runID)
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

func persistenceRunID(cfg EvolutionConfig, fallback string) string {
	if cfg.RunID != "" {
		return cfg.RunID
	}
	return fallback
}

func (p *Polis) mergeExistingRunHistory(ctx context.Context, runID string, current evo.RunResult) (evo.RunResult, error) {
	if runID == "" {
		return current, nil
	}

	if history, ok, err := p.store.GetFitnessHistory(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok {
		current.BestByGeneration = append(append([]float64{}, history...), current.BestByGeneration...)
	}

	if diagnostics, ok, err := p.store.GetGenerationDiagnostics(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok {
		prefix := make([]evo.GenerationDiagnostics, 0, len(diagnostics))
		for _, item := range diagnostics {
			prefix = append(prefix, evo.GenerationDiagnostics{
				Generation:            item.Generation,
				BestFitness:           item.BestFitness,
				MeanFitness:           item.MeanFitness,
				MinFitness:            item.MinFitness,
				SpeciesCount:          item.SpeciesCount,
				FingerprintDiversity:  item.FingerprintDiversity,
				SpeciationThreshold:   item.SpeciationThreshold,
				TargetSpeciesCount:    item.TargetSpeciesCount,
				MeanSpeciesSize:       item.MeanSpeciesSize,
				LargestSpeciesSize:    item.LargestSpeciesSize,
				TuningInvocations:     item.TuningInvocations,
				TuningAttempts:        item.TuningAttempts,
				TuningEvaluations:     item.TuningEvaluations,
				TuningAccepted:        item.TuningAccepted,
				TuningRejected:        item.TuningRejected,
				TuningGoalHits:        item.TuningGoalHits,
				TuningAcceptRate:      item.TuningAcceptRate,
				TuningEvalsPerAttempt: item.TuningEvalsPerAttempt,
			})
		}
		current.GenerationDiagnostics = append(prefix, current.GenerationDiagnostics...)
	}

	if speciesHistory, ok, err := p.store.GetSpeciesHistory(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok {
		prefix := make([]evo.SpeciesGeneration, 0, len(speciesHistory))
		for _, generation := range speciesHistory {
			species := make([]evo.SpeciesMetrics, 0, len(generation.Species))
			for _, metric := range generation.Species {
				species = append(species, evo.SpeciesMetrics{
					Key:         metric.Key,
					Size:        metric.Size,
					MeanFitness: metric.MeanFitness,
					BestFitness: metric.BestFitness,
				})
			}
			prefix = append(prefix, evo.SpeciesGeneration{
				Generation:     generation.Generation,
				Species:        species,
				NewSpecies:     append([]string{}, generation.NewSpecies...),
				ExtinctSpecies: append([]string{}, generation.ExtinctSpecies...),
			})
		}
		current.SpeciesHistory = append(prefix, current.SpeciesHistory...)
	}

	if lineage, ok, err := p.store.GetLineage(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok {
		prefix := make([]evo.LineageRecord, 0, len(lineage))
		for _, rec := range lineage {
			prefix = append(prefix, evo.LineageRecord{
				GenomeID:    rec.GenomeID,
				ParentID:    rec.ParentID,
				Generation:  rec.Generation,
				Operation:   rec.Operation,
				Fingerprint: rec.Fingerprint,
				Summary: evo.TopologySummary{
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
		current.Lineage = append(prefix, current.Lineage...)
	}

	if top, ok, err := p.store.GetTopGenomes(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok && len(top) > 0 {
		merged := make([]evo.ScoredGenome, 0, len(top)+len(current.FinalPopulation))
		for _, item := range top {
			merged = append(merged, evo.ScoredGenome{
				Genome:  item.Genome,
				Fitness: item.Fitness,
			})
		}
		merged = append(merged, current.FinalPopulation...)
		sort.Slice(merged, func(i, j int) bool {
			return merged[i].Fitness > merged[j].Fitness
		})
		seen := make(map[string]struct{}, len(merged))
		unique := make([]evo.ScoredGenome, 0, len(merged))
		for _, item := range merged {
			if item.Genome.ID != "" {
				if _, exists := seen[item.Genome.ID]; exists {
					continue
				}
				seen[item.Genome.ID] = struct{}{}
			}
			unique = append(unique, item)
		}
		current.FinalPopulation = unique
	}

	return current, nil
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
			Generation:            d.Generation,
			BestFitness:           d.BestFitness,
			MeanFitness:           d.MeanFitness,
			MinFitness:            d.MinFitness,
			SpeciesCount:          d.SpeciesCount,
			FingerprintDiversity:  d.FingerprintDiversity,
			SpeciationThreshold:   d.SpeciationThreshold,
			TargetSpeciesCount:    d.TargetSpeciesCount,
			MeanSpeciesSize:       d.MeanSpeciesSize,
			LargestSpeciesSize:    d.LargestSpeciesSize,
			TuningInvocations:     d.TuningInvocations,
			TuningAttempts:        d.TuningAttempts,
			TuningEvaluations:     d.TuningEvaluations,
			TuningAccepted:        d.TuningAccepted,
			TuningRejected:        d.TuningRejected,
			TuningGoalHits:        d.TuningGoalHits,
			TuningAcceptRate:      d.TuningAcceptRate,
			TuningEvalsPerAttempt: d.TuningEvalsPerAttempt,
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

func (p *Polis) ActiveSupportModules() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	names := make([]string, 0, len(p.supportModules))
	for name := range p.supportModules {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (p *Polis) ActivePublicScapes() []PublicScapeSummary {
	p.mu.RLock()
	defer p.mu.RUnlock()

	names := make([]string, 0, len(p.publicScapes))
	for name := range p.publicScapes {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]PublicScapeSummary, 0, len(names))
	for _, name := range names {
		summary := p.publicScapes[name]
		summary.Parameters = append([]any(nil), summary.Parameters...)
		out = append(out, summary)
	}
	return out
}

func (p *Polis) Started() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.started
}

func (p *Polis) LastStopReason() StopReason {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.lastStopReason
}

type managedScape interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
}

type reasonAwareManagedScape interface {
	managedScape
	StopWithReason(ctx context.Context, reason StopReason) error
}

type reasonAwareSupportModule interface {
	SupportModule
	StopWithReason(ctx context.Context, reason StopReason) error
}

func isValidStopReason(reason StopReason) bool {
	switch reason {
	case StopReasonNormal, StopReasonShutdown:
		return true
	default:
		return false
	}
}

func stopSupportModules(ctx context.Context, modules []SupportModule) {
	for i := len(modules) - 1; i >= 0; i-- {
		_ = modules[i].Stop(ctx)
	}
}

func stopManagedScapes(ctx context.Context, scapes []managedScape) {
	for i := len(scapes) - 1; i >= 0; i-- {
		_ = scapes[i].Stop(ctx)
	}
}
