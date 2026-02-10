package platform

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"protogonos/internal/evo"
	"protogonos/internal/model"
	"protogonos/internal/scape"
	"protogonos/internal/storage"
	"protogonos/internal/tuning"
)

type Config struct {
	Store storage.Store
}

type EvolutionConfig struct {
	ScapeName       string
	PopulationSize  int
	Generations     int
	EliteCount      int
	Workers         int
	Seed            int64
	InputNeuronIDs  []string
	OutputNeuronIDs []string
	Mutation        evo.Operator
	MutationPolicy  []evo.WeightedMutation
	Selector        evo.Selector
	Postprocessor   evo.FitnessPostprocessor
	Tuner           tuning.Tuner
	TuneAttempts    int
	Initial         []model.Genome
}

type EvolutionResult struct {
	BestByGeneration []float64
	BestFinalFitness float64
	TopFinal         []evo.ScoredGenome
	Lineage          []evo.LineageRecord
}

type Polis struct {
	store storage.Store

	mu      sync.RWMutex
	scapes  map[string]scape.Scape
	started bool
}

func NewPolis(cfg Config) *Polis {
	return &Polis{
		store:  cfg.Store,
		scapes: make(map[string]scape.Scape),
	}
}

func (p *Polis) Init(ctx context.Context) error {
	if p.store == nil {
		return fmt.Errorf("store is required")
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
	p.scapes[name] = s
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

	monitor, err := evo.NewPopulationMonitor(evo.MonitorConfig{
		Scape:           targetScape,
		Mutation:        cfg.Mutation,
		PopulationSize:  cfg.PopulationSize,
		EliteCount:      cfg.EliteCount,
		Generations:     cfg.Generations,
		Workers:         cfg.Workers,
		Seed:            cfg.Seed,
		InputNeuronIDs:  cfg.InputNeuronIDs,
		OutputNeuronIDs: cfg.OutputNeuronIDs,
		MutationPolicy:  cfg.MutationPolicy,
		Selector:        cfg.Selector,
		Postprocessor:   cfg.Postprocessor,
		Tuner:           cfg.Tuner,
		TuneAttempts:    cfg.TuneAttempts,
	})
	if err != nil {
		return EvolutionResult{}, err
	}

	result, err := monitor.Run(ctx, cfg.Initial)
	if err != nil {
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

	return EvolutionResult{
		BestByGeneration: result.BestByGeneration,
		BestFinalFitness: bestFinal,
		TopFinal:         topFinal,
		Lineage:          result.Lineage,
	}, nil
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
