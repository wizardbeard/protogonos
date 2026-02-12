package platform

import (
	"context"
	"math/rand"
	"testing"

	"protogonos/internal/evo"
	"protogonos/internal/model"
	"protogonos/internal/storage"
	"protogonos/internal/tuning"
)

func TestParityRegressionSelectionAndSpeciation(t *testing.T) {
	store := storage.NewMemoryStore()
	p := NewPolis(Config{Store: store})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init: %v", err)
	}
	if err := p.RegisterScape(linearScape{}); err != nil {
		t.Fatalf("register scape: %v", err)
	}

	initial := []model.Genome{
		linearGenome("g0", -1.0),
		linearGenome("g1", -0.8),
		linearGenome("g2", -0.6),
		linearGenome("g3", -0.4),
		linearGenome("g4", -0.2),
		linearGenome("g5", 0.0),
		linearGenome("g6", 0.2),
		linearGenome("g7", 0.4),
	}

	result, err := p.RunEvolution(context.Background(), EvolutionConfig{
		RunID:           "parity-selection-speciation",
		ScapeName:       "linear",
		PopulationSize:  len(initial),
		Generations:     8,
		EliteCount:      2,
		Workers:         2,
		Seed:            101,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
		Mutation:        &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(1001)), MaxDelta: 0.4},
		MutationPolicy: []evo.WeightedMutation{
			{Operator: &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(1001)), MaxDelta: 0.4}, Weight: 0.8},
			{Operator: &evo.AddRandomNeuron{Rand: rand.New(rand.NewSource(1002))}, Weight: 0.2},
		},
		Selector: &evo.SpeciesSharedTournamentSelector{
			Identifier:            evo.TopologySpecieIdentifier{},
			TournamentSize:        3,
			StagnationGenerations: 2,
		},
		Initial: initial,
	})
	if err != nil {
		t.Fatalf("run evolution: %v", err)
	}
	if len(result.BestByGeneration) != 8 {
		t.Fatalf("expected 8 generations, got %d", len(result.BestByGeneration))
	}
	if len(result.GenerationDiagnostics) != 8 {
		t.Fatalf("expected 8 generation diagnostics, got %d", len(result.GenerationDiagnostics))
	}
	if len(result.SpeciesHistory) != 8 {
		t.Fatalf("expected 8 species history rows, got %d", len(result.SpeciesHistory))
	}
	if len(result.SpeciesHistory[0].Species) == 0 {
		t.Fatal("expected non-empty species history at generation 1")
	}
	if result.BestFinalFitness <= result.BestByGeneration[0] {
		t.Fatalf("expected fitness improvement, first=%f final=%f", result.BestByGeneration[0], result.BestFinalFitness)
	}
}

func TestParityRegressionExoselfAndLifecycle(t *testing.T) {
	buildPolis := func() *Polis {
		store := storage.NewMemoryStore()
		p := NewPolis(Config{Store: store})
		if err := p.Init(context.Background()); err != nil {
			t.Fatalf("init: %v", err)
		}
		if err := p.RegisterScape(linearScape{}); err != nil {
			t.Fatalf("register scape: %v", err)
		}
		return p
	}
	initial := []model.Genome{
		linearGenome("g0", -1.0),
		linearGenome("g1", -0.8),
		linearGenome("g2", -0.5),
		linearGenome("g3", -0.2),
	}

	run := func(p *Polis, runID string, tuner tuning.Tuner, attempts int) EvolutionResult {
		result, err := p.RunEvolution(context.Background(), EvolutionConfig{
			RunID:           runID,
			ScapeName:       "linear",
			PopulationSize:  len(initial),
			Generations:     6,
			EliteCount:      1,
			Workers:         2,
			Seed:            202,
			InputNeuronIDs:  []string{"i"},
			OutputNeuronIDs: []string{"o"},
			Mutation:        &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(2001)), MaxDelta: 0.35},
			Selector:        evo.TournamentSelector{PoolSize: 0, TournamentSize: 3},
			Tuner:           tuner,
			TuneAttempts:    attempts,
			Initial: []model.Genome{
				initial[0],
				initial[1],
				initial[2],
				initial[3],
			},
		})
		if err != nil {
			t.Fatalf("run evolution %s: %v", runID, err)
		}
		return result
	}

	without := run(buildPolis(), "parity-exoself-off", nil, 0)
	with := run(buildPolis(), "parity-exoself-on", &tuning.Exoself{
		Rand:               rand.New(rand.NewSource(2200)),
		Steps:              8,
		StepSize:           0.35,
		CandidateSelection: tuning.CandidateSelectBestSoFar,
	}, 4)

	if with.BestFinalFitness < without.BestFinalFitness {
		t.Fatalf("expected exoself not to regress final best: without=%f with=%f", without.BestFinalFitness, with.BestFinalFitness)
	}
	if len(with.Lineage) == 0 {
		t.Fatal("expected lifecycle lineage records")
	}
	if len(with.GenerationDiagnostics) != len(with.BestByGeneration) {
		t.Fatalf("expected diagnostics and history lengths to match, diagnostics=%d history=%d", len(with.GenerationDiagnostics), len(with.BestByGeneration))
	}
}
