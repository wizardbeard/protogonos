package platform

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"protogonos/internal/evo"
	"protogonos/internal/genotype"
	"protogonos/internal/model"
	"protogonos/internal/scape"
	"protogonos/internal/storage"
)

type linearScape struct{}

func (linearScape) Name() string { return "linear" }

func (linearScape) Evaluate(ctx context.Context, a scape.Agent) (scape.Fitness, scape.Trace, error) {
	runner := a.(interface {
		RunStep(context.Context, []float64) ([]float64, error)
	})
	out, err := runner.RunStep(ctx, []float64{1})
	if err != nil {
		return 0, nil, err
	}
	delta := out[0] - 1
	mse := delta * delta
	return scape.Fitness(1 - mse), scape.Trace{"mse": mse}, nil
}

func TestPolisRunEvolution(t *testing.T) {
	store := storage.NewMemoryStore()
	p := NewPolis(Config{Store: store})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init: %v", err)
	}
	if err := p.RegisterScape(linearScape{}); err != nil {
		t.Fatalf("register scape: %v", err)
	}

	initial := []model.Genome{
		linearGenome("g0", -1),
		linearGenome("g1", -0.8),
		linearGenome("g2", -0.5),
		linearGenome("g3", -0.2),
	}

	result, err := p.RunEvolution(context.Background(), EvolutionConfig{
		ScapeName:       "linear",
		PopulationSize:  len(initial),
		Generations:     5,
		EliteCount:      2,
		Workers:         2,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
		Mutation:        &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(5)), MaxDelta: 0.4},
		Initial:         initial,
	})
	if err != nil {
		t.Fatalf("run evolution: %v", err)
	}

	if len(result.BestByGeneration) != 5 {
		t.Fatalf("expected 5 generations, got %d", len(result.BestByGeneration))
	}
	if result.BestFinalFitness == 0 {
		t.Fatalf("expected non-zero final fitness")
	}

	pop, ok, err := store.GetPopulation(context.Background(), "evo:linear:1")
	if err != nil {
		t.Fatalf("load persisted population: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted population snapshot")
	}
	if pop.Generation != 5 {
		t.Fatalf("expected persisted generation=5, got %d", pop.Generation)
	}
	if len(pop.AgentIDs) == 0 {
		t.Fatal("expected persisted population agent ids")
	}
	if _, ok, err := store.GetGenome(context.Background(), pop.AgentIDs[0]); err != nil || !ok {
		t.Fatalf("expected persisted genome, ok=%t err=%v", ok, err)
	}
	lineage, ok, err := store.GetLineage(context.Background(), "evo:linear:1")
	if err != nil {
		t.Fatalf("load persisted lineage: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted lineage")
	}
	if len(lineage) == 0 {
		t.Fatal("expected non-empty persisted lineage")
	}
	if len(lineage) != len(result.Lineage) {
		t.Fatalf("lineage count mismatch: persisted=%d result=%d", len(lineage), len(result.Lineage))
	}
	history, ok, err := store.GetFitnessHistory(context.Background(), "evo:linear:1")
	if err != nil {
		t.Fatalf("load persisted history: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted fitness history")
	}
	if len(history) != len(result.BestByGeneration) {
		t.Fatalf("history length mismatch: persisted=%d result=%d", len(history), len(result.BestByGeneration))
	}
	diagnostics, ok, err := store.GetGenerationDiagnostics(context.Background(), "evo:linear:1")
	if err != nil {
		t.Fatalf("load persisted diagnostics: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted generation diagnostics")
	}
	if len(diagnostics) != len(result.BestByGeneration) {
		t.Fatalf("diagnostics length mismatch: persisted=%d result=%d", len(diagnostics), len(result.BestByGeneration))
	}
	speciesHistory, ok, err := store.GetSpeciesHistory(context.Background(), "evo:linear:1")
	if err != nil {
		t.Fatalf("load persisted species history: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted species history")
	}
	if len(speciesHistory) != len(result.BestByGeneration) {
		t.Fatalf("species history length mismatch: persisted=%d result=%d", len(speciesHistory), len(result.BestByGeneration))
	}
	top, ok, err := store.GetTopGenomes(context.Background(), "evo:linear:1")
	if err != nil {
		t.Fatalf("load persisted top genomes: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted top genomes")
	}
	if len(top) == 0 {
		t.Fatal("expected at least one persisted top genome")
	}
	summary, ok, err := store.GetScapeSummary(context.Background(), "linear")
	if err != nil {
		t.Fatalf("load persisted scape summary: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted scape summary")
	}
	if summary.BestFitness != result.BestFinalFitness {
		t.Fatalf("scape summary best mismatch: got=%f want=%f", summary.BestFitness, result.BestFinalFitness)
	}
}

func TestPolisRunEvolutionRespectsFitnessGoal(t *testing.T) {
	store := storage.NewMemoryStore()
	p := NewPolis(Config{Store: store})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init: %v", err)
	}
	if err := p.RegisterScape(linearScape{}); err != nil {
		t.Fatalf("register scape: %v", err)
	}

	initial := []model.Genome{
		linearGenome("g0", 1.0),
		linearGenome("g1", 0.8),
		linearGenome("g2", 0.6),
		linearGenome("g3", 0.4),
	}

	result, err := p.RunEvolution(context.Background(), EvolutionConfig{
		ScapeName:       "linear",
		PopulationSize:  len(initial),
		Generations:     6,
		FitnessGoal:     0.99,
		EliteCount:      1,
		Workers:         2,
		Seed:            9,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
		Mutation:        &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(50)), MaxDelta: 0.4},
		Initial:         initial,
	})
	if err != nil {
		t.Fatalf("run evolution: %v", err)
	}
	if len(result.BestByGeneration) != 1 {
		t.Fatalf("expected early stop at fitness goal, got %d generations", len(result.BestByGeneration))
	}

	pop, ok, err := store.GetPopulation(context.Background(), "evo:linear:9")
	if err != nil {
		t.Fatalf("load persisted population: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted population snapshot")
	}
	if pop.Generation != 1 {
		t.Fatalf("expected persisted generation=1 after early stop, got %d", pop.Generation)
	}
}

func TestPolisRunEvolutionRespectsEvaluationLimit(t *testing.T) {
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
	}

	result, err := p.RunEvolution(context.Background(), EvolutionConfig{
		ScapeName:        "linear",
		PopulationSize:   len(initial),
		Generations:      6,
		EvaluationsLimit: len(initial),
		EliteCount:       1,
		Workers:          2,
		Seed:             10,
		InputNeuronIDs:   []string{"i"},
		OutputNeuronIDs:  []string{"o"},
		Mutation:         &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(51)), MaxDelta: 0.4},
		Initial:          initial,
	})
	if err != nil {
		t.Fatalf("run evolution: %v", err)
	}
	if len(result.BestByGeneration) != 1 {
		t.Fatalf("expected early stop at evaluation limit, got %d generations", len(result.BestByGeneration))
	}
}

func TestPolisRunControlPauseContinueStop(t *testing.T) {
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
	}
	control := make(chan evo.MonitorCommand, 2)
	control <- evo.CommandPause
	runID := "control-run"

	resultCh := make(chan EvolutionResult, 1)
	errCh := make(chan error, 1)
	go func() {
		result, err := p.RunEvolution(context.Background(), EvolutionConfig{
			RunID:           runID,
			ScapeName:       "linear",
			PopulationSize:  len(initial),
			Generations:     4,
			EliteCount:      1,
			Workers:         2,
			Seed:            77,
			InputNeuronIDs:  []string{"i"},
			OutputNeuronIDs: []string{"o"},
			Mutation:        &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(90)), MaxDelta: 0.3},
			Control:         control,
			Initial:         initial,
		})
		if err != nil {
			errCh <- err
			return
		}
		resultCh <- result
	}()

	select {
	case <-resultCh:
		t.Fatal("expected paused run not to complete before continue")
	case err := <-errCh:
		t.Fatalf("unexpected run error: %v", err)
	case <-time.After(30 * time.Millisecond):
	}

	if err := p.ContinueRun(runID); err != nil {
		t.Fatalf("continue run: %v", err)
	}
	if err := p.PauseRun(runID); err != nil {
		t.Fatalf("pause run: %v", err)
	}
	if err := p.StopRun(runID); err != nil {
		t.Fatalf("stop run: %v", err)
	}

	select {
	case err := <-errCh:
		t.Fatalf("unexpected run error after stop: %v", err)
	case result := <-resultCh:
		if len(result.BestByGeneration) == 0 || len(result.BestByGeneration) >= 4 {
			t.Fatalf("expected early stop with partial progress, got generations=%d", len(result.BestByGeneration))
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for controlled run completion")
	}
	if err := p.ContinueRun(runID); err == nil {
		t.Fatal("expected continue on inactive run to fail")
	}
}

func TestPolisRunEvolutionSupportsGenerationOffset(t *testing.T) {
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
	}

	_, err := p.RunEvolution(context.Background(), EvolutionConfig{
		RunID:           "offset-base",
		ScapeName:       "linear",
		PopulationSize:  len(initial),
		Generations:     2,
		EliteCount:      1,
		Workers:         2,
		Seed:            111,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
		Mutation:        &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(4001)), MaxDelta: 0.35},
		Initial:         initial,
	})
	if err != nil {
		t.Fatalf("base run evolution: %v", err)
	}

	pop, continued, err := genotype.LoadPopulationSnapshot(context.Background(), store, "offset-base")
	if err != nil {
		t.Fatalf("load base population: %v", err)
	}
	if pop.Generation != 2 {
		t.Fatalf("expected base generation=2, got %d", pop.Generation)
	}

	continuedResult, err := p.RunEvolution(context.Background(), EvolutionConfig{
		RunID:             "offset-continued",
		ScapeName:         "linear",
		PopulationSize:    len(continued),
		Generations:       2,
		InitialGeneration: pop.Generation,
		EliteCount:        1,
		Workers:           2,
		Seed:              112,
		InputNeuronIDs:    []string{"i"},
		OutputNeuronIDs:   []string{"o"},
		Mutation:          &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(4002)), MaxDelta: 0.35},
		Initial:           continued,
	})
	if err != nil {
		t.Fatalf("continued run evolution: %v", err)
	}
	if len(continuedResult.GenerationDiagnostics) != 2 {
		t.Fatalf("expected 2 diagnostics in continued run, got %d", len(continuedResult.GenerationDiagnostics))
	}
	if continuedResult.GenerationDiagnostics[0].Generation != 3 {
		t.Fatalf("expected continued first generation number=3, got %d", continuedResult.GenerationDiagnostics[0].Generation)
	}

	continuedPop, ok, err := store.GetPopulation(context.Background(), "offset-continued")
	if err != nil {
		t.Fatalf("load continued population: %v", err)
	}
	if !ok {
		t.Fatal("expected continued population snapshot")
	}
	if continuedPop.Generation != 4 {
		t.Fatalf("expected continued persisted generation=4, got %d", continuedPop.Generation)
	}
}

func linearGenome(id string, weight float64) model.Genome {
	return model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
		ID:              id,
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{{ID: "s", From: "i", To: "o", Weight: weight, Enabled: true}},
	}
}
