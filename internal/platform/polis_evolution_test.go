package platform

import (
	"context"
	"math/rand"
	"testing"

	"protogonos/internal/evo"
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
