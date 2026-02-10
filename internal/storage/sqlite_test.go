//go:build sqlite

package storage

import (
	"context"
	"path/filepath"
	"testing"

	"protogonos/internal/model"
)

func TestSQLiteStoreGenomeAndPopulationRoundTrip(t *testing.T) {
	ctx := context.Background()
	dbPath := filepath.Join(t.TempDir(), "protogonos.db")

	store := NewSQLiteStore(dbPath)
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})

	genome := model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		ID:              "g1",
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity", Bias: 0.5},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "n1", To: "n1", Weight: 1.25, Enabled: true},
		},
	}
	if err := store.SaveGenome(ctx, genome); err != nil {
		t.Fatalf("save genome: %v", err)
	}

	loadedGenome, ok, err := store.GetGenome(ctx, genome.ID)
	if err != nil {
		t.Fatalf("get genome: %v", err)
	}
	if !ok {
		t.Fatalf("expected genome %s", genome.ID)
	}
	if loadedGenome.ID != genome.ID || len(loadedGenome.Neurons) != len(genome.Neurons) {
		t.Fatalf("unexpected genome loaded: %+v", loadedGenome)
	}

	population := model.Population{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		ID:              "p1",
		AgentIDs:        []string{"a1", "a2"},
		Generation:      3,
	}
	if err := store.SavePopulation(ctx, population); err != nil {
		t.Fatalf("save population: %v", err)
	}

	loadedPopulation, ok, err := store.GetPopulation(ctx, population.ID)
	if err != nil {
		t.Fatalf("get population: %v", err)
	}
	if !ok {
		t.Fatalf("expected population %s", population.ID)
	}
	if loadedPopulation.ID != population.ID || loadedPopulation.Generation != population.Generation {
		t.Fatalf("unexpected population loaded: %+v", loadedPopulation)
	}
}

func TestSQLiteStorePersistsAcrossReopen(t *testing.T) {
	ctx := context.Background()
	dbPath := filepath.Join(t.TempDir(), "protogonos.db")

	first := NewSQLiteStore(dbPath)
	if err := first.Init(ctx); err != nil {
		t.Fatalf("first init: %v", err)
	}
	genome := model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		ID:              "persisted-genome",
	}
	if err := first.SaveGenome(ctx, genome); err != nil {
		t.Fatalf("first save: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}

	second := NewSQLiteStore(dbPath)
	if err := second.Init(ctx); err != nil {
		t.Fatalf("second init: %v", err)
	}
	t.Cleanup(func() {
		_ = second.Close()
	})

	loaded, ok, err := second.GetGenome(ctx, genome.ID)
	if err != nil {
		t.Fatalf("second get: %v", err)
	}
	if !ok || loaded.ID != genome.ID {
		t.Fatalf("expected persisted genome, got ok=%t value=%+v", ok, loaded)
	}
}
