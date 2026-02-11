package storage

import (
	"context"
	"testing"

	"protogonos/internal/model"
)

func TestMemoryStoreLineageRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	input := []model.LineageRecord{{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		GenomeID:        "g1",
		Generation:      1,
		Operation:       "mutate",
	}}
	if err := store.SaveLineage(ctx, "run-1", input); err != nil {
		t.Fatalf("save lineage: %v", err)
	}

	output, ok, err := store.GetLineage(ctx, "run-1")
	if err != nil {
		t.Fatalf("get lineage: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted lineage")
	}
	if len(output) != 1 || output[0].GenomeID != "g1" {
		t.Fatalf("unexpected lineage: %+v", output)
	}
}

func TestMemoryStoreFitnessHistoryRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	input := []float64{0.1, 0.2, 0.3}
	if err := store.SaveFitnessHistory(ctx, "run-1", input); err != nil {
		t.Fatalf("save history: %v", err)
	}
	output, ok, err := store.GetFitnessHistory(ctx, "run-1")
	if err != nil {
		t.Fatalf("get history: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted fitness history")
	}
	if len(output) != len(input) || output[2] != input[2] {
		t.Fatalf("unexpected history: %+v", output)
	}
}

func TestMemoryStoreGenerationDiagnosticsRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	input := []model.GenerationDiagnostics{
		{Generation: 1, BestFitness: 0.8, MeanFitness: 0.6, MinFitness: 0.2, SpeciesCount: 2, FingerprintDiversity: 2},
		{Generation: 2, BestFitness: 0.9, MeanFitness: 0.7, MinFitness: 0.3, SpeciesCount: 3, FingerprintDiversity: 3},
	}
	if err := store.SaveGenerationDiagnostics(ctx, "run-1", input); err != nil {
		t.Fatalf("save diagnostics: %v", err)
	}
	output, ok, err := store.GetGenerationDiagnostics(ctx, "run-1")
	if err != nil {
		t.Fatalf("get diagnostics: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted diagnostics")
	}
	if len(output) != len(input) || output[1].SpeciesCount != input[1].SpeciesCount {
		t.Fatalf("unexpected diagnostics: %+v", output)
	}
}
