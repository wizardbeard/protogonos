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

func TestMemoryStoreSpeciesHistoryRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	input := []model.SpeciesGeneration{
		{
			Generation:     1,
			Species:        []model.SpeciesMetrics{{Key: "sp-1", Size: 2, MeanFitness: 0.5, BestFitness: 0.7}},
			NewSpecies:     []string{"sp-1"},
			ExtinctSpecies: []string{},
		},
	}
	if err := store.SaveSpeciesHistory(ctx, "run-1", input); err != nil {
		t.Fatalf("save species history: %v", err)
	}
	output, ok, err := store.GetSpeciesHistory(ctx, "run-1")
	if err != nil {
		t.Fatalf("get species history: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted species history")
	}
	if len(output) != len(input) || output[0].Species[0].Key != input[0].Species[0].Key {
		t.Fatalf("unexpected species history: %+v", output)
	}
}

func TestMemoryStoreTopGenomesRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	input := []model.TopGenomeRecord{
		{Rank: 1, Fitness: 0.9, Genome: model.Genome{ID: "g1"}},
		{Rank: 2, Fitness: 0.8, Genome: model.Genome{ID: "g2"}},
	}
	if err := store.SaveTopGenomes(ctx, "run-1", input); err != nil {
		t.Fatalf("save top genomes: %v", err)
	}
	output, ok, err := store.GetTopGenomes(ctx, "run-1")
	if err != nil {
		t.Fatalf("get top genomes: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted top genomes")
	}
	if len(output) != len(input) || output[0].Genome.ID != input[0].Genome.ID {
		t.Fatalf("unexpected top genomes: %+v", output)
	}
}

func TestMemoryStoreScapeSummaryRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	input := model.ScapeSummary{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		Name:            "xor",
		Description:     "xor summary",
		BestFitness:     0.95,
	}
	if err := store.SaveScapeSummary(ctx, input); err != nil {
		t.Fatalf("save scape summary: %v", err)
	}
	output, ok, err := store.GetScapeSummary(ctx, "xor")
	if err != nil {
		t.Fatalf("get scape summary: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted scape summary")
	}
	if output.Name != input.Name || output.BestFitness != input.BestFitness {
		t.Fatalf("unexpected scape summary: %+v", output)
	}
}

func TestMemoryStoreDeletePopulation(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	pop := model.Population{ID: "pop-del", AgentIDs: []string{"a1"}}
	if err := store.SavePopulation(ctx, pop); err != nil {
		t.Fatalf("save population: %v", err)
	}
	if err := store.DeletePopulation(ctx, pop.ID); err != nil {
		t.Fatalf("delete population: %v", err)
	}
	_, ok, err := store.GetPopulation(ctx, pop.ID)
	if err != nil {
		t.Fatalf("get population after delete: %v", err)
	}
	if ok {
		t.Fatal("expected deleted population to be missing")
	}
}

func TestMemoryStoreDeleteGenome(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	genome := model.Genome{ID: "g-del"}
	if err := store.SaveGenome(ctx, genome); err != nil {
		t.Fatalf("save genome: %v", err)
	}
	if err := store.DeleteGenome(ctx, genome.ID); err != nil {
		t.Fatalf("delete genome: %v", err)
	}
	_, ok, err := store.GetGenome(ctx, genome.ID)
	if err != nil {
		t.Fatalf("get genome after delete: %v", err)
	}
	if ok {
		t.Fatal("expected deleted genome to be missing")
	}
}

func TestMemoryStoreResetClearsData(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	pop := model.Population{ID: "pop-reset", AgentIDs: []string{"a1"}}
	if err := store.SavePopulation(ctx, pop); err != nil {
		t.Fatalf("save population: %v", err)
	}
	if err := store.Reset(ctx); err != nil {
		t.Fatalf("reset: %v", err)
	}
	_, ok, err := store.GetPopulation(ctx, pop.ID)
	if err != nil {
		t.Fatalf("get population after reset: %v", err)
	}
	if ok {
		t.Fatal("expected reset to clear populations")
	}
}
