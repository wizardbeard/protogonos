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

	scapeSummary := model.ScapeSummary{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		Name:            "xor",
		Description:     "xor summary",
		BestFitness:     0.95,
	}
	if err := store.SaveScapeSummary(ctx, scapeSummary); err != nil {
		t.Fatalf("save scape summary: %v", err)
	}
	loadedSummary, ok, err := store.GetScapeSummary(ctx, "xor")
	if err != nil {
		t.Fatalf("get scape summary: %v", err)
	}
	if !ok {
		t.Fatal("expected scape summary xor")
	}
	if loadedSummary.BestFitness != scapeSummary.BestFitness {
		t.Fatalf("unexpected scape summary loaded: %+v", loadedSummary)
	}

	history := []float64{0.5, 0.7, 0.9}
	if err := store.SaveFitnessHistory(ctx, "run-1", history); err != nil {
		t.Fatalf("save history: %v", err)
	}
	loadedHistory, ok, err := store.GetFitnessHistory(ctx, "run-1")
	if err != nil {
		t.Fatalf("get history: %v", err)
	}
	if !ok {
		t.Fatal("expected fitness history run-1")
	}
	if len(loadedHistory) != len(history) || loadedHistory[1] != history[1] {
		t.Fatalf("unexpected history loaded: %+v", loadedHistory)
	}

	diagnostics := []model.GenerationDiagnostics{
		{Generation: 1, BestFitness: 0.7, MeanFitness: 0.5, MinFitness: 0.1, SpeciesCount: 2, FingerprintDiversity: 2},
	}
	if err := store.SaveGenerationDiagnostics(ctx, "run-1", diagnostics); err != nil {
		t.Fatalf("save diagnostics: %v", err)
	}
	loadedDiagnostics, ok, err := store.GetGenerationDiagnostics(ctx, "run-1")
	if err != nil {
		t.Fatalf("get diagnostics: %v", err)
	}
	if !ok {
		t.Fatal("expected diagnostics run-1")
	}
	if len(loadedDiagnostics) != 1 || loadedDiagnostics[0].Generation != 1 {
		t.Fatalf("unexpected diagnostics loaded: %+v", loadedDiagnostics)
	}

	top := []model.TopGenomeRecord{
		{Rank: 1, Fitness: 0.9, Genome: model.Genome{ID: "g1"}},
	}
	if err := store.SaveTopGenomes(ctx, "run-1", top); err != nil {
		t.Fatalf("save top genomes: %v", err)
	}
	loadedTop, ok, err := store.GetTopGenomes(ctx, "run-1")
	if err != nil {
		t.Fatalf("get top genomes: %v", err)
	}
	if !ok {
		t.Fatal("expected top genomes run-1")
	}
	if len(loadedTop) != 1 || loadedTop[0].Rank != 1 {
		t.Fatalf("unexpected top genomes loaded: %+v", loadedTop)
	}

	speciesHistory := []model.SpeciesGeneration{
		{
			Generation:     1,
			Species:        []model.SpeciesMetrics{{Key: "sp-1", Size: 2, MeanFitness: 0.5, BestFitness: 0.7}},
			NewSpecies:     []string{"sp-1"},
			ExtinctSpecies: []string{},
		},
	}
	if err := store.SaveSpeciesHistory(ctx, "run-1", speciesHistory); err != nil {
		t.Fatalf("save species history: %v", err)
	}
	loadedSpeciesHistory, ok, err := store.GetSpeciesHistory(ctx, "run-1")
	if err != nil {
		t.Fatalf("get species history: %v", err)
	}
	if !ok {
		t.Fatal("expected species history run-1")
	}
	if len(loadedSpeciesHistory) != 1 || loadedSpeciesHistory[0].Generation != 1 {
		t.Fatalf("unexpected species history loaded: %+v", loadedSpeciesHistory)
	}

	lineage := []model.LineageRecord{
		{
			VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
			GenomeID:        "g1",
			ParentID:        "",
			Generation:      0,
			Operation:       "seed",
			Fingerprint:     "abc",
			Summary: model.LineageSummary{
				TotalNeurons:           2,
				TotalSynapses:          1,
				TotalRecurrentSynapses: 0,
				TotalSensors:           1,
				TotalActuators:         1,
				ActivationDistribution: map[string]int{"identity": 2},
				AggregatorDistribution: map[string]int{"dot_product": 2},
			},
		},
	}
	if err := store.SaveLineage(ctx, "run-1", lineage); err != nil {
		t.Fatalf("save lineage: %v", err)
	}
	loadedLineage, ok, err := store.GetLineage(ctx, "run-1")
	if err != nil {
		t.Fatalf("get lineage: %v", err)
	}
	if !ok {
		t.Fatal("expected lineage run-1")
	}
	if len(loadedLineage) != 1 || loadedLineage[0].GenomeID != "g1" {
		t.Fatalf("unexpected lineage loaded: %+v", loadedLineage)
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
