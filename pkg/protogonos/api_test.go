package protogonos

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestClientRunRunsAndExport(t *testing.T) {
	base := t.TempDir()
	benchmarksDir := filepath.Join(base, "benchmarks")
	exportsDir := filepath.Join(base, "exports")

	client, err := New(Options{
		StoreKind:     "memory",
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	t.Cleanup(func() {
		_ = client.Close()
	})

	summary, err := client.Run(context.Background(), RunRequest{
		Scape:       "xor",
		Population:  8,
		Generations: 2,
		Seed:        42,
		Workers:     2,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if summary.RunID == "" {
		t.Fatal("expected run id")
	}
	if len(summary.BestByGeneration) != 2 {
		t.Fatalf("unexpected generation history length: %d", len(summary.BestByGeneration))
	}

	runs, err := client.Runs(context.Background(), RunsRequest{Limit: 5})
	if err != nil {
		t.Fatalf("runs: %v", err)
	}
	if len(runs) == 0 || runs[0].RunID != summary.RunID {
		t.Fatalf("expected latest run %s in runs list: %+v", summary.RunID, runs)
	}

	lineage, err := client.Lineage(context.Background(), LineageRequest{RunID: summary.RunID, Limit: 10})
	if err != nil {
		t.Fatalf("lineage: %v", err)
	}
	if len(lineage) == 0 {
		t.Fatal("expected non-empty lineage")
	}
	history, err := client.FitnessHistory(context.Background(), FitnessHistoryRequest{RunID: summary.RunID, Limit: 10})
	if err != nil {
		t.Fatalf("fitness history: %v", err)
	}
	if len(history) == 0 {
		t.Fatal("expected non-empty fitness history")
	}
	diagnostics, err := client.Diagnostics(context.Background(), DiagnosticsRequest{RunID: summary.RunID, Limit: 10})
	if err != nil {
		t.Fatalf("diagnostics: %v", err)
	}
	if len(diagnostics) == 0 {
		t.Fatal("expected non-empty diagnostics")
	}
	speciesHistory, err := client.SpeciesHistory(context.Background(), SpeciesHistoryRequest{RunID: summary.RunID, Limit: 10})
	if err != nil {
		t.Fatalf("species history: %v", err)
	}
	if len(speciesHistory) == 0 {
		t.Fatal("expected non-empty species history")
	}
	speciesDiff, err := client.SpeciesDiff(context.Background(), SpeciesDiffRequest{RunID: summary.RunID})
	if err != nil {
		t.Fatalf("species diff: %v", err)
	}
	if speciesDiff.RunID != summary.RunID {
		t.Fatalf("species diff run mismatch: got=%s want=%s", speciesDiff.RunID, summary.RunID)
	}
	if speciesDiff.FromGeneration <= 0 || speciesDiff.ToGeneration <= speciesDiff.FromGeneration {
		t.Fatalf("unexpected species diff generations: from=%d to=%d", speciesDiff.FromGeneration, speciesDiff.ToGeneration)
	}
	top, err := client.TopGenomes(context.Background(), TopGenomesRequest{RunID: summary.RunID, Limit: 5})
	if err != nil {
		t.Fatalf("top genomes: %v", err)
	}
	if len(top) == 0 {
		t.Fatal("expected non-empty top genomes")
	}
	scapeSummary, err := client.ScapeSummary(context.Background(), "xor")
	if err != nil {
		t.Fatalf("scape summary: %v", err)
	}
	if scapeSummary.Name != "xor" {
		t.Fatalf("unexpected scape summary: %+v", scapeSummary)
	}

	exported, err := client.Export(context.Background(), ExportRequest{Latest: true})
	if err != nil {
		t.Fatalf("export latest: %v", err)
	}
	if exported.RunID != summary.RunID {
		t.Fatalf("exported run mismatch: got=%s want=%s", exported.RunID, summary.RunID)
	}

	for _, file := range []string{"config.json", "fitness_history.json", "top_genomes.json", "lineage.json", "generation_diagnostics.json", "species_history.json"} {
		if _, err := os.Stat(filepath.Join(exported.Directory, file)); err != nil {
			t.Fatalf("expected exported file %s: %v", file, err)
		}
	}
}

func TestClientRunRejectsUnknownSelectionAndPostprocessor(t *testing.T) {
	client, err := New(Options{StoreKind: "memory", BenchmarksDir: t.TempDir(), ExportsDir: t.TempDir()})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	t.Cleanup(func() {
		_ = client.Close()
	})

	_, err = client.Run(context.Background(), RunRequest{
		Scape:       "xor",
		Population:  6,
		Generations: 1,
		Selection:   "unknown",
	})
	if err == nil {
		t.Fatal("expected selection validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:                "xor",
		Population:           6,
		Generations:          1,
		Selection:            "elite",
		FitnessPostprocessor: "unknown",
	})
	if err == nil {
		t.Fatal("expected postprocessor validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:             "xor",
		Population:        6,
		Generations:       1,
		Selection:         "elite",
		TopologicalPolicy: "unknown",
	})
	if err == nil {
		t.Fatal("expected topological policy validation error")
	}
}

func TestClientRunAcceptsReferenceStrategyAliases(t *testing.T) {
	base := t.TempDir()
	client, err := New(Options{
		StoreKind:     "memory",
		BenchmarksDir: filepath.Join(base, "benchmarks"),
		ExportsDir:    filepath.Join(base, "exports"),
	})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	t.Cleanup(func() {
		_ = client.Close()
	})

	for _, selection := range []string{"hof_competition", "competition", "top3"} {
		_, err = client.Run(context.Background(), RunRequest{
			Scape:           "xor",
			Population:      8,
			Generations:     2,
			Selection:       selection,
			EnableTuning:    true,
			TuneSelection:   "dynamic_random",
			TuneAttempts:    2,
			TuneSteps:       3,
			TuneStepSize:    0.25,
			WeightPerturb:   1,
			WeightAddNeuron: 0.2,
		})
		if err != nil {
			t.Fatalf("run with alias %s: %v", selection, err)
		}
	}
}

func TestClientRunRejectsNegativeNumericConfig(t *testing.T) {
	client, err := New(Options{StoreKind: "memory", BenchmarksDir: t.TempDir(), ExportsDir: t.TempDir()})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	t.Cleanup(func() {
		_ = client.Close()
	})

	_, err = client.Run(context.Background(), RunRequest{
		Scape:       "xor",
		Population:  -1,
		Generations: 2,
	})
	if err == nil {
		t.Fatal("expected population validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:             "xor",
		Population:        6,
		Generations:       2,
		TuneStepSize:      -0.1,
		EnableTuning:      true,
		TuneDurationParam: 1,
	})
	if err == nil {
		t.Fatal("expected tune step size validation error")
	}
}
