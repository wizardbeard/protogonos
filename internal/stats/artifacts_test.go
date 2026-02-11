package stats

import (
	"os"
	"path/filepath"
	"testing"

	"protogonos/internal/model"
)

func TestWriteAndExportRunArtifacts(t *testing.T) {
	baseDir := t.TempDir()
	outDir := filepath.Join(t.TempDir(), "exports")

	runID := "run-123"
	artifacts := RunArtifacts{
		Config: RunConfig{
			RunID:          runID,
			Scape:          "xor",
			PopulationSize: 4,
			Generations:    3,
			Seed:           1,
			Workers:        2,
			EliteCount:     1,
		},
		BestByGeneration: []float64{0.5, 0.6, 0.7},
		FinalBestFitness: 0.7,
		TopGenomes: []TopGenome{{
			Rank:    1,
			Fitness: 0.7,
			Genome:  model.Genome{ID: "g1"},
		}},
		Lineage: []LineageEntry{{
			GenomeID:   "g1",
			ParentID:   "",
			Generation: 0,
			Operation:  "seed",
		}},
	}

	runDir, err := WriteRunArtifacts(baseDir, artifacts)
	if err != nil {
		t.Fatalf("write artifacts: %v", err)
	}

	for _, file := range []string{"config.json", "fitness_history.json", "top_genomes.json", "lineage.json"} {
		if _, err := os.Stat(filepath.Join(runDir, file)); err != nil {
			t.Fatalf("expected file %s: %v", file, err)
		}
	}

	exportedDir, err := ExportRunArtifacts(baseDir, runID, outDir)
	if err != nil {
		t.Fatalf("export artifacts: %v", err)
	}

	for _, file := range []string{"config.json", "fitness_history.json", "top_genomes.json", "lineage.json"} {
		if _, err := os.Stat(filepath.Join(exportedDir, file)); err != nil {
			t.Fatalf("expected exported file %s: %v", file, err)
		}
	}

	if err := WriteTuningComparison(runDir, TuningComparison{
		Scape:            "xor",
		PopulationSize:   4,
		Generations:      3,
		Seed:             1,
		WithoutFinalBest: 0.7,
		WithFinalBest:    0.8,
		FinalImprovement: 0.1,
	}); err != nil {
		t.Fatalf("write compare report: %v", err)
	}

	exportedDirWithCompare, err := ExportRunArtifacts(baseDir, runID, outDir)
	if err != nil {
		t.Fatalf("export artifacts with compare: %v", err)
	}
	if _, err := os.Stat(filepath.Join(exportedDirWithCompare, "compare_tuning.json")); err != nil {
		t.Fatalf("expected exported compare report: %v", err)
	}

	if err := WriteBenchmarkSummary(runDir, BenchmarkSummary{
		RunID:          runID,
		Scape:          "xor",
		PopulationSize: 4,
		Generations:    3,
		Seed:           1,
		InitialBest:    0.5,
		FinalBest:      0.7,
		Improvement:    0.2,
		MinImprovement: 0.05,
		Passed:         true,
	}); err != nil {
		t.Fatalf("write benchmark summary: %v", err)
	}

	exportedDirWithBenchmark, err := ExportRunArtifacts(baseDir, runID, outDir)
	if err != nil {
		t.Fatalf("export artifacts with benchmark summary: %v", err)
	}
	if _, err := os.Stat(filepath.Join(exportedDirWithBenchmark, "benchmark_summary.json")); err != nil {
		t.Fatalf("expected exported benchmark summary: %v", err)
	}
}

func TestRunIndexAppendListAndUpsert(t *testing.T) {
	baseDir := t.TempDir()

	err := AppendRunIndex(baseDir, RunIndexEntry{
		RunID:            "run-1",
		Scape:            "xor",
		PopulationSize:   8,
		Generations:      3,
		Seed:             1,
		Workers:          2,
		EliteCount:       1,
		FinalBestFitness: 0.80,
		CreatedAtUTC:     "2026-02-10T10:00:00Z",
	})
	if err != nil {
		t.Fatalf("append run-1: %v", err)
	}

	err = AppendRunIndex(baseDir, RunIndexEntry{
		RunID:            "run-2",
		Scape:            "xor",
		PopulationSize:   8,
		Generations:      3,
		Seed:             2,
		Workers:          2,
		EliteCount:       1,
		FinalBestFitness: 0.82,
		CreatedAtUTC:     "2026-02-10T11:00:00Z",
	})
	if err != nil {
		t.Fatalf("append run-2: %v", err)
	}

	entries, err := ListRunIndex(baseDir)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if entries[0].RunID != "run-2" || entries[1].RunID != "run-1" {
		t.Fatalf("unexpected order: %+v", entries)
	}

	err = AppendRunIndex(baseDir, RunIndexEntry{
		RunID:            "run-1",
		Scape:            "xor",
		PopulationSize:   8,
		Generations:      3,
		Seed:             1,
		Workers:          2,
		EliteCount:       1,
		FinalBestFitness: 0.90,
		CreatedAtUTC:     "2026-02-10T12:00:00Z",
	})
	if err != nil {
		t.Fatalf("upsert run-1: %v", err)
	}

	entries, err = ListRunIndex(baseDir)
	if err != nil {
		t.Fatalf("list after upsert: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries after upsert, got %d", len(entries))
	}
	if entries[0].RunID != "run-1" || entries[0].FinalBestFitness != 0.90 {
		t.Fatalf("unexpected upsert result: %+v", entries[0])
	}
}

func TestRunIndexEqualTimestampPrefersLaterAppend(t *testing.T) {
	baseDir := t.TempDir()
	ts := "2026-02-10T12:00:00Z"

	if err := AppendRunIndex(baseDir, RunIndexEntry{RunID: "run-a", CreatedAtUTC: ts}); err != nil {
		t.Fatalf("append run-a: %v", err)
	}
	if err := AppendRunIndex(baseDir, RunIndexEntry{RunID: "run-b", CreatedAtUTC: ts}); err != nil {
		t.Fatalf("append run-b: %v", err)
	}

	entries, err := ListRunIndex(baseDir)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if entries[0].RunID != "run-b" {
		t.Fatalf("expected latest appended run-b first, got %+v", entries)
	}
}

func TestReadTuningComparison(t *testing.T) {
	baseDir := t.TempDir()
	runID := "run-compare"
	runDir := filepath.Join(baseDir, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir run dir: %v", err)
	}

	if _, ok, err := ReadTuningComparison(baseDir, runID); err != nil || ok {
		t.Fatalf("expected missing compare report; ok=%t err=%v", ok, err)
	}

	want := TuningComparison{
		Scape:            "xor",
		PopulationSize:   8,
		Generations:      2,
		Seed:             5,
		WithoutFinalBest: 0.75,
		WithFinalBest:    0.80,
		FinalImprovement: 0.05,
	}
	if err := WriteTuningComparison(runDir, want); err != nil {
		t.Fatalf("write compare report: %v", err)
	}

	got, ok, err := ReadTuningComparison(baseDir, runID)
	if err != nil {
		t.Fatalf("read compare report: %v", err)
	}
	if !ok {
		t.Fatal("expected compare report to exist")
	}
	if got.FinalImprovement != want.FinalImprovement {
		t.Fatalf("unexpected compare report: got=%+v want=%+v", got, want)
	}
}
