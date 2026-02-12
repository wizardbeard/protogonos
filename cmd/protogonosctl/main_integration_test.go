//go:build sqlite

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"protogonos/internal/stats"
)

func TestRunCommandSQLiteCreatesArtifacts(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "11",
		"--workers", "2",
	}

	if err := run(context.Background(), args); err != nil {
		t.Fatalf("run command: %v", err)
	}

	if _, err := os.Stat(dbPath); err != nil {
		t.Fatalf("expected sqlite db at %s: %v", dbPath, err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}

	runID := entries[0].RunID
	for _, file := range []string{"config.json", "fitness_history.json", "top_genomes.json", "lineage.json", "generation_diagnostics.json", "species_history.json"} {
		path := filepath.Join("benchmarks", runID, file)
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected artifact %s: %v", path, err)
		}
	}

	lineageData, err := os.ReadFile(filepath.Join("benchmarks", runID, "lineage.json"))
	if err != nil {
		t.Fatalf("read lineage: %v", err)
	}
	var lineage []stats.LineageEntry
	if err := json.Unmarshal(lineageData, &lineage); err != nil {
		t.Fatalf("decode lineage: %v", err)
	}
	seenStructural := false
	for _, record := range lineage {
		switch record.Operation {
		case "add_random_synapse", "remove_random_synapse", "add_random_neuron", "remove_random_neuron":
			seenStructural = true
		}
	}
	if !seenStructural {
		t.Fatalf("expected at least one structural mutation in lineage: %+v", lineage)
	}
}

func TestRunCommandSQLiteConfigLoadsMap2RecAndAllowsFlagOverrides(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	configPath := filepath.Join(workdir, "run_config.json")
	cfg := map[string]any{
		"scape": "xor",
		"seed":  71,
		"pmp": map[string]any{
			"init_specie_size": 8,
			"generation_limit": 5,
		},
		"constraint": map[string]any{
			"population_selection_f": "hof_competition",
			"tuning_selection_fs":    []any{"dynamic_random"},
			"mutation_operators": []any{
				[]any{"add_bias", 1},
				[]any{"add_outlink", 1},
				[]any{"add_neuron", 1},
			},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--config", configPath,
		"--gens", "2",
	}
	if err := run(context.Background(), args); err != nil {
		t.Fatalf("run command with config: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected run index entry")
	}
	if entries[0].PopulationSize != 8 {
		t.Fatalf("expected pmp-derived population size 8, got %d", entries[0].PopulationSize)
	}
	if entries[0].Generations != 2 {
		t.Fatalf("expected --gens override to 2, got %d", entries[0].Generations)
	}
}

func TestRunsCommandSQLiteListsPersistedRun(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "21",
		"--workers", "2",
	}

	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	expectedRunID := entries[0].RunID

	output, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"runs",
			"--limit", "1",
		})
	})
	if err != nil {
		t.Fatalf("runs command failed: %v", err)
	}

	if !strings.Contains(output, "run_id="+expectedRunID) {
		t.Fatalf("runs output missing expected run id %s: %s", expectedRunID, output)
	}
}

func TestExportLatestSQLiteCopiesArtifacts(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "31",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	runID := entries[0].RunID

	if err := run(context.Background(), []string{"export", "--latest"}); err != nil {
		t.Fatalf("export latest command: %v", err)
	}

	for _, file := range []string{"config.json", "fitness_history.json", "top_genomes.json", "lineage.json", "generation_diagnostics.json", "species_history.json"} {
		path := filepath.Join("exports", runID, file)
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected exported artifact %s: %v", path, err)
		}
	}
}

func TestLineageCommandSQLiteReadsPersistedLineage(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "41",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	out, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"lineage",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "3",
		})
	})
	if err != nil {
		t.Fatalf("lineage command: %v", err)
	}
	if !strings.Contains(out, "gen=") || !strings.Contains(out, "genome_id=") {
		t.Fatalf("unexpected lineage output: %s", out)
	}
}

func TestFitnessCommandSQLiteReadsPersistedHistory(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "42",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	out, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"fitness",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "2",
		})
	})
	if err != nil {
		t.Fatalf("fitness command: %v", err)
	}
	if !strings.Contains(out, "generation=1") || !strings.Contains(out, "best_fitness=") {
		t.Fatalf("unexpected fitness output: %s", out)
	}
}

func TestDiagnosticsCommandSQLiteReadsPersistedDiagnostics(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "43",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	out, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"diagnostics",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "2",
		})
	})
	if err != nil {
		t.Fatalf("diagnostics command: %v", err)
	}
	if !strings.Contains(out, "generation=1") || !strings.Contains(out, "species=") {
		t.Fatalf("unexpected diagnostics output: %s", out)
	}
}

func TestSpeciesCommandSQLiteReadsPersistedSpeciesHistory(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "430",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	out, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"species",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "1",
		})
	})
	if err != nil {
		t.Fatalf("species command: %v", err)
	}
	if !strings.Contains(out, "generation=1") || !strings.Contains(out, "species_key=") {
		t.Fatalf("unexpected species output: %s", out)
	}
}

func TestSpeciesDiffCommandSQLiteReadsPersistedSpeciesHistory(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "3",
		"--seed", "431",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	out, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"species-diff",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
		})
	})
	if err != nil {
		t.Fatalf("species-diff command: %v", err)
	}
	if !strings.Contains(out, "run_id=") || !strings.Contains(out, "changed") {
		t.Fatalf("unexpected species-diff output: %s", out)
	}
}

func TestTopCommandSQLiteReadsPersistedTopGenomes(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "44",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	out, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"top",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "2",
		})
	})
	if err != nil {
		t.Fatalf("top command: %v", err)
	}
	if !strings.Contains(out, "rank=1") || !strings.Contains(out, "genome_id=") {
		t.Fatalf("unexpected top output: %s", out)
	}
}

func TestScapeSummaryCommandSQLiteReadsPersistedSummary(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	runArgs := []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "45",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	out, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"scape-summary",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--scape", "xor",
		})
	})
	if err != nil {
		t.Fatalf("scape-summary command: %v", err)
	}
	if !strings.Contains(out, "scape=xor") || !strings.Contains(out, "best_fitness=") {
		t.Fatalf("unexpected scape-summary output: %s", out)
	}
}

func TestBenchmarkCommandWritesSummary(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"benchmark",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "8",
		"--gens", "3",
		"--seed", "9",
		"--workers", "2",
		"--min-improvement", "0.0001",
	}
	if err := run(context.Background(), args); err != nil {
		t.Fatalf("benchmark command: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	runID := entries[0].RunID

	summaryPath := filepath.Join("benchmarks", runID, "benchmark_summary.json")
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read benchmark summary: %v", err)
	}

	var summary stats.BenchmarkSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode benchmark summary: %v", err)
	}
	if summary.RunID != runID {
		t.Fatalf("run id mismatch: got=%s want=%s", summary.RunID, runID)
	}
	if summary.Scape != "xor" {
		t.Fatalf("unexpected scape in summary: %s", summary.Scape)
	}
}

func TestBenchmarkCommandConfigLoadsMap2RecAndAllowsFlagOverrides(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	configPath := filepath.Join(workdir, "benchmark_config.json")
	cfg := map[string]any{
		"scape": "xor",
		"seed":  515,
		"pmp": map[string]any{
			"init_specie_size": 9,
			"generation_limit": 6,
		},
		"constraint": map[string]any{
			"population_selection_f": "hof_competition",
			"tuning_selection_fs":    []any{"dynamic_random"},
			"mutation_operators": []any{
				[]any{"add_bias", 1},
				[]any{"add_outlink", 1},
				[]any{"add_neuron", 1},
			},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"benchmark",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--config", configPath,
		"--gens", "2",
		"--min-improvement", "-0.2",
	}
	if err := run(context.Background(), args); err != nil {
		t.Fatalf("benchmark command with config: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	if entries[0].PopulationSize != 9 {
		t.Fatalf("expected pmp-derived population size 9, got %d", entries[0].PopulationSize)
	}
	if entries[0].Generations != 2 {
		t.Fatalf("expected --gens override to 2, got %d", entries[0].Generations)
	}
}

func TestBenchmarkCommandWritesSummaryRegressionMimic(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"benchmark",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "regression-mimic",
		"--pop", "10",
		"--gens", "4",
		"--seed", "12",
		"--workers", "2",
		"--min-improvement", "0.0001",
	}
	if err := run(context.Background(), args); err != nil {
		t.Fatalf("benchmark command: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	runID := entries[0].RunID

	summaryPath := filepath.Join("benchmarks", runID, "benchmark_summary.json")
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read benchmark summary: %v", err)
	}

	var summary stats.BenchmarkSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode benchmark summary: %v", err)
	}
	if summary.RunID != runID {
		t.Fatalf("run id mismatch: got=%s want=%s", summary.RunID, runID)
	}
	if summary.Scape != "regression-mimic" {
		t.Fatalf("unexpected scape in summary: %s", summary.Scape)
	}
}

func TestBenchmarkCommandWritesSummaryCartPoleLite(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"benchmark",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "cart-pole-lite",
		"--pop", "10",
		"--gens", "4",
		"--seed", "13",
		"--workers", "2",
		"--min-improvement", "0.0001",
	}
	if err := run(context.Background(), args); err != nil {
		t.Fatalf("benchmark command: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	runID := entries[0].RunID

	summaryPath := filepath.Join("benchmarks", runID, "benchmark_summary.json")
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read benchmark summary: %v", err)
	}

	var summary stats.BenchmarkSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode benchmark summary: %v", err)
	}
	if summary.RunID != runID {
		t.Fatalf("run id mismatch: got=%s want=%s", summary.RunID, runID)
	}
	if summary.Scape != "cart-pole-lite" {
		t.Fatalf("unexpected scape in summary: %s", summary.Scape)
	}
}

func TestBenchmarkCommandWritesSummaryFlatlandStable(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"benchmark",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "flatland",
		"--pop", "10",
		"--gens", "4",
		"--seed", "1301",
		"--workers", "2",
		"--min-improvement", "-0.2",
	}
	if err := run(context.Background(), args); err != nil {
		t.Fatalf("benchmark command: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	runID := entries[0].RunID
	data, err := os.ReadFile(filepath.Join("benchmarks", runID, "benchmark_summary.json"))
	if err != nil {
		t.Fatalf("read benchmark summary: %v", err)
	}
	var summary stats.BenchmarkSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode benchmark summary: %v", err)
	}
	if summary.Scape != "flatland" {
		t.Fatalf("unexpected scape in summary: %s", summary.Scape)
	}
	if summary.Improvement < -0.2 {
		t.Fatalf("expected bounded stability, got improvement=%f", summary.Improvement)
	}
}

func TestBenchmarkCommandWritesSummaryGTSAStable(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"benchmark",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "gtsa",
		"--pop", "10",
		"--gens", "4",
		"--seed", "1302",
		"--workers", "2",
		"--min-improvement", "-0.2",
	}
	if err := run(context.Background(), args); err != nil {
		t.Fatalf("benchmark command: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	runID := entries[0].RunID
	data, err := os.ReadFile(filepath.Join("benchmarks", runID, "benchmark_summary.json"))
	if err != nil {
		t.Fatalf("read benchmark summary: %v", err)
	}
	var summary stats.BenchmarkSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode benchmark summary: %v", err)
	}
	if summary.Scape != "gtsa" {
		t.Fatalf("unexpected scape in summary: %s", summary.Scape)
	}
	if summary.Improvement < -0.2 {
		t.Fatalf("expected bounded stability, got improvement=%f", summary.Improvement)
	}
}

func TestBenchmarkCommandWritesSummaryFXStable(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	dbPath := filepath.Join(workdir, "protogonos.db")
	args := []string{
		"benchmark",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "fx",
		"--pop", "10",
		"--gens", "4",
		"--seed", "1303",
		"--workers", "2",
		"--min-improvement", "-0.2",
	}
	if err := run(context.Background(), args); err != nil {
		t.Fatalf("benchmark command: %v", err)
	}

	entries, err := stats.ListRunIndex("benchmarks")
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected at least one indexed run")
	}
	runID := entries[0].RunID
	data, err := os.ReadFile(filepath.Join("benchmarks", runID, "benchmark_summary.json"))
	if err != nil {
		t.Fatalf("read benchmark summary: %v", err)
	}
	var summary stats.BenchmarkSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode benchmark summary: %v", err)
	}
	if summary.Scape != "fx" {
		t.Fatalf("unexpected scape in summary: %s", summary.Scape)
	}
	if summary.Improvement < -0.2 {
		t.Fatalf("expected bounded stability, got improvement=%f", summary.Improvement)
	}
}

func TestProfileListCommand(t *testing.T) {
	out, err := captureStdout(func() error {
		return run(context.Background(), []string{"profile", "list"})
	})
	if err != nil {
		t.Fatalf("profile list command: %v", err)
	}
	if !strings.Contains(out, "id=ref-default-xorandxor") {
		t.Fatalf("unexpected profile list output: %s", out)
	}
}

func captureStdout(fn func() error) (string, error) {
	origStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		return "", err
	}

	os.Stdout = w
	runErr := fn()
	_ = w.Close()
	os.Stdout = origStdout

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		_ = r.Close()
		return "", err
	}
	_ = r.Close()
	return buf.String(), runErr
}
