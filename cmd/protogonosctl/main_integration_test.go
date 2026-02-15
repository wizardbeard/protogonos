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
		"scape":                   "xor",
		"seed":                    71,
		"enable_tuning":           true,
		"tune_perturbation_range": 1.6,
		"tune_annealing_factor":   0.9,
		"tune_min_improvement":    0.02,
		"pmp": map[string]any{
			"init_specie_size":    8,
			"specie_size_limit":   2,
			"generation_limit":    5,
			"survival_percentage": 0.6,
			"fitness_goal":        0.85,
			"evaluations_limit":   90,
		},
		"constraint": map[string]any{
			"population_selection_f":             "hof_competition",
			"population_fitness_postprocessor_f": "nsize_proportional",
			"tuning_selection_fs":                []any{"dynamic_random"},
			"tuning_duration_f":                  []any{"const", 2},
			"tot_topological_mutations_fs": []any{
				[]any{"ncount_exponential", 0.9},
			},
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
		"--run-id", "sqlite-config-override-run",
		"--gens", "2",
		"--topo-policy", "const",
		"--topo-count", "2",
		"--fitness-postprocessor", "none",
		"--survival-percentage", "0.4",
		"--specie-size-limit", "4",
		"--fitness-goal", "0.91",
		"--evaluations-limit", "123",
		"--start-paused",
		"--auto-continue-ms", "5",
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
	configData, err := os.ReadFile(filepath.Join("benchmarks", entries[0].RunID, "config.json"))
	if err != nil {
		t.Fatalf("read run config artifact: %v", err)
	}
	var runCfg stats.RunConfig
	if err := json.Unmarshal(configData, &runCfg); err != nil {
		t.Fatalf("decode run config artifact: %v", err)
	}
	if runCfg.TopologicalPolicy != "const" || runCfg.TopologicalCount != 2 {
		t.Fatalf("expected topo override const/2, got policy=%s count=%d", runCfg.TopologicalPolicy, runCfg.TopologicalCount)
	}
	if runCfg.RunID != "sqlite-config-override-run" {
		t.Fatalf("expected explicit run id override, got %s", runCfg.RunID)
	}
	if runCfg.FitnessPostprocessor != "none" {
		t.Fatalf("expected fitness postprocessor override none, got %s", runCfg.FitnessPostprocessor)
	}
	if runCfg.SurvivalPercentage != 0.4 {
		t.Fatalf("expected survival percentage override 0.4, got %f", runCfg.SurvivalPercentage)
	}
	if runCfg.SpecieSizeLimit != 4 {
		t.Fatalf("expected specie size limit override 4, got %d", runCfg.SpecieSizeLimit)
	}
	if runCfg.FitnessGoal != 0.91 {
		t.Fatalf("expected fitness goal override 0.91, got %f", runCfg.FitnessGoal)
	}
	if runCfg.EvaluationsLimit != 123 {
		t.Fatalf("expected evaluations limit override 123, got %d", runCfg.EvaluationsLimit)
	}
	if !runCfg.StartPaused || runCfg.AutoContinueAfterMS != 5 {
		t.Fatalf("expected pause control override start=true auto_ms=5, got start=%t auto_ms=%d", runCfg.StartPaused, runCfg.AutoContinueAfterMS)
	}
	if runCfg.TuneMinImprovement != 0.02 {
		t.Fatalf("expected tune min improvement from config 0.02, got %f", runCfg.TuneMinImprovement)
	}
	if runCfg.TunePerturbationRange != 1.6 || runCfg.TuneAnnealingFactor != 0.9 {
		t.Fatalf("expected tune spread params from config range=1.6 annealing=0.9, got range=%f annealing=%f", runCfg.TunePerturbationRange, runCfg.TuneAnnealingFactor)
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

func TestRunCommandSQLiteCanContinueFromPopulationSnapshot(t *testing.T) {
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
	baseRunID := "sqlite-base-pop"
	if err := run(context.Background(), []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--run-id", baseRunID,
		"--scape", "xor",
		"--pop", "8",
		"--gens", "2",
		"--seed", "51",
	}); err != nil {
		t.Fatalf("seed run command: %v", err)
	}

	continuedRunID := "sqlite-continued-pop"
	if err := run(context.Background(), []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--run-id", continuedRunID,
		"--continue-pop-id", baseRunID,
		"--scape", "xor",
		"--pop", "1",
		"--gens", "2",
		"--seed", "52",
	}); err != nil {
		t.Fatalf("continued run command: %v", err)
	}

	configData, err := os.ReadFile(filepath.Join("benchmarks", continuedRunID, "config.json"))
	if err != nil {
		t.Fatalf("read continued run config artifact: %v", err)
	}
	var runCfg stats.RunConfig
	if err := json.Unmarshal(configData, &runCfg); err != nil {
		t.Fatalf("decode continued run config artifact: %v", err)
	}
	if runCfg.ContinuePopulationID != baseRunID {
		t.Fatalf("expected continue population id %s, got %s", baseRunID, runCfg.ContinuePopulationID)
	}
	if runCfg.InitialGeneration != 2 {
		t.Fatalf("expected continued initial generation 2, got %d", runCfg.InitialGeneration)
	}
	if runCfg.PopulationSize != 8 {
		t.Fatalf("expected continued population size 8 from snapshot, got %d", runCfg.PopulationSize)
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
	if !strings.Contains(out, "generation=1") || !strings.Contains(out, "species=") || !strings.Contains(out, "tuning_invocations=") || !strings.Contains(out, "tuning_accept_rate=") || !strings.Contains(out, "tuning_evals_per_attempt=") {
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
			"--show-diagnostics",
		})
	})
	if err != nil {
		t.Fatalf("species-diff command: %v", err)
	}
	if !strings.Contains(out, "run_id=") || !strings.Contains(out, "changed") || !strings.Contains(out, "tuning_delta_attempts=") || !strings.Contains(out, "from_diag generation=") || !strings.Contains(out, "to_diag generation=") {
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
		"scape":         "xor",
		"seed":          515,
		"enable_tuning": true,
		"pmp": map[string]any{
			"init_specie_size": 9,
			"generation_limit": 6,
		},
		"constraint": map[string]any{
			"population_selection_f": "hof_competition",
			"tuning_selection_fs":    []any{"dynamic_random"},
			"tuning_duration_f":      []any{"const", 2},
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

func TestProfileShowCommand(t *testing.T) {
	out, err := captureStdout(func() error {
		return run(context.Background(), []string{"profile", "show", "--id", "ref-default-xorandxor"})
	})
	if err != nil {
		t.Fatalf("profile show command: %v", err)
	}
	if !strings.Contains(out, "id=ref-default-xorandxor") || !strings.Contains(out, "w_add_syn=") {
		t.Fatalf("unexpected profile show output: %s", out)
	}
}

func TestProfileShowCommandJSON(t *testing.T) {
	out, err := captureStdout(func() error {
		return run(context.Background(), []string{"profile", "show", "--id", "ref-default-xorandxor", "--json"})
	})
	if err != nil {
		t.Fatalf("profile show json command: %v", err)
	}
	if !strings.Contains(out, "\"ID\": \"ref-default-xorandxor\"") || !strings.Contains(out, "\"WeightAddSyn\"") {
		t.Fatalf("unexpected profile show json output: %s", out)
	}
}

func TestMonitorCommandReturnsRunNotActiveForUnknownRun(t *testing.T) {
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
	if err := run(context.Background(), []string{
		"monitor", "continue",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--run-id", "monitor-live",
	}); err == nil || !strings.Contains(err.Error(), "run not active") {
		t.Fatalf("expected run not active error, got %v", err)
	}
}

func TestMonitorCommandValidation(t *testing.T) {
	if err := run(context.Background(), []string{"monitor"}); err == nil {
		t.Fatal("expected missing action error")
	}

	if err := run(context.Background(), []string{"monitor", "pause"}); err == nil {
		t.Fatal("expected missing run-id error")
	}

	if err := run(context.Background(), []string{"monitor", "invalid", "--run-id", "x"}); err == nil {
		t.Fatal("expected unknown action error")
	}
}

func TestPopulationDeleteCommand(t *testing.T) {
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
	runID := "pop-del-cli"
	if err := run(context.Background(), []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--run-id", runID,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "1",
		"--seed", "61",
	}); err != nil {
		t.Fatalf("seed run command: %v", err)
	}

	if err := run(context.Background(), []string{
		"population", "delete",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--id", runID,
	}); err != nil {
		t.Fatalf("population delete command: %v", err)
	}
	if err := run(context.Background(), []string{
		"population", "delete",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--id", runID,
	}); err == nil {
		t.Fatal("expected deleting missing population to fail")
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
