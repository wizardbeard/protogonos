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
	"protogonos/internal/storage"
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
		case "add_outlink", "add_inlink", "remove_outlink", "remove_inlink", "add_neuron", "remove_neuron", "outsplice", "insplice":
			seenStructural = true
		}
	}
	if !seenStructural {
		t.Fatalf("expected at least one structural mutation in lineage: %+v", lineage)
	}
}

func TestResetCommandSQLiteClearsStore(t *testing.T) {
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
	runID := "reset-run"
	if err := run(context.Background(), []string{
		"run",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--run-id", runID,
		"--scape", "xor",
		"--pop", "4",
		"--gens", "1",
		"--seed", "17",
		"--workers", "2",
	}); err != nil {
		t.Fatalf("run command: %v", err)
	}

	storeBefore, err := storage.NewStore("sqlite", dbPath)
	if err != nil {
		t.Fatalf("new store before reset: %v", err)
	}
	if err := storeBefore.Init(context.Background()); err != nil {
		t.Fatalf("init store before reset: %v", err)
	}
	if _, ok, err := storeBefore.GetPopulation(context.Background(), runID); err != nil {
		t.Fatalf("get population before reset: %v", err)
	} else if !ok {
		t.Fatalf("expected population snapshot %q before reset", runID)
	}
	_ = storage.CloseIfSupported(storeBefore)

	if err := run(context.Background(), []string{
		"reset",
		"--store", "sqlite",
		"--db-path", dbPath,
	}); err != nil {
		t.Fatalf("reset command: %v", err)
	}

	storeAfter, err := storage.NewStore("sqlite", dbPath)
	if err != nil {
		t.Fatalf("new store after reset: %v", err)
	}
	if err := storeAfter.Init(context.Background()); err != nil {
		t.Fatalf("init store after reset: %v", err)
	}
	if _, ok, err := storeAfter.GetPopulation(context.Background(), runID); err != nil {
		t.Fatalf("get population after reset: %v", err)
	} else if ok {
		t.Fatalf("expected population snapshot %q to be cleared by reset", runID)
	}
	_ = storage.CloseIfSupported(storeAfter)
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
	llvmConfigPath := filepath.Join(workdir, "llvm_config_workflow.json")
	llvmOverridePath := filepath.Join(workdir, "llvm_override_workflow.json")
	workflowData := []byte(`{"name":"llvm.integration.v1","optimizations":["done","instcombine"],"modes":{"gt":{"program":"integration","max_phases":8,"initial_complexity":1.1,"target_complexity":0.5,"base_runtime":1.0}}}`)
	if err := os.WriteFile(llvmConfigPath, workflowData, 0o644); err != nil {
		t.Fatalf("write llvm config workflow: %v", err)
	}
	if err := os.WriteFile(llvmOverridePath, workflowData, 0o644); err != nil {
		t.Fatalf("write llvm override workflow: %v", err)
	}

	cfg := map[string]any{
		"scape":                   "xor",
		"seed":                    71,
		"enable_tuning":           true,
		"llvm_workflow_json":      llvmConfigPath,
		"validation_probe":        true,
		"test_probe":              true,
		"tune_perturbation_range": 1.6,
		"tune_annealing_factor":   0.9,
		"tune_min_improvement":    0.02,
		"trace": map[string]any{
			"step_size": 320,
		},
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
		"--trace-step-size", "444",
		"--start-paused",
		"--auto-continue-ms", "5",
		"--llvm-workflow-json", llvmOverridePath,
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
	if runCfg.TraceStepSize != 444 {
		t.Fatalf("expected trace step size override 444, got %d", runCfg.TraceStepSize)
	}
	if !runCfg.StartPaused || runCfg.AutoContinueAfterMS != 5 {
		t.Fatalf("expected pause control override start=true auto_ms=5, got start=%t auto_ms=%d", runCfg.StartPaused, runCfg.AutoContinueAfterMS)
	}
	if runCfg.TuneMinImprovement != 0.02 {
		t.Fatalf("expected tune min improvement from config 0.02, got %f", runCfg.TuneMinImprovement)
	}
	if !runCfg.ValidationProbe || !runCfg.TestProbe {
		t.Fatalf("expected validation/test probe flags from config, got validation=%t test=%t", runCfg.ValidationProbe, runCfg.TestProbe)
	}
	if runCfg.TunePerturbationRange != 1.6 || runCfg.TuneAnnealingFactor != 0.9 {
		t.Fatalf("expected tune spread params from config range=1.6 annealing=0.9, got range=%f annealing=%f", runCfg.TunePerturbationRange, runCfg.TuneAnnealingFactor)
	}
	if runCfg.LLVMWorkflowJSONPath != llvmOverridePath {
		t.Fatalf("expected llvm workflow flag override %q, got %q", llvmOverridePath, runCfg.LLVMWorkflowJSONPath)
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

	jsonOutput, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"runs",
			"--limit", "1",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("runs json command failed: %v", err)
	}
	var parsed []map[string]any
	if err := json.Unmarshal([]byte(jsonOutput), &parsed); err != nil {
		t.Fatalf("decode runs json output: %v\n%s", err, jsonOutput)
	}
	if len(parsed) == 0 {
		t.Fatalf("expected at least one item in runs json output: %s", jsonOutput)
	}
	if _, ok := parsed[0]["run_id"]; !ok {
		t.Fatalf("expected run_id field in runs json output: %v", parsed[0])
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

	jsonOut, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"lineage",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "3",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("lineage json command: %v", err)
	}
	var lineageJSON []map[string]any
	if err := json.Unmarshal([]byte(jsonOut), &lineageJSON); err != nil {
		t.Fatalf("decode lineage json output: %v\n%s", err, jsonOut)
	}
	if len(lineageJSON) == 0 {
		t.Fatalf("expected non-empty lineage json output: %s", jsonOut)
	}
	if _, ok := lineageJSON[0]["GenomeID"]; !ok {
		t.Fatalf("expected GenomeID in lineage json output: %v", lineageJSON[0])
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

	jsonOut, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"fitness",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "2",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("fitness json command: %v", err)
	}
	var parsed []float64
	if err := json.Unmarshal([]byte(jsonOut), &parsed); err != nil {
		t.Fatalf("decode fitness json output: %v\n%s", err, jsonOut)
	}
	if len(parsed) == 0 {
		t.Fatalf("expected non-empty fitness json output: %s", jsonOut)
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

	jsonOut, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"diagnostics",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "2",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("diagnostics json command: %v", err)
	}
	var parsed []map[string]any
	if err := json.Unmarshal([]byte(jsonOut), &parsed); err != nil {
		t.Fatalf("decode diagnostics json output: %v\n%s", err, jsonOut)
	}
	if len(parsed) == 0 {
		t.Fatalf("expected non-empty diagnostics json output: %s", jsonOut)
	}
	if _, ok := parsed[0]["generation"]; !ok {
		t.Fatalf("expected generation field in diagnostics json: %v", parsed[0])
	}
	if _, ok := parsed[0]["tuning_attempts"]; !ok {
		t.Fatalf("expected tuning telemetry field in diagnostics json: %v", parsed[0])
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

	jsonOut, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"species",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "1",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("species json command: %v", err)
	}
	var parsed []map[string]any
	if err := json.Unmarshal([]byte(jsonOut), &parsed); err != nil {
		t.Fatalf("decode species json output: %v\n%s", err, jsonOut)
	}
	if len(parsed) == 0 {
		t.Fatalf("expected non-empty species json output: %s", jsonOut)
	}
	if _, ok := parsed[0]["generation"]; !ok {
		t.Fatalf("expected generation field in species json: %v", parsed[0])
	}
	if _, ok := parsed[0]["species"]; !ok {
		t.Fatalf("expected species field in species json: %v", parsed[0])
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

	jsonOut, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"species-diff",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("species-diff json command: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(jsonOut), &parsed); err != nil {
		t.Fatalf("decode species-diff json output: %v\n%s", err, jsonOut)
	}
	if _, ok := parsed["run_id"]; !ok {
		t.Fatalf("expected run_id in species-diff json: %v", parsed)
	}
	if _, ok := parsed["tuning_attempts_delta"]; !ok {
		t.Fatalf("expected tuning delta in species-diff json: %v", parsed)
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

	jsonOut, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"top",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "2",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("top json command: %v", err)
	}
	var topJSON []map[string]any
	if err := json.Unmarshal([]byte(jsonOut), &topJSON); err != nil {
		t.Fatalf("decode top json output: %v\n%s", err, jsonOut)
	}
	if len(topJSON) == 0 {
		t.Fatalf("expected non-empty top json output: %s", jsonOut)
	}
	if _, ok := topJSON[0]["rank"]; !ok {
		t.Fatalf("expected rank in top json output: %v", topJSON[0])
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

func TestEpitopesTestCommandReplaysGenerationChampions(t *testing.T) {
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
		"--scape", "epitopes",
		"--epitopes-table", "abc_pred12",
		"--pop", "6",
		"--gens", "3",
		"--seed", "46",
		"--workers", "2",
	}
	if err := run(context.Background(), runArgs); err != nil {
		t.Fatalf("run command: %v", err)
	}

	out, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"epitopes-test",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "2",
		})
	})
	if err != nil {
		t.Fatalf("epitopes-test command: %v", err)
	}
	if !strings.Contains(out, "epitopes_test run_id=") || !strings.Contains(out, "source=trace_acc") || !strings.Contains(out, "mean_over_280=") || !strings.Contains(out, "table=abc_pred12") || !strings.Contains(out, "best_replay=") {
		t.Fatalf("unexpected epitopes-test output: %s", out)
	}

	jsonOut, err := captureStdout(func() error {
		return run(context.Background(), []string{
			"epitopes-test",
			"--store", "sqlite",
			"--db-path", dbPath,
			"--latest",
			"--limit", "2",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("epitopes-test json command: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(jsonOut), &parsed); err != nil {
		t.Fatalf("decode epitopes-test json output: %v\n%s", err, jsonOut)
	}
	if _, ok := parsed["BestGenomeID"]; !ok {
		t.Fatalf("expected BestGenomeID in epitopes-test json output: %v", parsed)
	}
	if _, ok := parsed["BestReplayFitness"]; !ok {
		t.Fatalf("expected BestReplayFitness in epitopes-test json output: %v", parsed)
	}
	items, ok := parsed["Items"].([]any)
	if !ok || len(items) == 0 {
		t.Fatalf("expected non-empty items in epitopes-test json output: %v", parsed)
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
	if summary.BestMax < summary.BestMin {
		t.Fatalf("expected best_max >= best_min, got max=%f min=%f", summary.BestMax, summary.BestMin)
	}
	if summary.BestStd < 0 {
		t.Fatalf("expected non-negative best_std, got %f", summary.BestStd)
	}
}

func TestBenchmarkExperimentStartListAndShow(t *testing.T) {
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
		"benchmark-experiment", "start",
		"--id", "exp-start",
		"--runs", "2",
		"--notes", "benchmarker parity",
		"--",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "21",
		"--workers", "2",
		"--min-improvement", "-1",
	}); err != nil {
		t.Fatalf("benchmark-experiment start: %v", err)
	}

	exp, ok, err := stats.ReadBenchmarkExperiment("benchmarks", "exp-start")
	if err != nil {
		t.Fatalf("read benchmark experiment: %v", err)
	}
	if !ok {
		t.Fatal("expected benchmark experiment to exist")
	}
	if exp.ProgressFlag != "completed" {
		t.Fatalf("expected completed experiment, got %+v", exp)
	}
	if exp.RunIndex != 3 || exp.TotalRuns != 2 {
		t.Fatalf("unexpected run progress: %+v", exp)
	}
	if len(exp.RunIDs) != 2 || len(exp.Summaries) != 2 {
		t.Fatalf("expected 2 run ids/summaries, got %+v", exp)
	}
	if exp.RunIDs[0] != "exp-start-run-001" || exp.RunIDs[1] != "exp-start-run-002" {
		t.Fatalf("unexpected run ids: %+v", exp.RunIDs)
	}

	listOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "list"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment list: %v", err)
	}
	if !strings.Contains(listOut, "id=exp-start") {
		t.Fatalf("unexpected list output: %s", listOut)
	}

	showOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "show", "--id", "exp-start"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment show: %v", err)
	}
	if !strings.Contains(showOut, "id=exp-start") || !strings.Contains(showOut, "progress=completed") {
		t.Fatalf("unexpected show output: %s", showOut)
	}
}

func TestBenchmarkExperimentContinue(t *testing.T) {
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
	benchmarkArgs := []string{
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "31",
		"--workers", "2",
		"--min-improvement", "-1",
	}
	if err := run(context.Background(), append([]string{"benchmark", "--run-id", "exp-continue-run-001"}, benchmarkArgs...)); err != nil {
		t.Fatalf("seed benchmark run: %v", err)
	}
	summary, ok, err := stats.ReadBenchmarkSummary("benchmarks", "exp-continue-run-001")
	if err != nil {
		t.Fatalf("read seed summary: %v", err)
	}
	if !ok {
		t.Fatal("expected seed benchmark summary")
	}
	exp := stats.BenchmarkExperiment{
		ID:            "exp-continue",
		Notes:         "resume test",
		ProgressFlag:  "in_progress",
		RunIndex:      2,
		TotalRuns:     2,
		StartedAtUTC:  "2026-02-27T00:00:00Z",
		BenchmarkArgs: benchmarkArgs,
		RunIDs:        []string{"exp-continue-run-001"},
		Summaries:     []stats.BenchmarkSummary{summary},
	}
	if err := stats.WriteBenchmarkExperiment("benchmarks", exp); err != nil {
		t.Fatalf("write experiment fixture: %v", err)
	}

	if err := run(context.Background(), []string{"benchmark-experiment", "continue", "--id", "exp-continue"}); err != nil {
		t.Fatalf("benchmark-experiment continue: %v", err)
	}
	got, ok, err := stats.ReadBenchmarkExperiment("benchmarks", "exp-continue")
	if err != nil {
		t.Fatalf("read continued experiment: %v", err)
	}
	if !ok {
		t.Fatal("expected continued experiment")
	}
	if got.ProgressFlag != "completed" || got.RunIndex != 3 {
		t.Fatalf("unexpected continued status: %+v", got)
	}
	if len(got.RunIDs) != 2 || got.RunIDs[1] != "exp-continue-run-002" {
		t.Fatalf("unexpected continued run ids: %+v", got.RunIDs)
	}
}

func TestBenchmarkExperimentEvaluationsAndReport(t *testing.T) {
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
		"benchmark-experiment", "start",
		"--id", "exp-reporting",
		"--runs", "2",
		"--",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--scape", "xor",
		"--pop", "6",
		"--gens", "2",
		"--seed", "41",
		"--workers", "2",
		"--min-improvement", "-1",
	}); err != nil {
		t.Fatalf("benchmark-experiment start: %v", err)
	}

	evalOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "evaluations", "--id", "exp-reporting", "--json"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment evaluations: %v", err)
	}
	var evalPayload struct {
		ID          string                         `json:"id"`
		Evaluations stats.BenchmarkEvaluationStats `json:"evaluations"`
	}
	if err := json.Unmarshal([]byte(evalOut), &evalPayload); err != nil {
		t.Fatalf("decode evaluations payload: %v", err)
	}
	if evalPayload.ID != "exp-reporting" {
		t.Fatalf("unexpected evaluations id: %+v", evalPayload)
	}
	if evalPayload.Evaluations.TotalRuns != 2 || len(evalPayload.Evaluations.Runs) != 2 {
		t.Fatalf("unexpected evaluations payload: %+v", evalPayload.Evaluations)
	}

	reportOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "report", "--id", "exp-reporting", "--name", "report2", "--json"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment report: %v", err)
	}
	var reportPayload struct {
		ID          string                         `json:"id"`
		Dir         string                         `json:"dir"`
		ReportName  string                         `json:"report_name"`
		Evaluations stats.BenchmarkEvaluationStats `json:"evaluations"`
		GraphFiles  []string                       `json:"graph_files"`
	}
	if err := json.Unmarshal([]byte(reportOut), &reportPayload); err != nil {
		t.Fatalf("decode report payload: %v", err)
	}
	if reportPayload.ID != "exp-reporting" || reportPayload.ReportName != "report2" {
		t.Fatalf("unexpected report payload: %+v", reportPayload)
	}
	if reportPayload.Dir == "" {
		t.Fatalf("expected non-empty report directory: %+v", reportPayload)
	}
	if reportPayload.Evaluations.TotalRuns != 2 {
		t.Fatalf("unexpected report evaluations payload: %+v", reportPayload.Evaluations)
	}
	if len(reportPayload.GraphFiles) == 0 {
		t.Fatalf("expected at least one graph output file: %+v", reportPayload)
	}

	reportDir := filepath.Join("benchmarks", "experiments", "exp-reporting")
	for _, name := range []string{
		"report2_Experiment.json",
		"report2_Trace_Acc.json",
		"report2_Evaluations.json",
		"report2_Report.json",
		"graph_xor_report2_Graphs",
	} {
		if _, err := os.Stat(filepath.Join(reportDir, name)); err != nil {
			t.Fatalf("expected report file %s: %v", name, err)
		}
	}

	trace2GraphOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "trace2graph", "--id", "exp-reporting", "--name", "trace_rebuild", "--json"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment trace2graph: %v", err)
	}
	var trace2GraphPayload struct {
		Files []string `json:"files"`
	}
	if err := json.Unmarshal([]byte(trace2GraphOut), &trace2GraphPayload); err != nil {
		t.Fatalf("decode trace2graph payload: %v", err)
	}
	if len(trace2GraphPayload.Files) == 0 {
		t.Fatalf("expected trace2graph files: %+v", trace2GraphPayload)
	}
	if _, err := os.Stat(filepath.Join(reportDir, "graph_xor_trace_rebuild")); err != nil {
		t.Fatalf("expected rebuilt graph file: %v", err)
	}

	plotOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "plot", "--id", "exp-reporting", "--mode", "avg", "--json"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment plot avg: %v", err)
	}
	var plotPayload struct {
		ID     string                       `json:"id"`
		Mode   string                       `json:"mode"`
		Points []stats.BenchmarkerPlotPoint `json:"points"`
	}
	if err := json.Unmarshal([]byte(plotOut), &plotPayload); err != nil {
		t.Fatalf("decode plot payload: %v", err)
	}
	if plotPayload.ID != "exp-reporting" || plotPayload.Mode != "avg" || len(plotPayload.Points) == 0 {
		t.Fatalf("unexpected plot payload: %+v", plotPayload)
	}
	if plotPayload.Points[0].Index != 500 {
		t.Fatalf("expected avg plot to start at 500, got %+v", plotPayload.Points[0])
	}

	plotMaxOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "plot", "--id", "exp-reporting", "--mode", "max", "--json"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment plot max: %v", err)
	}
	var plotMaxPayload struct {
		Mode   string                       `json:"mode"`
		Points []stats.BenchmarkerPlotPoint `json:"points"`
	}
	if err := json.Unmarshal([]byte(plotMaxOut), &plotMaxPayload); err != nil {
		t.Fatalf("decode max plot payload: %v", err)
	}
	if plotMaxPayload.Mode != "max" || len(plotMaxPayload.Points) == 0 {
		t.Fatalf("unexpected max plot payload: %+v", plotMaxPayload)
	}
	if plotMaxPayload.Points[0].Index != 0 {
		t.Fatalf("expected max plot to start at 0, got %+v", plotMaxPayload.Points[0])
	}

	chgOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "chg-mrph", "--id", "exp-reporting", "--run-id", "exp-reporting-run-001", "--scape", "regression-mimic"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment chg-mrph: %v", err)
	}
	if !strings.Contains(chgOut, "benchmark_experiment_chg_mrph id=exp-reporting scape=regression-mimic") {
		t.Fatalf("unexpected chg-mrph output: %s", chgOut)
	}
	expAfter, ok, err := stats.ReadBenchmarkExperiment("benchmarks", "exp-reporting")
	if err != nil {
		t.Fatalf("read experiment after chg-mrph: %v", err)
	}
	if !ok {
		t.Fatal("expected experiment to exist after chg-mrph")
	}
	joinedArgs := strings.Join(expAfter.BenchmarkArgs, " ")
	if !strings.Contains(joinedArgs, "--scape regression-mimic") && !strings.Contains(joinedArgs, "--scape=regression-mimic") {
		t.Fatalf("expected updated scape arg in benchmark args: %v", expAfter.BenchmarkArgs)
	}
	cfgAfter, ok, err := stats.ReadRunConfig("benchmarks", "exp-reporting-run-001")
	if err != nil {
		t.Fatalf("read run config after chg-mrph: %v", err)
	}
	if !ok {
		t.Fatal("expected run config to exist after chg-mrph")
	}
	if cfgAfter.Scape != "regression-mimic" {
		t.Fatalf("expected run config scape update, got %+v", cfgAfter)
	}

	vectorOut, err := captureStdout(func() error {
		return run(context.Background(), []string{"benchmark-experiment", "vector-compare", "--a", "1,2", "--b", "1,1", "--json"})
	})
	if err != nil {
		t.Fatalf("benchmark-experiment vector-compare: %v", err)
	}
	var vectorPayload struct {
		GT bool `json:"gt"`
		LT bool `json:"lt"`
		EQ bool `json:"eq"`
	}
	if err := json.Unmarshal([]byte(vectorOut), &vectorPayload); err != nil {
		t.Fatalf("decode vector payload: %v", err)
	}
	if !vectorPayload.GT || vectorPayload.LT || vectorPayload.EQ {
		t.Fatalf("unexpected vector comparison payload: %+v", vectorPayload)
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
	llvmConfigPath := filepath.Join(workdir, "llvm_benchmark_config_workflow.json")
	llvmOverridePath := filepath.Join(workdir, "llvm_benchmark_override_workflow.json")
	workflowData := []byte(`{"name":"llvm.benchmark.integration.v1","optimizations":["done","licm"],"modes":{"gt":{"program":"integration-bench","max_phases":8,"initial_complexity":1.1,"target_complexity":0.5,"base_runtime":1.0}}}`)
	if err := os.WriteFile(llvmConfigPath, workflowData, 0o644); err != nil {
		t.Fatalf("write llvm config workflow: %v", err)
	}
	if err := os.WriteFile(llvmOverridePath, workflowData, 0o644); err != nil {
		t.Fatalf("write llvm override workflow: %v", err)
	}

	cfg := map[string]any{
		"scape":              "xor",
		"seed":               515,
		"enable_tuning":      true,
		"llvm_workflow_json": llvmConfigPath,
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
		"--llvm-workflow-json", llvmOverridePath,
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
	configData, err := os.ReadFile(filepath.Join("benchmarks", entries[0].RunID, "config.json"))
	if err != nil {
		t.Fatalf("read benchmark config artifact: %v", err)
	}
	var runCfg stats.RunConfig
	if err := json.Unmarshal(configData, &runCfg); err != nil {
		t.Fatalf("decode benchmark config artifact: %v", err)
	}
	if runCfg.LLVMWorkflowJSONPath != llvmOverridePath {
		t.Fatalf("expected llvm workflow flag override %q, got %q", llvmOverridePath, runCfg.LLVMWorkflowJSONPath)
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
	if err := run(context.Background(), []string{
		"monitor", "print-trace",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--run-id", "monitor-live",
	}); err == nil || !strings.Contains(err.Error(), "run not active") {
		t.Fatalf("expected run not active error for print-trace, got %v", err)
	}
	if err := run(context.Background(), []string{
		"monitor", "goal-reached",
		"--store", "sqlite",
		"--db-path", dbPath,
		"--run-id", "monitor-live",
	}); err == nil || !strings.Contains(err.Error(), "run not active") {
		t.Fatalf("expected run not active error for goal-reached, got %v", err)
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
