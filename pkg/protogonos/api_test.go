package protogonos

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"protogonos/internal/stats"
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

	for _, selection := range []string{"hof_competition", "hof_rank", "hof_efficiency", "hof_random", "competition", "top3"} {
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

func TestClientRunAcceptsReferencePostprocessorAlias(t *testing.T) {
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

	_, err = client.Run(context.Background(), RunRequest{
		Scape:                "xor",
		Population:           8,
		Generations:          2,
		Selection:            "elite",
		FitnessPostprocessor: "nsize_proportional",
		WeightPerturb:        1,
		WeightAddNeuron:      0.2,
	})
	if err != nil {
		t.Fatalf("run with nsize_proportional alias: %v", err)
	}
}

func TestClientRunAcceptsBiasOnlyMutationPolicy(t *testing.T) {
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

	_, err = client.Run(context.Background(), RunRequest{
		Scape:         "xor",
		Population:    8,
		Generations:   2,
		Selection:     "elite",
		WeightBias:    1.0,
		WeightPerturb: 0,
	})
	if err != nil {
		t.Fatalf("run with bias-only mutation policy: %v", err)
	}
}

func TestClientRunAcceptsActivationAndAggregatorOnlyMutationPolicy(t *testing.T) {
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

	_, err = client.Run(context.Background(), RunRequest{
		Scape:            "xor",
		Population:       8,
		Generations:      2,
		Selection:        "elite",
		WeightActivation: 1.0,
		WeightAggregator: 1.0,
		WeightPerturb:    0,
	})
	if err != nil {
		t.Fatalf("run with activation/aggregator-only mutation policy: %v", err)
	}
}

func TestClientRunAcceptsRemoveBiasOnlyMutationPolicy(t *testing.T) {
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

	_, err = client.Run(context.Background(), RunRequest{
		Scape:            "xor",
		Population:       8,
		Generations:      2,
		Selection:        "elite",
		WeightRemoveBias: 1.0,
		WeightPerturb:    0,
	})
	if err != nil {
		t.Fatalf("run with remove-bias-only mutation policy: %v", err)
	}
}

func TestClientRunAcceptsPlasticityRuleOnlyMutationPolicy(t *testing.T) {
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

	_, err = client.Run(context.Background(), RunRequest{
		Scape:                "xor",
		Population:           8,
		Generations:          2,
		Selection:            "elite",
		WeightPlasticityRule: 1.0,
		WeightPerturb:        0,
	})
	if err != nil {
		t.Fatalf("run with plasticity-rule-only mutation policy: %v", err)
	}
}

func TestClientRunAcceptsReferenceTuningDurationAliases(t *testing.T) {
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

	for _, durationPolicy := range []string{"const", "nsize_proportional", "wsize_proportional"} {
		_, err = client.Run(context.Background(), RunRequest{
			Scape:              "xor",
			Population:         8,
			Generations:        2,
			Selection:          "elite",
			EnableTuning:       true,
			TuneDurationPolicy: durationPolicy,
			TuneDurationParam:  1.0,
			TuneAttempts:       2,
			TuneSteps:          2,
			TuneStepSize:       0.2,
			WeightPerturb:      1,
		})
		if err != nil {
			t.Fatalf("run with duration alias %s: %v", durationPolicy, err)
		}
	}
}

func TestClientRunAcceptsReferenceTuningSelectionModes(t *testing.T) {
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

	modes := []string{
		"all",
		"all_random",
		"recent",
		"recent_random",
		"lastgen",
		"lastgen_random",
	}
	for _, mode := range modes {
		_, err = client.Run(context.Background(), RunRequest{
			Scape:           "xor",
			Population:      8,
			Generations:     2,
			Selection:       "elite",
			EnableTuning:    true,
			TuneSelection:   mode,
			TuneAttempts:    2,
			TuneSteps:       2,
			TuneStepSize:    0.2,
			WeightPerturb:   1,
			WeightAddNeuron: 0.2,
		})
		if err != nil {
			t.Fatalf("run with tuning selection mode %s: %v", mode, err)
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

	_, err = client.Run(context.Background(), RunRequest{
		Scape:              "xor",
		Population:         6,
		Generations:        2,
		SurvivalPercentage: 1.1,
	})
	if err == nil {
		t.Fatal("expected survival percentage validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:         "xor",
		Population:    6,
		Generations:   2,
		FitnessGoal:   -0.01,
		Selection:     "elite",
		WeightPerturb: 1.0,
	})
	if err == nil {
		t.Fatal("expected fitness goal validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:            "xor",
		Population:       6,
		Generations:      2,
		EvaluationsLimit: -10,
		Selection:        "elite",
		WeightPerturb:    1.0,
	})
	if err == nil {
		t.Fatal("expected evaluations limit validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:           "xor",
		Population:      6,
		Generations:     2,
		SpecieSizeLimit: -1,
		Selection:       "elite",
		WeightPerturb:   1.0,
	})
	if err == nil {
		t.Fatal("expected specie size limit validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:             "xor",
		Population:        6,
		Generations:       2,
		Selection:         "elite",
		WeightPerturb:     1.0,
		AutoContinueAfter: -time.Millisecond,
	})
	if err == nil {
		t.Fatal("expected auto continue duration validation error")
	}
}

func TestClientRunStartPausedControls(t *testing.T) {
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

	pausedCtx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	_, err = client.Run(pausedCtx, RunRequest{
		Scape:         "xor",
		Population:    8,
		Generations:   2,
		StartPaused:   true,
		Selection:     "elite",
		WeightPerturb: 1.0,
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected paused run to block until context deadline, got err=%v", err)
	}

	resumed, err := client.Run(context.Background(), RunRequest{
		Scape:             "xor",
		Population:        8,
		Generations:       2,
		StartPaused:       true,
		AutoContinueAfter: 10 * time.Millisecond,
		Selection:         "elite",
		WeightPerturb:     1.0,
	})
	if err != nil {
		t.Fatalf("run with auto continue: %v", err)
	}
	if len(resumed.BestByGeneration) != 2 {
		t.Fatalf("expected full run after auto continue, got %d generations", len(resumed.BestByGeneration))
	}
}

func TestClientLiveRunControl(t *testing.T) {
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

	runID := "api-live-control"
	done := make(chan RunSummary, 1)
	errs := make(chan error, 1)
	go func() {
		summary, runErr := client.Run(context.Background(), RunRequest{
			RunID:         runID,
			Scape:         "xor",
			Population:    8,
			Generations:   4,
			StartPaused:   true,
			Selection:     "elite",
			WeightPerturb: 1.0,
		})
		if runErr != nil {
			errs <- runErr
			return
		}
		done <- summary
	}()

	select {
	case <-done:
		t.Fatal("expected paused run not to complete before continue")
	case err := <-errs:
		t.Fatalf("run failed while paused: %v", err)
	case <-time.After(30 * time.Millisecond):
	}

	if err := client.ContinueRun(context.Background(), MonitorControlRequest{RunID: runID}); err != nil {
		t.Fatalf("continue run: %v", err)
	}
	if err := client.PauseRun(context.Background(), MonitorControlRequest{RunID: runID}); err != nil {
		t.Fatalf("pause run: %v", err)
	}
	if err := client.StopRun(context.Background(), MonitorControlRequest{RunID: runID}); err != nil {
		t.Fatalf("stop run: %v", err)
	}

	select {
	case err := <-errs:
		t.Fatalf("run failed after stop: %v", err)
	case summary := <-done:
		if len(summary.BestByGeneration) == 0 || len(summary.BestByGeneration) >= 4 {
			t.Fatalf("expected early-stopped run with partial progress, got %d generations", len(summary.BestByGeneration))
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for controlled run completion")
	}

	if err := client.ContinueRun(context.Background(), MonitorControlRequest{RunID: runID}); err == nil {
		t.Fatal("expected continue on inactive run to fail")
	}
}

func TestClientRunCanContinueFromPopulationSnapshot(t *testing.T) {
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

	seedRun, err := client.Run(context.Background(), RunRequest{
		RunID:         "pop-seed",
		Scape:         "xor",
		Population:    8,
		Generations:   2,
		Selection:     "elite",
		WeightPerturb: 1.0,
	})
	if err != nil {
		t.Fatalf("seed run: %v", err)
	}
	if len(seedRun.BestByGeneration) != 2 {
		t.Fatalf("expected seed run generations=2, got %d", len(seedRun.BestByGeneration))
	}

	continued, err := client.Run(context.Background(), RunRequest{
		RunID:                "pop-continued",
		ContinuePopulationID: "pop-seed",
		Scape:                "xor",
		Population:           1,
		Generations:          2,
		Selection:            "elite",
		WeightPerturb:        1.0,
	})
	if err != nil {
		t.Fatalf("continued run: %v", err)
	}
	if len(continued.BestByGeneration) != 2 {
		t.Fatalf("expected continued run generations=2, got %d", len(continued.BestByGeneration))
	}

	configData, err := os.ReadFile(filepath.Join(base, "benchmarks", continued.RunID, "config.json"))
	if err != nil {
		t.Fatalf("read continued config: %v", err)
	}
	var config stats.RunConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		t.Fatalf("decode continued config: %v", err)
	}
	if config.ContinuePopulationID != "pop-seed" {
		t.Fatalf("expected continue population id pop-seed, got %s", config.ContinuePopulationID)
	}
	if config.PopulationSize != 8 {
		t.Fatalf("expected continued run to use snapshot population size 8, got %d", config.PopulationSize)
	}
}

func TestClientRunContinuePopulationScapeMismatchFailsFast(t *testing.T) {
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

	_, err = client.Run(context.Background(), RunRequest{
		RunID:         "reg-base",
		Scape:         "regression-mimic",
		Population:    6,
		Generations:   1,
		Selection:     "elite",
		WeightPerturb: 1.0,
	})
	if err != nil {
		t.Fatalf("seed regression run: %v", err)
	}

	_, err = client.Run(context.Background(), RunRequest{
		RunID:                "xor-continued",
		ContinuePopulationID: "reg-base",
		Scape:                "xor",
		Generations:          1,
		Selection:            "elite",
		WeightPerturb:        1.0,
	})
	if err == nil {
		t.Fatal("expected scape mismatch compatibility error")
	}
}
