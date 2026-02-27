package protogonos

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"protogonos/internal/model"
	internalscape "protogonos/internal/scape"
	"protogonos/internal/stats"
)

type flatlandRunStepAgent struct {
	id string
}

func (a flatlandRunStepAgent) ID() string { return a.id }

func (a flatlandRunStepAgent) RunStep(_ context.Context, input []float64) ([]float64, error) {
	if len(input) == 0 {
		return []float64{0}, nil
	}
	if input[0] > 0 {
		return []float64{1}, nil
	}
	if input[0] < 0 {
		return []float64{-1}, nil
	}
	return []float64{0}, nil
}

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
	if speciesDiff.FromDiagnostics.Generation != speciesDiff.FromGeneration {
		t.Fatalf("species diff from diagnostics generation mismatch: got=%d want=%d", speciesDiff.FromDiagnostics.Generation, speciesDiff.FromGeneration)
	}
	if speciesDiff.ToDiagnostics.Generation != speciesDiff.ToGeneration {
		t.Fatalf("species diff to diagnostics generation mismatch: got=%d want=%d", speciesDiff.ToDiagnostics.Generation, speciesDiff.ToGeneration)
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
	scapeSummaryAlias, err := client.ScapeSummary(context.Background(), "xor_sim")
	if err != nil {
		t.Fatalf("scape summary alias: %v", err)
	}
	if scapeSummaryAlias.Name != "xor" {
		t.Fatalf("unexpected scape summary alias: %+v", scapeSummaryAlias)
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

	for _, selection := range []string{"hof_competition", "hof_rank", "hof_top3", "hof_efficiency", "hof_random", "competition", "top3"} {
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

func TestClientRunValidatesSpecieIdentifier(t *testing.T) {
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
		Selection:        "species_tournament",
		SpecieIdentifier: "unknown",
		WeightPerturb:    1,
	})
	if err == nil {
		t.Fatal("expected specie identifier validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:            "xor",
		Population:       8,
		Generations:      2,
		Selection:        "species_tournament",
		SpecieIdentifier: "tot_n",
		WeightPerturb:    1,
	})
	if err != nil {
		t.Fatalf("run with tot_n specie identifier: %v", err)
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:            "xor",
		Population:       8,
		Generations:      2,
		Selection:        "species_tournament",
		SpecieIdentifier: "fingerprint",
		WeightPerturb:    1,
	})
	if err != nil {
		t.Fatalf("run with fingerprint specie identifier: %v", err)
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

func TestClientRunAcceptsSubstrateOnlyMutationPolicy(t *testing.T) {
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
		Scape:           "xor",
		Population:      8,
		Generations:     2,
		Selection:       "elite",
		WeightSubstrate: 1.0,
		WeightPerturb:   0,
	})
	if err != nil {
		t.Fatalf("run with substrate-only mutation policy: %v", err)
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
		"dynamic",
		"all",
		"all_random",
		"active",
		"active_random",
		"recent",
		"recent_random",
		"current",
		"current_random",
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
		TuneMinImprovement: -0.1,
		EnableTuning:       true,
		TuneDurationParam:  1,
	})
	if err == nil {
		t.Fatal("expected tune min improvement validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:                 "xor",
		Population:            6,
		Generations:           2,
		TunePerturbationRange: -1,
		EnableTuning:          true,
		TuneDurationParam:     1,
	})
	if err == nil {
		t.Fatal("expected tune perturbation range validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:               "xor",
		Population:          6,
		Generations:         2,
		TuneAnnealingFactor: -0.1,
		EnableTuning:        true,
		TuneDurationParam:   1,
	})
	if err == nil {
		t.Fatal("expected tune annealing factor validation error")
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
		Scape:         "xor",
		Population:    6,
		Generations:   2,
		TraceStepSize: -1,
		Selection:     "elite",
		WeightPerturb: 1.0,
	})
	if err == nil {
		t.Fatal("expected trace step size validation error")
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

	_, err = client.Run(context.Background(), RunRequest{
		Scape:       "xor",
		Population:  6,
		Generations: 2,
		OpMode:      "unknown_mode",
	})
	if err == nil {
		t.Fatal("expected op mode validation error")
	}
	_, err = client.Run(context.Background(), RunRequest{
		Scape:       "xor",
		Population:  6,
		Generations: 2,
		OpMode:      "validation,test",
	})
	if err == nil {
		t.Fatal("expected composite non-gt op mode validation error")
	}

	_, err = client.Run(context.Background(), RunRequest{
		Scape:         "xor",
		Population:    6,
		Generations:   2,
		EvolutionType: "unknown_mode",
	})
	if err == nil {
		t.Fatal("expected evolution type validation error")
	}
}

func TestMaterializeRunConfigFromRequestParsesCompositeOpModeForGTProbes(t *testing.T) {
	cfg, err := materializeRunConfigFromRequest(RunRequest{
		Scape:       "xor",
		Population:  6,
		Generations: 1,
		OpMode:      "[gt,validation,test]",
	})
	if err != nil {
		t.Fatalf("materialize run config: %v", err)
	}
	if cfg.Request.OpMode != "gt" {
		t.Fatalf("expected normalized op mode gt, got %s", cfg.Request.OpMode)
	}
	if !cfg.Request.ValidationProbe || !cfg.Request.TestProbe {
		t.Fatalf("expected validation/test probes implied by composite op mode, got validation=%t test=%t", cfg.Request.ValidationProbe, cfg.Request.TestProbe)
	}
}

func TestMaterializeRunConfigFromRequestNormalizesReferenceScapeAlias(t *testing.T) {
	cases := map[string]string{
		"scape_LLVMPhaseOrdering": "llvm-phase-ordering",
		"llvm_phase_ordering_sim": "llvm-phase-ordering",
		"flatland_sim":            "flatland",
		"scape_flatland":          "flatland",
		"epitopes_sim":            "epitopes",
		"scape_epitopes_sim":      "epitopes",
		"gtsa_sim":                "gtsa",
		"scape_fx_sim":            "fx",
	}
	for alias, want := range cases {
		cfg, err := materializeRunConfigFromRequest(RunRequest{
			Scape:       alias,
			Population:  6,
			Generations: 1,
			OpMode:      "gt",
		})
		if err != nil {
			t.Fatalf("materialize run config alias=%s: %v", alias, err)
		}
		if cfg.Request.Scape != want {
			t.Fatalf("expected normalized scape %s for alias=%s, got %s", want, alias, cfg.Request.Scape)
		}
	}
}

func TestMaterializeRunConfigFromRequestValidatesScapeDatasetBounds(t *testing.T) {
	_, err := materializeRunConfigFromRequest(RunRequest{
		Scape:        "gtsa",
		Population:   6,
		Generations:  1,
		GTSATrainEnd: -1,
	})
	if err == nil || !strings.Contains(err.Error(), "gtsa train end") {
		t.Fatalf("expected gtsa train-end validation error, got %v", err)
	}

	_, err = materializeRunConfigFromRequest(RunRequest{
		Scape:           "epitopes",
		Population:      6,
		Generations:     1,
		EpitopesGTStart: -2,
	})
	if err == nil || !strings.Contains(err.Error(), "epitopes gt start") {
		t.Fatalf("expected epitopes gt-start validation error, got %v", err)
	}
}

func TestMaterializeRunConfigFromRequestValidatesFlatlandOverrides(t *testing.T) {
	spread := 0.01
	_, err := materializeRunConfigFromRequest(RunRequest{
		Scape:                 "flatland",
		Population:            6,
		Generations:           1,
		FlatlandScannerSpread: &spread,
	})
	if err == nil || !strings.Contains(err.Error(), "scanner spread") {
		t.Fatalf("expected flatland scanner spread validation error, got %v", err)
	}

	trials := 0
	_, err = materializeRunConfigFromRequest(RunRequest{
		Scape:                   "flatland",
		Population:              6,
		Generations:             1,
		FlatlandBenchmarkTrials: &trials,
	})
	if err == nil || !strings.Contains(err.Error(), "benchmark trials") {
		t.Fatalf("expected flatland benchmark trials validation error, got %v", err)
	}

	maxAge := 0
	_, err = materializeRunConfigFromRequest(RunRequest{
		Scape:          "flatland",
		Population:     6,
		Generations:    1,
		FlatlandMaxAge: &maxAge,
	})
	if err == nil || !strings.Contains(err.Error(), "max age") {
		t.Fatalf("expected flatland max age validation error, got %v", err)
	}

	forageGoal := 0
	_, err = materializeRunConfigFromRequest(RunRequest{
		Scape:              "flatland",
		Population:         6,
		Generations:        1,
		FlatlandForageGoal: &forageGoal,
	})
	if err == nil || !strings.Contains(err.Error(), "forage goal") {
		t.Fatalf("expected flatland forage goal validation error, got %v", err)
	}
}

func TestApplyScapeDataSourcesAppliesFlatlandOverridesToContext(t *testing.T) {
	spread := 0.22
	offset := 0.12
	randomize := true
	layoutVariants := 5
	forcedLayout := 2
	trials := 3
	maxAge := 64
	forageGoal := 4
	ctx, err := applyScapeDataSources(context.Background(), RunRequest{
		FlatlandScannerProfile:  "forward",
		FlatlandScannerSpread:   &spread,
		FlatlandScannerOffset:   &offset,
		FlatlandLayoutRandomize: &randomize,
		FlatlandLayoutVariants:  &layoutVariants,
		FlatlandForceLayout:     &forcedLayout,
		FlatlandBenchmarkTrials: &trials,
		FlatlandMaxAge:          &maxAge,
		FlatlandForageGoal:      &forageGoal,
	})
	if err != nil {
		t.Fatalf("apply scape data sources: %v", err)
	}

	flatland := internalscape.FlatlandScape{}
	fitness, trace, err := flatland.EvaluateMode(ctx, flatlandRunStepAgent{id: "flatland-api-ctx"}, "benchmark")
	if err != nil {
		t.Fatalf("evaluate flatland benchmark with overrides: %v", err)
	}
	if profile, _ := trace["scanner_profile"].(string); profile != "forward5" {
		t.Fatalf("expected flatland scanner profile override forward5, got trace=%+v", trace)
	}
	if gotSpread, _ := trace["scanner_spread"].(float64); gotSpread != spread {
		t.Fatalf("expected flatland scanner spread=%f, got trace=%+v", spread, trace)
	}
	if gotOffset, _ := trace["scanner_offset"].(float64); gotOffset != offset {
		t.Fatalf("expected flatland scanner offset=%f, got trace=%+v", offset, trace)
	}
	if gotVariant, _ := trace["layout_variant"].(int); gotVariant != forcedLayout {
		t.Fatalf("expected forced flatland layout variant=%d, got trace=%+v", forcedLayout, trace)
	}
	if gotVariants, _ := trace["layout_variants"].(int); gotVariants != layoutVariants {
		t.Fatalf("expected flatland layout variants=%d, got trace=%+v", layoutVariants, trace)
	}
	if forced, _ := trace["layout_forced"].(bool); !forced {
		t.Fatalf("expected forced flatland layout flag, got trace=%+v", trace)
	}
	if gotTrials, _ := trace["benchmark_trials"].(int); gotTrials != trials {
		t.Fatalf("expected benchmark trials=%d, got trace=%+v", trials, trace)
	}
	if aggregated, _ := trace["benchmark_aggregated"].(bool); !aggregated {
		t.Fatalf("expected benchmark aggregation enabled, got trace=%+v", trace)
	}
	if trialFitness, ok := trace["benchmark_trial_fitnesses"].([]float64); !ok || len(trialFitness) != trials {
		t.Fatalf("expected benchmark_trial_fitnesses len=%d, got trace=%+v", trials, trace)
	}
	if mean, ok := trace["benchmark_fitness_mean"].(float64); !ok || float64(fitness) != mean {
		t.Fatalf("expected returned fitness to equal benchmark_fitness_mean, fitness=%f trace=%+v", fitness, trace)
	}
	if gotMaxAge, _ := trace["max_age"].(int); gotMaxAge != maxAge {
		t.Fatalf("expected max_age=%d, got trace=%+v", maxAge, trace)
	}
	if gotForageGoal, _ := trace["forage_goal"].(int); gotForageGoal != forageGoal {
		t.Fatalf("expected forage_goal=%d, got trace=%+v", forageGoal, trace)
	}
}

func TestApplyScapeDataSourcesAppliesEpitopesTableSelectionToContext(t *testing.T) {
	internalscape.ResetEpitopesTableSource()
	t.Cleanup(internalscape.ResetEpitopesTableSource)

	ctx, err := applyScapeDataSources(context.Background(), RunRequest{
		EpitopesTableName: "abc_pred12",
	})
	if err != nil {
		t.Fatalf("apply scape data sources: %v", err)
	}

	epitopes := internalscape.EpitopesScape{}
	_, scopedTrace, err := epitopes.EvaluateMode(ctx, flatlandRunStepAgent{id: "epitopes-api-ctx"}, "benchmark")
	if err != nil {
		t.Fatalf("evaluate scoped epitopes: %v", err)
	}
	_, defaultTrace, err := epitopes.EvaluateMode(context.Background(), flatlandRunStepAgent{id: "epitopes-api-default"}, "benchmark")
	if err != nil {
		t.Fatalf("evaluate default epitopes: %v", err)
	}
	if table, _ := scopedTrace["table_name"].(string); table != "abc_pred12" {
		t.Fatalf("expected scoped epitopes table abc_pred12, got trace=%+v", scopedTrace)
	}
	if seqLen, _ := scopedTrace["sequence_length"].(int); seqLen != 12 {
		t.Fatalf("expected scoped epitopes sequence_length=12, got trace=%+v", scopedTrace)
	}
	if table, _ := defaultTrace["table_name"].(string); table != "abc_pred16" {
		t.Fatalf("expected default epitopes table abc_pred16, got trace=%+v", defaultTrace)
	}
}

func TestClientEpitopesReplayReplaysTopGenomesFromArtifacts(t *testing.T) {
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

	runSummary, err := client.Run(context.Background(), RunRequest{
		Scape:       "epitopes",
		Population:  6,
		Generations: 1,
		Seed:        31,
		Workers:     2,
	})
	if err != nil {
		t.Fatalf("run epitopes: %v", err)
	}

	replay, err := client.EpitopesReplay(context.Background(), EpitopesReplayRequest{
		RunID: runSummary.RunID,
		Limit: 3,
		Mode:  "benchmark",
	})
	if err != nil {
		t.Fatalf("epitopes replay: %v", err)
	}
	if replay.RunID != runSummary.RunID {
		t.Fatalf("unexpected replay run id: %+v", replay)
	}
	if replay.Evaluated != 3 || len(replay.Items) != 3 {
		t.Fatalf("expected 3 replayed genomes, got %+v", replay)
	}
	if replay.BestGenomeID == "" {
		t.Fatalf("expected best genome id in replay summary, got %+v", replay)
	}
	if replay.TableName != "abc_pred16" {
		t.Fatalf("expected default epitopes table in replay summary, got %+v", replay)
	}
}

func TestClientEpitopesReplayRejectsNonEpitopesRun(t *testing.T) {
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

	runSummary, err := client.Run(context.Background(), RunRequest{
		Scape:       "xor",
		Population:  6,
		Generations: 1,
		Seed:        41,
		Workers:     2,
	})
	if err != nil {
		t.Fatalf("run xor: %v", err)
	}

	_, err = client.EpitopesReplay(context.Background(), EpitopesReplayRequest{
		RunID: runSummary.RunID,
		Mode:  "benchmark",
	})
	if err == nil || !strings.Contains(err.Error(), "not an epitopes run") {
		t.Fatalf("expected non-epitopes replay error, got %v", err)
	}
}

func TestClientRunAppliesFXCSVSourceFromRunRequest(t *testing.T) {
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

	fxCSV := filepath.Join(base, "fx_prices.csv")
	var b strings.Builder
	b.WriteString("t,close\n")
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&b, "%d,%0.6f\n", i, 1.01+0.0003*float64(i))
	}
	if err := os.WriteFile(fxCSV, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write fx csv: %v", err)
	}

	summary, err := client.Run(context.Background(), RunRequest{
		Scape:       "fx",
		Population:  8,
		Generations: 1,
		Seed:        11,
		Workers:     2,
		FXCSVPath:   fxCSV,
	})
	if err != nil {
		t.Fatalf("run with fx csv source: %v", err)
	}

	var artifacts stats.RunArtifacts
	configData, err := os.ReadFile(filepath.Join(summary.ArtifactsDir, "config.json"))
	if err != nil {
		t.Fatalf("read config artifact: %v", err)
	}
	if err := json.Unmarshal(configData, &artifacts.Config); err != nil {
		t.Fatalf("decode config artifact: %v", err)
	}
	if artifacts.Config.FXCSVPath != fxCSV {
		t.Fatalf("expected fx_csv_path in artifacts, got %+v", artifacts.Config)
	}
}

func TestClientRunRejectsInvalidScapeCSVSource(t *testing.T) {
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
		Scape:       "fx",
		Population:  8,
		Generations: 1,
		FXCSVPath:   filepath.Join(base, "does-not-exist.csv"),
	})
	if err == nil || !strings.Contains(err.Error(), "configure fx data source") {
		t.Fatalf("expected fx csv load error, got %v", err)
	}
}

func TestClientRunAppliesLLVMWorkflowJSONSourceFromRunRequest(t *testing.T) {
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

	workflowJSON := filepath.Join(base, "llvm_workflow.json")
	workflowData := `{
  "name": "llvm.run.v1",
  "optimizations": ["done", "instcombine", "gvn"],
  "modes": {
    "gt": {
      "program": "llvm-test",
      "max_phases": 10,
      "initial_complexity": 1.2,
      "target_complexity": 0.5,
      "base_runtime": 1.0
    }
  }
}`
	if err := os.WriteFile(workflowJSON, []byte(workflowData), 0o644); err != nil {
		t.Fatalf("write workflow json: %v", err)
	}

	summary, err := client.Run(context.Background(), RunRequest{
		Scape:                "llvm-phase-ordering",
		Population:           8,
		Generations:          1,
		Seed:                 13,
		Workers:              2,
		LLVMWorkflowJSONPath: workflowJSON,
	})
	if err != nil {
		t.Fatalf("run with llvm workflow source: %v", err)
	}

	var artifacts stats.RunArtifacts
	configData, err := os.ReadFile(filepath.Join(summary.ArtifactsDir, "config.json"))
	if err != nil {
		t.Fatalf("read config artifact: %v", err)
	}
	if err := json.Unmarshal(configData, &artifacts.Config); err != nil {
		t.Fatalf("decode config artifact: %v", err)
	}
	if artifacts.Config.LLVMWorkflowJSONPath != workflowJSON {
		t.Fatalf("expected llvm workflow source in artifacts, got %+v", artifacts.Config)
	}
}

func TestClientRunRejectsInvalidLLVMWorkflowSource(t *testing.T) {
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
		Scape:                "llvm-phase-ordering",
		Population:           8,
		Generations:          1,
		LLVMWorkflowJSONPath: filepath.Join(base, "missing_workflow.json"),
	})
	if err == nil || !strings.Contains(err.Error(), "configure llvm workflow source") {
		t.Fatalf("expected llvm workflow load error, got %v", err)
	}
}

func TestClientRunValidationOpModeSkipsEvolutionAndTuning(t *testing.T) {
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

	summary, err := client.Run(context.Background(), RunRequest{
		Scape:             "xor",
		Population:        8,
		Generations:       5,
		OpMode:            "validation",
		EnableTuning:      true,
		TuneAttempts:      3,
		TuneSteps:         3,
		TuneStepSize:      0.25,
		TuneDurationParam: 1.0,
		Selection:         "elite",
		WeightPerturb:     1.0,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(summary.BestByGeneration) != 1 {
		t.Fatalf("expected validation mode to evaluate exactly one generation, got %d", len(summary.BestByGeneration))
	}

	diagnostics, err := client.Diagnostics(context.Background(), DiagnosticsRequest{RunID: summary.RunID})
	if err != nil {
		t.Fatalf("diagnostics: %v", err)
	}
	if len(diagnostics) != 1 {
		t.Fatalf("expected single diagnostics generation in validation mode, got %d", len(diagnostics))
	}
	if diagnostics[0].TuningInvocations != 0 || diagnostics[0].TuningAttempts != 0 {
		t.Fatalf("expected tuning to be skipped outside gt mode, got diagnostics=%+v", diagnostics[0])
	}
}

func TestClientRunValidationOpModeForcesTuningFlagsOffInArtifacts(t *testing.T) {
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

	summary, err := client.Run(context.Background(), RunRequest{
		Scape:             "xor",
		Population:        8,
		Generations:       3,
		OpMode:            "validation",
		EnableTuning:      true,
		CompareTuning:     true,
		TuneAttempts:      4,
		TuneSteps:         3,
		TuneStepSize:      0.25,
		TuneDurationParam: 1.0,
		Selection:         "elite",
		WeightPerturb:     1.0,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if summary.Compare != nil {
		t.Fatalf("expected compare summary to be disabled outside gt mode, got %+v", summary.Compare)
	}

	runs, err := client.Runs(context.Background(), RunsRequest{Limit: 1})
	if err != nil {
		t.Fatalf("runs: %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("expected one run entry, got %d", len(runs))
	}
	if runs[0].RunID != summary.RunID {
		t.Fatalf("unexpected run id in index: got=%s want=%s", runs[0].RunID, summary.RunID)
	}
	if runs[0].TuningEnabled {
		t.Fatal("expected tuning_enabled=false in run index for validation mode")
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
	if err := client.PrintTraceRun(context.Background(), MonitorControlRequest{RunID: runID}); err != nil {
		t.Fatalf("print trace run: %v", err)
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
	if err := client.PrintTraceRun(context.Background(), MonitorControlRequest{RunID: runID}); err == nil {
		t.Fatal("expected print trace on inactive run to fail")
	}
	if err := client.GoalReachedRun(context.Background(), MonitorControlRequest{RunID: runID}); err == nil {
		t.Fatal("expected goal reached on inactive run to fail")
	}
}

func TestClientLiveRunGoalReachedControl(t *testing.T) {
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

	runID := "api-goal-control"
	done := make(chan RunSummary, 1)
	errs := make(chan error, 1)
	go func() {
		summary, runErr := client.Run(context.Background(), RunRequest{
			RunID:         runID,
			Scape:         "xor",
			Population:    8,
			Generations:   5,
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

	if err := client.GoalReachedRun(context.Background(), MonitorControlRequest{RunID: runID}); err != nil {
		t.Fatalf("goal reached run: %v", err)
	}
	if err := client.ContinueRun(context.Background(), MonitorControlRequest{RunID: runID}); err != nil {
		t.Fatalf("continue run: %v", err)
	}

	select {
	case err := <-errs:
		t.Fatalf("run failed after goal reached: %v", err)
	case summary := <-done:
		if len(summary.BestByGeneration) == 0 || len(summary.BestByGeneration) >= 5 {
			t.Fatalf("expected goal-reached run to stop early with partial progress, got %d generations", len(summary.BestByGeneration))
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for goal-reached run completion")
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
	if config.InitialGeneration != 2 {
		t.Fatalf("expected initial generation 2 for continued run, got %d", config.InitialGeneration)
	}
	if config.PopulationSize != 8 {
		t.Fatalf("expected continued run to use snapshot population size 8, got %d", config.PopulationSize)
	}

	diagData, err := os.ReadFile(filepath.Join(base, "benchmarks", continued.RunID, "generation_diagnostics.json"))
	if err != nil {
		t.Fatalf("read continued diagnostics: %v", err)
	}
	var diags []model.GenerationDiagnostics
	if err := json.Unmarshal(diagData, &diags); err != nil {
		t.Fatalf("decode continued diagnostics: %v", err)
	}
	if len(diags) == 0 {
		t.Fatal("expected continued diagnostics")
	}
	if diags[0].Generation != 3 {
		t.Fatalf("expected continued diagnostics to start at generation 3, got %d", diags[0].Generation)
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

func TestClientRunContinueDefaultsRunIDToPopulationID(t *testing.T) {
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
		RunID:         "cont-pop-id",
		Scape:         "xor",
		Population:    8,
		Generations:   1,
		Selection:     "elite",
		WeightPerturb: 1.0,
	})
	if err != nil {
		t.Fatalf("seed run: %v", err)
	}

	continued, err := client.Run(context.Background(), RunRequest{
		ContinuePopulationID: "cont-pop-id",
		Scape:                "xor",
		Generations:          1,
		Selection:            "elite",
		WeightPerturb:        1.0,
	})
	if err != nil {
		t.Fatalf("continued run: %v", err)
	}
	if continued.RunID != "cont-pop-id" {
		t.Fatalf("expected continued run id to default to population id cont-pop-id, got %s", continued.RunID)
	}
}

func TestClientDeletePopulation(t *testing.T) {
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
		RunID:         "pop-delete",
		Scape:         "xor",
		Population:    6,
		Generations:   1,
		Selection:     "elite",
		WeightPerturb: 1.0,
	})
	if err != nil {
		t.Fatalf("seed run: %v", err)
	}

	if err := client.DeletePopulation(context.Background(), DeletePopulationRequest{PopulationID: "pop-delete"}); err != nil {
		t.Fatalf("delete population: %v", err)
	}
	if err := client.DeletePopulation(context.Background(), DeletePopulationRequest{PopulationID: "pop-delete"}); err == nil {
		t.Fatal("expected delete population to fail when population is missing")
	}
}
