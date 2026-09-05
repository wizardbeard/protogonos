package stats

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"protogonos/internal/model"
	"protogonos/internal/substrate"
)

func TestWriteAndExportRunArtifacts(t *testing.T) {
	baseDir := t.TempDir()
	outDir := filepath.Join(t.TempDir(), "exports")

	runID := "run-123"
	referenceSensors := []model.IORecordSpec{{
		Name:          "gtsa_input",
		ReferenceName: "general_predictor",
		Type:          "standard",
		ScapeKind:     "private",
		ScapeName:     "scape_GTSA",
		Format:        "no_geo",
		VL:            30,
		Parameters:    []string{"10"},
	}}
	referenceActuators := []model.IORecordSpec{{
		Name:          "gtsa_predict",
		ReferenceName: "general_predictor",
		Type:          "standard",
		ScapeKind:     "private",
		ScapeName:     "scape_GTSA",
		Format:        "no_geo",
		VL:            1,
		Parameters:    []string{"1"},
	}}
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
		GenerationDiagnostics: []model.GenerationDiagnostics{
			{Generation: 1, BestFitness: 0.5, MeanFitness: 0.4, MinFitness: 0.3, SpeciesCount: 2, FingerprintDiversity: 2},
		},
		SpeciesHistory: []model.SpeciesGeneration{
			{
				Generation:     1,
				Species:        []model.SpeciesMetrics{{Key: "sp-1", Size: 2, MeanFitness: 0.4, BestFitness: 0.5}},
				NewSpecies:     []string{"sp-1"},
				ExtinctSpecies: nil,
			},
		},
		TraceAcc: []TraceGeneration{
			{
				Generation: 1,
				Stats: []TraceStatEntry{
					{
						SpeciesKey:        "sp-1",
						ChampionGenomeID:  "g1",
						ChampionGenome:    model.Genome{ID: "g1", ReferenceSensors: referenceSensors, ReferenceActuators: referenceActuators},
						BestFitness:       0.5,
						ValidationFitness: float64Ptr(0.45),
					},
				},
			},
		},
		FinalBestFitness: 0.7,
		TopGenomes: []TopGenome{{
			Rank:    1,
			Fitness: 0.7,
			Genome:  model.Genome{ID: "g1", ReferenceSensors: referenceSensors, ReferenceActuators: referenceActuators},
			SubstrateSnapshot: &substrate.LayerRuntimeSnapshot{
				Plasticity: substrate.SubstratePlasticityABCN,
				LinkForm:   substrate.LinkFormL2LFeedforward,
				StateMode:  substrate.SubstrateStateReset,
				ABCN: substrate.ABCNSubstrate{
					InputLayer: substrate.CoordinateHyperlayer{{Coords: []float64{0}}},
					Layers: []substrate.ABCNCoordinateHyperlayer{
						{{Coords: []float64{1}, Weights: []substrate.ABCNWeight{{Weight: 0.5, A: 0.1, B: 0.2, C: 0.3, N: 0.4}}}},
					},
				},
				Weights: []float64{0.5},
			},
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

	for _, file := range []string{"config.json", "fitness_history.json", "top_genomes.json", "lineage.json", "generation_diagnostics.json", "species_history.json", "trace_acc.json"} {
		if _, err := os.Stat(filepath.Join(runDir, file)); err != nil {
			t.Fatalf("expected file %s: %v", file, err)
		}
	}

	exportedDir, err := ExportRunArtifacts(baseDir, runID, outDir)
	if err != nil {
		t.Fatalf("export artifacts: %v", err)
	}

	for _, file := range []string{"config.json", "fitness_history.json", "top_genomes.json", "lineage.json", "generation_diagnostics.json", "species_history.json", "trace_acc.json"} {
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
		Morphology:     "xor",
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
	if err := WriteBenchmarkSeries(runDir, []float64{0.5, 0.6, 0.7}); err != nil {
		t.Fatalf("write benchmark series: %v", err)
	}

	exportedDirWithBenchmark, err := ExportRunArtifacts(baseDir, runID, outDir)
	if err != nil {
		t.Fatalf("export artifacts with benchmark summary: %v", err)
	}
	if _, err := os.Stat(filepath.Join(exportedDirWithBenchmark, "benchmark_summary.json")); err != nil {
		t.Fatalf("expected exported benchmark summary: %v", err)
	}
	if _, err := os.Stat(filepath.Join(exportedDirWithBenchmark, "benchmark_series.csv")); err != nil {
		t.Fatalf("expected exported benchmark series: %v", err)
	}

	readCfg, ok, err := ReadRunConfig(baseDir, runID)
	if err != nil {
		t.Fatalf("read run config: %v", err)
	}
	if !ok {
		t.Fatalf("expected run config to exist for run id %s", runID)
	}
	if readCfg.RunID != runID || readCfg.Scape != "xor" {
		t.Fatalf("unexpected run config payload: %+v", readCfg)
	}

	readTop, ok, err := ReadTopGenomes(baseDir, runID)
	if err != nil {
		t.Fatalf("read top genomes: %v", err)
	}
	if !ok {
		t.Fatalf("expected top genomes to exist for run id %s", runID)
	}
	if len(readTop) != 1 || readTop[0].Genome.ID != "g1" {
		t.Fatalf("unexpected top genomes payload: %+v", readTop)
	}
	if readTop[0].SubstrateSnapshot == nil {
		t.Fatalf("expected substrate snapshot in top genome artifact")
	}
	if len(readTop[0].Genome.ReferenceSensors) != 1 || readTop[0].Genome.ReferenceSensors[0].ReferenceName != "general_predictor" || readTop[0].Genome.ReferenceSensors[0].VL != 30 {
		t.Fatalf("expected reference sensor metadata in top genome artifact, got %+v", readTop[0].Genome.ReferenceSensors)
	}
	if len(readTop[0].Genome.ReferenceActuators) != 1 || readTop[0].Genome.ReferenceActuators[0].ReferenceName != "general_predictor" {
		t.Fatalf("expected reference actuator metadata in top genome artifact, got %+v", readTop[0].Genome.ReferenceActuators)
	}
	if got := readTop[0].SubstrateSnapshot.ABCN.Layers[0][0].Weights[0]; got.A != 0.1 || got.N != 0.4 {
		t.Fatalf("unexpected substrate snapshot coefficients: %+v", readTop[0].SubstrateSnapshot)
	}

	readTraceAcc, ok, err := ReadTraceAcc(baseDir, runID)
	if err != nil {
		t.Fatalf("read trace acc: %v", err)
	}
	if !ok {
		t.Fatalf("expected trace acc to exist for run id %s", runID)
	}
	if len(readTraceAcc) != 1 || len(readTraceAcc[0].Stats) != 1 || readTraceAcc[0].Stats[0].ChampionGenomeID != "g1" {
		t.Fatalf("unexpected trace acc payload: %+v", readTraceAcc)
	}
	if len(readTraceAcc[0].Stats[0].ChampionGenome.ReferenceSensors) != 1 {
		t.Fatalf("expected reference sensor metadata in trace champion genome, got %+v", readTraceAcc[0].Stats[0].ChampionGenome.ReferenceSensors)
	}
	exportedTop, ok, err := ReadTopGenomes(outDir, runID)
	if err != nil {
		t.Fatalf("read exported top genomes: %v", err)
	}
	if !ok {
		t.Fatalf("expected exported top genomes to exist for run id %s", runID)
	}
	if len(exportedTop) != 1 || len(exportedTop[0].Genome.ReferenceSensors) != 1 || exportedTop[0].Genome.ReferenceSensors[0].VL != 30 {
		t.Fatalf("expected exported reference sensor metadata, got %+v", exportedTop)
	}
}

func TestRunIndexAppendListAndUpsert(t *testing.T) {
	baseDir := t.TempDir()

	err := AppendRunIndex(baseDir, RunIndexEntry{
		RunID:            "run-1",
		Scape:            "xor",
		Morphology:       "xor",
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
		Morphology:       "xor",
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
		Morphology:       "xor",
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
	if entries[0].Morphology != "xor" {
		t.Fatalf("expected morphology to survive run index persistence, got %+v", entries[0])
	}
}

func TestReadRunConfigAndTopGenomesMissingReturnsNotFound(t *testing.T) {
	baseDir := t.TempDir()

	if _, ok, err := ReadRunConfig(baseDir, "missing-run"); err != nil {
		t.Fatalf("read missing run config: %v", err)
	} else if ok {
		t.Fatal("expected missing run config to report not found")
	}

	if top, ok, err := ReadTopGenomes(baseDir, "missing-run"); err != nil {
		t.Fatalf("read missing top genomes: %v", err)
	} else if ok || top != nil {
		t.Fatalf("expected missing top genomes to report not found, got top=%+v ok=%t", top, ok)
	}

	if traceAcc, ok, err := ReadTraceAcc(baseDir, "missing-run"); err != nil {
		t.Fatalf("read missing trace acc: %v", err)
	} else if ok || traceAcc != nil {
		t.Fatalf("expected missing trace acc to report not found, got trace_acc=%+v ok=%t", traceAcc, ok)
	}
}

func TestReadTopGenomesAllowsLegacyRecordsWithoutSubstrateSnapshot(t *testing.T) {
	baseDir := t.TempDir()
	runID := "legacy-top"
	runDir := filepath.Join(baseDir, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir run dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "top_genomes.json"), []byte(`[
		{"rank":1,"fitness":0.9,"genome":{"id":"g1"}},
		{"rank":2,"fitness":0.8,"genome":{"id":"g2"}}
	]`), 0o644); err != nil {
		t.Fatalf("write top genomes: %v", err)
	}

	top, ok, err := ReadTopGenomes(baseDir, runID)
	if err != nil {
		t.Fatalf("read top genomes: %v", err)
	}
	if !ok || len(top) != 2 {
		t.Fatalf("unexpected top genomes: ok=%t top=%+v", ok, top)
	}
	if top[0].SubstrateSnapshot != nil {
		t.Fatalf("expected legacy record without substrate snapshot, got %+v", top[0].SubstrateSnapshot)
	}
}

func TestReadTopGenomesValidatesSubstrateSnapshotShape(t *testing.T) {
	baseDir := t.TempDir()
	runID := "snapshot-top"
	runDir := filepath.Join(baseDir, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir run dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "top_genomes.json"), []byte(`[
		{
			"rank":1,
			"fitness":0.9,
			"genome":{"id":"g1"},
			"substrate_snapshot":{
				"plasticity":"none",
				"link_form":"l2l_feedforward",
				"state_mode":"hold",
				"substrate":[
					[{"coords":[0],"output":1}],
					[{"coords":[1],"output":0.5,"weights":[0.25]}]
				],
				"weights":[0.25]
			}
		}
	]`), 0o644); err != nil {
		t.Fatalf("write top genomes: %v", err)
	}

	top, ok, err := ReadTopGenomes(baseDir, runID)
	if err != nil {
		t.Fatalf("read top genomes: %v", err)
	}
	if !ok || len(top) != 1 || top[0].SubstrateSnapshot == nil {
		t.Fatalf("unexpected top genomes: ok=%t top=%+v", ok, top)
	}
	if top[0].SubstrateSnapshot.StateMode != substrate.SubstrateStateHold {
		t.Fatalf("unexpected snapshot state mode: %+v", top[0].SubstrateSnapshot)
	}
}

func TestReadTopGenomesRejectsInvalidSubstrateSnapshotShape(t *testing.T) {
	baseDir := t.TempDir()
	runID := "bad-snapshot-top"
	runDir := filepath.Join(baseDir, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir run dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "top_genomes.json"), []byte(`[
		{
			"rank":1,
			"fitness":0.9,
			"genome":{"id":"g1"},
			"substrate_snapshot":{
				"plasticity":"none",
				"link_form":"l2l_feedforward",
				"state_mode":"hold",
				"substrate":[
					[{"coords":[0],"output":1}],
					[{"coords":[1],"output":0.5,"weights":[0.25,0.5]}]
				],
				"weights":[0.25]
			}
		}
	]`), 0o644); err != nil {
		t.Fatalf("write top genomes: %v", err)
	}

	if _, _, err := ReadTopGenomes(baseDir, runID); err == nil || !strings.Contains(err.Error(), "weight count mismatch") {
		t.Fatalf("expected weight count validation error, got %v", err)
	}
}

func TestWriteRunConfig(t *testing.T) {
	baseDir := t.TempDir()
	runID := "write-config-run"
	cfg := RunConfig{
		RunID:          runID,
		Scape:          "xor",
		PopulationSize: 8,
		Generations:    2,
		Seed:           3,
	}
	if err := WriteRunConfig(baseDir, runID, cfg); err != nil {
		t.Fatalf("write run config: %v", err)
	}
	loaded, ok, err := ReadRunConfig(baseDir, runID)
	if err != nil {
		t.Fatalf("read run config: %v", err)
	}
	if !ok {
		t.Fatal("expected written config to exist")
	}
	if loaded.RunID != runID || loaded.Scape != "xor" || loaded.PopulationSize != 8 {
		t.Fatalf("unexpected loaded run config: %+v", loaded)
	}
}

func TestFillRunConfigProfileHintsFromStoredMorphology(t *testing.T) {
	baseDir := t.TempDir()
	runID := "legacy-profiled-run"
	cfg := RunConfig{RunID: runID, Scape: "epitopes"}
	if err := WriteRunConfig(baseDir, runID, cfg); err != nil {
		t.Fatalf("write config: %v", err)
	}
	runDir := filepath.Join(baseDir, runID)
	if err := WriteBenchmarkSummary(runDir, BenchmarkSummary{
		RunID:      runID,
		Scape:      "epitopes",
		Morphology: "epitopes[core]",
	}); err != nil {
		t.Fatalf("write benchmark summary: %v", err)
	}

	filled, err := FillRunConfigProfileHints(baseDir, runID, cfg)
	if err != nil {
		t.Fatalf("fill profile hints: %v", err)
	}
	if filled.EpitopesProfile != "core" {
		t.Fatalf("expected epitopes core profile from morphology label, got %+v", filled)
	}
}

func TestResolveRunMorphologyLabelFallsBackToRunIndexForLegacyConfig(t *testing.T) {
	baseDir := t.TempDir()
	runID := "legacy-fx-run"
	cfg := RunConfig{RunID: runID, Scape: "fx"}
	if err := WriteRunConfig(baseDir, runID, cfg); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := AppendRunIndex(baseDir, RunIndexEntry{
		RunID:      runID,
		Scape:      "fx",
		Morphology: "fx[market]",
	}); err != nil {
		t.Fatalf("append run index: %v", err)
	}

	label, err := ResolveRunMorphologyLabel(baseDir, runID, cfg)
	if err != nil {
		t.Fatalf("resolve run morphology: %v", err)
	}
	if label != "fx[market]" {
		t.Fatalf("expected fx[market] label, got %q", label)
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

func float64Ptr(v float64) *float64 {
	return &v
}

func TestWriteBenchmarkSeries(t *testing.T) {
	runDir := t.TempDir()
	if err := WriteBenchmarkSeries(runDir, []float64{1.5, 1.75}); err != nil {
		t.Fatalf("write benchmark series: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(runDir, "benchmark_series.csv"))
	if err != nil {
		t.Fatalf("read benchmark series: %v", err)
	}
	got := string(data)
	if got != "generation,best_fitness\n1,1.5\n2,1.75\n" {
		t.Fatalf("unexpected benchmark series output:\n%s", got)
	}
	if strings.Count(got, "\n") != 3 {
		t.Fatalf("expected 3 csv lines, got %d", strings.Count(got, "\n"))
	}

	loaded, ok, err := ReadBenchmarkSeries(filepath.Dir(runDir), filepath.Base(runDir))
	if err != nil {
		t.Fatalf("read benchmark series: %v", err)
	}
	if !ok {
		t.Fatalf("expected benchmark series to exist")
	}
	if len(loaded) != 2 || loaded[0] != 1.5 || loaded[1] != 1.75 {
		t.Fatalf("unexpected loaded benchmark series: %v", loaded)
	}

	missing, ok, err := ReadBenchmarkSeries(t.TempDir(), "missing-run")
	if err != nil {
		t.Fatalf("read missing benchmark series: %v", err)
	}
	if ok || len(missing) != 0 {
		t.Fatalf("expected missing benchmark series, got ok=%t values=%v", ok, missing)
	}
}

func TestReadBenchmarkSummary(t *testing.T) {
	base := t.TempDir()
	runID := "bench-run"
	runDir := filepath.Join(base, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir run dir: %v", err)
	}
	want := BenchmarkSummary{
		RunID:          runID,
		Scape:          "xor",
		Morphology:     "xor",
		PopulationSize: 8,
		Generations:    3,
		Seed:           9,
		InitialBest:    0.1,
		FinalBest:      0.2,
		BestMean:       0.15,
		BestStd:        0.05,
		BestMax:        0.2,
		BestMin:        0.1,
		Improvement:    0.1,
		MinImprovement: 0.01,
		Passed:         true,
	}
	if err := WriteBenchmarkSummary(runDir, want); err != nil {
		t.Fatalf("write benchmark summary: %v", err)
	}
	got, ok, err := ReadBenchmarkSummary(base, runID)
	if err != nil {
		t.Fatalf("read benchmark summary: %v", err)
	}
	if !ok {
		t.Fatalf("expected benchmark summary to exist")
	}
	if got.RunID != want.RunID || got.Scape != want.Scape || got.Morphology != want.Morphology || got.BestMean != want.BestMean {
		t.Fatalf("unexpected benchmark summary: %+v", got)
	}
}

func TestBenchmarkMorphologyLabel(t *testing.T) {
	cases := []struct {
		scape    string
		gtsa     string
		fx       string
		epitopes string
		llvm     string
		flatland string
		want     string
	}{
		{scape: "gtsa", gtsa: "core", want: "gtsa[core]"},
		{scape: "fx", fx: "market", want: "fx[market]"},
		{scape: "epitopes", epitopes: "core", want: "epitopes[core]"},
		{scape: "llvm-phase-ordering", llvm: "core", want: "llvm-phase-ordering[core]"},
		{scape: "flatland", flatland: "core3", want: "flatland[core3]"},
		{scape: "fx", fx: "default", want: "fx"},
		{scape: "xor", want: "xor"},
		{scape: "", want: "unknown"},
	}
	for _, tc := range cases {
		if got := BenchmarkMorphologyLabel(tc.scape, tc.gtsa, tc.fx, tc.epitopes, tc.llvm, tc.flatland); got != tc.want {
			t.Fatalf("BenchmarkMorphologyLabel(%q,%q,%q,%q,%q,%q)=%q want=%q", tc.scape, tc.gtsa, tc.fx, tc.epitopes, tc.llvm, tc.flatland, got, tc.want)
		}
	}
}
