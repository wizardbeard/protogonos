package stats

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBuildBenchmarkEvaluationStats(t *testing.T) {
	base := t.TempDir()
	exp := BenchmarkExperiment{
		ID:           "exp-eval",
		ProgressFlag: "completed",
		RunIDs:       []string{"exp-eval-run-001", "exp-eval-run-002"},
		Summaries: []BenchmarkSummary{
			{RunID: "exp-eval-run-001", FinalBest: 0.8},
			{RunID: "exp-eval-run-002", FinalBest: 0.4},
		},
	}

	for _, runID := range exp.RunIDs {
		runDir := filepath.Join(base, runID)
		if _, err := WriteRunArtifacts(base, RunArtifacts{
			Config: RunConfig{
				RunID:          runID,
				Scape:          "xor",
				PopulationSize: 10,
				Generations:    3,
				Seed:           1,
			},
			BestByGeneration: []float64{0.1, 0.2, 0.3},
			TopGenomes:       nil,
			Lineage:          nil,
		}); err != nil {
			t.Fatalf("write run artifacts for %s: %v", runID, err)
		}
		switch runID {
		case "exp-eval-run-001":
			if err := WriteBenchmarkSeries(runDir, []float64{0.2, 0.6, 0.8}); err != nil {
				t.Fatalf("write run1 series: %v", err)
			}
		case "exp-eval-run-002":
			if err := WriteBenchmarkSeries(runDir, []float64{0.1, 0.3, 0.4}); err != nil {
				t.Fatalf("write run2 series: %v", err)
			}
		}
	}

	goal := 0.5
	stats, err := BuildBenchmarkEvaluationStats(base, exp, &goal, nil)
	if err != nil {
		t.Fatalf("build evaluation stats: %v", err)
	}
	if stats.TotalRuns != 2 || stats.SuccessRuns != 1 {
		t.Fatalf("unexpected success accounting: %+v", stats)
	}
	if stats.SuccessRate != 0.5 {
		t.Fatalf("unexpected success rate: %+v", stats)
	}
	if len(stats.Runs) != 2 {
		t.Fatalf("unexpected run details: %+v", stats)
	}
	if !stats.Runs[0].Success || stats.Runs[0].Evaluations != 20 {
		t.Fatalf("unexpected first run evaluation detail: %+v", stats.Runs[0])
	}
	if stats.Runs[1].Success {
		t.Fatalf("expected second run to fail goal: %+v", stats.Runs[1])
	}
}

func TestBuildBenchmarkEvaluationStatsNoGoalUsesFullSeries(t *testing.T) {
	base := t.TempDir()
	runID := "exp-full-run-001"
	if _, err := WriteRunArtifacts(base, RunArtifacts{
		Config: RunConfig{
			RunID:          runID,
			Scape:          "xor",
			PopulationSize: 7,
			Generations:    3,
			Seed:           1,
		},
		BestByGeneration: []float64{0.1, 0.2, 0.3},
	}); err != nil {
		t.Fatalf("write run artifacts: %v", err)
	}
	if err := WriteBenchmarkSeries(filepath.Join(base, runID), []float64{0.1, 0.2, 0.3}); err != nil {
		t.Fatalf("write benchmark series: %v", err)
	}

	exp := BenchmarkExperiment{
		ID:        "exp-full",
		RunIDs:    []string{runID},
		Summaries: []BenchmarkSummary{{RunID: runID, FinalBest: 0.3}},
	}
	stats, err := BuildBenchmarkEvaluationStats(base, exp, nil, nil)
	if err != nil {
		t.Fatalf("build evaluation stats: %v", err)
	}
	if stats.SuccessRuns != 1 {
		t.Fatalf("expected success run when no goal is provided, got %+v", stats)
	}
	if stats.Runs[0].Evaluations != 21 {
		t.Fatalf("expected full-series evaluations (21), got %+v", stats.Runs[0])
	}
}

func TestWriteBenchmarkerReport(t *testing.T) {
	base := t.TempDir()
	report := BenchmarkerReport{
		ExperimentID: "exp-report",
		ReportName:   "report",
		Experiment: BenchmarkExperiment{
			ID:           "exp-report",
			ProgressFlag: "completed",
			RunIndex:     3,
			TotalRuns:    2,
		},
		TraceAcc: []BenchmarkSummary{
			{RunID: "exp-report-run-001", FinalBest: 0.5},
			{RunID: "exp-report-run-002", FinalBest: 0.7},
		},
		Evaluations: BenchmarkEvaluationStats{
			TotalRuns:   2,
			SuccessRuns: 2,
			SuccessRate: 1.0,
		},
	}
	dir, err := WriteBenchmarkerReport(base, report)
	if err != nil {
		t.Fatalf("write benchmarker report: %v", err)
	}
	for _, name := range []string{
		"report_Experiment.json",
		"report_Trace_Acc.json",
		"report_Evaluations.json",
		"report_Report.json",
	} {
		path := filepath.Join(dir, name)
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected report file %s: %v", name, err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read report file %s: %v", name, err)
		}
		if len(data) == 0 {
			t.Fatalf("expected non-empty report file %s", name)
		}
	}

	reportData, err := os.ReadFile(filepath.Join(dir, "report_Report.json"))
	if err != nil {
		t.Fatalf("read report envelope: %v", err)
	}
	var loaded BenchmarkerReport
	if err := json.Unmarshal(reportData, &loaded); err != nil {
		t.Fatalf("decode report envelope: %v", err)
	}
	if loaded.GeneratedAt == "" {
		t.Fatalf("expected generated timestamp in report envelope: %+v", loaded)
	}
}
