package stats

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"protogonos/internal/nn"
)

type BenchmarkEvaluationRun struct {
	RunID             string  `json:"run_id"`
	Evaluations       int     `json:"evaluations"`
	Success           bool    `json:"success"`
	ReachedGeneration int     `json:"reached_generation,omitempty"`
	FinalBest         float64 `json:"final_best"`
	Goal              float64 `json:"goal,omitempty"`
	EvalLimit         int     `json:"eval_limit,omitempty"`
}

type BenchmarkEvaluationStats struct {
	TotalRuns        int                      `json:"total_runs"`
	SuccessRuns      int                      `json:"success_runs"`
	SuccessRate      float64                  `json:"success_rate"`
	AvgEvaluations   float64                  `json:"avg_evaluations"`
	StdEvaluations   float64                  `json:"std_evaluations"`
	MinEvaluations   float64                  `json:"min_evaluations"`
	MaxEvaluations   float64                  `json:"max_evaluations"`
	FitnessGoal      *float64                 `json:"fitness_goal,omitempty"`
	EvaluationsLimit *int                     `json:"evaluations_limit,omitempty"`
	Runs             []BenchmarkEvaluationRun `json:"runs"`
}

type BenchmarkerReport struct {
	ExperimentID string                   `json:"experiment_id"`
	ReportName   string                   `json:"report_name"`
	GeneratedAt  string                   `json:"generated_at_utc"`
	Experiment   BenchmarkExperiment      `json:"experiment"`
	TraceAcc     []BenchmarkSummary       `json:"trace_acc"`
	Evaluations  BenchmarkEvaluationStats `json:"evaluations"`
}

func BuildBenchmarkEvaluationStats(baseDir string, exp BenchmarkExperiment, fitnessGoal *float64, evalLimit *int) (BenchmarkEvaluationStats, error) {
	result := BenchmarkEvaluationStats{
		TotalRuns:        len(exp.RunIDs),
		FitnessGoal:      cloneFloat64Ptr(fitnessGoal),
		EvaluationsLimit: cloneIntPtr(evalLimit),
		Runs:             make([]BenchmarkEvaluationRun, 0, len(exp.RunIDs)),
	}
	successValues := make([]float64, 0, len(exp.RunIDs))
	for i, runID := range exp.RunIDs {
		cfg, ok, err := ReadRunConfig(baseDir, runID)
		if err != nil {
			return BenchmarkEvaluationStats{}, err
		}
		if !ok {
			return BenchmarkEvaluationStats{}, fmt.Errorf("run config not found for run id: %s", runID)
		}
		series, ok, err := ReadBenchmarkSeries(baseDir, runID)
		if err != nil {
			return BenchmarkEvaluationStats{}, err
		}
		if !ok {
			return BenchmarkEvaluationStats{}, fmt.Errorf("benchmark series not found for run id: %s", runID)
		}

		run := evaluateBenchmarkSeries(runID, series, cfg.PopulationSize, fitnessGoal, evalLimit)
		if i < len(exp.Summaries) {
			run.FinalBest = exp.Summaries[i].FinalBest
		}
		result.Runs = append(result.Runs, run)
		if run.Success {
			result.SuccessRuns++
			successValues = append(successValues, float64(run.Evaluations))
		}
	}
	if result.TotalRuns > 0 {
		result.SuccessRate = float64(result.SuccessRuns) / float64(result.TotalRuns)
	}
	if len(successValues) > 0 {
		result.AvgEvaluations, _ = nn.Avg(successValues)
		result.StdEvaluations, _ = nn.Std(successValues)
		result.MinEvaluations = successValues[0]
		result.MaxEvaluations = successValues[0]
		for _, value := range successValues[1:] {
			if value < result.MinEvaluations {
				result.MinEvaluations = value
			}
			if value > result.MaxEvaluations {
				result.MaxEvaluations = value
			}
		}
	}
	return result, nil
}

func WriteBenchmarkerReport(baseDir string, report BenchmarkerReport) (string, error) {
	if report.ExperimentID == "" {
		return "", fmt.Errorf("report experiment id is required")
	}
	name := report.ReportName
	if name == "" {
		name = "report"
	}
	reportDir := filepath.Join(baseDir, benchmarkExperimentsDir, report.ExperimentID)
	if err := os.MkdirAll(reportDir, 0o755); err != nil {
		return "", err
	}
	if report.GeneratedAt == "" {
		report.GeneratedAt = time.Now().UTC().Format(time.RFC3339Nano)
	}
	if err := writeJSON(filepath.Join(reportDir, name+"_Experiment.json"), report.Experiment); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(reportDir, name+"_Trace_Acc.json"), report.TraceAcc); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(reportDir, name+"_Evaluations.json"), report.Evaluations); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(reportDir, name+"_Report.json"), report); err != nil {
		return "", err
	}
	return reportDir, nil
}

func evaluateBenchmarkSeries(runID string, series []float64, populationSize int, fitnessGoal *float64, evalLimit *int) BenchmarkEvaluationRun {
	if populationSize <= 0 {
		populationSize = 1
	}
	run := BenchmarkEvaluationRun{
		RunID:       runID,
		FinalBest:   0,
		Evaluations: 0,
		Success:     false,
	}
	if fitnessGoal != nil {
		run.Goal = *fitnessGoal
	}
	if evalLimit != nil {
		run.EvalLimit = *evalLimit
	}
	if len(series) > 0 {
		run.FinalBest = series[len(series)-1]
	}

	for generation, best := range series {
		run.Evaluations += populationSize
		run.ReachedGeneration = generation + 1
		if fitnessGoal != nil && best >= *fitnessGoal {
			run.Success = true
			return run
		}
		if fitnessGoal != nil && evalLimit != nil && run.Evaluations > *evalLimit {
			run.Success = false
			return run
		}
	}
	run.Success = fitnessGoal == nil
	return run
}

func cloneFloat64Ptr(v *float64) *float64 {
	if v == nil || math.IsNaN(*v) {
		return nil
	}
	value := *v
	return &value
}

func cloneIntPtr(v *int) *int {
	if v == nil {
		return nil
	}
	value := *v
	return &value
}
