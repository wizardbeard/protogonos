package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"protogonos/internal/stats"
)

const (
	benchmarkExperimentProgressInProgress = "in_progress"
	benchmarkExperimentProgressCompleted  = "completed"
)

func runBenchmarkExperiment(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return errors.New("benchmark-experiment requires a subcommand: start|continue|show|list|evaluations|report|trace2graph|plot")
	}
	switch args[0] {
	case "start":
		return runBenchmarkExperimentStart(ctx, args[1:])
	case "continue":
		return runBenchmarkExperimentContinue(ctx, args[1:])
	case "show":
		return runBenchmarkExperimentShow(args[1:])
	case "list":
		return runBenchmarkExperimentList(args[1:])
	case "evaluations":
		return runBenchmarkExperimentEvaluations(args[1:])
	case "report":
		return runBenchmarkExperimentReport(args[1:])
	case "trace2graph":
		return runBenchmarkExperimentTraceToGraph(args[1:])
	case "plot":
		return runBenchmarkExperimentPlot(args[1:])
	default:
		return fmt.Errorf("unknown benchmark-experiment subcommand: %s", args[0])
	}
}

func runBenchmarkExperimentStart(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("benchmark-experiment start", flag.ContinueOnError)
	id := fs.String("id", "", "experiment id")
	runs := fs.Int("runs", 1, "total benchmark runs")
	notes := fs.String("notes", "", "optional experiment notes")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*id) == "" {
		return errors.New("benchmark-experiment start requires --id")
	}
	if *runs <= 0 {
		return errors.New("benchmark-experiment start requires --runs > 0")
	}
	if existing, ok, err := stats.ReadBenchmarkExperiment(benchmarksDir, strings.TrimSpace(*id)); err != nil {
		return err
	} else if ok {
		return fmt.Errorf("benchmark experiment already exists: %s (progress=%s run_index=%d total_runs=%d)", existing.ID, existing.ProgressFlag, existing.RunIndex, existing.TotalRuns)
	}

	benchmarkArgs := sanitizeExperimentBenchmarkArgs(fs.Args())
	exp := stats.BenchmarkExperiment{
		ID:           strings.TrimSpace(*id),
		Notes:        strings.TrimSpace(*notes),
		ProgressFlag: benchmarkExperimentProgressInProgress,
		RunIndex:     1,
		TotalRuns:    *runs,
		StartedAtUTC: time.Now().UTC().Format(time.RFC3339Nano),
		BenchmarkArgs: append(
			[]string(nil),
			benchmarkArgs...,
		),
	}
	if err := stats.WriteBenchmarkExperiment(benchmarksDir, exp); err != nil {
		return err
	}
	return executeBenchmarkExperiment(ctx, &exp)
}

func runBenchmarkExperimentContinue(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("benchmark-experiment continue", flag.ContinueOnError)
	id := fs.String("id", "", "experiment id")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*id) == "" {
		return errors.New("benchmark-experiment continue requires --id")
	}
	exp, ok, err := stats.ReadBenchmarkExperiment(benchmarksDir, strings.TrimSpace(*id))
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("benchmark experiment not found: %s", strings.TrimSpace(*id))
	}
	if exp.ProgressFlag == benchmarkExperimentProgressCompleted {
		fmt.Printf("benchmark_experiment id=%s progress=%s run_index=%d total_runs=%d\n", exp.ID, exp.ProgressFlag, exp.RunIndex, exp.TotalRuns)
		return nil
	}
	if exp.RunIndex < 1 {
		exp.RunIndex = 1
	}
	if override := sanitizeExperimentBenchmarkArgs(fs.Args()); len(override) > 0 {
		exp.BenchmarkArgs = override
	}
	exp.Interruptions = append(exp.Interruptions, time.Now().UTC().Format(time.RFC3339Nano))
	exp.ProgressFlag = benchmarkExperimentProgressInProgress
	if err := stats.WriteBenchmarkExperiment(benchmarksDir, exp); err != nil {
		return err
	}
	return executeBenchmarkExperiment(ctx, &exp)
}

func runBenchmarkExperimentShow(args []string) error {
	fs := flag.NewFlagSet("benchmark-experiment show", flag.ContinueOnError)
	id := fs.String("id", "", "experiment id")
	jsonOut := fs.Bool("json", false, "emit experiment as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*id) == "" {
		return errors.New("benchmark-experiment show requires --id")
	}
	exp, ok, err := stats.ReadBenchmarkExperiment(benchmarksDir, strings.TrimSpace(*id))
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("benchmark experiment not found: %s", strings.TrimSpace(*id))
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(exp)
	}
	fmt.Printf("id=%s progress=%s run_index=%d total_runs=%d started=%s completed=%s interruptions=%d notes=%s\n",
		exp.ID,
		exp.ProgressFlag,
		exp.RunIndex,
		exp.TotalRuns,
		exp.StartedAtUTC,
		exp.CompletedAtUTC,
		len(exp.Interruptions),
		exp.Notes,
	)
	for i, runID := range exp.RunIDs {
		finalBest := 0.0
		passed := false
		if i < len(exp.Summaries) {
			finalBest = exp.Summaries[i].FinalBest
			passed = exp.Summaries[i].Passed
		}
		fmt.Printf("run=%d run_id=%s final_best=%.6f passed=%t\n", i+1, runID, finalBest, passed)
	}
	return nil
}

func runBenchmarkExperimentList(args []string) error {
	fs := flag.NewFlagSet("benchmark-experiment list", flag.ContinueOnError)
	jsonOut := fs.Bool("json", false, "emit experiments as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	exps, err := stats.ListBenchmarkExperiments(benchmarksDir)
	if err != nil {
		return err
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(exps)
	}
	if len(exps) == 0 {
		fmt.Println("no benchmark experiments")
		return nil
	}
	for _, exp := range exps {
		fmt.Printf("id=%s progress=%s run_index=%d total_runs=%d started=%s completed=%s interruptions=%d notes=%s\n",
			exp.ID,
			exp.ProgressFlag,
			exp.RunIndex,
			exp.TotalRuns,
			exp.StartedAtUTC,
			exp.CompletedAtUTC,
			len(exp.Interruptions),
			exp.Notes,
		)
	}
	return nil
}

func runBenchmarkExperimentEvaluations(args []string) error {
	fs := flag.NewFlagSet("benchmark-experiment evaluations", flag.ContinueOnError)
	id := fs.String("id", "", "experiment id")
	fitnessGoal := fs.Float64("fitness-goal", math.NaN(), "optional success fitness goal")
	evalLimit := fs.Int("evaluations-limit", 0, "optional success evaluation limit (>0)")
	jsonOut := fs.Bool("json", false, "emit evaluations as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	setFlags := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		setFlags[f.Name] = true
	})

	exp, evalStats, err := loadBenchmarkExperimentEvaluationStats(strings.TrimSpace(*id), *fitnessGoal, *evalLimit, setFlags)
	if err != nil {
		return err
	}
	if *jsonOut {
		payload := struct {
			ID          string                         `json:"id"`
			Evaluations stats.BenchmarkEvaluationStats `json:"evaluations"`
		}{
			ID:          exp.ID,
			Evaluations: evalStats,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(payload)
	}

	fmt.Printf(
		"benchmark_experiment_evaluations id=%s success=%d/%d success_rate=%.6f avg=%.6f std=%.6f min=%.6f max=%.6f\n",
		exp.ID,
		evalStats.SuccessRuns,
		evalStats.TotalRuns,
		evalStats.SuccessRate,
		evalStats.AvgEvaluations,
		evalStats.StdEvaluations,
		evalStats.MinEvaluations,
		evalStats.MaxEvaluations,
	)
	for i, run := range evalStats.Runs {
		fmt.Printf(
			"run=%d run_id=%s success=%t evaluations=%d generation=%d final_best=%.6f\n",
			i+1,
			run.RunID,
			run.Success,
			run.Evaluations,
			run.ReachedGeneration,
			run.FinalBest,
		)
	}
	return nil
}

func runBenchmarkExperimentReport(args []string) error {
	fs := flag.NewFlagSet("benchmark-experiment report", flag.ContinueOnError)
	id := fs.String("id", "", "experiment id")
	name := fs.String("name", "report", "report output prefix")
	fitnessGoal := fs.Float64("fitness-goal", math.NaN(), "optional success fitness goal")
	evalLimit := fs.Int("evaluations-limit", 0, "optional success evaluation limit (>0)")
	jsonOut := fs.Bool("json", false, "emit report metadata as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	setFlags := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		setFlags[f.Name] = true
	})

	exp, evalStats, err := loadBenchmarkExperimentEvaluationStats(strings.TrimSpace(*id), *fitnessGoal, *evalLimit, setFlags)
	if err != nil {
		return err
	}
	report := stats.BenchmarkerReport{
		ExperimentID: exp.ID,
		ReportName:   strings.TrimSpace(*name),
		Experiment:   exp,
		TraceAcc:     append([]stats.BenchmarkSummary(nil), exp.Summaries...),
		Evaluations:  evalStats,
	}
	reportDir, err := stats.WriteBenchmarkerReport(benchmarksDir, report)
	if err != nil {
		return err
	}
	graphs, err := stats.BuildBenchmarkerGraphs(benchmarksDir, exp)
	if err != nil {
		return err
	}
	graphFiles, err := stats.WriteBenchmarkerGraphs(benchmarksDir, exp.ID, report.ReportName+"_Graphs", graphs)
	if err != nil {
		return err
	}

	if *jsonOut {
		payload := struct {
			ID          string                         `json:"id"`
			Dir         string                         `json:"dir"`
			ReportName  string                         `json:"report_name"`
			Evaluations stats.BenchmarkEvaluationStats `json:"evaluations"`
			GraphFiles  []string                       `json:"graph_files"`
		}{
			ID:          exp.ID,
			Dir:         reportDir,
			ReportName:  report.ReportName,
			Evaluations: evalStats,
			GraphFiles:  append([]string(nil), graphFiles...),
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(payload)
	}

	fmt.Printf(
		"benchmark_experiment_report id=%s name=%s dir=%s graphs=%d success=%d/%d success_rate=%.6f\n",
		exp.ID,
		report.ReportName,
		reportDir,
		len(graphFiles),
		evalStats.SuccessRuns,
		evalStats.TotalRuns,
		evalStats.SuccessRate,
	)
	return nil
}

func loadBenchmarkExperimentEvaluationStats(
	id string,
	fitnessGoal float64,
	evaluationsLimit int,
	setFlags map[string]bool,
) (stats.BenchmarkExperiment, stats.BenchmarkEvaluationStats, error) {
	if id == "" {
		return stats.BenchmarkExperiment{}, stats.BenchmarkEvaluationStats{}, errors.New("benchmark-experiment requires --id")
	}
	exp, ok, err := stats.ReadBenchmarkExperiment(benchmarksDir, id)
	if err != nil {
		return stats.BenchmarkExperiment{}, stats.BenchmarkEvaluationStats{}, err
	}
	if !ok {
		return stats.BenchmarkExperiment{}, stats.BenchmarkEvaluationStats{}, fmt.Errorf("benchmark experiment not found: %s", id)
	}
	var goalPtr *float64
	if setFlags["fitness-goal"] {
		value := fitnessGoal
		goalPtr = &value
	}
	var evalLimitPtr *int
	if setFlags["evaluations-limit"] {
		if evaluationsLimit <= 0 {
			return stats.BenchmarkExperiment{}, stats.BenchmarkEvaluationStats{}, errors.New("benchmark-experiment requires --evaluations-limit > 0 when provided")
		}
		value := evaluationsLimit
		evalLimitPtr = &value
	}
	evalStats, err := stats.BuildBenchmarkEvaluationStats(benchmarksDir, exp, goalPtr, evalLimitPtr)
	if err != nil {
		return stats.BenchmarkExperiment{}, stats.BenchmarkEvaluationStats{}, err
	}
	return exp, evalStats, nil
}

func runBenchmarkExperimentTraceToGraph(args []string) error {
	fs := flag.NewFlagSet("benchmark-experiment trace2graph", flag.ContinueOnError)
	id := fs.String("id", "", "experiment id")
	traceFile := fs.String("trace-file", "", "trace_acc.json path for standalone trace->graph conversion")
	name := fs.String("name", "__Graph", "graph postfix")
	morphology := fs.String("morphology", "trace", "morphology label for --trace-file mode")
	outDir := fs.String("out-dir", benchmarksDir, "output directory for --trace-file mode")
	jsonOut := fs.Bool("json", false, "emit generated graph files as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	useID := strings.TrimSpace(*id) != ""
	useTraceFile := strings.TrimSpace(*traceFile) != ""
	if useID == useTraceFile {
		return errors.New("benchmark-experiment trace2graph requires exactly one of --id or --trace-file")
	}

	postfix := strings.TrimSpace(*name)
	if postfix == "" {
		postfix = "__Graph"
	}

	var graphFiles []string
	if useID {
		exp, ok, err := stats.ReadBenchmarkExperiment(benchmarksDir, strings.TrimSpace(*id))
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("benchmark experiment not found: %s", strings.TrimSpace(*id))
		}
		graphs, err := stats.BuildBenchmarkerGraphs(benchmarksDir, exp)
		if err != nil {
			return err
		}
		graphFiles, err = stats.WriteBenchmarkerGraphs(benchmarksDir, exp.ID, postfix, graphs)
		if err != nil {
			return err
		}
	} else {
		tracePath := strings.TrimSpace(*traceFile)
		traceAcc, err := stats.ReadTraceAccFile(tracePath)
		if err != nil {
			return err
		}
		graph := stats.BuildBenchmarkerGraphFromTrace(traceAcc, strings.TrimSpace(*morphology))
		graphFiles, err = stats.WriteBenchmarkerGraphsToDir(filepath.Clean(*outDir), postfix, []stats.BenchmarkerGraph{graph})
		if err != nil {
			return err
		}
	}

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(struct {
			Files []string `json:"files"`
		}{
			Files: append([]string(nil), graphFiles...),
		})
	}
	for _, file := range graphFiles {
		fmt.Println(file)
	}
	return nil
}

func runBenchmarkExperimentPlot(args []string) error {
	fs := flag.NewFlagSet("benchmark-experiment plot", flag.ContinueOnError)
	id := fs.String("id", "", "experiment id")
	mode := fs.String("mode", "avg", "plot mode: avg|max")
	startIndex := fs.Int("start-index", -1, "index for first point (default 500 for avg, 0 for max)")
	step := fs.Int("step", 500, "index step")
	jsonOut := fs.Bool("json", false, "emit plot points as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*id) == "" {
		return errors.New("benchmark-experiment plot requires --id")
	}
	exp, ok, err := stats.ReadBenchmarkExperiment(benchmarksDir, strings.TrimSpace(*id))
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("benchmark experiment not found: %s", strings.TrimSpace(*id))
	}
	series := make([][]float64, 0, len(exp.RunIDs))
	for _, runID := range exp.RunIDs {
		runSeries, ok, err := stats.ReadBenchmarkSeries(benchmarksDir, runID)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("benchmark series not found for run id: %s", runID)
		}
		series = append(series, runSeries)
	}

	modeValue := strings.ToLower(strings.TrimSpace(*mode))
	start := *startIndex
	var points []stats.BenchmarkerPlotPoint
	switch modeValue {
	case "avg":
		if start < 0 {
			start = 500
		}
		points = stats.BuildBenchmarkerAveragePlot(series, start, *step)
	case "max":
		if start < 0 {
			start = 0
		}
		points = stats.BuildBenchmarkerMaxPlot(series, start, *step)
	default:
		return fmt.Errorf("unknown benchmark-experiment plot mode: %s", *mode)
	}

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(struct {
			ID     string                       `json:"id"`
			Mode   string                       `json:"mode"`
			Points []stats.BenchmarkerPlotPoint `json:"points"`
		}{
			ID:     exp.ID,
			Mode:   modeValue,
			Points: points,
		})
	}
	for _, point := range points {
		fmt.Printf("%d %g\n", point.Index, point.Value)
	}
	return nil
}

func executeBenchmarkExperiment(ctx context.Context, exp *stats.BenchmarkExperiment) error {
	if exp == nil {
		return errors.New("experiment is required")
	}
	if exp.ID == "" {
		return errors.New("experiment id is required")
	}
	if exp.TotalRuns <= 0 {
		return errors.New("experiment total_runs must be > 0")
	}
	if exp.RunIndex < 1 {
		exp.RunIndex = 1
	}
	keep := exp.RunIndex - 1
	if keep < 0 {
		keep = 0
	}
	if len(exp.RunIDs) > keep {
		exp.RunIDs = exp.RunIDs[:keep]
	}
	if len(exp.Summaries) > keep {
		exp.Summaries = exp.Summaries[:keep]
	}

	for runIdx := exp.RunIndex; runIdx <= exp.TotalRuns; runIdx++ {
		runID := fmt.Sprintf("%s-run-%03d", exp.ID, runIdx)
		runArgs := append([]string(nil), exp.BenchmarkArgs...)
		runArgs = append(runArgs, "--run-id", runID)
		if err := runBenchmark(ctx, runArgs); err != nil {
			exp.ProgressFlag = benchmarkExperimentProgressInProgress
			exp.RunIndex = runIdx
			exp.Interruptions = append(exp.Interruptions, time.Now().UTC().Format(time.RFC3339Nano))
			_ = stats.WriteBenchmarkExperiment(benchmarksDir, *exp)
			return err
		}

		summary, ok, err := stats.ReadBenchmarkSummary(benchmarksDir, runID)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("missing benchmark summary for run id: %s", runID)
		}
		exp.RunIDs = append(exp.RunIDs, runID)
		exp.Summaries = append(exp.Summaries, summary)
		exp.RunIndex = runIdx + 1
		exp.ProgressFlag = benchmarkExperimentProgressInProgress
		if err := stats.WriteBenchmarkExperiment(benchmarksDir, *exp); err != nil {
			return err
		}
		fmt.Printf("benchmark_experiment id=%s run=%d/%d run_id=%s final_best=%.6f passed=%t\n",
			exp.ID,
			runIdx,
			exp.TotalRuns,
			runID,
			summary.FinalBest,
			summary.Passed,
		)
	}

	exp.ProgressFlag = benchmarkExperimentProgressCompleted
	exp.CompletedAtUTC = time.Now().UTC().Format(time.RFC3339Nano)
	if err := stats.WriteBenchmarkExperiment(benchmarksDir, *exp); err != nil {
		return err
	}
	fmt.Printf("benchmark_experiment id=%s progress=%s runs=%d\n", exp.ID, exp.ProgressFlag, exp.TotalRuns)
	return nil
}

func sanitizeExperimentBenchmarkArgs(args []string) []string {
	if len(args) == 0 {
		return nil
	}
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "--run-id":
			if i+1 < len(args) {
				i++
			}
			continue
		case strings.HasPrefix(arg, "--run-id="):
			continue
		default:
			out = append(out, arg)
		}
	}
	return out
}
