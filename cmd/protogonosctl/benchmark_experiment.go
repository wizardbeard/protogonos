package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
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
		return errors.New("benchmark-experiment requires a subcommand: start|continue|show|list")
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
