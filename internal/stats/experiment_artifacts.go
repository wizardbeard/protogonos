package stats

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

const benchmarkExperimentsDir = "experiments"

type BenchmarkExperiment struct {
	ID             string             `json:"id"`
	Notes          string             `json:"notes,omitempty"`
	ProgressFlag   string             `json:"progress_flag"`
	RunIndex       int                `json:"run_index"`
	TotalRuns      int                `json:"total_runs"`
	StartedAtUTC   string             `json:"started_at_utc,omitempty"`
	CompletedAtUTC string             `json:"completed_at_utc,omitempty"`
	Interruptions  []string           `json:"interruptions,omitempty"`
	BenchmarkArgs  []string           `json:"benchmark_args,omitempty"`
	RunIDs         []string           `json:"run_ids,omitempty"`
	Summaries      []BenchmarkSummary `json:"summaries,omitempty"`
}

func WriteBenchmarkExperiment(baseDir string, exp BenchmarkExperiment) error {
	if exp.ID == "" {
		return fmt.Errorf("experiment id is required")
	}
	path := benchmarkExperimentPath(baseDir, exp.ID)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(exp, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func ReadBenchmarkExperiment(baseDir, id string) (BenchmarkExperiment, bool, error) {
	if id == "" {
		return BenchmarkExperiment{}, false, fmt.Errorf("experiment id is required")
	}
	path := benchmarkExperimentPath(baseDir, id)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return BenchmarkExperiment{}, false, nil
		}
		return BenchmarkExperiment{}, false, err
	}
	var exp BenchmarkExperiment
	if err := json.Unmarshal(data, &exp); err != nil {
		return BenchmarkExperiment{}, false, err
	}
	return exp, true, nil
}

func ListBenchmarkExperiments(baseDir string) ([]BenchmarkExperiment, error) {
	root := filepath.Join(baseDir, benchmarkExperimentsDir)
	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			return []BenchmarkExperiment{}, nil
		}
		return nil, err
	}

	exps := make([]BenchmarkExperiment, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		exp, ok, err := ReadBenchmarkExperiment(baseDir, entry.Name())
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		exps = append(exps, exp)
	}
	sort.Slice(exps, func(i, j int) bool {
		switch {
		case exps[i].StartedAtUTC == exps[j].StartedAtUTC:
			return exps[i].ID < exps[j].ID
		case exps[i].StartedAtUTC == "":
			return false
		case exps[j].StartedAtUTC == "":
			return true
		default:
			return exps[i].StartedAtUTC > exps[j].StartedAtUTC
		}
	})
	return exps, nil
}

func benchmarkExperimentPath(baseDir, id string) string {
	return filepath.Join(baseDir, benchmarkExperimentsDir, id, "experiment.json")
}
