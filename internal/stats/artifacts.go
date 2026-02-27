package stats

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"protogonos/internal/model"
)

const runIndexFile = "run_index.json"

type RunConfig struct {
	RunID                   string   `json:"run_id"`
	ContinuePopulationID    string   `json:"continue_population_id,omitempty"`
	SpecieIdentifier        string   `json:"specie_identifier,omitempty"`
	OpMode                  string   `json:"op_mode,omitempty"`
	EvolutionType           string   `json:"evolution_type,omitempty"`
	InitialGeneration       int      `json:"initial_generation"`
	Scape                   string   `json:"scape"`
	GTSACSVPath             string   `json:"gtsa_csv_path,omitempty"`
	GTSATrainEnd            int      `json:"gtsa_train_end,omitempty"`
	GTSAValidationEnd       int      `json:"gtsa_validation_end,omitempty"`
	GTSATestEnd             int      `json:"gtsa_test_end,omitempty"`
	FXCSVPath               string   `json:"fx_csv_path,omitempty"`
	EpitopesCSVPath         string   `json:"epitopes_csv_path,omitempty"`
	EpitopesTableName       string   `json:"epitopes_table_name,omitempty"`
	LLVMWorkflowJSONPath    string   `json:"llvm_workflow_json_path,omitempty"`
	EpitopesGTStart         int      `json:"epitopes_gt_start,omitempty"`
	EpitopesGTEnd           int      `json:"epitopes_gt_end,omitempty"`
	EpitopesValidationStart int      `json:"epitopes_validation_start,omitempty"`
	EpitopesValidationEnd   int      `json:"epitopes_validation_end,omitempty"`
	EpitopesTestStart       int      `json:"epitopes_test_start,omitempty"`
	EpitopesTestEnd         int      `json:"epitopes_test_end,omitempty"`
	EpitopesBenchmarkStart  int      `json:"epitopes_benchmark_start,omitempty"`
	EpitopesBenchmarkEnd    int      `json:"epitopes_benchmark_end,omitempty"`
	FlatlandScannerProfile  string   `json:"flatland_scanner_profile,omitempty"`
	FlatlandScannerSpread   *float64 `json:"flatland_scanner_spread,omitempty"`
	FlatlandScannerOffset   *float64 `json:"flatland_scanner_offset,omitempty"`
	FlatlandLayoutRandomize *bool    `json:"flatland_layout_randomize,omitempty"`
	FlatlandLayoutVariants  *int     `json:"flatland_layout_variants,omitempty"`
	FlatlandForceLayout     *int     `json:"flatland_force_layout_variant,omitempty"`
	FlatlandBenchmarkTrials *int     `json:"flatland_benchmark_trials,omitempty"`
	FlatlandMaxAge          *int     `json:"flatland_max_age,omitempty"`
	FlatlandForageGoal      *int     `json:"flatland_forage_goal,omitempty"`
	PopulationSize          int      `json:"population_size"`
	Generations             int      `json:"generations"`
	SurvivalPercentage      float64  `json:"survival_percentage"`
	SpecieSizeLimit         int      `json:"specie_size_limit"`
	FitnessGoal             float64  `json:"fitness_goal"`
	EvaluationsLimit        int      `json:"evaluations_limit"`
	TraceStepSize           int      `json:"trace_step_size"`
	StartPaused             bool     `json:"start_paused"`
	AutoContinueAfterMS     int64    `json:"auto_continue_after_ms"`
	Seed                    int64    `json:"seed"`
	Workers                 int      `json:"workers"`
	EliteCount              int      `json:"elite_count"`
	Selection               string   `json:"selection"`
	FitnessPostprocessor    string   `json:"fitness_postprocessor"`
	TopologicalPolicy       string   `json:"topological_policy"`
	TopologicalCount        int      `json:"topological_count"`
	TopologicalParam        float64  `json:"topological_param"`
	TopologicalMax          int      `json:"topological_max"`
	TuningEnabled           bool     `json:"tuning_enabled"`
	ValidationProbe         bool     `json:"validation_probe"`
	TestProbe               bool     `json:"test_probe"`
	TuneSelection           string   `json:"tune_selection"`
	TuneDurationPolicy      string   `json:"tune_duration_policy"`
	TuneDurationParam       float64  `json:"tune_duration_param"`
	TuneAttempts            int      `json:"tune_attempts"`
	TuneSteps               int      `json:"tune_steps"`
	TuneStepSize            float64  `json:"tune_step_size"`
	TunePerturbationRange   float64  `json:"tune_perturbation_range"`
	TuneAnnealingFactor     float64  `json:"tune_annealing_factor"`
	TuneMinImprovement      float64  `json:"tune_min_improvement"`
	WeightPerturb           float64  `json:"weight_perturb"`
	WeightBias              float64  `json:"weight_bias"`
	WeightRemoveBias        float64  `json:"weight_remove_bias"`
	WeightActivation        float64  `json:"weight_activation"`
	WeightAggregator        float64  `json:"weight_aggregator"`
	WeightAddSynapse        float64  `json:"weight_add_synapse"`
	WeightRemoveSynapse     float64  `json:"weight_remove_synapse"`
	WeightAddNeuron         float64  `json:"weight_add_neuron"`
	WeightRemoveNeuron      float64  `json:"weight_remove_neuron"`
	WeightPlasticityRule    float64  `json:"weight_plasticity_rule"`
	WeightPlasticity        float64  `json:"weight_plasticity"`
	WeightSubstrate         float64  `json:"weight_substrate"`
}

type TopGenome struct {
	Rank    int          `json:"rank"`
	Fitness float64      `json:"fitness"`
	Genome  model.Genome `json:"genome"`
}

type TraceGeneration struct {
	Generation int              `json:"generation"`
	Stats      []TraceStatEntry `json:"stats"`
}

type TraceStatEntry struct {
	SpeciesKey        string       `json:"species_key"`
	ChampionGenomeID  string       `json:"champion_genome_id,omitempty"`
	ChampionGenome    model.Genome `json:"champion_genome,omitempty"`
	BestFitness       float64      `json:"best_fitness"`
	ValidationFitness *float64     `json:"validation_fitness,omitempty"`
	TestFitness       *float64     `json:"test_fitness,omitempty"`
}

type RunArtifacts struct {
	Config                RunConfig                     `json:"config"`
	BestByGeneration      []float64                     `json:"best_by_generation"`
	GenerationDiagnostics []model.GenerationDiagnostics `json:"generation_diagnostics,omitempty"`
	SpeciesHistory        []model.SpeciesGeneration     `json:"species_history,omitempty"`
	TraceAcc              []TraceGeneration             `json:"trace_acc,omitempty"`
	FinalBestFitness      float64                       `json:"final_best_fitness"`
	TopGenomes            []TopGenome                   `json:"top_genomes"`
	Lineage               []LineageEntry                `json:"lineage"`
}

type LineageEntry struct {
	GenomeID    string                  `json:"genome_id"`
	ParentID    string                  `json:"parent_id"`
	Generation  int                     `json:"generation"`
	Operation   string                  `json:"operation"`
	Events      []model.EvoHistoryEvent `json:"events,omitempty"`
	Fingerprint string                  `json:"fingerprint,omitempty"`
	Summary     map[string]any          `json:"summary,omitempty"`
}

type TuningComparison struct {
	Scape             string    `json:"scape"`
	PopulationSize    int       `json:"population_size"`
	Generations       int       `json:"generations"`
	Seed              int64     `json:"seed"`
	WithoutTuningBest []float64 `json:"without_tuning_best"`
	WithTuningBest    []float64 `json:"with_tuning_best"`
	WithoutFinalBest  float64   `json:"without_final_best"`
	WithFinalBest     float64   `json:"with_final_best"`
	FinalImprovement  float64   `json:"final_improvement"`
}

type BenchmarkSummary struct {
	RunID          string  `json:"run_id"`
	Scape          string  `json:"scape"`
	PopulationSize int     `json:"population_size"`
	Generations    int     `json:"generations"`
	Seed           int64   `json:"seed"`
	InitialBest    float64 `json:"initial_best"`
	FinalBest      float64 `json:"final_best"`
	BestMean       float64 `json:"best_mean"`
	BestStd        float64 `json:"best_std"`
	BestMax        float64 `json:"best_max"`
	BestMin        float64 `json:"best_min"`
	Improvement    float64 `json:"improvement"`
	MinImprovement float64 `json:"min_improvement"`
	Passed         bool    `json:"passed"`
}

type RunIndexEntry struct {
	RunID            string  `json:"run_id"`
	Scape            string  `json:"scape"`
	PopulationSize   int     `json:"population_size"`
	Generations      int     `json:"generations"`
	Seed             int64   `json:"seed"`
	Workers          int     `json:"workers"`
	EliteCount       int     `json:"elite_count"`
	TuningEnabled    bool    `json:"tuning_enabled"`
	FinalBestFitness float64 `json:"final_best_fitness"`
	CreatedAtUTC     string  `json:"created_at_utc"`
}

func WriteRunArtifacts(baseDir string, artifacts RunArtifacts) (string, error) {
	if artifacts.Config.RunID == "" {
		return "", fmt.Errorf("run id is required")
	}

	runDir := filepath.Join(baseDir, artifacts.Config.RunID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return "", err
	}

	if err := writeJSON(filepath.Join(runDir, "config.json"), artifacts.Config); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(runDir, "fitness_history.json"), map[string]any{"best_by_generation": artifacts.BestByGeneration, "final_best_fitness": artifacts.FinalBestFitness}); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(runDir, "top_genomes.json"), artifacts.TopGenomes); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(runDir, "lineage.json"), artifacts.Lineage); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(runDir, "generation_diagnostics.json"), artifacts.GenerationDiagnostics); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(runDir, "species_history.json"), artifacts.SpeciesHistory); err != nil {
		return "", err
	}
	if err := writeJSON(filepath.Join(runDir, "trace_acc.json"), artifacts.TraceAcc); err != nil {
		return "", err
	}

	return runDir, nil
}

func AppendRunIndex(baseDir string, entry RunIndexEntry) error {
	if entry.RunID == "" {
		return fmt.Errorf("run id is required")
	}
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return err
	}

	index, err := ListRunIndex(baseDir)
	if err != nil {
		return err
	}

	for i := range index {
		if index[i].RunID == entry.RunID {
			index[i] = entry
			return writeJSON(filepath.Join(baseDir, runIndexFile), index)
		}
	}

	index = append(index, entry)
	return writeJSON(filepath.Join(baseDir, runIndexFile), index)
}

func ListRunIndex(baseDir string) ([]RunIndexEntry, error) {
	path := filepath.Join(baseDir, runIndexFile)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return []RunIndexEntry{}, nil
		}
		return nil, err
	}

	var entries []RunIndexEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}

	type indexedEntry struct {
		entry RunIndexEntry
		idx   int
	}
	indexed := make([]indexedEntry, len(entries))
	for i := range entries {
		indexed[i] = indexedEntry{entry: entries[i], idx: i}
	}
	sort.Slice(indexed, func(i, j int) bool {
		if indexed[i].entry.CreatedAtUTC == indexed[j].entry.CreatedAtUTC {
			// Prefer later appended entries for equal timestamps.
			return indexed[i].idx > indexed[j].idx
		}
		return indexed[i].entry.CreatedAtUTC > indexed[j].entry.CreatedAtUTC
	})

	sorted := make([]RunIndexEntry, 0, len(indexed))
	for _, item := range indexed {
		sorted = append(sorted, item.entry)
	}
	return sorted, nil
}

func ExportRunArtifacts(baseDir, runID, outDir string) (string, error) {
	if runID == "" {
		return "", fmt.Errorf("run id is required")
	}

	src := filepath.Join(baseDir, runID)
	if _, err := os.Stat(src); err != nil {
		return "", err
	}

	dst := filepath.Join(outDir, runID)
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return "", err
	}

	files := []string{"config.json", "fitness_history.json", "top_genomes.json", "lineage.json", "generation_diagnostics.json", "species_history.json"}
	for _, file := range files {
		if err := copyFile(filepath.Join(src, file), filepath.Join(dst, file)); err != nil {
			return "", err
		}
	}
	traceAccPath := filepath.Join(src, "trace_acc.json")
	if _, err := os.Stat(traceAccPath); err == nil {
		if err := copyFile(traceAccPath, filepath.Join(dst, "trace_acc.json")); err != nil {
			return "", err
		}
	} else if err != nil && !os.IsNotExist(err) {
		return "", err
	}
	comparePath := filepath.Join(src, "compare_tuning.json")
	if _, err := os.Stat(comparePath); err == nil {
		if err := copyFile(comparePath, filepath.Join(dst, "compare_tuning.json")); err != nil {
			return "", err
		}
	} else if err != nil && !os.IsNotExist(err) {
		return "", err
	}
	benchmarkPath := filepath.Join(src, "benchmark_summary.json")
	if _, err := os.Stat(benchmarkPath); err == nil {
		if err := copyFile(benchmarkPath, filepath.Join(dst, "benchmark_summary.json")); err != nil {
			return "", err
		}
	} else if err != nil && !os.IsNotExist(err) {
		return "", err
	}
	seriesPath := filepath.Join(src, "benchmark_series.csv")
	if _, err := os.Stat(seriesPath); err == nil {
		if err := copyFile(seriesPath, filepath.Join(dst, "benchmark_series.csv")); err != nil {
			return "", err
		}
	} else if err != nil && !os.IsNotExist(err) {
		return "", err
	}

	return dst, nil
}

func ReadRunConfig(baseDir, runID string) (RunConfig, bool, error) {
	path := filepath.Join(baseDir, runID, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return RunConfig{}, false, nil
		}
		return RunConfig{}, false, err
	}

	var cfg RunConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return RunConfig{}, false, err
	}
	return cfg, true, nil
}

func WriteRunConfig(baseDir, runID string, cfg RunConfig) error {
	if strings.TrimSpace(runID) == "" {
		return fmt.Errorf("run id is required")
	}
	if strings.TrimSpace(cfg.RunID) == "" {
		cfg.RunID = strings.TrimSpace(runID)
	}
	if cfg.RunID != strings.TrimSpace(runID) {
		return fmt.Errorf("run config run id mismatch: got=%s want=%s", cfg.RunID, strings.TrimSpace(runID))
	}
	runDir := filepath.Join(baseDir, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return err
	}
	return writeJSON(filepath.Join(runDir, "config.json"), cfg)
}

func ReadTopGenomes(baseDir, runID string) ([]TopGenome, bool, error) {
	path := filepath.Join(baseDir, runID, "top_genomes.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}

	var top []TopGenome
	if err := json.Unmarshal(data, &top); err != nil {
		return nil, false, err
	}
	return top, true, nil
}

func ReadTraceAcc(baseDir, runID string) ([]TraceGeneration, bool, error) {
	path := filepath.Join(baseDir, runID, "trace_acc.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}

	var traceAcc []TraceGeneration
	if err := json.Unmarshal(data, &traceAcc); err != nil {
		return nil, false, err
	}
	return traceAcc, true, nil
}

func WriteTuningComparison(runDir string, report TuningComparison) error {
	return writeJSON(filepath.Join(runDir, "compare_tuning.json"), report)
}

func ReadTuningComparison(baseDir, runID string) (TuningComparison, bool, error) {
	path := filepath.Join(baseDir, runID, "compare_tuning.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return TuningComparison{}, false, nil
		}
		return TuningComparison{}, false, err
	}

	var report TuningComparison
	if err := json.Unmarshal(data, &report); err != nil {
		return TuningComparison{}, false, err
	}
	return report, true, nil
}

func WriteBenchmarkSummary(runDir string, summary BenchmarkSummary) error {
	return writeJSON(filepath.Join(runDir, "benchmark_summary.json"), summary)
}

func ReadBenchmarkSummary(baseDir, runID string) (BenchmarkSummary, bool, error) {
	path := filepath.Join(baseDir, runID, "benchmark_summary.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return BenchmarkSummary{}, false, nil
		}
		return BenchmarkSummary{}, false, err
	}
	var summary BenchmarkSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		return BenchmarkSummary{}, false, err
	}
	return summary, true, nil
}

func WriteBenchmarkSeries(runDir string, bestByGeneration []float64) error {
	path := filepath.Join(runDir, "benchmark_series.csv")
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	if err := writer.Write([]string{"generation", "best_fitness"}); err != nil {
		return err
	}
	for i, best := range bestByGeneration {
		if err := writer.Write([]string{
			strconv.Itoa(i + 1),
			strconv.FormatFloat(best, 'f', -1, 64),
		}); err != nil {
			return err
		}
	}
	writer.Flush()
	return writer.Error()
}

func ReadBenchmarkSeries(baseDir, runID string) ([]float64, bool, error) {
	path := filepath.Join(baseDir, runID, "benchmark_series.csv")
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		if err == io.EOF {
			return []float64{}, true, nil
		}
		return nil, false, err
	}
	if len(header) < 2 {
		return nil, false, fmt.Errorf("benchmark series header must have at least 2 columns")
	}

	series := make([]float64, 0, 128)
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, false, err
		}
		if len(record) < 2 {
			return nil, false, fmt.Errorf("benchmark series row must have at least 2 columns")
		}
		value, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			return nil, false, err
		}
		series = append(series, value)
	}
	return series, true, nil
}

func writeJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Sync()
}
