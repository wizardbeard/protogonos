package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"protogonos/internal/evo"
	"protogonos/internal/morphology"
	"protogonos/internal/platform"
	"protogonos/internal/scape"
	"protogonos/internal/stats"
	"protogonos/internal/storage"
	"protogonos/internal/tuning"
	protoapi "protogonos/pkg/protogonos"
)

const (
	benchmarksDir = "benchmarks"
	exportsDir    = "exports"
)

func main() {
	if err := run(context.Background(), os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return usageError("missing command")
	}

	switch args[0] {
	case "init":
		return runInit(ctx, args[1:])
	case "reset":
		return runReset(ctx, args[1:])
	case "start":
		return runStart(ctx, args[1:])
	case "run":
		return runRun(ctx, args[1:])
	case "benchmark":
		return runBenchmark(ctx, args[1:])
	case "profile":
		return runProfile(ctx, args[1:])
	case "runs":
		return runRuns(ctx, args[1:])
	case "lineage":
		return runLineage(ctx, args[1:])
	case "fitness":
		return runFitness(ctx, args[1:])
	case "diagnostics":
		return runDiagnostics(ctx, args[1:])
	case "species":
		return runSpecies(ctx, args[1:])
	case "species-diff":
		return runSpeciesDiff(ctx, args[1:])
	case "monitor":
		return runMonitor(ctx, args[1:])
	case "population":
		return runPopulation(ctx, args[1:])
	case "top":
		return runTop(ctx, args[1:])
	case "scape-summary":
		return runScapeSummary(ctx, args[1:])
	case "export":
		return runExport(ctx, args[1:])
	default:
		return usageError(fmt.Sprintf("unknown command: %s", args[0]))
	}
}

func runInit(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}

	store, err := storage.NewStore(*storeKind, *dbPath)
	if err != nil {
		return err
	}
	defer func() {
		_ = storage.CloseIfSupported(store)
	}()

	polis := platform.NewPolis(platform.Config{Store: store})
	if err := polis.Init(ctx); err != nil {
		return err
	}

	fmt.Printf("initialized store=%s\n", *storeKind)
	return nil
}

func runReset(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("reset", flag.ContinueOnError)
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}

	store, err := storage.NewStore(*storeKind, *dbPath)
	if err != nil {
		return err
	}
	defer func() {
		_ = storage.CloseIfSupported(store)
	}()

	polis := platform.NewPolis(platform.Config{Store: store})
	if err := polis.Reset(ctx); err != nil {
		return err
	}

	fmt.Printf("reset store=%s\n", *storeKind)
	return nil
}

func runStart(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("start", flag.ContinueOnError)
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}

	store, err := storage.NewStore(*storeKind, *dbPath)
	if err != nil {
		return err
	}
	defer func() {
		_ = storage.CloseIfSupported(store)
	}()

	polis := platform.NewPolis(platform.Config{Store: store})
	if err := polis.Init(ctx); err != nil {
		return err
	}
	if err := registerDefaultScapes(polis); err != nil {
		return err
	}

	fmt.Printf("started store=%s scapes=%v\n", *storeKind, polis.RegisteredScapes())
	return nil
}

func runRun(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	configPath := fs.String("config", "", "optional run config JSON path (map2rec-backed)")
	runID := fs.String("run-id", "", "explicit run id (optional)")
	continuePopID := fs.String("continue-pop-id", "", "continue from persisted population snapshot id")
	specieIdentifier := fs.String("specie-identifier", "topology", "species identifier: topology|tot_n|fingerprint")
	opMode := fs.String("op-mode", "gt", "operation mode: gt|validation|test (or composite gt+validation/test)")
	evolutionType := fs.String("evolution-type", "generational", "evolution type: generational|steady_state")
	scapeName := fs.String("scape", "xor", "scape name")
	population := fs.Int("pop", 50, "population size")
	generations := fs.Int("gens", 100, "generation count")
	survivalPercentage := fs.Float64("survival-percentage", 0.0, "survival percentage used to derive elite retention when elite count is unset")
	specieSizeLimit := fs.Int("specie-size-limit", 0, "maximum parent-pool size retained per species (0 disables)")
	fitnessGoal := fs.Float64("fitness-goal", 0.0, "early-stop best fitness goal (0 disables)")
	evaluationsLimit := fs.Int("evaluations-limit", 0, "early-stop total evaluation limit (0 disables)")
	traceStepSize := fs.Int("trace-step-size", 500, "trace update cadence in total evaluations (0 uses runtime default)")
	startPaused := fs.Bool("start-paused", false, "start monitor in paused state (requires continue)")
	autoContinueMS := fs.Int("auto-continue-ms", 0, "auto-send continue after N milliseconds when start-paused is set (0 disables)")
	seed := fs.Int64("seed", 1, "rng seed")
	workers := fs.Int("workers", 4, "worker count")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	enableTuning := fs.Bool("tuning", false, "enable exoself tuning")
	compareTuning := fs.Bool("compare-tuning", false, "run with and without tuning and emit side-by-side metrics")
	validationProbe := fs.Bool("validation-probe", false, "evaluate per-species champions in validation probe during gt runs")
	testProbe := fs.Bool("test-probe", false, "evaluate per-species champions in test probe during gt runs")
	profileName := fs.String("profile", "", "optional parity profile id (from testdata/fixtures/parity/ref_benchmarker_profiles.json)")
	selectionName := fs.String("selection", "elite", "parent selection strategy: elite|tournament|species_tournament|species_shared_tournament|hof_competition|hof_rank|hof_top3|hof_efficiency|hof_random|competition|top3")
	postprocessorName := fs.String("fitness-postprocessor", "none", "fitness postprocessor: none|size_proportional|nsize_proportional|novelty_proportional")
	topoPolicyName := fs.String("topo-policy", "const", "topological mutation count policy: const|ncount_linear|ncount_exponential")
	topoCount := fs.Int("topo-count", 1, "mutation count for topo-policy=const")
	topoParam := fs.Float64("topo-param", 0.5, "policy parameter (multiplier/power) for topo-policy")
	topoMax := fs.Int("topo-max", 8, "maximum mutation count for non-const topo policies (<=0 disables cap)")
	tuneAttempts := fs.Int("attempts", 4, "tuning attempts per agent evaluation")
	tuneSteps := fs.Int("tune-steps", 6, "tuning perturbation steps per attempt")
	tuneStepSize := fs.Float64("tune-step-size", 0.35, "tuning perturbation magnitude")
	tunePerturbationRange := fs.Float64("tune-perturbation-range", 1.0, "tuning perturbation spread multiplier")
	tuneAnnealingFactor := fs.Float64("tune-annealing-factor", 1.0, "tuning per-step annealing factor")
	tuneMinImprovement := fs.Float64("tune-min-improvement", 0.0, "minimum fitness gain required to accept a tuning candidate")
	tuneSelection := fs.String("tune-selection", tuning.CandidateSelectBestSoFar, "tuner candidate selection: best_so_far|original|dynamic|dynamic_random|all|all_random|active|active_random|recent|recent_random|current|current_random|lastgen|lastgen_random")
	tuneDurationPolicy := fs.String("tune-duration-policy", "fixed", "tuning attempt policy: fixed|const|linear_decay|topology_scaled|nsize_proportional|wsize_proportional")
	tuneDurationParam := fs.Float64("tune-duration-param", 1.0, "tuning attempt policy parameter")
	wPerturb := fs.Float64("w-perturb", 0.70, "weight for perturb_random_weight mutation")
	wBias := fs.Float64("w-bias", 0.00, "weight for perturb_random_bias mutation")
	wRemoveBias := fs.Float64("w-remove-bias", 0.00, "weight for remove_random_bias mutation")
	wActivation := fs.Float64("w-activation", 0.00, "weight for change_random_activation mutation")
	wAggregator := fs.Float64("w-aggregator", 0.00, "weight for change_random_aggregator mutation")
	wAddSynapse := fs.Float64("w-add-synapse", 0.10, "weight for add_random_synapse mutation")
	wRemoveSynapse := fs.Float64("w-remove-synapse", 0.08, "weight for remove_random_synapse mutation")
	wAddNeuron := fs.Float64("w-add-neuron", 0.07, "weight for add_random_neuron mutation")
	wRemoveNeuron := fs.Float64("w-remove-neuron", 0.05, "weight for remove_random_neuron mutation")
	wPlasticityRule := fs.Float64("w-plasticity-rule", 0.00, "weight for change_plasticity_rule mutation")
	wPlasticity := fs.Float64("w-plasticity", 0.03, "weight for perturb_plasticity_rate mutation")
	wSubstrate := fs.Float64("w-substrate", 0.02, "weight for perturb_substrate_parameter mutation")
	if err := fs.Parse(args); err != nil {
		return err
	}
	setFlags := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		setFlags[f.Name] = true
	})

	req, err := loadOrDefaultRunRequest(*configPath)
	if err != nil {
		return err
	}
	if *configPath == "" {
		req = protoapi.RunRequest{
			Scape:                 *scapeName,
			OpMode:                *opMode,
			EvolutionType:         *evolutionType,
			RunID:                 *runID,
			ContinuePopulationID:  *continuePopID,
			SpecieIdentifier:      *specieIdentifier,
			Population:            *population,
			Generations:           *generations,
			SurvivalPercentage:    *survivalPercentage,
			SpecieSizeLimit:       *specieSizeLimit,
			FitnessGoal:           *fitnessGoal,
			EvaluationsLimit:      *evaluationsLimit,
			TraceStepSize:         *traceStepSize,
			StartPaused:           *startPaused,
			AutoContinueAfter:     time.Duration(*autoContinueMS) * time.Millisecond,
			Seed:                  *seed,
			Workers:               *workers,
			Selection:             *selectionName,
			FitnessPostprocessor:  *postprocessorName,
			TopologicalPolicy:     *topoPolicyName,
			TopologicalCount:      *topoCount,
			TopologicalParam:      *topoParam,
			TopologicalMax:        *topoMax,
			EnableTuning:          *enableTuning,
			CompareTuning:         *compareTuning,
			ValidationProbe:       *validationProbe,
			TestProbe:             *testProbe,
			TuneSelection:         *tuneSelection,
			TuneDurationPolicy:    *tuneDurationPolicy,
			TuneDurationParam:     *tuneDurationParam,
			TuneAttempts:          *tuneAttempts,
			TuneSteps:             *tuneSteps,
			TuneStepSize:          *tuneStepSize,
			TunePerturbationRange: *tunePerturbationRange,
			TuneAnnealingFactor:   *tuneAnnealingFactor,
			TuneMinImprovement:    *tuneMinImprovement,
			WeightPerturb:         *wPerturb,
			WeightBias:            *wBias,
			WeightRemoveBias:      *wRemoveBias,
			WeightActivation:      *wActivation,
			WeightAggregator:      *wAggregator,
			WeightAddSynapse:      *wAddSynapse,
			WeightRemoveSynapse:   *wRemoveSynapse,
			WeightAddNeuron:       *wAddNeuron,
			WeightRemoveNeuron:    *wRemoveNeuron,
			WeightPlasticityRule:  *wPlasticityRule,
			WeightPlasticity:      *wPlasticity,
			WeightSubstrate:       *wSubstrate,
		}
	} else {
		err := overrideFromFlags(&req, setFlags, map[string]any{
			"scape":                   *scapeName,
			"op-mode":                 *opMode,
			"evolution-type":          *evolutionType,
			"run-id":                  *runID,
			"continue-pop-id":         *continuePopID,
			"specie-identifier":       *specieIdentifier,
			"pop":                     *population,
			"gens":                    *generations,
			"survival-percentage":     *survivalPercentage,
			"specie-size-limit":       *specieSizeLimit,
			"fitness-goal":            *fitnessGoal,
			"evaluations-limit":       *evaluationsLimit,
			"trace-step-size":         *traceStepSize,
			"start-paused":            *startPaused,
			"auto-continue-ms":        *autoContinueMS,
			"seed":                    *seed,
			"workers":                 *workers,
			"tuning":                  *enableTuning,
			"compare-tuning":          *compareTuning,
			"validation-probe":        *validationProbe,
			"test-probe":              *testProbe,
			"selection":               *selectionName,
			"fitness-postprocessor":   *postprocessorName,
			"topo-policy":             *topoPolicyName,
			"topo-count":              *topoCount,
			"topo-param":              *topoParam,
			"topo-max":                *topoMax,
			"attempts":                *tuneAttempts,
			"tune-steps":              *tuneSteps,
			"tune-step-size":          *tuneStepSize,
			"tune-perturbation-range": *tunePerturbationRange,
			"tune-annealing-factor":   *tuneAnnealingFactor,
			"tune-min-improvement":    *tuneMinImprovement,
			"tune-selection":          *tuneSelection,
			"tune-duration-policy":    *tuneDurationPolicy,
			"tune-duration-param":     *tuneDurationParam,
			"w-perturb":               *wPerturb,
			"w-bias":                  *wBias,
			"w-remove-bias":           *wRemoveBias,
			"w-activation":            *wActivation,
			"w-aggregator":            *wAggregator,
			"w-add-synapse":           *wAddSynapse,
			"w-remove-synapse":        *wRemoveSynapse,
			"w-add-neuron":            *wAddNeuron,
			"w-remove-neuron":         *wRemoveNeuron,
			"w-plasticity-rule":       *wPlasticityRule,
			"w-plasticity":            *wPlasticity,
			"w-substrate":             *wSubstrate,
		})
		if err != nil {
			return err
		}
	}
	if *profileName != "" {
		preset, err := loadParityPreset(*profileName)
		if err != nil {
			return err
		}
		req.Selection = preset.Selection
		req.TuneSelection = preset.TuneSelection
		req.WeightPerturb = preset.WeightPerturb
		req.WeightBias = preset.WeightBias
		req.WeightRemoveBias = preset.WeightRemoveBias
		req.WeightActivation = preset.WeightActivation
		req.WeightAggregator = preset.WeightAggregator
		req.WeightAddSynapse = preset.WeightAddSyn
		req.WeightRemoveSynapse = preset.WeightRemoveSyn
		req.WeightAddNeuron = preset.WeightAddNeuro
		req.WeightRemoveNeuron = preset.WeightRemoveNeuro
		req.WeightPlasticityRule = preset.WeightPlasticityRule
		req.WeightPlasticity = preset.WeightPlasticity
		req.WeightSubstrate = preset.WeightSubstrate
	}
	req.TuneSelection = normalizeTuneSelection(req.TuneSelection)
	if req.WeightPerturb < 0 || req.WeightBias < 0 || req.WeightRemoveBias < 0 || req.WeightActivation < 0 || req.WeightAggregator < 0 || req.WeightAddSynapse < 0 || req.WeightRemoveSynapse < 0 || req.WeightAddNeuron < 0 || req.WeightRemoveNeuron < 0 || req.WeightPlasticityRule < 0 || req.WeightPlasticity < 0 || req.WeightSubstrate < 0 {
		return errors.New("mutation weights must be >= 0")
	}
	weightSum := req.WeightPerturb + req.WeightBias + req.WeightRemoveBias + req.WeightActivation + req.WeightAggregator + req.WeightAddSynapse + req.WeightRemoveSynapse + req.WeightAddNeuron + req.WeightRemoveNeuron + req.WeightPlasticityRule + req.WeightPlasticity + req.WeightSubstrate
	if weightSum <= 0 && (*configPath == "" || *profileName != "" || hasAnyWeightOverrideFlag(setFlags)) {
		return errors.New("at least one mutation weight must be > 0")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()
	if err := morphology.EnsureScapeCompatibility(req.Scape); err != nil {
		return err
	}

	runSummary, err := client.Run(ctx, req)
	if err != nil {
		return err
	}
	fmt.Printf("run completed run_id=%s scape=%s pop=%d gens=%d seed=%d\n", runSummary.RunID, req.Scape, req.Population, req.Generations, req.Seed)
	for i, best := range runSummary.BestByGeneration {
		fmt.Printf("generation=%d best_fitness=%.6f\n", i+1, best)
	}
	fmt.Printf("final_best_fitness=%.6f\n", runSummary.FinalBestFitness)
	if runSummary.Compare != nil {
		fmt.Printf("compare_tuning without_final=%.6f with_final=%.6f improvement=%.6f\n",
			runSummary.Compare.WithoutFinalBest,
			runSummary.Compare.WithFinalBest,
			runSummary.Compare.FinalImprovement,
		)
	}
	fmt.Printf("artifacts_dir=%s\n", filepath.Clean(runSummary.ArtifactsDir))
	return nil
}

func runRuns(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("runs", flag.ContinueOnError)
	limit := fs.Int("limit", 20, "max runs to list")
	showCompare := fs.Bool("show-compare", false, "show compare-tuning improvement when available")
	jsonOut := fs.Bool("json", false, "emit runs list as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *limit <= 0 {
		return errors.New("limit must be > 0")
	}

	entries, err := stats.ListRunIndex(benchmarksDir)
	if err != nil {
		return err
	}
	if len(entries) == 0 {
		fmt.Println("no runs found")
		return nil
	}

	if len(entries) > *limit {
		entries = entries[:*limit]
	}
	if *jsonOut {
		type runsItem struct {
			RunID              string   `json:"run_id"`
			CreatedAtUTC       string   `json:"created_at_utc"`
			Scape              string   `json:"scape"`
			Seed               int64    `json:"seed"`
			PopulationSize     int      `json:"population_size"`
			Generations        int      `json:"generations"`
			TuningEnabled      bool     `json:"tuning_enabled"`
			FinalBestFitness   float64  `json:"final_best_fitness"`
			CompareImprovement *float64 `json:"compare_improvement,omitempty"`
		}
		items := make([]runsItem, 0, len(entries))
		for _, e := range entries {
			var compare *float64
			if *showCompare {
				report, ok, err := stats.ReadTuningComparison(benchmarksDir, e.RunID)
				if err != nil {
					return err
				}
				if ok {
					v := report.FinalImprovement
					compare = &v
				}
			}
			items = append(items, runsItem{
				RunID:              e.RunID,
				CreatedAtUTC:       e.CreatedAtUTC,
				Scape:              e.Scape,
				Seed:               e.Seed,
				PopulationSize:     e.PopulationSize,
				Generations:        e.Generations,
				TuningEnabled:      e.TuningEnabled,
				FinalBestFitness:   e.FinalBestFitness,
				CompareImprovement: compare,
			})
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(items)
	}

	for _, e := range entries {
		compareDisplay := "n/a"
		if *showCompare {
			report, ok, err := stats.ReadTuningComparison(benchmarksDir, e.RunID)
			if err != nil {
				return err
			}
			if ok {
				compareDisplay = fmt.Sprintf("%.6f", report.FinalImprovement)
			}
		}

		fmt.Printf("run_id=%s created_at=%s scape=%s seed=%d pop=%d gens=%d tuning=%t final_best_fitness=%.6f compare_improvement=%s\n",
			e.RunID,
			e.CreatedAtUTC,
			e.Scape,
			e.Seed,
			e.PopulationSize,
			e.Generations,
			e.TuningEnabled,
			e.FinalBestFitness,
			compareDisplay,
		)
	}
	return nil
}

func runLineage(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("lineage", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "show lineage for the most recent run from run index")
	limit := fs.Int("limit", 50, "max lineage rows to print (<=0 for all)")
	jsonOut := fs.Bool("json", false, "emit lineage rows as JSON")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID != "" && *latest {
		return errors.New("use either --run-id or --latest, not both")
	}
	if *runID == "" && !*latest {
		return errors.New("lineage requires --run-id or --latest")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()

	lineage, err := client.Lineage(ctx, protoapi.LineageRequest{
		RunID:  *runID,
		Latest: *latest,
		Limit:  *limit,
	})
	if err != nil {
		return err
	}
	if len(lineage) == 0 {
		fmt.Println("no lineage records")
		return nil
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(lineage)
	}

	for _, rec := range lineage {
		fmt.Printf("gen=%d genome_id=%s parent_id=%s op=%s fingerprint=%s neurons=%d synapses=%d\n",
			rec.Generation,
			rec.GenomeID,
			rec.ParentID,
			rec.Operation,
			rec.Fingerprint,
			rec.Summary.TotalNeurons,
			rec.Summary.TotalSynapses,
		)
	}
	return nil
}

func runFitness(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("fitness", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "show fitness history for the most recent run from run index")
	limit := fs.Int("limit", 50, "max generations to print (<=0 for all)")
	jsonOut := fs.Bool("json", false, "emit fitness history as JSON")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID != "" && *latest {
		return errors.New("use either --run-id or --latest, not both")
	}
	if *runID == "" && !*latest {
		return errors.New("fitness requires --run-id or --latest")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()

	history, err := client.FitnessHistory(ctx, protoapi.FitnessHistoryRequest{
		RunID:  *runID,
		Latest: *latest,
		Limit:  *limit,
	})
	if err != nil {
		return err
	}
	if len(history) == 0 {
		fmt.Println("no fitness history")
		return nil
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(history)
	}

	for i, best := range history {
		fmt.Printf("generation=%d best_fitness=%.6f\n", i+1, best)
	}
	return nil
}

func runDiagnostics(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("diagnostics", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "show diagnostics for the most recent run from run index")
	limit := fs.Int("limit", 50, "max generations to print (<=0 for all)")
	jsonOut := fs.Bool("json", false, "emit diagnostics as JSON")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID != "" && *latest {
		return errors.New("use either --run-id or --latest, not both")
	}
	if *runID == "" && !*latest {
		return errors.New("diagnostics requires --run-id or --latest")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()

	diagnostics, err := client.Diagnostics(ctx, protoapi.DiagnosticsRequest{
		RunID:  *runID,
		Latest: *latest,
		Limit:  *limit,
	})
	if err != nil {
		return err
	}
	if len(diagnostics) == 0 {
		fmt.Println("no diagnostics")
		return nil
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(diagnostics)
	}

	for _, d := range diagnostics {
		fmt.Printf("generation=%d best=%.6f mean=%.6f min=%.6f species=%d fingerprints=%d threshold=%.4f target_species=%d mean_species_size=%.2f largest_species=%d tuning_invocations=%d tuning_attempts=%d tuning_evaluations=%d tuning_accepted=%d tuning_rejected=%d tuning_goal_hits=%d tuning_accept_rate=%.4f tuning_evals_per_attempt=%.4f\n",
			d.Generation,
			d.BestFitness,
			d.MeanFitness,
			d.MinFitness,
			d.SpeciesCount,
			d.FingerprintDiversity,
			d.SpeciationThreshold,
			d.TargetSpeciesCount,
			d.MeanSpeciesSize,
			d.LargestSpeciesSize,
			d.TuningInvocations,
			d.TuningAttempts,
			d.TuningEvaluations,
			d.TuningAccepted,
			d.TuningRejected,
			d.TuningGoalHits,
			d.TuningAcceptRate,
			d.TuningEvalsPerAttempt,
		)
	}
	return nil
}

func runTop(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("top", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "show top genomes for the most recent run from run index")
	limit := fs.Int("limit", 5, "max top genomes to print (<=0 for all)")
	jsonOut := fs.Bool("json", false, "emit top genomes as JSON")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID != "" && *latest {
		return errors.New("use either --run-id or --latest, not both")
	}
	if *runID == "" && !*latest {
		return errors.New("top requires --run-id or --latest")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()

	top, err := client.TopGenomes(ctx, protoapi.TopGenomesRequest{
		RunID:  *runID,
		Latest: *latest,
		Limit:  *limit,
	})
	if err != nil {
		return err
	}
	if len(top) == 0 {
		fmt.Println("no top genomes")
		return nil
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(top)
	}

	for _, item := range top {
		fmt.Printf("rank=%d fitness=%.6f genome_id=%s neurons=%d synapses=%d\n",
			item.Rank,
			item.Fitness,
			item.Genome.ID,
			len(item.Genome.Neurons),
			len(item.Genome.Synapses),
		)
	}
	return nil
}

func runSpecies(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("species", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "show species history for the most recent run from run index")
	limit := fs.Int("limit", 50, "max generations to print (<=0 for all)")
	jsonOut := fs.Bool("json", false, "emit species history as JSON")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID != "" && *latest {
		return errors.New("use either --run-id or --latest, not both")
	}
	if *runID == "" && !*latest {
		return errors.New("species requires --run-id or --latest")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()

	history, err := client.SpeciesHistory(ctx, protoapi.SpeciesHistoryRequest{
		RunID:  *runID,
		Latest: *latest,
		Limit:  *limit,
	})
	if err != nil {
		return err
	}
	if len(history) == 0 {
		fmt.Println("no species history")
		return nil
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(history)
	}
	for _, generation := range history {
		fmt.Printf("generation=%d species=%d new=%d extinct=%d\n",
			generation.Generation,
			len(generation.Species),
			len(generation.NewSpecies),
			len(generation.ExtinctSpecies),
		)
		for _, item := range generation.Species {
			fmt.Printf("species_key=%s size=%d mean=%.6f best=%.6f\n", item.Key, item.Size, item.MeanFitness, item.BestFitness)
		}
	}
	return nil
}

func runSpeciesDiff(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("species-diff", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "diff species history for the most recent run from run index")
	fromGen := fs.Int("from-gen", 0, "from generation (default: previous generation)")
	toGen := fs.Int("to-gen", 0, "to generation (default: latest generation)")
	showDiagnostics := fs.Bool("show-diagnostics", false, "print from/to generation diagnostics snapshots alongside species diff")
	jsonOut := fs.Bool("json", false, "emit species diff as JSON")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID != "" && *latest {
		return errors.New("use either --run-id or --latest, not both")
	}
	if *runID == "" && !*latest {
		return errors.New("species-diff requires --run-id or --latest")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()

	diff, err := client.SpeciesDiff(ctx, protoapi.SpeciesDiffRequest{
		RunID:          *runID,
		Latest:         *latest,
		FromGeneration: *fromGen,
		ToGeneration:   *toGen,
	})
	if err != nil {
		return err
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(diff)
	}

	fmt.Printf("run_id=%s from=%d to=%d added=%d removed=%d changed=%d unchanged=%d tuning_delta_invocations=%+d tuning_delta_attempts=%+d tuning_delta_evaluations=%+d tuning_delta_accepted=%+d tuning_delta_rejected=%+d tuning_delta_goal_hits=%+d tuning_delta_accept_rate=%+.4f tuning_delta_evals_per_attempt=%+.4f\n",
		diff.RunID,
		diff.FromGeneration,
		diff.ToGeneration,
		len(diff.Added),
		len(diff.Removed),
		len(diff.Changed),
		diff.UnchangedCount,
		diff.TuningInvocationsDelta,
		diff.TuningAttemptsDelta,
		diff.TuningEvaluationsDelta,
		diff.TuningAcceptedDelta,
		diff.TuningRejectedDelta,
		diff.TuningGoalHitsDelta,
		diff.TuningAcceptRateDelta,
		diff.TuningEvalsPerAttemptDelta,
	)
	if *showDiagnostics {
		fmt.Printf("from_diag generation=%d best=%.6f mean=%.6f min=%.6f species=%d fingerprints=%d tuning_invocations=%d tuning_attempts=%d tuning_evaluations=%d tuning_accepted=%d tuning_rejected=%d tuning_goal_hits=%d tuning_accept_rate=%.4f tuning_evals_per_attempt=%.4f\n",
			diff.FromDiagnostics.Generation,
			diff.FromDiagnostics.BestFitness,
			diff.FromDiagnostics.MeanFitness,
			diff.FromDiagnostics.MinFitness,
			diff.FromDiagnostics.SpeciesCount,
			diff.FromDiagnostics.FingerprintDiversity,
			diff.FromDiagnostics.TuningInvocations,
			diff.FromDiagnostics.TuningAttempts,
			diff.FromDiagnostics.TuningEvaluations,
			diff.FromDiagnostics.TuningAccepted,
			diff.FromDiagnostics.TuningRejected,
			diff.FromDiagnostics.TuningGoalHits,
			diff.FromDiagnostics.TuningAcceptRate,
			diff.FromDiagnostics.TuningEvalsPerAttempt,
		)
		fmt.Printf("to_diag generation=%d best=%.6f mean=%.6f min=%.6f species=%d fingerprints=%d tuning_invocations=%d tuning_attempts=%d tuning_evaluations=%d tuning_accepted=%d tuning_rejected=%d tuning_goal_hits=%d tuning_accept_rate=%.4f tuning_evals_per_attempt=%.4f\n",
			diff.ToDiagnostics.Generation,
			diff.ToDiagnostics.BestFitness,
			diff.ToDiagnostics.MeanFitness,
			diff.ToDiagnostics.MinFitness,
			diff.ToDiagnostics.SpeciesCount,
			diff.ToDiagnostics.FingerprintDiversity,
			diff.ToDiagnostics.TuningInvocations,
			diff.ToDiagnostics.TuningAttempts,
			diff.ToDiagnostics.TuningEvaluations,
			diff.ToDiagnostics.TuningAccepted,
			diff.ToDiagnostics.TuningRejected,
			diff.ToDiagnostics.TuningGoalHits,
			diff.ToDiagnostics.TuningAcceptRate,
			diff.ToDiagnostics.TuningEvalsPerAttempt,
		)
	}
	for _, item := range diff.Added {
		fmt.Printf("added species_key=%s size=%d mean=%.6f best=%.6f\n", item.Key, item.Size, item.MeanFitness, item.BestFitness)
	}
	for _, item := range diff.Removed {
		fmt.Printf("removed species_key=%s size=%d mean=%.6f best=%.6f\n", item.Key, item.Size, item.MeanFitness, item.BestFitness)
	}
	for _, item := range diff.Changed {
		fmt.Printf("changed species_key=%s size=%d->%d delta=%+d mean=%.6f->%.6f delta=%+.6f best=%.6f->%.6f delta=%+.6f\n",
			item.Key,
			item.FromSize,
			item.ToSize,
			item.SizeDelta,
			item.FromMeanFitness,
			item.ToMeanFitness,
			item.MeanDelta,
			item.FromBestFitness,
			item.ToBestFitness,
			item.BestDelta,
		)
	}
	return nil
}

func runScapeSummary(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("scape-summary", flag.ContinueOnError)
	scapeName := fs.String("scape", "", "scape name")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *scapeName == "" {
		return errors.New("scape-summary requires --scape")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()

	summary, err := client.ScapeSummary(ctx, *scapeName)
	if err != nil {
		return err
	}
	fmt.Printf("scape=%s best_fitness=%.6f description=%s\n",
		summary.Name,
		summary.BestFitness,
		summary.Description,
	)
	return nil
}

func runBenchmark(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	configPath := fs.String("config", "", "optional run config JSON path (map2rec-backed)")
	runID := fs.String("run-id", "", "explicit run id (optional)")
	continuePopID := fs.String("continue-pop-id", "", "continue from persisted population snapshot id")
	specieIdentifier := fs.String("specie-identifier", "topology", "species identifier: topology|tot_n|fingerprint")
	opMode := fs.String("op-mode", "gt", "operation mode: gt|validation|test (or composite gt+validation/test)")
	evolutionType := fs.String("evolution-type", "generational", "evolution type: generational|steady_state")
	scapeName := fs.String("scape", "xor", "scape name")
	population := fs.Int("pop", 50, "population size")
	generations := fs.Int("gens", 100, "generation count")
	survivalPercentage := fs.Float64("survival-percentage", 0.0, "survival percentage used to derive elite retention when elite count is unset")
	specieSizeLimit := fs.Int("specie-size-limit", 0, "maximum parent-pool size retained per species (0 disables)")
	fitnessGoal := fs.Float64("fitness-goal", 0.0, "early-stop best fitness goal (0 disables)")
	evaluationsLimit := fs.Int("evaluations-limit", 0, "early-stop total evaluation limit (0 disables)")
	traceStepSize := fs.Int("trace-step-size", 500, "trace update cadence in total evaluations (0 uses runtime default)")
	startPaused := fs.Bool("start-paused", false, "start monitor in paused state (requires continue)")
	autoContinueMS := fs.Int("auto-continue-ms", 0, "auto-send continue after N milliseconds when start-paused is set (0 disables)")
	seed := fs.Int64("seed", 1, "rng seed")
	workers := fs.Int("workers", 4, "worker count")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	enableTuning := fs.Bool("tuning", false, "enable exoself tuning")
	validationProbe := fs.Bool("validation-probe", false, "evaluate per-species champions in validation probe during gt runs")
	testProbe := fs.Bool("test-probe", false, "evaluate per-species champions in test probe during gt runs")
	profileName := fs.String("profile", "", "optional parity profile id (from testdata/fixtures/parity/ref_benchmarker_profiles.json)")
	selectionName := fs.String("selection", "elite", "parent selection strategy: elite|tournament|species_tournament|species_shared_tournament|hof_competition|hof_rank|hof_top3|hof_efficiency|hof_random|competition|top3")
	postprocessorName := fs.String("fitness-postprocessor", "none", "fitness postprocessor: none|size_proportional|nsize_proportional|novelty_proportional")
	topoPolicyName := fs.String("topo-policy", "const", "topological mutation count policy: const|ncount_linear|ncount_exponential")
	topoCount := fs.Int("topo-count", 1, "mutation count for topo-policy=const")
	topoParam := fs.Float64("topo-param", 0.5, "policy parameter (multiplier/power) for topo-policy")
	topoMax := fs.Int("topo-max", 8, "maximum mutation count for non-const topo policies (<=0 disables cap)")
	tuneAttempts := fs.Int("attempts", 4, "tuning attempts per agent evaluation")
	tuneSteps := fs.Int("tune-steps", 6, "tuning perturbation steps per attempt")
	tuneStepSize := fs.Float64("tune-step-size", 0.35, "tuning perturbation magnitude")
	tunePerturbationRange := fs.Float64("tune-perturbation-range", 1.0, "tuning perturbation spread multiplier")
	tuneAnnealingFactor := fs.Float64("tune-annealing-factor", 1.0, "tuning per-step annealing factor")
	tuneMinImprovement := fs.Float64("tune-min-improvement", 0.0, "minimum fitness gain required to accept a tuning candidate")
	tuneSelection := fs.String("tune-selection", tuning.CandidateSelectBestSoFar, "tuner candidate selection: best_so_far|original|dynamic|dynamic_random|all|all_random|active|active_random|recent|recent_random|current|current_random|lastgen|lastgen_random")
	tuneDurationPolicy := fs.String("tune-duration-policy", "fixed", "tuning attempt policy: fixed|const|linear_decay|topology_scaled|nsize_proportional|wsize_proportional")
	tuneDurationParam := fs.Float64("tune-duration-param", 1.0, "tuning attempt policy parameter")
	wPerturb := fs.Float64("w-perturb", 0.70, "weight for perturb_random_weight mutation")
	wBias := fs.Float64("w-bias", 0.00, "weight for perturb_random_bias mutation")
	wRemoveBias := fs.Float64("w-remove-bias", 0.00, "weight for remove_random_bias mutation")
	wActivation := fs.Float64("w-activation", 0.00, "weight for change_random_activation mutation")
	wAggregator := fs.Float64("w-aggregator", 0.00, "weight for change_random_aggregator mutation")
	wAddSynapse := fs.Float64("w-add-synapse", 0.10, "weight for add_random_synapse mutation")
	wRemoveSynapse := fs.Float64("w-remove-synapse", 0.08, "weight for remove_random_synapse mutation")
	wAddNeuron := fs.Float64("w-add-neuron", 0.07, "weight for add_random_neuron mutation")
	wRemoveNeuron := fs.Float64("w-remove-neuron", 0.05, "weight for remove_random_neuron mutation")
	wPlasticityRule := fs.Float64("w-plasticity-rule", 0.00, "weight for change_plasticity_rule mutation")
	wPlasticity := fs.Float64("w-plasticity", 0.03, "weight for perturb_plasticity_rate mutation")
	wSubstrate := fs.Float64("w-substrate", 0.02, "weight for perturb_substrate_parameter mutation")
	minImprovement := fs.Float64("min-improvement", 0.001, "minimum expected fitness improvement")
	if err := fs.Parse(args); err != nil {
		return err
	}
	setFlags := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		setFlags[f.Name] = true
	})

	req, err := loadOrDefaultRunRequest(*configPath)
	if err != nil {
		return err
	}
	if *configPath == "" {
		req = protoapi.RunRequest{
			Scape:                 *scapeName,
			OpMode:                *opMode,
			EvolutionType:         *evolutionType,
			RunID:                 *runID,
			ContinuePopulationID:  *continuePopID,
			SpecieIdentifier:      *specieIdentifier,
			Population:            *population,
			Generations:           *generations,
			SurvivalPercentage:    *survivalPercentage,
			SpecieSizeLimit:       *specieSizeLimit,
			FitnessGoal:           *fitnessGoal,
			EvaluationsLimit:      *evaluationsLimit,
			TraceStepSize:         *traceStepSize,
			StartPaused:           *startPaused,
			AutoContinueAfter:     time.Duration(*autoContinueMS) * time.Millisecond,
			Seed:                  *seed,
			Workers:               *workers,
			Selection:             *selectionName,
			FitnessPostprocessor:  *postprocessorName,
			TopologicalPolicy:     *topoPolicyName,
			TopologicalCount:      *topoCount,
			TopologicalParam:      *topoParam,
			TopologicalMax:        *topoMax,
			EnableTuning:          *enableTuning,
			ValidationProbe:       *validationProbe,
			TestProbe:             *testProbe,
			TuneSelection:         *tuneSelection,
			TuneDurationPolicy:    *tuneDurationPolicy,
			TuneDurationParam:     *tuneDurationParam,
			TuneAttempts:          *tuneAttempts,
			TuneSteps:             *tuneSteps,
			TuneStepSize:          *tuneStepSize,
			TunePerturbationRange: *tunePerturbationRange,
			TuneAnnealingFactor:   *tuneAnnealingFactor,
			TuneMinImprovement:    *tuneMinImprovement,
			WeightPerturb:         *wPerturb,
			WeightBias:            *wBias,
			WeightRemoveBias:      *wRemoveBias,
			WeightActivation:      *wActivation,
			WeightAggregator:      *wAggregator,
			WeightAddSynapse:      *wAddSynapse,
			WeightRemoveSynapse:   *wRemoveSynapse,
			WeightAddNeuron:       *wAddNeuron,
			WeightRemoveNeuron:    *wRemoveNeuron,
			WeightPlasticityRule:  *wPlasticityRule,
			WeightPlasticity:      *wPlasticity,
			WeightSubstrate:       *wSubstrate,
		}
	} else {
		err := overrideFromFlags(&req, setFlags, map[string]any{
			"scape":                   *scapeName,
			"op-mode":                 *opMode,
			"evolution-type":          *evolutionType,
			"run-id":                  *runID,
			"continue-pop-id":         *continuePopID,
			"specie-identifier":       *specieIdentifier,
			"pop":                     *population,
			"gens":                    *generations,
			"survival-percentage":     *survivalPercentage,
			"specie-size-limit":       *specieSizeLimit,
			"fitness-goal":            *fitnessGoal,
			"evaluations-limit":       *evaluationsLimit,
			"trace-step-size":         *traceStepSize,
			"start-paused":            *startPaused,
			"auto-continue-ms":        *autoContinueMS,
			"seed":                    *seed,
			"workers":                 *workers,
			"tuning":                  *enableTuning,
			"validation-probe":        *validationProbe,
			"test-probe":              *testProbe,
			"selection":               *selectionName,
			"fitness-postprocessor":   *postprocessorName,
			"topo-policy":             *topoPolicyName,
			"topo-count":              *topoCount,
			"topo-param":              *topoParam,
			"topo-max":                *topoMax,
			"attempts":                *tuneAttempts,
			"tune-steps":              *tuneSteps,
			"tune-step-size":          *tuneStepSize,
			"tune-perturbation-range": *tunePerturbationRange,
			"tune-annealing-factor":   *tuneAnnealingFactor,
			"tune-min-improvement":    *tuneMinImprovement,
			"tune-selection":          *tuneSelection,
			"tune-duration-policy":    *tuneDurationPolicy,
			"tune-duration-param":     *tuneDurationParam,
			"w-perturb":               *wPerturb,
			"w-bias":                  *wBias,
			"w-remove-bias":           *wRemoveBias,
			"w-activation":            *wActivation,
			"w-aggregator":            *wAggregator,
			"w-add-synapse":           *wAddSynapse,
			"w-remove-synapse":        *wRemoveSynapse,
			"w-add-neuron":            *wAddNeuron,
			"w-remove-neuron":         *wRemoveNeuron,
			"w-plasticity-rule":       *wPlasticityRule,
			"w-plasticity":            *wPlasticity,
			"w-substrate":             *wSubstrate,
		})
		if err != nil {
			return err
		}
	}
	if *profileName != "" {
		preset, err := loadParityPreset(*profileName)
		if err != nil {
			return err
		}
		req.Selection = preset.Selection
		req.TuneSelection = preset.TuneSelection
		req.WeightPerturb = preset.WeightPerturb
		req.WeightBias = preset.WeightBias
		req.WeightRemoveBias = preset.WeightRemoveBias
		req.WeightActivation = preset.WeightActivation
		req.WeightAggregator = preset.WeightAggregator
		req.WeightAddSynapse = preset.WeightAddSyn
		req.WeightRemoveSynapse = preset.WeightRemoveSyn
		req.WeightAddNeuron = preset.WeightAddNeuro
		req.WeightRemoveNeuron = preset.WeightRemoveNeuro
		req.WeightPlasticityRule = preset.WeightPlasticityRule
		req.WeightPlasticity = preset.WeightPlasticity
		req.WeightSubstrate = preset.WeightSubstrate
	}
	req.TuneSelection = normalizeTuneSelection(req.TuneSelection)
	if req.WeightPerturb < 0 || req.WeightBias < 0 || req.WeightRemoveBias < 0 || req.WeightActivation < 0 || req.WeightAggregator < 0 || req.WeightAddSynapse < 0 || req.WeightRemoveSynapse < 0 || req.WeightAddNeuron < 0 || req.WeightRemoveNeuron < 0 || req.WeightPlasticityRule < 0 || req.WeightPlasticity < 0 || req.WeightSubstrate < 0 {
		return errors.New("mutation weights must be >= 0")
	}
	weightSum := req.WeightPerturb + req.WeightBias + req.WeightRemoveBias + req.WeightActivation + req.WeightAggregator + req.WeightAddSynapse + req.WeightRemoveSynapse + req.WeightAddNeuron + req.WeightRemoveNeuron + req.WeightPlasticityRule + req.WeightPlasticity + req.WeightSubstrate
	if weightSum <= 0 && (*configPath == "" || *profileName != "" || hasAnyWeightOverrideFlag(setFlags)) {
		return errors.New("at least one mutation weight must be > 0")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()
	if err := morphology.EnsureScapeCompatibility(req.Scape); err != nil {
		return err
	}

	runSummary, err := client.Run(ctx, req)
	if err != nil {
		return err
	}
	if len(runSummary.BestByGeneration) == 0 {
		return errors.New("benchmark run produced empty fitness history")
	}

	initialBest := runSummary.BestByGeneration[0]
	bestMean, bestStd, bestMax, bestMin := bestSeriesStats(runSummary.BestByGeneration)
	improvement := runSummary.FinalBestFitness - initialBest
	passed := improvement >= *minImprovement
	report := stats.BenchmarkSummary{
		RunID:          runSummary.RunID,
		Scape:          req.Scape,
		PopulationSize: req.Population,
		Generations:    req.Generations,
		Seed:           req.Seed,
		InitialBest:    initialBest,
		FinalBest:      runSummary.FinalBestFitness,
		BestMean:       bestMean,
		BestStd:        bestStd,
		BestMax:        bestMax,
		BestMin:        bestMin,
		Improvement:    improvement,
		MinImprovement: *minImprovement,
		Passed:         passed,
	}
	if err := stats.WriteBenchmarkSummary(runSummary.ArtifactsDir, report); err != nil {
		return err
	}

	fmt.Printf("benchmark run_id=%s scape=%s initial_best=%.6f final_best=%.6f mean_best=%.6f std_best=%.6f best_min=%.6f best_max=%.6f improvement=%.6f threshold=%.6f passed=%t\n",
		runSummary.RunID,
		req.Scape,
		initialBest,
		runSummary.FinalBestFitness,
		bestMean,
		bestStd,
		bestMin,
		bestMax,
		improvement,
		*minImprovement,
		passed,
	)
	fmt.Printf("benchmark_summary=%s\n", filepath.Join(runSummary.ArtifactsDir, "benchmark_summary.json"))
	return nil
}

func bestSeriesStats(values []float64) (mean, std, max, min float64) {
	if len(values) == 0 {
		return 0, 0, 0, 0
	}
	min = values[0]
	max = values[0]
	total := 0.0
	for _, value := range values {
		total += value
		if value > max {
			max = value
		}
		if value < min {
			min = value
		}
	}
	mean = total / float64(len(values))
	sumSq := 0.0
	for _, value := range values {
		diff := mean - value
		sumSq += diff * diff
	}
	std = math.Sqrt(sumSq / float64(len(values)))
	return mean, std, max, min
}

func runProfile(_ context.Context, args []string) error {
	if len(args) == 0 {
		return errors.New("profile requires a subcommand: list|show")
	}
	switch args[0] {
	case "list":
		profiles, err := listParityProfiles()
		if err != nil {
			return err
		}
		if len(profiles) == 0 {
			fmt.Println("no profiles")
			return nil
		}
		for _, profile := range profiles {
			fmt.Printf("id=%s selection=%s expected_selection=%s tune_selection=%s expected_tune_selection=%s mutation_ops=%d\n",
				profile.ID,
				profile.PopulationSelection,
				profile.ExpectedSelection,
				profile.TuningSelection,
				profile.ExpectedTuning,
				profile.MutationOperatorLen,
			)
		}
		return nil
	case "show":
		fs := flag.NewFlagSet("profile show", flag.ContinueOnError)
		id := fs.String("id", "", "profile id")
		asJSON := fs.Bool("json", false, "print resolved profile as JSON")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if *id == "" {
			return errors.New("profile show requires --id")
		}
		resolved, err := resolveParityProfile(*id)
		if err != nil {
			return err
		}
		if *asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(resolved)
		}
		fmt.Printf("id=%s selection=%s expected_selection=%s tune_selection=%s expected_tune_selection=%s mutation_ops=%d w_perturb=%.3f w_bias=%.3f w_remove_bias=%.3f w_activation=%.3f w_aggregator=%.3f w_add_syn=%.3f w_remove_syn=%.3f w_add_neuron=%.3f w_remove_neuron=%.3f w_plasticity_rule=%.3f w_plasticity=%.3f w_substrate=%.3f\n",
			resolved.ID,
			resolved.PopulationSelection,
			resolved.ExpectedSelection,
			resolved.TuningSelection,
			resolved.ExpectedTuning,
			resolved.MutationOperatorLen,
			resolved.WeightPerturb,
			resolved.WeightBias,
			resolved.WeightRemoveBias,
			resolved.WeightActivation,
			resolved.WeightAggregator,
			resolved.WeightAddSyn,
			resolved.WeightRemoveSyn,
			resolved.WeightAddNeuro,
			resolved.WeightRemoveNeuro,
			resolved.WeightPlasticityRule,
			resolved.WeightPlasticity,
			resolved.WeightSubstrate,
		)
		return nil
	default:
		return fmt.Errorf("unsupported profile subcommand: %s", args[0])
	}
}

func runExport(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "export the most recent run from run index")
	outDir := fs.String("out", exportsDir, "export output directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID != "" && *latest {
		return errors.New("use either --run-id or --latest, not both")
	}
	if *runID == "" && !*latest {
		return errors.New("export requires --run-id or --latest")
	}
	if *latest {
		entries, err := stats.ListRunIndex(benchmarksDir)
		if err != nil {
			return err
		}
		if len(entries) == 0 {
			return errors.New("no runs available to export")
		}
		*runID = entries[0].RunID
	}

	exportedDir, err := stats.ExportRunArtifacts(benchmarksDir, *runID, *outDir)
	if err != nil {
		return err
	}

	fmt.Printf("exported run_id=%s to=%s\n", *runID, filepath.Clean(exportedDir))
	return nil
}

func runMonitor(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return errors.New("monitor requires an action: pause|continue|stop|goal-reached|print-trace")
	}
	action := args[0]
	fs := flag.NewFlagSet("monitor", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	if *runID == "" {
		return errors.New("monitor requires --run-id")
	}

	client, err := protoapi.New(protoapi.Options{
		StoreKind:     *storeKind,
		DBPath:        *dbPath,
		BenchmarksDir: benchmarksDir,
		ExportsDir:    exportsDir,
	})
	if err != nil {
		return err
	}
	defer func() {
		_ = client.Close()
	}()

	req := protoapi.MonitorControlRequest{RunID: *runID}
	switch action {
	case "pause":
		err = client.PauseRun(ctx, req)
	case "continue":
		err = client.ContinueRun(ctx, req)
	case "stop":
		err = client.StopRun(ctx, req)
	case "goal-reached":
		err = client.GoalReachedRun(ctx, req)
	case "print-trace":
		err = client.PrintTraceRun(ctx, req)
	default:
		return fmt.Errorf("unknown monitor action: %s", action)
	}
	if err != nil {
		return err
	}

	fmt.Printf("monitor action=%s run_id=%s\n", action, *runID)
	return nil
}

func runPopulation(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return errors.New("population requires a subcommand: delete")
	}
	switch args[0] {
	case "delete":
		fs := flag.NewFlagSet("population delete", flag.ContinueOnError)
		populationID := fs.String("id", "", "population id")
		storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
		dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if *populationID == "" {
			return errors.New("population delete requires --id")
		}

		client, err := protoapi.New(protoapi.Options{
			StoreKind:     *storeKind,
			DBPath:        *dbPath,
			BenchmarksDir: benchmarksDir,
			ExportsDir:    exportsDir,
		})
		if err != nil {
			return err
		}
		defer func() {
			_ = client.Close()
		}()

		if err := client.DeletePopulation(ctx, protoapi.DeletePopulationRequest{PopulationID: *populationID}); err != nil {
			return err
		}
		fmt.Printf("population deleted id=%s\n", *populationID)
		return nil
	default:
		return fmt.Errorf("unsupported population subcommand: %s", args[0])
	}
}

func registerDefaultScapes(p *platform.Polis) error {
	if err := p.RegisterScape(scape.XORScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.RegressionMimicScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.CartPoleLiteScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.Pole2BalancingScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.FlatlandScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.DTMScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.GTSAScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.FXScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.EpitopesScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.LLVMPhaseOrderingScape{}); err != nil {
		return err
	}
	return nil
}

func defaultMutationPolicy(
	seed int64,
	scapeName string,
	inputNeuronIDs, outputNeuronIDs []string,
	wPerturb, wBias, wRemoveBias, wActivation, wAggregator, wAddSynapse, wRemoveSynapse, wAddNeuron, wRemoveNeuron, wPlasticityRule, wPlasticity, wSubstrate float64,
) []evo.WeightedMutation {
	protected := make(map[string]struct{}, len(inputNeuronIDs)+len(outputNeuronIDs))
	for _, id := range inputNeuronIDs {
		protected[id] = struct{}{}
	}
	for _, id := range outputNeuronIDs {
		protected[id] = struct{}{}
	}

	return []evo.WeightedMutation{
		{Operator: &evo.MutateWeights{Rand: rand.New(rand.NewSource(seed + 1000)), MaxDelta: 1.0}, Weight: wPerturb},
		{Operator: &evo.AddBias{Rand: rand.New(rand.NewSource(seed + 1007)), MaxDelta: 0.3}, Weight: wBias},
		{Operator: &evo.RemoveBias{Rand: rand.New(rand.NewSource(seed + 1010))}, Weight: wRemoveBias},
		{Operator: &evo.MutateAF{Rand: rand.New(rand.NewSource(seed + 1008))}, Weight: wActivation},
		{Operator: &evo.MutateAggrF{Rand: rand.New(rand.NewSource(seed + 1009))}, Weight: wAggregator},
		{Operator: &evo.AddRandomInlink{Rand: rand.New(rand.NewSource(seed + 1001)), MaxAbsWeight: 1.0, InputNeuronIDs: inputNeuronIDs, FeedForwardOnly: true}, Weight: wAddSynapse / 2},
		{Operator: &evo.AddRandomOutlink{Rand: rand.New(rand.NewSource(seed + 1002)), MaxAbsWeight: 1.0, OutputNeuronIDs: outputNeuronIDs, FeedForwardOnly: true}, Weight: wAddSynapse / 2},
		{Operator: &evo.RemoveRandomInlink{Rand: rand.New(rand.NewSource(seed + 1003)), InputNeuronIDs: inputNeuronIDs, FeedForwardOnly: true}, Weight: wRemoveSynapse / 3},
		{Operator: &evo.RemoveRandomOutlink{Rand: rand.New(rand.NewSource(seed + 1004)), OutputNeuronIDs: outputNeuronIDs, FeedForwardOnly: true}, Weight: wRemoveSynapse / 3},
		{Operator: &evo.CutlinkFromNeuronToNeuron{Rand: rand.New(rand.NewSource(seed + 1005))}, Weight: wRemoveSynapse / 3},
		{Operator: &evo.AddNeuron{Rand: rand.New(rand.NewSource(seed + 1005))}, Weight: wAddNeuron * 0.40},
		{Operator: &evo.AddRandomOutsplice{Rand: rand.New(rand.NewSource(seed + 1006)), OutputNeuronIDs: outputNeuronIDs, FeedForwardOnly: true}, Weight: wAddNeuron * 0.30},
		{Operator: &evo.AddRandomInsplice{Rand: rand.New(rand.NewSource(seed + 1007)), InputNeuronIDs: inputNeuronIDs, FeedForwardOnly: true}, Weight: wAddNeuron * 0.30},
		{Operator: &evo.RemoveNeuronMutation{Rand: rand.New(rand.NewSource(seed + 1020)), Protected: protected}, Weight: wRemoveNeuron},
		{Operator: &evo.MutatePF{Rand: rand.New(rand.NewSource(seed + 1021))}, Weight: wPlasticityRule},
		{Operator: &evo.MutatePlasticityParameters{Rand: rand.New(rand.NewSource(seed + 1022)), MaxDelta: 0.15}, Weight: wPlasticity},
		{Operator: &evo.AddRandomSensor{Rand: rand.New(rand.NewSource(seed + 1008)), ScapeName: scapeName}, Weight: wSubstrate * 0.07},
		{Operator: &evo.AddRandomSensorLink{Rand: rand.New(rand.NewSource(seed + 1009)), ScapeName: scapeName}, Weight: wSubstrate * 0.07},
		{Operator: &evo.AddRandomActuator{Rand: rand.New(rand.NewSource(seed + 1010)), ScapeName: scapeName}, Weight: wSubstrate * 0.07},
		{Operator: &evo.AddRandomActuatorLink{Rand: rand.New(rand.NewSource(seed + 1011)), ScapeName: scapeName}, Weight: wSubstrate * 0.07},
		{Operator: &evo.RemoveRandomSensor{Rand: rand.New(rand.NewSource(seed + 1012))}, Weight: wSubstrate * 0.06},
		{Operator: &evo.CutlinkFromSensorToNeuron{Rand: rand.New(rand.NewSource(seed + 1013))}, Weight: wSubstrate * 0.06},
		{Operator: &evo.RemoveRandomActuator{Rand: rand.New(rand.NewSource(seed + 1014))}, Weight: wSubstrate * 0.06},
		{Operator: &evo.CutlinkFromNeuronToActuator{Rand: rand.New(rand.NewSource(seed + 1015))}, Weight: wSubstrate * 0.06},
		{Operator: &evo.AddRandomCPP{Rand: rand.New(rand.NewSource(seed + 1016))}, Weight: wSubstrate * 0.05},
		{Operator: &evo.RemoveRandomCPP{}, Weight: wSubstrate * 0.03},
		{Operator: &evo.AddRandomCEP{Rand: rand.New(rand.NewSource(seed + 1017))}, Weight: wSubstrate * 0.05},
		{Operator: &evo.RemoveRandomCEP{}, Weight: wSubstrate * 0.03},
		{Operator: &evo.AddCircuitNode{Rand: rand.New(rand.NewSource(seed + 1018))}, Weight: wSubstrate * 0.05},
		{Operator: &evo.DeleteCircuitNode{Rand: rand.New(rand.NewSource(seed + 1019))}, Weight: wSubstrate * 0.05},
		{Operator: &evo.AddCircuitLayer{Rand: rand.New(rand.NewSource(seed + 1020))}, Weight: wSubstrate * 0.05},
		{Operator: &evo.PerturbSubstrateParameter{Rand: rand.New(rand.NewSource(seed + 1021)), MaxDelta: 0.15}, Weight: wSubstrate * 0.05},
		{Operator: &evo.MutateTuningSelection{Rand: rand.New(rand.NewSource(seed + 1022))}, Weight: wSubstrate * 0.03},
		{Operator: &evo.MutateTuningAnnealing{Rand: rand.New(rand.NewSource(seed + 1023))}, Weight: wSubstrate * 0.03},
		{Operator: &evo.MutateTotTopologicalMutations{Rand: rand.New(rand.NewSource(seed + 1024))}, Weight: wSubstrate * 0.03},
		{Operator: &evo.MutateHeredityType{Rand: rand.New(rand.NewSource(seed + 1025))}, Weight: wSubstrate * 0.03},
	}
}

func usageError(msg string) error {
	return fmt.Errorf("%s\nusage: protogonosctl <init|reset|start|run|benchmark|profile|runs|lineage|fitness|diagnostics|species|species-diff|monitor|population|top|scape-summary|export> [flags]", msg)
}

func selectionFromName(name string) (evo.Selector, error) {
	switch name {
	case "elite":
		return evo.EliteSelector{}, nil
	case "tournament":
		return evo.TournamentSelector{PoolSize: 0, TournamentSize: 3}, nil
	case "species_tournament":
		return evo.SpeciesTournamentSelector{
			Identifier:     evo.TopologySpecieIdentifier{},
			PoolSize:       0,
			TournamentSize: 3,
		}, nil
	case "species_shared_tournament":
		return &evo.SpeciesSharedTournamentSelector{
			Identifier:     evo.TopologySpecieIdentifier{},
			PoolSize:       0,
			TournamentSize: 3,
		}, nil
	case "hof_competition":
		return &evo.SpeciesSharedTournamentSelector{
			Identifier:            evo.TopologySpecieIdentifier{},
			PoolSize:              0,
			TournamentSize:        3,
			StagnationGenerations: 2,
		}, nil
	case "hof_rank":
		return evo.RankSelector{PoolSize: 0}, nil
	case "hof_top3":
		return evo.TopKFitnessSelector{K: 3}, nil
	case "hof_efficiency":
		return evo.EfficiencySelector{PoolSize: 0}, nil
	case "hof_random":
		return evo.RandomSelector{PoolSize: 0}, nil
	case "competition":
		return &evo.SpeciesSharedTournamentSelector{
			Identifier:     evo.TopologySpecieIdentifier{},
			PoolSize:       0,
			TournamentSize: 3,
		}, nil
	case "top3":
		return evo.TopKFitnessSelector{K: 3}, nil
	case "rank":
		return evo.RankSelector{PoolSize: 0}, nil
	case "efficiency":
		return evo.EfficiencySelector{PoolSize: 0}, nil
	case "random":
		return evo.RandomSelector{PoolSize: 0}, nil
	default:
		return nil, fmt.Errorf("unsupported selection strategy: %s", name)
	}
}

func normalizeTuneSelection(name string) string {
	return tuning.NormalizeCandidateSelectionName(name)
}

func postprocessorFromName(name string) (evo.FitnessPostprocessor, error) {
	switch name {
	case "none":
		return evo.NoopFitnessPostprocessor{}, nil
	case "size_proportional":
		return evo.SizeProportionalPostprocessor{}, nil
	case "nsize_proportional":
		return evo.SizeProportionalPostprocessor{}, nil
	case "novelty_proportional":
		return evo.NoveltyProportionalPostprocessor{}, nil
	default:
		return nil, fmt.Errorf("unsupported fitness postprocessor: %s", name)
	}
}

func topologicalPolicyFromConfig(name string, count int, param float64, maxCount int) (evo.TopologicalMutationPolicy, error) {
	switch name {
	case "const":
		if count <= 0 {
			return nil, fmt.Errorf("topo-count must be > 0 for const policy")
		}
		return evo.ConstTopologicalMutations{Count: count}, nil
	case "ncount_linear":
		if param <= 0 {
			return nil, fmt.Errorf("topo-param must be > 0 for ncount_linear policy")
		}
		return evo.NCountLinearTopologicalMutations{
			Multiplier: param,
			MaxCount:   maxCount,
		}, nil
	case "ncount_exponential":
		if param <= 0 {
			return nil, fmt.Errorf("topo-param must be > 0 for ncount_exponential policy")
		}
		return evo.NCountExponentialTopologicalMutations{
			Power:    param,
			MaxCount: maxCount,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported topological mutation policy: %s", name)
	}
}
