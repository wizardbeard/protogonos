package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"protogonos/internal/evo"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
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
	case "start":
		return runStart(ctx, args[1:])
	case "run":
		return runRun(ctx, args[1:])
	case "benchmark":
		return runBenchmark(ctx, args[1:])
	case "runs":
		return runRuns(ctx, args[1:])
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
	scapeName := fs.String("scape", "xor", "scape name")
	population := fs.Int("pop", 50, "population size")
	generations := fs.Int("gens", 100, "generation count")
	seed := fs.Int64("seed", 1, "rng seed")
	workers := fs.Int("workers", 4, "worker count")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	enableTuning := fs.Bool("tuning", false, "enable exoself tuning")
	compareTuning := fs.Bool("compare-tuning", false, "run with and without tuning and emit side-by-side metrics")
	selectionName := fs.String("selection", "elite", "parent selection strategy: elite|tournament|species_tournament")
	postprocessorName := fs.String("fitness-postprocessor", "none", "fitness postprocessor: none|size_proportional|novelty_proportional")
	topoPolicyName := fs.String("topo-policy", "const", "topological mutation count policy: const|ncount_linear|ncount_exponential")
	topoCount := fs.Int("topo-count", 1, "mutation count for topo-policy=const")
	topoParam := fs.Float64("topo-param", 0.5, "policy parameter (multiplier/power) for topo-policy")
	topoMax := fs.Int("topo-max", 8, "maximum mutation count for non-const topo policies (<=0 disables cap)")
	tuneAttempts := fs.Int("attempts", 4, "tuning attempts per agent evaluation")
	tuneSteps := fs.Int("tune-steps", 6, "tuning perturbation steps per attempt")
	tuneStepSize := fs.Float64("tune-step-size", 0.35, "tuning perturbation magnitude")
	wPerturb := fs.Float64("w-perturb", 0.70, "weight for perturb_random_weight mutation")
	wAddSynapse := fs.Float64("w-add-synapse", 0.10, "weight for add_random_synapse mutation")
	wRemoveSynapse := fs.Float64("w-remove-synapse", 0.08, "weight for remove_random_synapse mutation")
	wAddNeuron := fs.Float64("w-add-neuron", 0.07, "weight for add_random_neuron mutation")
	wRemoveNeuron := fs.Float64("w-remove-neuron", 0.05, "weight for remove_random_neuron mutation")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *wPerturb < 0 || *wAddSynapse < 0 || *wRemoveSynapse < 0 || *wAddNeuron < 0 || *wRemoveNeuron < 0 {
		return errors.New("mutation weights must be >= 0")
	}
	if *wPerturb+*wAddSynapse+*wRemoveSynapse+*wAddNeuron+*wRemoveNeuron <= 0 {
		return errors.New("at least one mutation weight must be > 0")
	}
	selector, err := selectionFromName(*selectionName)
	if err != nil {
		return err
	}
	postprocessor, err := postprocessorFromName(*postprocessorName)
	if err != nil {
		return err
	}
	topoPolicy, err := topologicalPolicyFromConfig(*topoPolicyName, *topoCount, *topoParam, *topoMax)
	if err != nil {
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

	_, inIDs, outIDs, err := seedPopulationForScape(*scapeName, *population, *seed)
	if err != nil {
		return err
	}
	if err := morphology.EnsureScapeCompatibility(*scapeName); err != nil {
		return err
	}

	eliteCount := *population / 5
	if eliteCount < 1 {
		eliteCount = 1
	}

	runEvolution := func(useTuning bool) (platform.EvolutionResult, error) {
		mutation := &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(*seed + 1000)), MaxDelta: 1.0}
		policy := defaultMutationPolicy(*seed, inIDs, outIDs, *wPerturb, *wAddSynapse, *wRemoveSynapse, *wAddNeuron, *wRemoveNeuron)
		var tuner tuning.Tuner
		if useTuning {
			tuner = &tuning.Exoself{
				Rand:     rand.New(rand.NewSource(*seed + 2000)),
				Steps:    *tuneSteps,
				StepSize: *tuneStepSize,
			}
		}
		initialRun, _, _, err := seedPopulationForScape(*scapeName, *population, *seed)
		if err != nil {
			return platform.EvolutionResult{}, err
		}
		return polis.RunEvolution(ctx, platform.EvolutionConfig{
			ScapeName:       *scapeName,
			PopulationSize:  *population,
			Generations:     *generations,
			EliteCount:      eliteCount,
			Workers:         *workers,
			Seed:            *seed,
			InputNeuronIDs:  inIDs,
			OutputNeuronIDs: outIDs,
			Mutation:        mutation,
			MutationPolicy:  policy,
			Selector:        selector,
			Postprocessor:   postprocessor,
			TopologicalMutations: topoPolicy,
			Tuner:           tuner,
			TuneAttempts:    *tuneAttempts,
			Initial:         initialRun,
		})
	}

	var result platform.EvolutionResult
	var compareReport *stats.TuningComparison
	if *compareTuning {
		withoutTuning, err := runEvolution(false)
		if err != nil {
			return err
		}
		withTuning, err := runEvolution(true)
		if err != nil {
			return err
		}
		compareReport = &stats.TuningComparison{
			Scape:             *scapeName,
			PopulationSize:    *population,
			Generations:       *generations,
			Seed:              *seed,
			WithoutTuningBest: withoutTuning.BestByGeneration,
			WithTuningBest:    withTuning.BestByGeneration,
			WithoutFinalBest:  withoutTuning.BestFinalFitness,
			WithFinalBest:     withTuning.BestFinalFitness,
			FinalImprovement:  withTuning.BestFinalFitness - withoutTuning.BestFinalFitness,
		}
		if *enableTuning {
			result = withTuning
		} else {
			result = withoutTuning
		}
	} else {
		result, err = runEvolution(*enableTuning)
		if err != nil {
			return err
		}
	}

	now := time.Now().UTC()
	runID := fmt.Sprintf("%s-%d-%d", *scapeName, *seed, now.Unix())
	top := make([]stats.TopGenome, 0, len(result.TopFinal))
	for i, scored := range result.TopFinal {
		top = append(top, stats.TopGenome{Rank: i + 1, Fitness: scored.Fitness, Genome: scored.Genome})
	}
	lineage := make([]stats.LineageEntry, 0, len(result.Lineage))
	for _, record := range result.Lineage {
		lineage = append(lineage, stats.LineageEntry{
			GenomeID:   record.GenomeID,
			ParentID:   record.ParentID,
			Generation: record.Generation,
			Operation:  record.Operation,
		})
	}

	runDir, err := stats.WriteRunArtifacts(benchmarksDir, stats.RunArtifacts{
		Config: stats.RunConfig{
			RunID:                runID,
			Scape:                *scapeName,
			PopulationSize:       *population,
			Generations:          *generations,
			Seed:                 *seed,
			Workers:              *workers,
			EliteCount:           eliteCount,
			Selection:            *selectionName,
			FitnessPostprocessor: *postprocessorName,
			TopologicalPolicy:    *topoPolicyName,
			TopologicalCount:     *topoCount,
			TopologicalParam:     *topoParam,
			TopologicalMax:       *topoMax,
			TuningEnabled:        *enableTuning,
			TuneAttempts:         *tuneAttempts,
			TuneSteps:            *tuneSteps,
			TuneStepSize:         *tuneStepSize,
			WeightPerturb:        *wPerturb,
			WeightAddSynapse:     *wAddSynapse,
			WeightRemoveSynapse:  *wRemoveSynapse,
			WeightAddNeuron:      *wAddNeuron,
			WeightRemoveNeuron:   *wRemoveNeuron,
		},
		BestByGeneration: result.BestByGeneration,
		FinalBestFitness: result.BestFinalFitness,
		TopGenomes:       top,
		Lineage:          lineage,
	})
	if err != nil {
		return err
	}

	if err := stats.AppendRunIndex(benchmarksDir, stats.RunIndexEntry{
		RunID:            runID,
		Scape:            *scapeName,
		PopulationSize:   *population,
		Generations:      *generations,
		Seed:             *seed,
		Workers:          *workers,
		EliteCount:       eliteCount,
		TuningEnabled:    *enableTuning,
		FinalBestFitness: result.BestFinalFitness,
		CreatedAtUTC:     now.Format(time.RFC3339),
	}); err != nil {
		return err
	}
	if compareReport != nil {
		if err := stats.WriteTuningComparison(runDir, *compareReport); err != nil {
			return err
		}
	}

	fmt.Printf("run completed run_id=%s scape=%s pop=%d gens=%d seed=%d\n", runID, *scapeName, *population, *generations, *seed)
	for i, best := range result.BestByGeneration {
		fmt.Printf("generation=%d best_fitness=%.6f\n", i+1, best)
	}
	fmt.Printf("final_best_fitness=%.6f\n", result.BestFinalFitness)
	if compareReport != nil {
		fmt.Printf("compare_tuning without_final=%.6f with_final=%.6f improvement=%.6f\n",
			compareReport.WithoutFinalBest,
			compareReport.WithFinalBest,
			compareReport.FinalImprovement,
		)
	}
	fmt.Printf("artifacts_dir=%s\n", filepath.Clean(runDir))
	return nil
}

func runRuns(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("runs", flag.ContinueOnError)
	limit := fs.Int("limit", 20, "max runs to list")
	showCompare := fs.Bool("show-compare", false, "show compare-tuning improvement when available")
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

func runBenchmark(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	scapeName := fs.String("scape", "xor", "scape name")
	population := fs.Int("pop", 50, "population size")
	generations := fs.Int("gens", 100, "generation count")
	seed := fs.Int64("seed", 1, "rng seed")
	workers := fs.Int("workers", 4, "worker count")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	enableTuning := fs.Bool("tuning", false, "enable exoself tuning")
	selectionName := fs.String("selection", "elite", "parent selection strategy: elite|tournament|species_tournament")
	postprocessorName := fs.String("fitness-postprocessor", "none", "fitness postprocessor: none|size_proportional|novelty_proportional")
	topoPolicyName := fs.String("topo-policy", "const", "topological mutation count policy: const|ncount_linear|ncount_exponential")
	topoCount := fs.Int("topo-count", 1, "mutation count for topo-policy=const")
	topoParam := fs.Float64("topo-param", 0.5, "policy parameter (multiplier/power) for topo-policy")
	topoMax := fs.Int("topo-max", 8, "maximum mutation count for non-const topo policies (<=0 disables cap)")
	tuneAttempts := fs.Int("attempts", 4, "tuning attempts per agent evaluation")
	tuneSteps := fs.Int("tune-steps", 6, "tuning perturbation steps per attempt")
	tuneStepSize := fs.Float64("tune-step-size", 0.35, "tuning perturbation magnitude")
	wPerturb := fs.Float64("w-perturb", 0.70, "weight for perturb_random_weight mutation")
	wAddSynapse := fs.Float64("w-add-synapse", 0.10, "weight for add_random_synapse mutation")
	wRemoveSynapse := fs.Float64("w-remove-synapse", 0.08, "weight for remove_random_synapse mutation")
	wAddNeuron := fs.Float64("w-add-neuron", 0.07, "weight for add_random_neuron mutation")
	wRemoveNeuron := fs.Float64("w-remove-neuron", 0.05, "weight for remove_random_neuron mutation")
	minImprovement := fs.Float64("min-improvement", 0.001, "minimum expected fitness improvement")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *wPerturb < 0 || *wAddSynapse < 0 || *wRemoveSynapse < 0 || *wAddNeuron < 0 || *wRemoveNeuron < 0 {
		return errors.New("mutation weights must be >= 0")
	}
	if *wPerturb+*wAddSynapse+*wRemoveSynapse+*wAddNeuron+*wRemoveNeuron <= 0 {
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
	if err := morphology.EnsureScapeCompatibility(*scapeName); err != nil {
		return err
	}

	runSummary, err := client.Run(ctx, protoapi.RunRequest{
		Scape:                *scapeName,
		Population:           *population,
		Generations:          *generations,
		Seed:                 *seed,
		Workers:              *workers,
		Selection:            *selectionName,
		FitnessPostprocessor: *postprocessorName,
		TopologicalPolicy:    *topoPolicyName,
		TopologicalCount:     *topoCount,
		TopologicalParam:     *topoParam,
		TopologicalMax:       *topoMax,
		EnableTuning:         *enableTuning,
		TuneAttempts:         *tuneAttempts,
		TuneSteps:            *tuneSteps,
		TuneStepSize:         *tuneStepSize,
		WeightPerturb:        *wPerturb,
		WeightAddSynapse:     *wAddSynapse,
		WeightRemoveSynapse:  *wRemoveSynapse,
		WeightAddNeuron:      *wAddNeuron,
		WeightRemoveNeuron:   *wRemoveNeuron,
	})
	if err != nil {
		return err
	}
	if len(runSummary.BestByGeneration) == 0 {
		return errors.New("benchmark run produced empty fitness history")
	}

	initialBest := runSummary.BestByGeneration[0]
	improvement := runSummary.FinalBestFitness - initialBest
	passed := improvement >= *minImprovement
	report := stats.BenchmarkSummary{
		RunID:          runSummary.RunID,
		Scape:          *scapeName,
		PopulationSize: *population,
		Generations:    *generations,
		Seed:           *seed,
		InitialBest:    initialBest,
		FinalBest:      runSummary.FinalBestFitness,
		Improvement:    improvement,
		MinImprovement: *minImprovement,
		Passed:         passed,
	}
	if err := stats.WriteBenchmarkSummary(runSummary.ArtifactsDir, report); err != nil {
		return err
	}

	fmt.Printf("benchmark run_id=%s scape=%s initial_best=%.6f final_best=%.6f improvement=%.6f threshold=%.6f passed=%t\n",
		runSummary.RunID,
		*scapeName,
		initialBest,
		runSummary.FinalBestFitness,
		improvement,
		*minImprovement,
		passed,
	)
	fmt.Printf("benchmark_summary=%s\n", filepath.Join(runSummary.ArtifactsDir, "benchmark_summary.json"))
	return nil
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

func seedPopulationForScape(scapeName string, size int, seed int64) ([]model.Genome, []string, []string, error) {
	switch scapeName {
	case "xor":
		return seedXORPopulation(size, seed), []string{"i1", "i2"}, []string{"o"}, nil
	case "regression-mimic":
		return seedRegressionMimicPopulation(size, seed), []string{"i"}, []string{"o"}, nil
	default:
		return nil, nil, nil, fmt.Errorf("unsupported scape: %s", scapeName)
	}
}

func seedXORPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("xor-g0-%d", i),
			SensorIDs:       []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
			ActuatorIDs:     []string{protoio.XOROutputActuatorName},
			Neurons: []model.Neuron{
				{ID: "i1", Activation: "identity", Bias: 0},
				{ID: "i2", Activation: "identity", Bias: 0},
				{ID: "h1", Activation: "sigmoid", Bias: jitter(rng, 2)},
				{ID: "h2", Activation: "sigmoid", Bias: jitter(rng, 2)},
				{ID: "o", Activation: "sigmoid", Bias: jitter(rng, 2)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "i1", To: "h1", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s2", From: "i2", To: "h1", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s3", From: "i1", To: "h2", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s4", From: "i2", To: "h2", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s5", From: "h1", To: "o", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s6", From: "h2", To: "o", Weight: jitter(rng, 6), Enabled: true},
			},
		})
	}
	return population
}

func jitter(rng *rand.Rand, amplitude float64) float64 {
	return (rng.Float64()*2 - 1) * amplitude
}

func seedRegressionMimicPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("reg-g0-%d", i),
			SensorIDs:       []string{protoio.ScalarInputSensorName},
			ActuatorIDs:     []string{protoio.ScalarOutputActuatorName},
			Neurons: []model.Neuron{
				{ID: "i", Activation: "identity", Bias: 0},
				{ID: "o", Activation: "identity", Bias: jitter(rng, 1)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "i", To: "o", Weight: jitter(rng, 2), Enabled: true},
			},
		})
	}
	return population
}

func registerDefaultScapes(p *platform.Polis) error {
	if err := p.RegisterScape(scape.XORScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.RegressionMimicScape{}); err != nil {
		return err
	}
	return nil
}

func defaultMutationPolicy(
	seed int64,
	inputNeuronIDs, outputNeuronIDs []string,
	wPerturb, wAddSynapse, wRemoveSynapse, wAddNeuron, wRemoveNeuron float64,
) []evo.WeightedMutation {
	protected := make(map[string]struct{}, len(inputNeuronIDs)+len(outputNeuronIDs))
	for _, id := range inputNeuronIDs {
		protected[id] = struct{}{}
	}
	for _, id := range outputNeuronIDs {
		protected[id] = struct{}{}
	}

	return []evo.WeightedMutation{
		{Operator: &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(seed + 1000)), MaxDelta: 1.0}, Weight: wPerturb},
		{Operator: &evo.AddRandomSynapse{Rand: rand.New(rand.NewSource(seed + 1001)), MaxAbsWeight: 1.0}, Weight: wAddSynapse},
		{Operator: &evo.RemoveRandomSynapse{Rand: rand.New(rand.NewSource(seed + 1002))}, Weight: wRemoveSynapse},
		{Operator: &evo.AddRandomNeuron{Rand: rand.New(rand.NewSource(seed + 1003))}, Weight: wAddNeuron},
		{Operator: &evo.RemoveRandomNeuron{Rand: rand.New(rand.NewSource(seed + 1004)), Protected: protected}, Weight: wRemoveNeuron},
	}
}

func usageError(msg string) error {
	return fmt.Errorf("%s\nusage: protogonosctl <init|start|run|benchmark|runs|export> [flags]", msg)
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
	default:
		return nil, fmt.Errorf("unsupported selection strategy: %s", name)
	}
}

func postprocessorFromName(name string) (evo.FitnessPostprocessor, error) {
	switch name {
	case "none":
		return evo.NoopFitnessPostprocessor{}, nil
	case "size_proportional":
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
