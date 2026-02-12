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
	"protogonos/internal/genotype"
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
	profileName := fs.String("profile", "", "optional parity profile id (from testdata/fixtures/parity/ref_benchmarker_profiles.json)")
	selectionName := fs.String("selection", "elite", "parent selection strategy: elite|tournament|species_tournament|species_shared_tournament")
	postprocessorName := fs.String("fitness-postprocessor", "none", "fitness postprocessor: none|size_proportional|novelty_proportional")
	topoPolicyName := fs.String("topo-policy", "const", "topological mutation count policy: const|ncount_linear|ncount_exponential")
	topoCount := fs.Int("topo-count", 1, "mutation count for topo-policy=const")
	topoParam := fs.Float64("topo-param", 0.5, "policy parameter (multiplier/power) for topo-policy")
	topoMax := fs.Int("topo-max", 8, "maximum mutation count for non-const topo policies (<=0 disables cap)")
	tuneAttempts := fs.Int("attempts", 4, "tuning attempts per agent evaluation")
	tuneSteps := fs.Int("tune-steps", 6, "tuning perturbation steps per attempt")
	tuneStepSize := fs.Float64("tune-step-size", 0.35, "tuning perturbation magnitude")
	tuneSelection := fs.String("tune-selection", tuning.CandidateSelectBestSoFar, "tuner candidate selection: best_so_far|original|dynamic_random")
	tuneDurationPolicy := fs.String("tune-duration-policy", "fixed", "tuning attempt policy: fixed|linear_decay|topology_scaled")
	tuneDurationParam := fs.Float64("tune-duration-param", 1.0, "tuning attempt policy parameter")
	wPerturb := fs.Float64("w-perturb", 0.70, "weight for perturb_random_weight mutation")
	wAddSynapse := fs.Float64("w-add-synapse", 0.10, "weight for add_random_synapse mutation")
	wRemoveSynapse := fs.Float64("w-remove-synapse", 0.08, "weight for remove_random_synapse mutation")
	wAddNeuron := fs.Float64("w-add-neuron", 0.07, "weight for add_random_neuron mutation")
	wRemoveNeuron := fs.Float64("w-remove-neuron", 0.05, "weight for remove_random_neuron mutation")
	wPlasticity := fs.Float64("w-plasticity", 0.03, "weight for perturb_plasticity_rate mutation")
	wSubstrate := fs.Float64("w-substrate", 0.02, "weight for perturb_substrate_parameter mutation")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *profileName != "" {
		preset, err := loadParityPreset(*profileName)
		if err != nil {
			return err
		}
		*selectionName = preset.Selection
		*tuneSelection = preset.TuneSelection
		*wPerturb = preset.WeightPerturb
		*wAddSynapse = preset.WeightAddSyn
		*wRemoveSynapse = preset.WeightRemoveSyn
		*wAddNeuron = preset.WeightAddNeuro
		*wRemoveNeuron = preset.WeightRemoveNeuro
		*wPlasticity = preset.WeightPlasticity
		*wSubstrate = preset.WeightSubstrate
	}
	*tuneSelection = normalizeTuneSelection(*tuneSelection)
	if *wPerturb < 0 || *wAddSynapse < 0 || *wRemoveSynapse < 0 || *wAddNeuron < 0 || *wRemoveNeuron < 0 || *wPlasticity < 0 || *wSubstrate < 0 {
		return errors.New("mutation weights must be >= 0")
	}
	if *wPerturb+*wAddSynapse+*wRemoveSynapse+*wAddNeuron+*wRemoveNeuron+*wPlasticity+*wSubstrate <= 0 {
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

	seedPopulation, err := genotype.ConstructSeedPopulation(*scapeName, *population, *seed)
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
	now := time.Now().UTC()
	runID := fmt.Sprintf("%s-%d-%d", *scapeName, *seed, now.Unix())

	runEvolution := func(useTuning bool) (platform.EvolutionResult, error) {
		mutation := &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(*seed + 1000)), MaxDelta: 1.0}
		policy := defaultMutationPolicy(
			*seed,
			seedPopulation.InputNeuronIDs,
			seedPopulation.OutputNeuronIDs,
			*wPerturb,
			*wAddSynapse,
			*wRemoveSynapse,
			*wAddNeuron,
			*wRemoveNeuron,
			*wPlasticity,
			*wSubstrate,
		)
		var tuner tuning.Tuner
		var attemptPolicy tuning.AttemptPolicy
		if useTuning {
			var err error
			attemptPolicy, err = tuning.AttemptPolicyFromConfig(*tuneDurationPolicy, *tuneDurationParam)
			if err != nil {
				return platform.EvolutionResult{}, err
			}
			tuner = &tuning.Exoself{
				Rand:               rand.New(rand.NewSource(*seed + 2000)),
				Steps:              *tuneSteps,
				StepSize:           *tuneStepSize,
				CandidateSelection: *tuneSelection,
			}
		}
		return polis.RunEvolution(ctx, platform.EvolutionConfig{
			RunID:                runID,
			ScapeName:            *scapeName,
			PopulationSize:       *population,
			Generations:          *generations,
			EliteCount:           eliteCount,
			Workers:              *workers,
			Seed:                 *seed,
			InputNeuronIDs:       seedPopulation.InputNeuronIDs,
			OutputNeuronIDs:      seedPopulation.OutputNeuronIDs,
			Mutation:             mutation,
			MutationPolicy:       policy,
			Selector:             selector,
			Postprocessor:        postprocessor,
			TopologicalMutations: topoPolicy,
			Tuner:                tuner,
			TuneAttempts:         *tuneAttempts,
			TuneAttemptPolicy:    attemptPolicy,
			Initial:              seedPopulation.Genomes,
		})
	}

	var result platform.EvolutionResult
	var compareReport *stats.TuningComparison
	if *compareTuning {
		if *enableTuning {
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
			result = withTuning
		} else {
			withTuning, err := runEvolution(true)
			if err != nil {
				return err
			}
			withoutTuning, err := runEvolution(false)
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
			result = withoutTuning
		}
	} else {
		result, err = runEvolution(*enableTuning)
		if err != nil {
			return err
		}
	}

	top := make([]stats.TopGenome, 0, len(result.TopFinal))
	for i, scored := range result.TopFinal {
		top = append(top, stats.TopGenome{Rank: i + 1, Fitness: scored.Fitness, Genome: scored.Genome})
	}
	lineage := make([]stats.LineageEntry, 0, len(result.Lineage))
	for _, record := range result.Lineage {
		lineage = append(lineage, stats.LineageEntry{
			GenomeID:    record.GenomeID,
			ParentID:    record.ParentID,
			Generation:  record.Generation,
			Operation:   record.Operation,
			Fingerprint: record.Fingerprint,
			Summary: map[string]any{
				"total_neurons":            record.Summary.TotalNeurons,
				"total_synapses":           record.Summary.TotalSynapses,
				"total_recurrent_synapses": record.Summary.TotalRecurrentSynapses,
				"total_sensors":            record.Summary.TotalSensors,
				"total_actuators":          record.Summary.TotalActuators,
				"activation_distribution":  record.Summary.ActivationDistribution,
				"aggregator_distribution":  record.Summary.AggregatorDistribution,
			},
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
			TuneSelection:        *tuneSelection,
			TuneDurationPolicy:   *tuneDurationPolicy,
			TuneDurationParam:    *tuneDurationParam,
			TuneAttempts:         *tuneAttempts,
			TuneSteps:            *tuneSteps,
			TuneStepSize:         *tuneStepSize,
			WeightPerturb:        *wPerturb,
			WeightAddSynapse:     *wAddSynapse,
			WeightRemoveSynapse:  *wRemoveSynapse,
			WeightAddNeuron:      *wAddNeuron,
			WeightRemoveNeuron:   *wRemoveNeuron,
			WeightPlasticity:     *wPlasticity,
			WeightSubstrate:      *wSubstrate,
		},
		BestByGeneration:      result.BestByGeneration,
		GenerationDiagnostics: result.GenerationDiagnostics,
		SpeciesHistory:        result.SpeciesHistory,
		FinalBestFitness:      result.BestFinalFitness,
		TopGenomes:            top,
		Lineage:               lineage,
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
		CreatedAtUTC:     now.Format(time.RFC3339Nano),
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

func runLineage(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("lineage", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "show lineage for the most recent run from run index")
	limit := fs.Int("limit", 50, "max lineage rows to print (<=0 for all)")
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

	for _, d := range diagnostics {
		fmt.Printf("generation=%d best=%.6f mean=%.6f min=%.6f species=%d fingerprints=%d threshold=%.4f target_species=%d mean_species_size=%.2f largest_species=%d\n",
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
		)
	}
	return nil
}

func runTop(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("top", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id")
	latest := fs.Bool("latest", false, "show top genomes for the most recent run from run index")
	limit := fs.Int("limit", 5, "max top genomes to print (<=0 for all)")
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
	scapeName := fs.String("scape", "xor", "scape name")
	population := fs.Int("pop", 50, "population size")
	generations := fs.Int("gens", 100, "generation count")
	seed := fs.Int64("seed", 1, "rng seed")
	workers := fs.Int("workers", 4, "worker count")
	storeKind := fs.String("store", storage.DefaultStoreKind(), "store backend: memory|sqlite")
	dbPath := fs.String("db-path", "protogonos.db", "sqlite database path")
	enableTuning := fs.Bool("tuning", false, "enable exoself tuning")
	profileName := fs.String("profile", "", "optional parity profile id (from testdata/fixtures/parity/ref_benchmarker_profiles.json)")
	selectionName := fs.String("selection", "elite", "parent selection strategy: elite|tournament|species_tournament|species_shared_tournament")
	postprocessorName := fs.String("fitness-postprocessor", "none", "fitness postprocessor: none|size_proportional|novelty_proportional")
	topoPolicyName := fs.String("topo-policy", "const", "topological mutation count policy: const|ncount_linear|ncount_exponential")
	topoCount := fs.Int("topo-count", 1, "mutation count for topo-policy=const")
	topoParam := fs.Float64("topo-param", 0.5, "policy parameter (multiplier/power) for topo-policy")
	topoMax := fs.Int("topo-max", 8, "maximum mutation count for non-const topo policies (<=0 disables cap)")
	tuneAttempts := fs.Int("attempts", 4, "tuning attempts per agent evaluation")
	tuneSteps := fs.Int("tune-steps", 6, "tuning perturbation steps per attempt")
	tuneStepSize := fs.Float64("tune-step-size", 0.35, "tuning perturbation magnitude")
	tuneSelection := fs.String("tune-selection", tuning.CandidateSelectBestSoFar, "tuner candidate selection: best_so_far|original|dynamic_random")
	tuneDurationPolicy := fs.String("tune-duration-policy", "fixed", "tuning attempt policy: fixed|linear_decay|topology_scaled")
	tuneDurationParam := fs.Float64("tune-duration-param", 1.0, "tuning attempt policy parameter")
	wPerturb := fs.Float64("w-perturb", 0.70, "weight for perturb_random_weight mutation")
	wAddSynapse := fs.Float64("w-add-synapse", 0.10, "weight for add_random_synapse mutation")
	wRemoveSynapse := fs.Float64("w-remove-synapse", 0.08, "weight for remove_random_synapse mutation")
	wAddNeuron := fs.Float64("w-add-neuron", 0.07, "weight for add_random_neuron mutation")
	wRemoveNeuron := fs.Float64("w-remove-neuron", 0.05, "weight for remove_random_neuron mutation")
	wPlasticity := fs.Float64("w-plasticity", 0.03, "weight for perturb_plasticity_rate mutation")
	wSubstrate := fs.Float64("w-substrate", 0.02, "weight for perturb_substrate_parameter mutation")
	minImprovement := fs.Float64("min-improvement", 0.001, "minimum expected fitness improvement")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *profileName != "" {
		preset, err := loadParityPreset(*profileName)
		if err != nil {
			return err
		}
		*selectionName = preset.Selection
		*tuneSelection = preset.TuneSelection
		*wPerturb = preset.WeightPerturb
		*wAddSynapse = preset.WeightAddSyn
		*wRemoveSynapse = preset.WeightRemoveSyn
		*wAddNeuron = preset.WeightAddNeuro
		*wRemoveNeuron = preset.WeightRemoveNeuro
		*wPlasticity = preset.WeightPlasticity
		*wSubstrate = preset.WeightSubstrate
	}
	*tuneSelection = normalizeTuneSelection(*tuneSelection)
	if *wPerturb < 0 || *wAddSynapse < 0 || *wRemoveSynapse < 0 || *wAddNeuron < 0 || *wRemoveNeuron < 0 || *wPlasticity < 0 || *wSubstrate < 0 {
		return errors.New("mutation weights must be >= 0")
	}
	if *wPerturb+*wAddSynapse+*wRemoveSynapse+*wAddNeuron+*wRemoveNeuron+*wPlasticity+*wSubstrate <= 0 {
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
		TuneSelection:        *tuneSelection,
		TuneDurationPolicy:   *tuneDurationPolicy,
		TuneDurationParam:    *tuneDurationParam,
		TuneAttempts:         *tuneAttempts,
		TuneSteps:            *tuneSteps,
		TuneStepSize:         *tuneStepSize,
		WeightPerturb:        *wPerturb,
		WeightAddSynapse:     *wAddSynapse,
		WeightRemoveSynapse:  *wRemoveSynapse,
		WeightAddNeuron:      *wAddNeuron,
		WeightRemoveNeuron:   *wRemoveNeuron,
		WeightPlasticity:     *wPlasticity,
		WeightSubstrate:      *wSubstrate,
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

func runProfile(_ context.Context, args []string) error {
	if len(args) == 0 {
		return errors.New("profile requires a subcommand: list")
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
			fmt.Printf("id=%s selection=%s tune_selection=%s mutation_ops=%d\n",
				profile.ID,
				profile.PopulationSelection,
				profile.TuningSelection,
				profile.MutationOperatorLen,
			)
		}
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
	if err := p.RegisterScape(scape.FlatlandScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.GTSAScape{}); err != nil {
		return err
	}
	if err := p.RegisterScape(scape.FXScape{}); err != nil {
		return err
	}
	return nil
}

func defaultMutationPolicy(
	seed int64,
	inputNeuronIDs, outputNeuronIDs []string,
	wPerturb, wAddSynapse, wRemoveSynapse, wAddNeuron, wRemoveNeuron, wPlasticity, wSubstrate float64,
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
		{Operator: &evo.PerturbPlasticityRate{Rand: rand.New(rand.NewSource(seed + 1005)), MaxDelta: 0.15}, Weight: wPlasticity},
		{Operator: &evo.PerturbSubstrateParameter{Rand: rand.New(rand.NewSource(seed + 1006)), MaxDelta: 0.15}, Weight: wSubstrate},
	}
}

func usageError(msg string) error {
	return fmt.Errorf("%s\nusage: protogonosctl <init|start|run|benchmark|profile|runs|lineage|fitness|diagnostics|species|top|scape-summary|export> [flags]", msg)
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
	default:
		return nil, fmt.Errorf("unsupported selection strategy: %s", name)
	}
}

func normalizeTuneSelection(name string) string {
	switch name {
	case "", tuning.CandidateSelectBestSoFar:
		return tuning.CandidateSelectBestSoFar
	case tuning.CandidateSelectOriginal:
		return tuning.CandidateSelectOriginal
	case "dynamic_random":
		return tuning.CandidateSelectBestSoFar
	default:
		return name
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
