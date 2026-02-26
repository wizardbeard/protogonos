package protogonos

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"protogonos/internal/evo"
	"protogonos/internal/genotype"
	"protogonos/internal/model"
	"protogonos/internal/morphology"
	"protogonos/internal/platform"
	"protogonos/internal/scape"
	"protogonos/internal/scapeid"
	"protogonos/internal/stats"
	"protogonos/internal/storage"
	"protogonos/internal/tuning"
)

const (
	defaultBenchmarksDir = "benchmarks"
	defaultExportsDir    = "exports"
	defaultDBPath        = "protogonos.db"
)

type Options struct {
	StoreKind     string
	DBPath        string
	BenchmarksDir string
	ExportsDir    string
}

type Client struct {
	store storage.Store
	polis *platform.Polis

	benchmarksDir string
	exportsDir    string
}

type RunRequest struct {
	RunID                   string
	ContinuePopulationID    string
	SpecieIdentifier        string
	OpMode                  string
	EvolutionType           string
	Scape                   string
	GTSACSVPath             string
	GTSATrainEnd            int
	GTSAValidationEnd       int
	GTSATestEnd             int
	FXCSVPath               string
	EpitopesCSVPath         string
	LLVMWorkflowJSONPath    string
	EpitopesGTStart         int
	EpitopesGTEnd           int
	EpitopesValidationStart int
	EpitopesValidationEnd   int
	EpitopesTestStart       int
	EpitopesTestEnd         int
	EpitopesBenchmarkStart  int
	EpitopesBenchmarkEnd    int
	FlatlandScannerProfile  string
	FlatlandScannerSpread   *float64
	FlatlandScannerOffset   *float64
	FlatlandLayoutRandomize *bool
	FlatlandLayoutVariants  *int
	FlatlandForceLayout     *int
	FlatlandBenchmarkTrials *int
	Population              int
	Generations             int
	SurvivalPercentage      float64
	SpecieSizeLimit         int
	FitnessGoal             float64
	EvaluationsLimit        int
	TraceStepSize           int
	StartPaused             bool
	AutoContinueAfter       time.Duration
	Seed                    int64
	Workers                 int
	Selection               string
	FitnessPostprocessor    string
	TopologicalPolicy       string
	TopologicalCount        int
	TopologicalParam        float64
	TopologicalMax          int
	EnableTuning            bool
	CompareTuning           bool
	ValidationProbe         bool
	TestProbe               bool
	TuneSelection           string
	TuneDurationPolicy      string
	TuneDurationParam       float64
	TuneAttempts            int
	TuneSteps               int
	TuneStepSize            float64
	TunePerturbationRange   float64
	TuneAnnealingFactor     float64
	TuneMinImprovement      float64
	WeightPerturb           float64
	WeightBias              float64
	WeightRemoveBias        float64
	WeightActivation        float64
	WeightAggregator        float64
	WeightAddSynapse        float64
	WeightRemoveSynapse     float64
	WeightAddNeuron         float64
	WeightRemoveNeuron      float64
	WeightPlasticityRule    float64
	WeightPlasticity        float64
	WeightSubstrate         float64
}

type CompareSummary struct {
	WithoutFinalBest float64
	WithFinalBest    float64
	FinalImprovement float64
}

type RunSummary struct {
	RunID            string
	ArtifactsDir     string
	BestByGeneration []float64
	FinalBestFitness float64
	Compare          *CompareSummary
}

type materializedRunConfig struct {
	Request           RunRequest
	Selector          evo.Selector
	Postprocessor     evo.FitnessPostprocessor
	TopologicalPolicy evo.TopologicalMutationPolicy
	TuneAttemptPolicy tuning.AttemptPolicy
	SpeciationMode    string
}

type RunsRequest struct {
	Limit       int
	ShowCompare bool
}

type RunItem struct {
	RunID              string
	CreatedAtUTC       string
	Scape              string
	Seed               int64
	Population         int
	Generations        int
	TuningEnabled      bool
	FinalBestFitness   float64
	CompareImprovement *float64
}

type ExportRequest struct {
	RunID  string
	Latest bool
	OutDir string
}

type ExportSummary struct {
	RunID     string
	Directory string
}

type LineageRequest struct {
	RunID  string
	Latest bool
	Limit  int
}

type LineageItem struct {
	GenomeID    string
	ParentID    string
	Generation  int
	Operation   string
	Events      []model.EvoHistoryEvent
	Fingerprint string
	Summary     model.LineageSummary
}

type FitnessHistoryRequest struct {
	RunID  string
	Latest bool
	Limit  int
}

type DiagnosticsRequest struct {
	RunID  string
	Latest bool
	Limit  int
}

type SpeciesHistoryRequest struct {
	RunID  string
	Latest bool
	Limit  int
}

type SpeciesDiffRequest struct {
	RunID          string
	Latest         bool
	FromGeneration int
	ToGeneration   int
}

type SpeciesDelta struct {
	Key             string  `json:"key"`
	FromSize        int     `json:"from_size"`
	ToSize          int     `json:"to_size"`
	SizeDelta       int     `json:"size_delta"`
	FromMeanFitness float64 `json:"from_mean_fitness"`
	ToMeanFitness   float64 `json:"to_mean_fitness"`
	MeanDelta       float64 `json:"mean_delta"`
	FromBestFitness float64 `json:"from_best_fitness"`
	ToBestFitness   float64 `json:"to_best_fitness"`
	BestDelta       float64 `json:"best_delta"`
}

type SpeciesDiff struct {
	RunID                      string                      `json:"run_id"`
	FromGeneration             int                         `json:"from_generation"`
	ToGeneration               int                         `json:"to_generation"`
	Added                      []model.SpeciesMetrics      `json:"added"`
	Removed                    []model.SpeciesMetrics      `json:"removed"`
	Changed                    []SpeciesDelta              `json:"changed"`
	UnchangedCount             int                         `json:"unchanged_count"`
	FromDiagnostics            model.GenerationDiagnostics `json:"from_diagnostics"`
	ToDiagnostics              model.GenerationDiagnostics `json:"to_diagnostics"`
	TuningInvocationsDelta     int                         `json:"tuning_invocations_delta"`
	TuningAttemptsDelta        int                         `json:"tuning_attempts_delta"`
	TuningEvaluationsDelta     int                         `json:"tuning_evaluations_delta"`
	TuningAcceptedDelta        int                         `json:"tuning_accepted_delta"`
	TuningRejectedDelta        int                         `json:"tuning_rejected_delta"`
	TuningGoalHitsDelta        int                         `json:"tuning_goal_hits_delta"`
	TuningAcceptRateDelta      float64                     `json:"tuning_accept_rate_delta"`
	TuningEvalsPerAttemptDelta float64                     `json:"tuning_evals_per_attempt_delta"`
}

type TopGenomesRequest struct {
	RunID  string
	Latest bool
	Limit  int
}

type MonitorControlRequest struct {
	RunID string
}

type DeletePopulationRequest struct {
	PopulationID string
}

type ScapeSummaryItem struct {
	Name        string
	Description string
	BestFitness float64
}

func New(opts Options) (*Client, error) {
	storeKind := opts.StoreKind
	if storeKind == "" {
		storeKind = storage.DefaultStoreKind()
	}
	dbPath := opts.DBPath
	if dbPath == "" {
		dbPath = defaultDBPath
	}
	benchmarksDir := opts.BenchmarksDir
	if benchmarksDir == "" {
		benchmarksDir = defaultBenchmarksDir
	}
	exportsDir := opts.ExportsDir
	if exportsDir == "" {
		exportsDir = defaultExportsDir
	}

	store, err := storage.NewStore(storeKind, dbPath)
	if err != nil {
		return nil, err
	}

	return &Client{
		store:         store,
		benchmarksDir: benchmarksDir,
		exportsDir:    exportsDir,
	}, nil
}

func (c *Client) Close() error {
	if c.polis != nil {
		c.polis.Shutdown()
		c.polis = nil
	}
	return storage.CloseIfSupported(c.store)
}

func (c *Client) Init(ctx context.Context) error {
	_, err := c.ensurePolis(ctx)
	return err
}

func (c *Client) Start(ctx context.Context) error {
	p, err := c.ensurePolis(ctx)
	if err != nil {
		return err
	}
	return registerDefaultScapes(p)
}

func (c *Client) Run(ctx context.Context, req RunRequest) (RunSummary, error) {
	cfg, err := materializeRunConfigFromRequest(req)
	if err != nil {
		return RunSummary{}, err
	}
	req = cfg.Request
	runCtx, err := applyScapeDataSources(ctx, req)
	if err != nil {
		return RunSummary{}, err
	}

	p, err := c.ensurePolis(ctx)
	if err != nil {
		return RunSummary{}, err
	}
	if err := registerDefaultScapes(p); err != nil {
		return RunSummary{}, err
	}

	seedPopulation, err := genotype.ConstructSeedPopulation(req.Scape, req.Population, req.Seed)
	if err != nil {
		return RunSummary{}, err
	}
	initialPopulation := seedPopulation.Genomes
	initialGeneration := 0
	if req.ContinuePopulationID != "" {
		popSnapshot, continued, err := genotype.LoadPopulationSnapshot(ctx, c.store, req.ContinuePopulationID)
		if err != nil {
			return RunSummary{}, err
		}
		if len(continued) == 0 {
			return RunSummary{}, fmt.Errorf("continued population is empty: %s", req.ContinuePopulationID)
		}
		initialPopulation = continued
		req.Population = len(continued)
		initialGeneration = popSnapshot.Generation
	}
	if err := morphology.EnsureScapeCompatibility(req.Scape); err != nil {
		return RunSummary{}, err
	}
	if err := morphology.EnsurePopulationIOCompatibility(req.Scape, initialPopulation); err != nil {
		return RunSummary{}, err
	}

	eliteCount := req.Population / 5
	if eliteCount < 1 {
		eliteCount = 1
	}
	if req.SurvivalPercentage > 0 {
		eliteCount = 0
	}
	now := time.Now().UTC()
	runID := req.RunID
	if runID == "" && req.ContinuePopulationID != "" {
		runID = req.ContinuePopulationID
	}
	if runID == "" {
		runID = fmt.Sprintf("%s-%d-%d", req.Scape, req.Seed, now.Unix())
	}

	runEvolution := func(useTuning bool) (platform.EvolutionResult, error) {
		mutation := &evo.PerturbWeightsProportional{Rand: rand.New(rand.NewSource(req.Seed + 1000)), MaxDelta: 1.0}
		policy := defaultMutationPolicy(req.Seed, req.Scape, seedPopulation.InputNeuronIDs, seedPopulation.OutputNeuronIDs, req)
		var tuner tuning.Tuner
		var attemptPolicy tuning.AttemptPolicy
		if useTuning {
			attemptPolicy = cfg.TuneAttemptPolicy
			tuner = &tuning.Exoself{
				Rand:               rand.New(rand.NewSource(req.Seed + 2000)),
				Steps:              req.TuneSteps,
				StepSize:           req.TuneStepSize,
				PerturbationRange:  req.TunePerturbationRange,
				AnnealingFactor:    req.TuneAnnealingFactor,
				MinImprovement:     req.TuneMinImprovement,
				CandidateSelection: req.TuneSelection,
			}
		}
		var controlCh chan evo.MonitorCommand
		if req.StartPaused {
			controlCh = make(chan evo.MonitorCommand, 2)
			controlCh <- evo.CommandPause
			if req.AutoContinueAfter > 0 {
				go func() {
					timer := time.NewTimer(req.AutoContinueAfter)
					defer timer.Stop()
					select {
					case <-runCtx.Done():
						return
					case <-timer.C:
						select {
						case controlCh <- evo.CommandContinue:
						case <-runCtx.Done():
						}
					}
				}()
			}
		}
		return p.RunEvolution(runCtx, platform.EvolutionConfig{
			RunID:                runID,
			OpMode:               req.OpMode,
			EvolutionType:        req.EvolutionType,
			SpeciationMode:       cfg.SpeciationMode,
			ScapeName:            req.Scape,
			PopulationSize:       req.Population,
			Generations:          req.Generations,
			InitialGeneration:    initialGeneration,
			SurvivalPercentage:   req.SurvivalPercentage,
			SpecieSizeLimit:      req.SpecieSizeLimit,
			FitnessGoal:          req.FitnessGoal,
			EvaluationsLimit:     req.EvaluationsLimit,
			TraceStepSize:        req.TraceStepSize,
			Control:              controlCh,
			EliteCount:           eliteCount,
			Workers:              req.Workers,
			Seed:                 req.Seed,
			InputNeuronIDs:       seedPopulation.InputNeuronIDs,
			OutputNeuronIDs:      seedPopulation.OutputNeuronIDs,
			Mutation:             mutation,
			MutationPolicy:       policy,
			Selector:             cfg.Selector,
			Postprocessor:        cfg.Postprocessor,
			TopologicalMutations: cfg.TopologicalPolicy,
			Tuner:                tuner,
			TuneAttempts:         req.TuneAttempts,
			TuneAttemptPolicy:    attemptPolicy,
			ValidationProbe:      req.ValidationProbe,
			TestProbe:            req.TestProbe,
			Initial:              initialPopulation,
		})
	}

	var result platform.EvolutionResult
	var compareReport *stats.TuningComparison
	if req.CompareTuning {
		if req.EnableTuning {
			withoutTuning, err := runEvolution(false)
			if err != nil {
				return RunSummary{}, err
			}
			withTuning, err := runEvolution(true)
			if err != nil {
				return RunSummary{}, err
			}
			compareReport = &stats.TuningComparison{
				Scape:             req.Scape,
				PopulationSize:    req.Population,
				Generations:       req.Generations,
				Seed:              req.Seed,
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
				return RunSummary{}, err
			}
			withoutTuning, err := runEvolution(false)
			if err != nil {
				return RunSummary{}, err
			}
			compareReport = &stats.TuningComparison{
				Scape:             req.Scape,
				PopulationSize:    req.Population,
				Generations:       req.Generations,
				Seed:              req.Seed,
				WithoutTuningBest: withoutTuning.BestByGeneration,
				WithTuningBest:    withTuning.BestByGeneration,
				WithoutFinalBest:  withoutTuning.BestFinalFitness,
				WithFinalBest:     withTuning.BestFinalFitness,
				FinalImprovement:  withTuning.BestFinalFitness - withoutTuning.BestFinalFitness,
			}
			result = withoutTuning
		}
	} else {
		result, err = runEvolution(req.EnableTuning)
		if err != nil {
			return RunSummary{}, err
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
			Events:      toModelEvoHistoryEvents(record.Events),
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

	runDir, err := stats.WriteRunArtifacts(c.benchmarksDir, stats.RunArtifacts{
		Config: stats.RunConfig{
			RunID:                   runID,
			OpMode:                  req.OpMode,
			EvolutionType:           req.EvolutionType,
			Scape:                   req.Scape,
			GTSACSVPath:             req.GTSACSVPath,
			GTSATrainEnd:            req.GTSATrainEnd,
			GTSAValidationEnd:       req.GTSAValidationEnd,
			GTSATestEnd:             req.GTSATestEnd,
			FXCSVPath:               req.FXCSVPath,
			EpitopesCSVPath:         req.EpitopesCSVPath,
			LLVMWorkflowJSONPath:    req.LLVMWorkflowJSONPath,
			EpitopesGTStart:         req.EpitopesGTStart,
			EpitopesGTEnd:           req.EpitopesGTEnd,
			EpitopesValidationStart: req.EpitopesValidationStart,
			EpitopesValidationEnd:   req.EpitopesValidationEnd,
			EpitopesTestStart:       req.EpitopesTestStart,
			EpitopesTestEnd:         req.EpitopesTestEnd,
			EpitopesBenchmarkStart:  req.EpitopesBenchmarkStart,
			EpitopesBenchmarkEnd:    req.EpitopesBenchmarkEnd,
			FlatlandScannerProfile:  req.FlatlandScannerProfile,
			FlatlandScannerSpread:   cloneFloat64Ptr(req.FlatlandScannerSpread),
			FlatlandScannerOffset:   cloneFloat64Ptr(req.FlatlandScannerOffset),
			FlatlandLayoutRandomize: cloneBoolPtr(req.FlatlandLayoutRandomize),
			FlatlandLayoutVariants:  cloneIntPtr(req.FlatlandLayoutVariants),
			FlatlandForceLayout:     cloneIntPtr(req.FlatlandForceLayout),
			FlatlandBenchmarkTrials: cloneIntPtr(req.FlatlandBenchmarkTrials),
			ContinuePopulationID:    req.ContinuePopulationID,
			SpecieIdentifier:        req.SpecieIdentifier,
			InitialGeneration:       initialGeneration,
			PopulationSize:          req.Population,
			Generations:             req.Generations,
			SurvivalPercentage:      req.SurvivalPercentage,
			SpecieSizeLimit:         req.SpecieSizeLimit,
			FitnessGoal:             req.FitnessGoal,
			EvaluationsLimit:        req.EvaluationsLimit,
			TraceStepSize:           req.TraceStepSize,
			StartPaused:             req.StartPaused,
			AutoContinueAfterMS:     req.AutoContinueAfter.Milliseconds(),
			Seed:                    req.Seed,
			Workers:                 req.Workers,
			EliteCount:              eliteCount,
			Selection:               req.Selection,
			FitnessPostprocessor:    req.FitnessPostprocessor,
			TopologicalPolicy:       req.TopologicalPolicy,
			TopologicalCount:        req.TopologicalCount,
			TopologicalParam:        req.TopologicalParam,
			TopologicalMax:          req.TopologicalMax,
			TuningEnabled:           req.EnableTuning,
			ValidationProbe:         req.ValidationProbe,
			TestProbe:               req.TestProbe,
			TuneSelection:           req.TuneSelection,
			TuneDurationPolicy:      req.TuneDurationPolicy,
			TuneDurationParam:       req.TuneDurationParam,
			TuneAttempts:            req.TuneAttempts,
			TuneSteps:               req.TuneSteps,
			TuneStepSize:            req.TuneStepSize,
			TunePerturbationRange:   req.TunePerturbationRange,
			TuneAnnealingFactor:     req.TuneAnnealingFactor,
			TuneMinImprovement:      req.TuneMinImprovement,
			WeightPerturb:           req.WeightPerturb,
			WeightBias:              req.WeightBias,
			WeightRemoveBias:        req.WeightRemoveBias,
			WeightActivation:        req.WeightActivation,
			WeightAggregator:        req.WeightAggregator,
			WeightAddSynapse:        req.WeightAddSynapse,
			WeightRemoveSynapse:     req.WeightRemoveSynapse,
			WeightAddNeuron:         req.WeightAddNeuron,
			WeightRemoveNeuron:      req.WeightRemoveNeuron,
			WeightPlasticityRule:    req.WeightPlasticityRule,
			WeightPlasticity:        req.WeightPlasticity,
			WeightSubstrate:         req.WeightSubstrate,
		},
		BestByGeneration:      result.BestByGeneration,
		GenerationDiagnostics: result.GenerationDiagnostics,
		SpeciesHistory:        result.SpeciesHistory,
		FinalBestFitness:      result.BestFinalFitness,
		TopGenomes:            top,
		Lineage:               lineage,
	})
	if err != nil {
		return RunSummary{}, err
	}

	if err := stats.AppendRunIndex(c.benchmarksDir, stats.RunIndexEntry{
		RunID:            runID,
		Scape:            req.Scape,
		PopulationSize:   req.Population,
		Generations:      req.Generations,
		Seed:             req.Seed,
		Workers:          req.Workers,
		EliteCount:       eliteCount,
		TuningEnabled:    req.EnableTuning,
		FinalBestFitness: result.BestFinalFitness,
		CreatedAtUTC:     now.Format(time.RFC3339Nano),
	}); err != nil {
		return RunSummary{}, err
	}
	if compareReport != nil {
		if err := stats.WriteTuningComparison(runDir, *compareReport); err != nil {
			return RunSummary{}, err
		}
	}

	summary := RunSummary{
		RunID:            runID,
		ArtifactsDir:     filepath.Clean(runDir),
		BestByGeneration: append([]float64(nil), result.BestByGeneration...),
		FinalBestFitness: result.BestFinalFitness,
	}
	if compareReport != nil {
		summary.Compare = &CompareSummary{
			WithoutFinalBest: compareReport.WithoutFinalBest,
			WithFinalBest:    compareReport.WithFinalBest,
			FinalImprovement: compareReport.FinalImprovement,
		}
	}
	return summary, nil
}

func applyScapeDataSources(ctx context.Context, req RunRequest) (context.Context, error) {
	scopedCtx, err := scape.WithDataSources(ctx, scape.DataSources{
		GTSA: scape.GTSADataSource{
			CSVPath: req.GTSACSVPath,
			Bounds: scape.GTSATableBounds{
				TrainEnd:      req.GTSATrainEnd,
				ValidationEnd: req.GTSAValidationEnd,
				TestEnd:       req.GTSATestEnd,
			},
		},
		FX: scape.FXDataSource{
			CSVPath: req.FXCSVPath,
		},
		Epitopes: scape.EpitopesDataSource{
			CSVPath: req.EpitopesCSVPath,
			Bounds: scape.EpitopesTableBounds{
				GTStart:         req.EpitopesGTStart,
				GTEnd:           req.EpitopesGTEnd,
				ValidationStart: req.EpitopesValidationStart,
				ValidationEnd:   req.EpitopesValidationEnd,
				TestStart:       req.EpitopesTestStart,
				TestEnd:         req.EpitopesTestEnd,
				BenchmarkStart:  req.EpitopesBenchmarkStart,
				BenchmarkEnd:    req.EpitopesBenchmarkEnd,
			},
		},
		LLVM: scape.LLVMDataSource{
			WorkflowJSONPath: req.LLVMWorkflowJSONPath,
		},
	})
	if err != nil {
		return nil, err
	}
	if !hasFlatlandOverrideConfig(req) {
		return scopedCtx, nil
	}
	scopedCtx, err = scape.WithFlatlandOverrides(scopedCtx, toFlatlandOverrides(req))
	if err != nil {
		return nil, fmt.Errorf("configure flatland overrides: %w", err)
	}
	return scopedCtx, nil
}

func (c *Client) Runs(_ context.Context, req RunsRequest) ([]RunItem, error) {
	if req.Limit <= 0 {
		req.Limit = 20
	}

	entries, err := stats.ListRunIndex(c.benchmarksDir)
	if err != nil {
		return nil, err
	}
	if len(entries) > req.Limit {
		entries = entries[:req.Limit]
	}

	out := make([]RunItem, 0, len(entries))
	for _, e := range entries {
		item := RunItem{
			RunID:            e.RunID,
			CreatedAtUTC:     e.CreatedAtUTC,
			Scape:            e.Scape,
			Seed:             e.Seed,
			Population:       e.PopulationSize,
			Generations:      e.Generations,
			TuningEnabled:    e.TuningEnabled,
			FinalBestFitness: e.FinalBestFitness,
		}
		if req.ShowCompare {
			report, ok, err := stats.ReadTuningComparison(c.benchmarksDir, e.RunID)
			if err != nil {
				return nil, err
			}
			if ok {
				improvement := report.FinalImprovement
				item.CompareImprovement = &improvement
			}
		}
		out = append(out, item)
	}
	return out, nil
}

func (c *Client) Export(_ context.Context, req ExportRequest) (ExportSummary, error) {
	if req.RunID != "" && req.Latest {
		return ExportSummary{}, errors.New("use either run id or latest")
	}
	if req.RunID == "" && !req.Latest {
		return ExportSummary{}, errors.New("export requires run id or latest")
	}
	if req.OutDir == "" {
		req.OutDir = c.exportsDir
	}

	runID := req.RunID
	if req.Latest {
		entries, err := stats.ListRunIndex(c.benchmarksDir)
		if err != nil {
			return ExportSummary{}, err
		}
		if len(entries) == 0 {
			return ExportSummary{}, errors.New("no runs available to export")
		}
		runID = entries[0].RunID
	}

	exportedDir, err := stats.ExportRunArtifacts(c.benchmarksDir, runID, req.OutDir)
	if err != nil {
		return ExportSummary{}, err
	}
	return ExportSummary{RunID: runID, Directory: filepath.Clean(exportedDir)}, nil
}

func (c *Client) Lineage(ctx context.Context, req LineageRequest) ([]LineageItem, error) {
	if req.RunID != "" && req.Latest {
		return nil, errors.New("use either run id or latest")
	}
	if req.Limit < 0 {
		return nil, errors.New("limit must be >= 0")
	}

	runID := req.RunID
	if req.Latest {
		entries, err := stats.ListRunIndex(c.benchmarksDir)
		if err != nil {
			return nil, err
		}
		if len(entries) == 0 {
			return nil, errors.New("no runs available")
		}
		runID = entries[0].RunID
	}
	if runID == "" {
		return nil, errors.New("lineage requires run id or latest")
	}

	if _, err := c.ensurePolis(ctx); err != nil {
		return nil, err
	}
	lineage, ok, err := c.store.GetLineage(ctx, runID)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("lineage not found for run id: %s", runID)
	}

	if req.Limit > 0 && len(lineage) > req.Limit {
		lineage = lineage[:req.Limit]
	}

	out := make([]LineageItem, 0, len(lineage))
	for _, rec := range lineage {
		out = append(out, LineageItem{
			GenomeID:    rec.GenomeID,
			ParentID:    rec.ParentID,
			Generation:  rec.Generation,
			Operation:   rec.Operation,
			Events:      cloneModelEvoHistoryEvents(rec.Events),
			Fingerprint: rec.Fingerprint,
			Summary:     rec.Summary,
		})
	}
	return out, nil
}

func toModelEvoHistoryEvents(events []genotype.EvoHistoryEvent) []model.EvoHistoryEvent {
	if len(events) == 0 {
		return nil
	}
	out := make([]model.EvoHistoryEvent, 0, len(events))
	for _, event := range events {
		out = append(out, model.EvoHistoryEvent{
			Mutation: event.Mutation,
			IDs:      append([]string{}, event.IDs...),
		})
	}
	return out
}

func cloneModelEvoHistoryEvents(events []model.EvoHistoryEvent) []model.EvoHistoryEvent {
	if len(events) == 0 {
		return nil
	}
	out := make([]model.EvoHistoryEvent, 0, len(events))
	for _, event := range events {
		out = append(out, model.EvoHistoryEvent{
			Mutation: event.Mutation,
			IDs:      append([]string{}, event.IDs...),
		})
	}
	return out
}

func (c *Client) FitnessHistory(ctx context.Context, req FitnessHistoryRequest) ([]float64, error) {
	if req.RunID != "" && req.Latest {
		return nil, errors.New("use either run id or latest")
	}
	if req.Limit < 0 {
		return nil, errors.New("limit must be >= 0")
	}

	runID := req.RunID
	if req.Latest {
		entries, err := stats.ListRunIndex(c.benchmarksDir)
		if err != nil {
			return nil, err
		}
		if len(entries) == 0 {
			return nil, errors.New("no runs available")
		}
		runID = entries[0].RunID
	}
	if runID == "" {
		return nil, errors.New("fitness history requires run id or latest")
	}

	if _, err := c.ensurePolis(ctx); err != nil {
		return nil, err
	}
	history, ok, err := c.store.GetFitnessHistory(ctx, runID)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("fitness history not found for run id: %s", runID)
	}
	if req.Limit > 0 && len(history) > req.Limit {
		history = history[:req.Limit]
	}
	return append([]float64(nil), history...), nil
}

func (c *Client) Diagnostics(ctx context.Context, req DiagnosticsRequest) ([]model.GenerationDiagnostics, error) {
	if req.RunID != "" && req.Latest {
		return nil, errors.New("use either run id or latest")
	}
	if req.Limit < 0 {
		return nil, errors.New("limit must be >= 0")
	}

	runID := req.RunID
	if req.Latest {
		entries, err := stats.ListRunIndex(c.benchmarksDir)
		if err != nil {
			return nil, err
		}
		if len(entries) == 0 {
			return nil, errors.New("no runs available")
		}
		runID = entries[0].RunID
	}
	if runID == "" {
		return nil, errors.New("diagnostics requires run id or latest")
	}

	if _, err := c.ensurePolis(ctx); err != nil {
		return nil, err
	}
	diagnostics, ok, err := c.store.GetGenerationDiagnostics(ctx, runID)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("diagnostics not found for run id: %s", runID)
	}
	if req.Limit > 0 && len(diagnostics) > req.Limit {
		diagnostics = diagnostics[:req.Limit]
	}
	out := make([]model.GenerationDiagnostics, len(diagnostics))
	copy(out, diagnostics)
	return out, nil
}

func (c *Client) SpeciesHistory(ctx context.Context, req SpeciesHistoryRequest) ([]model.SpeciesGeneration, error) {
	if req.RunID != "" && req.Latest {
		return nil, errors.New("use either run id or latest")
	}
	if req.Limit < 0 {
		return nil, errors.New("limit must be >= 0")
	}

	runID := req.RunID
	if req.Latest {
		entries, err := stats.ListRunIndex(c.benchmarksDir)
		if err != nil {
			return nil, err
		}
		if len(entries) == 0 {
			return nil, errors.New("no runs available")
		}
		runID = entries[0].RunID
	}
	if runID == "" {
		return nil, errors.New("species history requires run id or latest")
	}

	if _, err := c.ensurePolis(ctx); err != nil {
		return nil, err
	}
	history, ok, err := c.store.GetSpeciesHistory(ctx, runID)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("species history not found for run id: %s", runID)
	}
	if req.Limit > 0 && len(history) > req.Limit {
		history = history[:req.Limit]
	}
	out := make([]model.SpeciesGeneration, 0, len(history))
	for _, generation := range history {
		species := make([]model.SpeciesMetrics, len(generation.Species))
		copy(species, generation.Species)
		out = append(out, model.SpeciesGeneration{
			Generation:     generation.Generation,
			Species:        species,
			NewSpecies:     append([]string(nil), generation.NewSpecies...),
			ExtinctSpecies: append([]string(nil), generation.ExtinctSpecies...),
		})
	}
	return out, nil
}

func (c *Client) SpeciesDiff(ctx context.Context, req SpeciesDiffRequest) (SpeciesDiff, error) {
	if req.RunID != "" && req.Latest {
		return SpeciesDiff{}, errors.New("use either run id or latest")
	}
	if (req.FromGeneration > 0 && req.ToGeneration <= 0) || (req.FromGeneration <= 0 && req.ToGeneration > 0) {
		return SpeciesDiff{}, errors.New("from and to generations must both be set")
	}
	if req.FromGeneration > 0 && req.ToGeneration > 0 && req.FromGeneration == req.ToGeneration {
		return SpeciesDiff{}, errors.New("from and to generations must differ")
	}

	runID := req.RunID
	if req.Latest {
		entries, err := stats.ListRunIndex(c.benchmarksDir)
		if err != nil {
			return SpeciesDiff{}, err
		}
		if len(entries) == 0 {
			return SpeciesDiff{}, errors.New("no runs available")
		}
		runID = entries[0].RunID
	}
	if runID == "" {
		return SpeciesDiff{}, errors.New("species diff requires run id or latest")
	}

	if _, err := c.ensurePolis(ctx); err != nil {
		return SpeciesDiff{}, err
	}
	history, ok, err := c.store.GetSpeciesHistory(ctx, runID)
	if err != nil {
		return SpeciesDiff{}, err
	}
	if !ok {
		return SpeciesDiff{}, fmt.Errorf("species history not found for run id: %s", runID)
	}
	if len(history) < 2 {
		return SpeciesDiff{}, fmt.Errorf("species history for run id %s has fewer than 2 generations", runID)
	}

	fromGen := req.FromGeneration
	toGen := req.ToGeneration
	if fromGen <= 0 && toGen <= 0 {
		fromGen = history[len(history)-2].Generation
		toGen = history[len(history)-1].Generation
	}

	byGeneration := make(map[int]model.SpeciesGeneration, len(history))
	for _, generation := range history {
		byGeneration[generation.Generation] = generation
	}
	fromHistory, ok := byGeneration[fromGen]
	if !ok {
		return SpeciesDiff{}, fmt.Errorf("species generation not found: %d", fromGen)
	}
	toHistory, ok := byGeneration[toGen]
	if !ok {
		return SpeciesDiff{}, fmt.Errorf("species generation not found: %d", toGen)
	}

	fromByKey := make(map[string]model.SpeciesMetrics, len(fromHistory.Species))
	for _, item := range fromHistory.Species {
		fromByKey[item.Key] = item
	}
	toByKey := make(map[string]model.SpeciesMetrics, len(toHistory.Species))
	for _, item := range toHistory.Species {
		toByKey[item.Key] = item
	}

	diff := SpeciesDiff{
		RunID:          runID,
		FromGeneration: fromGen,
		ToGeneration:   toGen,
	}
	if diagnostics, ok, err := c.store.GetGenerationDiagnostics(ctx, runID); err != nil {
		return SpeciesDiff{}, err
	} else if ok {
		diagByGen := make(map[int]model.GenerationDiagnostics, len(diagnostics))
		for _, d := range diagnostics {
			diagByGen[d.Generation] = d
		}
		diff.FromDiagnostics = diagByGen[fromGen]
		diff.ToDiagnostics = diagByGen[toGen]
		diff.TuningInvocationsDelta = diff.ToDiagnostics.TuningInvocations - diff.FromDiagnostics.TuningInvocations
		diff.TuningAttemptsDelta = diff.ToDiagnostics.TuningAttempts - diff.FromDiagnostics.TuningAttempts
		diff.TuningEvaluationsDelta = diff.ToDiagnostics.TuningEvaluations - diff.FromDiagnostics.TuningEvaluations
		diff.TuningAcceptedDelta = diff.ToDiagnostics.TuningAccepted - diff.FromDiagnostics.TuningAccepted
		diff.TuningRejectedDelta = diff.ToDiagnostics.TuningRejected - diff.FromDiagnostics.TuningRejected
		diff.TuningGoalHitsDelta = diff.ToDiagnostics.TuningGoalHits - diff.FromDiagnostics.TuningGoalHits
		diff.TuningAcceptRateDelta = diff.ToDiagnostics.TuningAcceptRate - diff.FromDiagnostics.TuningAcceptRate
		diff.TuningEvalsPerAttemptDelta = diff.ToDiagnostics.TuningEvalsPerAttempt - diff.FromDiagnostics.TuningEvalsPerAttempt
	}
	for key, from := range fromByKey {
		to, ok := toByKey[key]
		if !ok {
			diff.Removed = append(diff.Removed, from)
			continue
		}
		sizeChanged := from.Size != to.Size
		meanChanged := from.MeanFitness != to.MeanFitness
		bestChanged := from.BestFitness != to.BestFitness
		if sizeChanged || meanChanged || bestChanged {
			diff.Changed = append(diff.Changed, SpeciesDelta{
				Key:             key,
				FromSize:        from.Size,
				ToSize:          to.Size,
				SizeDelta:       to.Size - from.Size,
				FromMeanFitness: from.MeanFitness,
				ToMeanFitness:   to.MeanFitness,
				MeanDelta:       to.MeanFitness - from.MeanFitness,
				FromBestFitness: from.BestFitness,
				ToBestFitness:   to.BestFitness,
				BestDelta:       to.BestFitness - from.BestFitness,
			})
		} else {
			diff.UnchangedCount++
		}
	}
	for key, to := range toByKey {
		if _, ok := fromByKey[key]; !ok {
			diff.Added = append(diff.Added, to)
		}
	}

	sort.Slice(diff.Added, func(i, j int) bool { return diff.Added[i].Key < diff.Added[j].Key })
	sort.Slice(diff.Removed, func(i, j int) bool { return diff.Removed[i].Key < diff.Removed[j].Key })
	sort.Slice(diff.Changed, func(i, j int) bool { return diff.Changed[i].Key < diff.Changed[j].Key })
	return diff, nil
}

func (c *Client) TopGenomes(ctx context.Context, req TopGenomesRequest) ([]model.TopGenomeRecord, error) {
	if req.RunID != "" && req.Latest {
		return nil, errors.New("use either run id or latest")
	}
	if req.Limit < 0 {
		return nil, errors.New("limit must be >= 0")
	}

	runID := req.RunID
	if req.Latest {
		entries, err := stats.ListRunIndex(c.benchmarksDir)
		if err != nil {
			return nil, err
		}
		if len(entries) == 0 {
			return nil, errors.New("no runs available")
		}
		runID = entries[0].RunID
	}
	if runID == "" {
		return nil, errors.New("top genomes requires run id or latest")
	}

	if _, err := c.ensurePolis(ctx); err != nil {
		return nil, err
	}
	top, ok, err := c.store.GetTopGenomes(ctx, runID)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("top genomes not found for run id: %s", runID)
	}
	if req.Limit > 0 && len(top) > req.Limit {
		top = top[:req.Limit]
	}
	out := make([]model.TopGenomeRecord, len(top))
	copy(out, top)
	return out, nil
}

func (c *Client) ScapeSummary(ctx context.Context, scapeName string) (ScapeSummaryItem, error) {
	if strings.TrimSpace(scapeName) == "" {
		return ScapeSummaryItem{}, errors.New("scape name is required")
	}
	scapeName = scapeid.Normalize(scapeName)
	if _, err := c.ensurePolis(ctx); err != nil {
		return ScapeSummaryItem{}, err
	}
	summary, ok, err := c.store.GetScapeSummary(ctx, scapeName)
	if err != nil {
		return ScapeSummaryItem{}, err
	}
	if !ok {
		return ScapeSummaryItem{}, fmt.Errorf("scape summary not found: %s", scapeName)
	}
	return ScapeSummaryItem{
		Name:        summary.Name,
		Description: summary.Description,
		BestFitness: summary.BestFitness,
	}, nil
}

func (c *Client) PauseRun(ctx context.Context, req MonitorControlRequest) error {
	if req.RunID == "" {
		return errors.New("run id is required")
	}
	p, err := c.ensurePolis(ctx)
	if err != nil {
		return err
	}
	return p.PauseRun(req.RunID)
}

func (c *Client) ContinueRun(ctx context.Context, req MonitorControlRequest) error {
	if req.RunID == "" {
		return errors.New("run id is required")
	}
	p, err := c.ensurePolis(ctx)
	if err != nil {
		return err
	}
	return p.ContinueRun(req.RunID)
}

func (c *Client) StopRun(ctx context.Context, req MonitorControlRequest) error {
	if req.RunID == "" {
		return errors.New("run id is required")
	}
	p, err := c.ensurePolis(ctx)
	if err != nil {
		return err
	}
	return p.StopRun(req.RunID)
}

func (c *Client) GoalReachedRun(ctx context.Context, req MonitorControlRequest) error {
	if req.RunID == "" {
		return errors.New("run id is required")
	}
	p, err := c.ensurePolis(ctx)
	if err != nil {
		return err
	}
	return p.GoalReachedRun(req.RunID)
}

func (c *Client) PrintTraceRun(ctx context.Context, req MonitorControlRequest) error {
	if req.RunID == "" {
		return errors.New("run id is required")
	}
	p, err := c.ensurePolis(ctx)
	if err != nil {
		return err
	}
	return p.PrintTraceRun(req.RunID)
}

func (c *Client) DeletePopulation(ctx context.Context, req DeletePopulationRequest) error {
	if req.PopulationID == "" {
		return errors.New("population id is required")
	}
	if _, err := c.ensurePolis(ctx); err != nil {
		return err
	}
	return genotype.DeletePopulationSnapshot(ctx, c.store, req.PopulationID)
}

func (c *Client) ensurePolis(ctx context.Context) (*platform.Polis, error) {
	if c.polis != nil {
		return c.polis, nil
	}
	p := platform.NewPolis(platform.Config{Store: c.store})
	if err := p.Init(ctx); err != nil {
		return nil, err
	}
	c.polis = p
	return c.polis, nil
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

func materializeRunConfigFromRequest(req RunRequest) (materializedRunConfig, error) {
	if req.OpMode == "" {
		req.OpMode = evo.OpModeGT
	}
	parsedOpMode, impliedValidationProbe, impliedTestProbe, err := parseOpMode(req.OpMode)
	if err != nil {
		return materializedRunConfig{}, err
	}
	req.OpMode = parsedOpMode
	req.ValidationProbe = req.ValidationProbe || impliedValidationProbe
	req.TestProbe = req.TestProbe || impliedTestProbe
	if req.OpMode != evo.OpModeGT {
		req.EnableTuning = false
		req.CompareTuning = false
		req.ValidationProbe = false
		req.TestProbe = false
	}
	if req.EvolutionType == "" {
		req.EvolutionType = evo.EvolutionTypeGenerational
	}
	switch req.EvolutionType {
	case evo.EvolutionTypeGenerational, evo.EvolutionTypeSteadyState:
	default:
		return materializedRunConfig{}, errors.New("evolution type must be one of generational|steady_state")
	}
	if req.Scape == "" {
		req.Scape = "xor"
	}
	req.Scape = scapeid.Normalize(req.Scape)
	if req.GTSATrainEnd < 0 {
		return materializedRunConfig{}, errors.New("gtsa train end must be >= 0")
	}
	if req.GTSAValidationEnd < 0 {
		return materializedRunConfig{}, errors.New("gtsa validation end must be >= 0")
	}
	if req.GTSATestEnd < 0 {
		return materializedRunConfig{}, errors.New("gtsa test end must be >= 0")
	}
	if req.EpitopesGTStart < 0 {
		return materializedRunConfig{}, errors.New("epitopes gt start must be >= 0")
	}
	if req.EpitopesGTEnd < 0 {
		return materializedRunConfig{}, errors.New("epitopes gt end must be >= 0")
	}
	if req.EpitopesValidationStart < 0 {
		return materializedRunConfig{}, errors.New("epitopes validation start must be >= 0")
	}
	if req.EpitopesValidationEnd < 0 {
		return materializedRunConfig{}, errors.New("epitopes validation end must be >= 0")
	}
	if req.EpitopesTestStart < 0 {
		return materializedRunConfig{}, errors.New("epitopes test start must be >= 0")
	}
	if req.EpitopesTestEnd < 0 {
		return materializedRunConfig{}, errors.New("epitopes test end must be >= 0")
	}
	if req.EpitopesBenchmarkStart < 0 {
		return materializedRunConfig{}, errors.New("epitopes benchmark start must be >= 0")
	}
	if req.EpitopesBenchmarkEnd < 0 {
		return materializedRunConfig{}, errors.New("epitopes benchmark end must be >= 0")
	}
	if hasFlatlandOverrideConfig(req) {
		if _, err := scape.WithFlatlandOverrides(context.Background(), toFlatlandOverrides(req)); err != nil {
			return materializedRunConfig{}, err
		}
	}
	if req.Population < 0 {
		return materializedRunConfig{}, errors.New("population must be >= 0")
	}
	if req.Population == 0 {
		req.Population = 50
	}
	if req.Generations < 0 {
		return materializedRunConfig{}, errors.New("generations must be >= 0")
	}
	if req.Generations == 0 {
		req.Generations = 100
	}
	if req.SurvivalPercentage < 0 || req.SurvivalPercentage > 1 {
		return materializedRunConfig{}, errors.New("survival percentage must be in [0, 1]")
	}
	if req.FitnessGoal < 0 {
		return materializedRunConfig{}, errors.New("fitness goal must be >= 0")
	}
	if req.SpecieSizeLimit < 0 {
		return materializedRunConfig{}, errors.New("specie size limit must be >= 0")
	}
	if req.EvaluationsLimit < 0 {
		return materializedRunConfig{}, errors.New("evaluations limit must be >= 0")
	}
	if req.TraceStepSize < 0 {
		return materializedRunConfig{}, errors.New("trace step size must be >= 0")
	}
	if req.TraceStepSize == 0 {
		req.TraceStepSize = 500
	}
	if req.AutoContinueAfter < 0 {
		return materializedRunConfig{}, errors.New("auto continue after must be >= 0")
	}
	if req.Workers < 0 {
		return materializedRunConfig{}, errors.New("workers must be >= 0")
	}
	if req.Workers == 0 {
		req.Workers = 4
	}
	if req.Selection == "" {
		req.Selection = "elite"
	}
	if req.FitnessPostprocessor == "" {
		req.FitnessPostprocessor = "none"
	}
	if req.TopologicalPolicy == "" {
		req.TopologicalPolicy = "const"
	}
	if req.TopologicalCount < 0 {
		return materializedRunConfig{}, errors.New("topological count must be >= 0")
	}
	if req.TopologicalCount == 0 {
		req.TopologicalCount = 1
	}
	if req.TopologicalParam < 0 {
		return materializedRunConfig{}, errors.New("topological param must be >= 0")
	}
	if req.TopologicalParam == 0 {
		req.TopologicalParam = 0.5
	}
	if req.TopologicalMax == 0 {
		req.TopologicalMax = 8
	}
	if req.TuneAttempts < 0 {
		return materializedRunConfig{}, errors.New("tune attempts must be >= 0")
	}
	if req.TuneAttempts == 0 {
		req.TuneAttempts = 4
	}
	if req.TuneSelection == "" {
		req.TuneSelection = tuning.CandidateSelectBestSoFar
	}
	req.TuneSelection = normalizeTuneSelection(req.TuneSelection)
	if req.TuneDurationPolicy == "" {
		req.TuneDurationPolicy = "fixed"
	}
	req.TuneDurationPolicy = tuning.NormalizeAttemptPolicyName(req.TuneDurationPolicy)
	if req.TuneDurationParam < 0 {
		return materializedRunConfig{}, errors.New("tune duration param must be >= 0")
	}
	if req.TuneDurationParam == 0 {
		req.TuneDurationParam = 1.0
	}
	if req.TuneSteps < 0 {
		return materializedRunConfig{}, errors.New("tune steps must be >= 0")
	}
	if req.TuneSteps == 0 {
		req.TuneSteps = 6
	}
	if req.TuneStepSize < 0 {
		return materializedRunConfig{}, errors.New("tune step size must be >= 0")
	}
	if req.TuneStepSize == 0 {
		req.TuneStepSize = 0.35
	}
	if req.TunePerturbationRange < 0 {
		return materializedRunConfig{}, errors.New("tune perturbation range must be >= 0")
	}
	if req.TunePerturbationRange == 0 {
		req.TunePerturbationRange = 1.0
	}
	if req.TuneAnnealingFactor < 0 {
		return materializedRunConfig{}, errors.New("tune annealing factor must be >= 0")
	}
	if req.TuneAnnealingFactor == 0 {
		req.TuneAnnealingFactor = 1.0
	}
	if req.TuneMinImprovement < 0 {
		return materializedRunConfig{}, errors.New("tune min improvement must be >= 0")
	}
	if req.WeightPerturb == 0 && req.WeightBias == 0 && req.WeightRemoveBias == 0 && req.WeightActivation == 0 && req.WeightAggregator == 0 && req.WeightAddSynapse == 0 && req.WeightRemoveSynapse == 0 && req.WeightAddNeuron == 0 && req.WeightRemoveNeuron == 0 && req.WeightPlasticityRule == 0 && req.WeightPlasticity == 0 && req.WeightSubstrate == 0 {
		req.WeightPerturb = 0.70
		req.WeightBias = 0.00
		req.WeightRemoveBias = 0.00
		req.WeightActivation = 0.00
		req.WeightAggregator = 0.00
		req.WeightAddSynapse = 0.10
		req.WeightRemoveSynapse = 0.08
		req.WeightAddNeuron = 0.07
		req.WeightRemoveNeuron = 0.05
		req.WeightPlasticityRule = 0.00
		req.WeightPlasticity = 0.03
		req.WeightSubstrate = 0.02
	}
	if req.WeightPerturb < 0 || req.WeightBias < 0 || req.WeightRemoveBias < 0 || req.WeightActivation < 0 || req.WeightAggregator < 0 || req.WeightAddSynapse < 0 || req.WeightRemoveSynapse < 0 || req.WeightAddNeuron < 0 || req.WeightRemoveNeuron < 0 || req.WeightPlasticityRule < 0 || req.WeightPlasticity < 0 || req.WeightSubstrate < 0 {
		return materializedRunConfig{}, errors.New("mutation weights must be >= 0")
	}
	if req.WeightPerturb+req.WeightBias+req.WeightRemoveBias+req.WeightActivation+req.WeightAggregator+req.WeightAddSynapse+req.WeightRemoveSynapse+req.WeightAddNeuron+req.WeightRemoveNeuron+req.WeightPlasticityRule+req.WeightPlasticity+req.WeightSubstrate <= 0 {
		return materializedRunConfig{}, errors.New("at least one mutation weight must be > 0")
	}

	if req.SpecieIdentifier == "" {
		req.SpecieIdentifier = "topology"
	}
	specieIdentifier, err := evo.SpecieIdentifierFromName(req.SpecieIdentifier)
	if err != nil {
		return materializedRunConfig{}, err
	}

	selector, err := selectionFromName(req.Selection, specieIdentifier)
	if err != nil {
		return materializedRunConfig{}, err
	}
	postprocessor, err := postprocessorFromName(req.FitnessPostprocessor)
	if err != nil {
		return materializedRunConfig{}, err
	}
	topologicalPolicy, err := topologicalPolicyFromConfig(req.TopologicalPolicy, req.TopologicalCount, req.TopologicalParam, req.TopologicalMax)
	if err != nil {
		return materializedRunConfig{}, err
	}

	var attemptPolicy tuning.AttemptPolicy
	if req.EnableTuning || req.CompareTuning {
		attemptPolicy, err = tuning.AttemptPolicyFromConfig(req.TuneDurationPolicy, req.TuneDurationParam)
		if err != nil {
			return materializedRunConfig{}, err
		}
	}

	return materializedRunConfig{
		Request:           req,
		Selector:          selector,
		Postprocessor:     postprocessor,
		TopologicalPolicy: topologicalPolicy,
		TuneAttemptPolicy: attemptPolicy,
		SpeciationMode:    speciationModeFromIdentifier(req.SpecieIdentifier),
	}, nil
}

func speciationModeFromIdentifier(name string) string {
	switch strings.TrimSpace(strings.ToLower(name)) {
	case "fingerprint", "exact_fingerprint":
		return evo.SpeciationModeFingerprint
	default:
		return evo.SpeciationModeAdaptive
	}
}

func parseOpMode(raw string) (string, bool, bool, error) {
	normalized := strings.ToLower(strings.TrimSpace(raw))
	if normalized == "" {
		return evo.OpModeGT, false, false, nil
	}
	switch normalized {
	case evo.OpModeGT:
		return evo.OpModeGT, false, false, nil
	case evo.OpModeValidation:
		return evo.OpModeValidation, false, false, nil
	case evo.OpModeTest:
		return evo.OpModeTest, false, false, nil
	}

	clean := strings.TrimPrefix(normalized, "[")
	clean = strings.TrimSuffix(clean, "]")
	if clean == "" {
		return "", false, false, errors.New("op mode must be one of gt|validation|test or include gt with validation/test probes")
	}
	replacer := strings.NewReplacer("|", ",", "+", ",", ";", ",", " ", ",")
	clean = replacer.Replace(clean)
	parts := strings.Split(clean, ",")

	modes := map[string]struct{}{}
	for _, part := range parts {
		mode := strings.TrimSpace(part)
		if mode == "" {
			continue
		}
		switch mode {
		case evo.OpModeGT, evo.OpModeValidation, evo.OpModeTest:
			modes[mode] = struct{}{}
		default:
			return "", false, false, fmt.Errorf("unsupported op mode component: %s", mode)
		}
	}
	if len(modes) == 0 {
		return "", false, false, errors.New("op mode must be one of gt|validation|test or include gt with validation/test probes")
	}
	if _, ok := modes[evo.OpModeGT]; ok {
		_, validationProbe := modes[evo.OpModeValidation]
		_, testProbe := modes[evo.OpModeTest]
		return evo.OpModeGT, validationProbe, testProbe, nil
	}
	if len(modes) == 1 {
		if _, ok := modes[evo.OpModeValidation]; ok {
			return evo.OpModeValidation, false, false, nil
		}
		if _, ok := modes[evo.OpModeTest]; ok {
			return evo.OpModeTest, false, false, nil
		}
	}
	return "", false, false, errors.New("composite non-gt op mode is unsupported; include gt for probe combinations")
}

func hasFlatlandOverrideConfig(req RunRequest) bool {
	return strings.TrimSpace(req.FlatlandScannerProfile) != "" ||
		req.FlatlandScannerSpread != nil ||
		req.FlatlandScannerOffset != nil ||
		req.FlatlandLayoutRandomize != nil ||
		req.FlatlandLayoutVariants != nil ||
		req.FlatlandForceLayout != nil ||
		req.FlatlandBenchmarkTrials != nil
}

func toFlatlandOverrides(req RunRequest) scape.FlatlandOverrides {
	return scape.FlatlandOverrides{
		ScannerProfile:     req.FlatlandScannerProfile,
		ScannerSpread:      cloneFloat64Ptr(req.FlatlandScannerSpread),
		ScannerOffset:      cloneFloat64Ptr(req.FlatlandScannerOffset),
		RandomizeLayout:    cloneBoolPtr(req.FlatlandLayoutRandomize),
		LayoutVariants:     cloneIntPtr(req.FlatlandLayoutVariants),
		ForceLayoutVariant: cloneIntPtr(req.FlatlandForceLayout),
		BenchmarkTrials:    cloneIntPtr(req.FlatlandBenchmarkTrials),
	}
}

func cloneFloat64Ptr(v *float64) *float64 {
	if v == nil {
		return nil
	}
	out := *v
	return &out
}

func cloneBoolPtr(v *bool) *bool {
	if v == nil {
		return nil
	}
	out := *v
	return &out
}

func cloneIntPtr(v *int) *int {
	if v == nil {
		return nil
	}
	out := *v
	return &out
}

func defaultMutationPolicy(seed int64, scapeName string, inputNeuronIDs, outputNeuronIDs []string, req RunRequest) []evo.WeightedMutation {
	protected := make(map[string]struct{}, len(inputNeuronIDs)+len(outputNeuronIDs))
	for _, id := range inputNeuronIDs {
		protected[id] = struct{}{}
	}
	for _, id := range outputNeuronIDs {
		protected[id] = struct{}{}
	}

	return []evo.WeightedMutation{
		{Operator: &evo.MutateWeights{Rand: rand.New(rand.NewSource(seed + 1000)), MaxDelta: 1.0}, Weight: req.WeightPerturb},
		{Operator: &evo.AddBias{Rand: rand.New(rand.NewSource(seed + 1007)), MaxDelta: 0.3}, Weight: req.WeightBias},
		{Operator: &evo.RemoveBias{Rand: rand.New(rand.NewSource(seed + 1010))}, Weight: req.WeightRemoveBias},
		{Operator: &evo.MutateAF{Rand: rand.New(rand.NewSource(seed + 1008))}, Weight: req.WeightActivation},
		{Operator: &evo.MutateAggrF{Rand: rand.New(rand.NewSource(seed + 1009))}, Weight: req.WeightAggregator},
		{Operator: &evo.AddRandomInlink{Rand: rand.New(rand.NewSource(seed + 1001)), MaxAbsWeight: 1.0, InputNeuronIDs: inputNeuronIDs, FeedForwardOnly: true}, Weight: req.WeightAddSynapse / 2},
		{Operator: &evo.AddRandomOutlink{Rand: rand.New(rand.NewSource(seed + 1002)), MaxAbsWeight: 1.0, OutputNeuronIDs: outputNeuronIDs, FeedForwardOnly: true}, Weight: req.WeightAddSynapse / 2},
		{Operator: &evo.RemoveRandomInlink{Rand: rand.New(rand.NewSource(seed + 1003)), InputNeuronIDs: inputNeuronIDs, FeedForwardOnly: true}, Weight: req.WeightRemoveSynapse / 3},
		{Operator: &evo.RemoveRandomOutlink{Rand: rand.New(rand.NewSource(seed + 1004)), OutputNeuronIDs: outputNeuronIDs, FeedForwardOnly: true}, Weight: req.WeightRemoveSynapse / 3},
		{Operator: &evo.CutlinkFromNeuronToNeuron{Rand: rand.New(rand.NewSource(seed + 1005))}, Weight: req.WeightRemoveSynapse / 3},
		{Operator: &evo.AddNeuron{Rand: rand.New(rand.NewSource(seed + 1005))}, Weight: req.WeightAddNeuron * 0.40},
		{Operator: &evo.AddRandomOutsplice{Rand: rand.New(rand.NewSource(seed + 1006)), OutputNeuronIDs: outputNeuronIDs, FeedForwardOnly: true}, Weight: req.WeightAddNeuron * 0.30},
		{Operator: &evo.AddRandomInsplice{Rand: rand.New(rand.NewSource(seed + 1007)), InputNeuronIDs: inputNeuronIDs, FeedForwardOnly: true}, Weight: req.WeightAddNeuron * 0.30},
		{Operator: &evo.RemoveNeuronMutation{Rand: rand.New(rand.NewSource(seed + 1020)), Protected: protected}, Weight: req.WeightRemoveNeuron},
		{Operator: &evo.MutatePF{Rand: rand.New(rand.NewSource(seed + 1021))}, Weight: req.WeightPlasticityRule},
		{Operator: &evo.MutatePlasticityParameters{Rand: rand.New(rand.NewSource(seed + 1022)), MaxDelta: 0.15}, Weight: req.WeightPlasticity},
		{Operator: &evo.AddRandomSensor{Rand: rand.New(rand.NewSource(seed + 1008)), ScapeName: scapeName}, Weight: req.WeightSubstrate * 0.07},
		{Operator: &evo.AddRandomSensorLink{Rand: rand.New(rand.NewSource(seed + 1009)), ScapeName: scapeName}, Weight: req.WeightSubstrate * 0.07},
		{Operator: &evo.AddRandomActuator{Rand: rand.New(rand.NewSource(seed + 1010)), ScapeName: scapeName}, Weight: req.WeightSubstrate * 0.07},
		{Operator: &evo.AddRandomActuatorLink{Rand: rand.New(rand.NewSource(seed + 1011)), ScapeName: scapeName}, Weight: req.WeightSubstrate * 0.07},
		{Operator: &evo.RemoveRandomSensor{Rand: rand.New(rand.NewSource(seed + 1012))}, Weight: req.WeightSubstrate * 0.06},
		{Operator: &evo.CutlinkFromSensorToNeuron{Rand: rand.New(rand.NewSource(seed + 1013))}, Weight: req.WeightSubstrate * 0.06},
		{Operator: &evo.RemoveRandomActuator{Rand: rand.New(rand.NewSource(seed + 1014))}, Weight: req.WeightSubstrate * 0.06},
		{Operator: &evo.CutlinkFromNeuronToActuator{Rand: rand.New(rand.NewSource(seed + 1015))}, Weight: req.WeightSubstrate * 0.06},
		{Operator: &evo.AddRandomCPP{Rand: rand.New(rand.NewSource(seed + 1016))}, Weight: req.WeightSubstrate * 0.05},
		{Operator: &evo.RemoveRandomCPP{}, Weight: req.WeightSubstrate * 0.03},
		{Operator: &evo.AddRandomCEP{Rand: rand.New(rand.NewSource(seed + 1017))}, Weight: req.WeightSubstrate * 0.05},
		{Operator: &evo.RemoveRandomCEP{}, Weight: req.WeightSubstrate * 0.03},
		{Operator: &evo.AddCircuitNode{Rand: rand.New(rand.NewSource(seed + 1018))}, Weight: req.WeightSubstrate * 0.05},
		{Operator: &evo.DeleteCircuitNode{Rand: rand.New(rand.NewSource(seed + 1019))}, Weight: req.WeightSubstrate * 0.05},
		{Operator: &evo.AddCircuitLayer{Rand: rand.New(rand.NewSource(seed + 1020))}, Weight: req.WeightSubstrate * 0.05},
		{Operator: &evo.PerturbSubstrateParameter{Rand: rand.New(rand.NewSource(seed + 1021)), MaxDelta: 0.15}, Weight: req.WeightSubstrate * 0.05},
		{Operator: &evo.MutateTuningSelection{Rand: rand.New(rand.NewSource(seed + 1022))}, Weight: req.WeightSubstrate * 0.03},
		{Operator: &evo.MutateTuningAnnealing{Rand: rand.New(rand.NewSource(seed + 1023))}, Weight: req.WeightSubstrate * 0.03},
		{Operator: &evo.MutateTotTopologicalMutations{Rand: rand.New(rand.NewSource(seed + 1024))}, Weight: req.WeightSubstrate * 0.03},
		{Operator: &evo.MutateHeredityType{Rand: rand.New(rand.NewSource(seed + 1025))}, Weight: req.WeightSubstrate * 0.03},
	}
}

func selectionFromName(name string, specieIdentifier evo.SpecieIdentifier) (evo.Selector, error) {
	switch name {
	case "elite":
		return evo.EliteSelector{}, nil
	case "tournament":
		return evo.TournamentSelector{PoolSize: 0, TournamentSize: 3}, nil
	case "species_tournament":
		return evo.SpeciesTournamentSelector{
			Identifier:     specieIdentifier,
			PoolSize:       0,
			TournamentSize: 3,
		}, nil
	case "species_shared_tournament":
		return &evo.SpeciesSharedTournamentSelector{
			Identifier:     specieIdentifier,
			PoolSize:       0,
			TournamentSize: 3,
		}, nil
	case "hof_competition":
		return &evo.SpeciesSharedTournamentSelector{
			Identifier:            specieIdentifier,
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
			Identifier:     specieIdentifier,
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
			return nil, fmt.Errorf("topological count must be > 0 for const policy")
		}
		return evo.ConstTopologicalMutations{Count: count}, nil
	case "ncount_linear":
		if param <= 0 {
			return nil, fmt.Errorf("topological param must be > 0 for ncount_linear")
		}
		return evo.NCountLinearTopologicalMutations{
			Multiplier: param,
			MaxCount:   maxCount,
		}, nil
	case "ncount_exponential":
		if param <= 0 {
			return nil, fmt.Errorf("topological param must be > 0 for ncount_exponential")
		}
		return evo.NCountExponentialTopologicalMutations{
			Power:    param,
			MaxCount: maxCount,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported topological mutation policy: %s", name)
	}
}
