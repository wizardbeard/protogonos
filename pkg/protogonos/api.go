package protogonos

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"path/filepath"
	"time"

	"protogonos/internal/evo"
	"protogonos/internal/genotype"
	"protogonos/internal/model"
	"protogonos/internal/morphology"
	"protogonos/internal/platform"
	"protogonos/internal/scape"
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
	Scape                string
	Population           int
	Generations          int
	Seed                 int64
	Workers              int
	Selection            string
	FitnessPostprocessor string
	TopologicalPolicy    string
	TopologicalCount     int
	TopologicalParam     float64
	TopologicalMax       int
	EnableTuning         bool
	CompareTuning        bool
	TuneAttempts         int
	TuneSteps            int
	TuneStepSize         float64
	WeightPerturb        float64
	WeightAddSynapse     float64
	WeightRemoveSynapse  float64
	WeightAddNeuron      float64
	WeightRemoveNeuron   float64
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

type TopGenomesRequest struct {
	RunID  string
	Latest bool
	Limit  int
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
	if req.Scape == "" {
		req.Scape = "xor"
	}
	if req.Population <= 0 {
		req.Population = 50
	}
	if req.Generations <= 0 {
		req.Generations = 100
	}
	if req.Workers <= 0 {
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
	if req.TopologicalCount <= 0 {
		req.TopologicalCount = 1
	}
	if req.TopologicalParam <= 0 {
		req.TopologicalParam = 0.5
	}
	if req.TopologicalMax == 0 {
		req.TopologicalMax = 8
	}
	if req.TuneAttempts <= 0 {
		req.TuneAttempts = 4
	}
	if req.TuneSteps <= 0 {
		req.TuneSteps = 6
	}
	if req.TuneStepSize <= 0 {
		req.TuneStepSize = 0.35
	}
	if req.WeightPerturb == 0 && req.WeightAddSynapse == 0 && req.WeightRemoveSynapse == 0 && req.WeightAddNeuron == 0 && req.WeightRemoveNeuron == 0 {
		req.WeightPerturb = 0.70
		req.WeightAddSynapse = 0.10
		req.WeightRemoveSynapse = 0.08
		req.WeightAddNeuron = 0.07
		req.WeightRemoveNeuron = 0.05
	}
	if req.WeightPerturb < 0 || req.WeightAddSynapse < 0 || req.WeightRemoveSynapse < 0 || req.WeightAddNeuron < 0 || req.WeightRemoveNeuron < 0 {
		return RunSummary{}, errors.New("mutation weights must be >= 0")
	}
	if req.WeightPerturb+req.WeightAddSynapse+req.WeightRemoveSynapse+req.WeightAddNeuron+req.WeightRemoveNeuron <= 0 {
		return RunSummary{}, errors.New("at least one mutation weight must be > 0")
	}
	selector, err := selectionFromName(req.Selection)
	if err != nil {
		return RunSummary{}, err
	}
	postprocessor, err := postprocessorFromName(req.FitnessPostprocessor)
	if err != nil {
		return RunSummary{}, err
	}
	topologicalPolicy, err := topologicalPolicyFromConfig(req.TopologicalPolicy, req.TopologicalCount, req.TopologicalParam, req.TopologicalMax)
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
	if err := morphology.EnsureScapeCompatibility(req.Scape); err != nil {
		return RunSummary{}, err
	}

	eliteCount := req.Population / 5
	if eliteCount < 1 {
		eliteCount = 1
	}
	now := time.Now().UTC()
	runID := fmt.Sprintf("%s-%d-%d", req.Scape, req.Seed, now.Unix())

	runEvolution := func(useTuning bool) (platform.EvolutionResult, error) {
		mutation := &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(req.Seed + 1000)), MaxDelta: 1.0}
		policy := defaultMutationPolicy(req.Seed, seedPopulation.InputNeuronIDs, seedPopulation.OutputNeuronIDs, req)
		var tuner tuning.Tuner
		if useTuning {
			tuner = &tuning.Exoself{
				Rand:     rand.New(rand.NewSource(req.Seed + 2000)),
				Steps:    req.TuneSteps,
				StepSize: req.TuneStepSize,
			}
		}
		return p.RunEvolution(ctx, platform.EvolutionConfig{
			RunID:                runID,
			ScapeName:            req.Scape,
			PopulationSize:       req.Population,
			Generations:          req.Generations,
			EliteCount:           eliteCount,
			Workers:              req.Workers,
			Seed:                 req.Seed,
			InputNeuronIDs:       seedPopulation.InputNeuronIDs,
			OutputNeuronIDs:      seedPopulation.OutputNeuronIDs,
			Mutation:             mutation,
			MutationPolicy:       policy,
			Selector:             selector,
			Postprocessor:        postprocessor,
			TopologicalMutations: topologicalPolicy,
			Tuner:                tuner,
			TuneAttempts:         req.TuneAttempts,
			Initial:              seedPopulation.Genomes,
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
			RunID:                runID,
			Scape:                req.Scape,
			PopulationSize:       req.Population,
			Generations:          req.Generations,
			Seed:                 req.Seed,
			Workers:              req.Workers,
			EliteCount:           eliteCount,
			Selection:            req.Selection,
			FitnessPostprocessor: req.FitnessPostprocessor,
			TopologicalPolicy:    req.TopologicalPolicy,
			TopologicalCount:     req.TopologicalCount,
			TopologicalParam:     req.TopologicalParam,
			TopologicalMax:       req.TopologicalMax,
			TuningEnabled:        req.EnableTuning,
			TuneAttempts:         req.TuneAttempts,
			TuneSteps:            req.TuneSteps,
			TuneStepSize:         req.TuneStepSize,
			WeightPerturb:        req.WeightPerturb,
			WeightAddSynapse:     req.WeightAddSynapse,
			WeightRemoveSynapse:  req.WeightRemoveSynapse,
			WeightAddNeuron:      req.WeightAddNeuron,
			WeightRemoveNeuron:   req.WeightRemoveNeuron,
		},
		BestByGeneration:      result.BestByGeneration,
		GenerationDiagnostics: result.GenerationDiagnostics,
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
			Fingerprint: rec.Fingerprint,
			Summary:     rec.Summary,
		})
	}
	return out, nil
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
	if scapeName == "" {
		return ScapeSummaryItem{}, errors.New("scape name is required")
	}
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
	return nil
}

func defaultMutationPolicy(seed int64, inputNeuronIDs, outputNeuronIDs []string, req RunRequest) []evo.WeightedMutation {
	protected := make(map[string]struct{}, len(inputNeuronIDs)+len(outputNeuronIDs))
	for _, id := range inputNeuronIDs {
		protected[id] = struct{}{}
	}
	for _, id := range outputNeuronIDs {
		protected[id] = struct{}{}
	}

	return []evo.WeightedMutation{
		{Operator: &evo.PerturbRandomWeight{Rand: rand.New(rand.NewSource(seed + 1000)), MaxDelta: 1.0}, Weight: req.WeightPerturb},
		{Operator: &evo.AddRandomSynapse{Rand: rand.New(rand.NewSource(seed + 1001)), MaxAbsWeight: 1.0}, Weight: req.WeightAddSynapse},
		{Operator: &evo.RemoveRandomSynapse{Rand: rand.New(rand.NewSource(seed + 1002))}, Weight: req.WeightRemoveSynapse},
		{Operator: &evo.AddRandomNeuron{Rand: rand.New(rand.NewSource(seed + 1003))}, Weight: req.WeightAddNeuron},
		{Operator: &evo.RemoveRandomNeuron{Rand: rand.New(rand.NewSource(seed + 1004)), Protected: protected}, Weight: req.WeightRemoveNeuron},
	}
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
