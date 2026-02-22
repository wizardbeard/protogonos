package platform

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"protogonos/internal/evo"
	"protogonos/internal/genotype"
	"protogonos/internal/model"
	"protogonos/internal/scape"
	"protogonos/internal/storage"
	"protogonos/internal/tuning"
)

type Config struct {
	Store                       storage.Store
	SupportModules              []SupportModule
	SupportModuleRestartPolicy  SupervisorRestartPolicy
	PublicScapes                []PublicScapeSpec
	SupervisorPolicy            SupervisorPolicy
	EscalateOnSupervisorFailure bool
	SupervisorFailureReason     StopReason
}

type SupportModule interface {
	Name() string
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
}

type PublicScapeSpec struct {
	Scape         scape.Scape
	Type          string
	Parameters    []any
	Metabolics    any
	Physics       any
	RestartPolicy SupervisorRestartPolicy
}

type PublicScapeSummary struct {
	Name          string                  `json:"name"`
	Type          string                  `json:"type,omitempty"`
	Parameters    []any                   `json:"parameters,omitempty"`
	Metabolics    any                     `json:"metabolics,omitempty"`
	Physics       any                     `json:"physics,omitempty"`
	RestartPolicy SupervisorRestartPolicy `json:"restart_policy,omitempty"`
}

type StopReason string

const (
	StopReasonNormal   StopReason = "normal"
	StopReasonShutdown StopReason = "shutdown"
)

type CallMessage interface {
	isPolisCallMessage()
}

type CastMessage interface {
	isPolisCastMessage()
}

type GetScapeCall struct {
	Type string
}

func (GetScapeCall) isPolisCallMessage() {}

type StopCall struct {
	Reason StopReason
}

func (StopCall) isPolisCallMessage() {}

type GetScapeCallResult struct {
	Scape scape.Scape
	Found bool
}

type StopCast struct {
	Reason StopReason
}

func (StopCast) isPolisCastMessage() {}

type PolisInitState struct {
	SupportModules []SupportModule
	PublicScapes   []PublicScapeSpec
}

type InitCast struct {
	State *PolisInitState
}

func (InitCast) isPolisCastMessage() {}

type polisCallEnvelope struct {
	ctx   context.Context
	msg   CallMessage
	reply chan polisCallResponse
}

type polisCallResponse struct {
	value any
	err   error
}

type polisCastEnvelope struct {
	ctx   context.Context
	msg   CastMessage
	reply chan error
}

type EvolutionConfig struct {
	RunID                string
	OpMode               string
	EvolutionType        string
	ScapeName            string
	PopulationSize       int
	Generations          int
	InitialGeneration    int
	SurvivalPercentage   float64
	SpecieSizeLimit      int
	FitnessGoal          float64
	EvaluationsLimit     int
	EliteCount           int
	Workers              int
	Seed                 int64
	InputNeuronIDs       []string
	OutputNeuronIDs      []string
	Mutation             evo.Operator
	MutationPolicy       []evo.WeightedMutation
	Selector             evo.Selector
	Postprocessor        evo.FitnessPostprocessor
	TopologicalMutations evo.TopologicalMutationPolicy
	Tuner                tuning.Tuner
	TuneAttempts         int
	TuneAttemptPolicy    tuning.AttemptPolicy
	Control              chan evo.MonitorCommand
	Initial              []model.Genome
}

type EvolutionResult struct {
	BestByGeneration      []float64
	GenerationDiagnostics []model.GenerationDiagnostics
	SpeciesHistory        []model.SpeciesGeneration
	BestFinalFitness      float64
	TopFinal              []evo.ScoredGenome
	Lineage               []evo.LineageRecord
}

type SupervisionFailure struct {
	TaskName     string    `json:"task_name"`
	ErrorMessage string    `json:"error_message"`
	RestartCount int       `json:"restart_count"`
	ObservedAt   time.Time `json:"observed_at"`
}

type Polis struct {
	store storage.Store

	mu sync.RWMutex

	scapes               map[string]scape.Scape
	supportModules       map[string]SupportModule
	publicScapes         map[string]PublicScapeSummary
	publicScapeByType    map[string]string
	publicScapeTypeOrder map[string][]string
	started              bool
	lastStopReason       StopReason
	runs                 map[string]chan evo.MonitorCommand

	mailboxActive bool
	mailboxCallCh chan polisCallEnvelope
	mailboxCastCh chan polisCastEnvelope
	mailboxCancel context.CancelFunc
	mailboxDone   chan struct{}

	supervisor          *Supervisor
	supervisionFailures []SupervisionFailure
	config              Config
}

var (
	defaultPolisMu sync.Mutex
	defaultPolis   *Polis
)

func NewPolis(cfg Config) *Polis {
	p := &Polis{
		store:                cfg.Store,
		scapes:               make(map[string]scape.Scape),
		supportModules:       make(map[string]SupportModule),
		publicScapes:         make(map[string]PublicScapeSummary),
		publicScapeByType:    make(map[string]string),
		publicScapeTypeOrder: make(map[string][]string),
		runs:                 make(map[string]chan evo.MonitorCommand),
		config:               cfg,
		lastStopReason:       StopReasonNormal,
	}
	p.supervisor = p.newSupervisor()
	return p
}

func (p *Polis) newSupervisor() *Supervisor {
	return NewSupervisorWithHooks(p.config.SupervisorPolicy, SupervisorHooks{
		OnTaskPermanentFailure: p.handleSupervisionFailure,
	})
}

func StartDefault(ctx context.Context, cfg Config) (*Polis, error) {
	defaultPolisMu.Lock()
	defer defaultPolisMu.Unlock()

	if defaultPolis != nil && defaultPolis.Started() {
		return defaultPolis, nil
	}

	p := NewPolis(cfg)
	if err := p.Init(ctx); err != nil {
		return nil, err
	}
	defaultPolis = p
	return defaultPolis, nil
}

func Default() (*Polis, bool) {
	defaultPolisMu.Lock()
	p := defaultPolis
	defaultPolisMu.Unlock()

	if p == nil || !p.Started() {
		return nil, false
	}
	return p, true
}

func StopDefault(reason StopReason) error {
	defaultPolisMu.Lock()
	p := defaultPolis
	defaultPolisMu.Unlock()
	if p == nil {
		return nil
	}
	if err := p.StopWithReason(reason); err != nil {
		return err
	}
	defaultPolisMu.Lock()
	if defaultPolis == p {
		defaultPolis = nil
	}
	defaultPolisMu.Unlock()
	return nil
}

func SyncDefault(ctx context.Context) error {
	defaultPolisMu.Lock()
	p := defaultPolis
	defaultPolisMu.Unlock()
	if p == nil || !p.Started() {
		return fmt.Errorf("default polis is not initialized")
	}
	return p.Sync(ctx)
}

func (p *Polis) Init(ctx context.Context) error {
	if p.store == nil {
		return fmt.Errorf("store is required")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.started {
		return nil
	}
	return p.initLocked(ctx, p.config.SupportModules, p.config.PublicScapes)
}

func (p *Polis) InitWithState(ctx context.Context, state PolisInitState) error {
	if p.store == nil {
		return fmt.Errorf("store is required")
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.started {
		p.stopRuntimeLocked(StopReasonNormal)
	}
	supportModules := append([]SupportModule(nil), state.SupportModules...)
	publicScapes := append([]PublicScapeSpec(nil), state.PublicScapes...)
	p.config.SupportModules = supportModules
	p.config.PublicScapes = publicScapes
	return p.initLocked(ctx, supportModules, publicScapes)
}

func (p *Polis) initLocked(
	ctx context.Context,
	supportModules []SupportModule,
	publicScapes []PublicScapeSpec,
) error {
	if err := p.store.Init(ctx); err != nil {
		return err
	}
	if p.supervisor == nil {
		p.supervisor = p.newSupervisor()
	}

	startedModules := make([]SupportModule, 0, len(supportModules))
	for i, module := range supportModules {
		if module == nil {
			stopSupportModules(ctx, startedModules)
			p.resetRuntimeStateLocked()
			return fmt.Errorf("support module is nil at index %d", i)
		}
		name := module.Name()
		if name == "" {
			stopSupportModules(ctx, startedModules)
			p.resetRuntimeStateLocked()
			return fmt.Errorf("support module name is required at index %d", i)
		}
		if _, exists := p.supportModules[name]; exists {
			stopSupportModules(ctx, startedModules)
			p.resetRuntimeStateLocked()
			return fmt.Errorf("duplicate support module: %s", name)
		}
		if err := p.startSupportModule(ctx, name, module, p.config.SupportModuleRestartPolicy); err != nil {
			stopSupportModules(ctx, startedModules)
			p.resetRuntimeStateLocked()
			return fmt.Errorf("start support module %s: %w", name, err)
		}
		p.supportModules[name] = module
		startedModules = append(startedModules, module)
	}

	startedScapes := make([]managedScape, 0, len(publicScapes))
	for i, spec := range publicScapes {
		if spec.Scape == nil {
			stopManagedScapes(ctx, startedScapes)
			stopSupportModules(ctx, startedModules)
			p.resetRuntimeStateLocked()
			return fmt.Errorf("public scape is nil at index %d", i)
		}
		name := spec.Scape.Name()
		if name == "" {
			stopManagedScapes(ctx, startedScapes)
			stopSupportModules(ctx, startedModules)
			p.resetRuntimeStateLocked()
			return fmt.Errorf("public scape name is required at index %d", i)
		}
		if _, exists := p.scapes[name]; exists {
			stopManagedScapes(ctx, startedScapes)
			stopSupportModules(ctx, startedModules)
			p.resetRuntimeStateLocked()
			return fmt.Errorf("duplicate public scape: %s", name)
		}
		summary := publicScapeSummaryFromSpec(name, spec)
		if err := p.startPublicScape(ctx, name, spec.Scape, summary, summary.RestartPolicy); err != nil {
			stopManagedScapes(ctx, startedScapes)
			stopSupportModules(ctx, startedModules)
			p.resetRuntimeStateLocked()
			return fmt.Errorf("start public scape %s: %w", name, err)
		}
		if managed, ok := spec.Scape.(managedScape); ok {
			startedScapes = append(startedScapes, managed)
		}
		p.scapes[name] = spec.Scape
		p.publicScapes[name] = summary
		p.publicScapeTypeOrder[summary.Type] = append(p.publicScapeTypeOrder[summary.Type], name)
		if _, exists := p.publicScapeByType[summary.Type]; !exists {
			p.publicScapeByType[summary.Type] = name
		}
	}

	p.started = true
	p.ensureMailboxLocked()
	return nil
}

func (p *Polis) Create(ctx context.Context) error {
	return p.Init(ctx)
}

func (p *Polis) Reset(ctx context.Context) error {
	_ = p.StopWithReason(StopReasonShutdown)
	if resetter, ok := p.store.(storage.Resetter); ok {
		if err := resetter.Reset(ctx); err != nil {
			return err
		}
	}
	return p.Init(ctx)
}

func (p *Polis) RegisterScape(s scape.Scape) error {
	if s == nil {
		return fmt.Errorf("scape is nil")
	}

	name := s.Name()
	if name == "" {
		return fmt.Errorf("scape name is required")
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.started {
		return fmt.Errorf("polis is not initialized")
	}
	p.scapes[name] = s
	return nil
}

func (p *Polis) AddSupportModule(ctx context.Context, module SupportModule) error {
	return p.AddSupportModuleWithPolicy(ctx, module, p.config.SupportModuleRestartPolicy)
}

func (p *Polis) AddSupportModuleWithPolicy(
	ctx context.Context,
	module SupportModule,
	restartPolicy SupervisorRestartPolicy,
) error {
	if module == nil {
		return fmt.Errorf("support module is nil")
	}
	name := module.Name()
	if name == "" {
		return fmt.Errorf("support module name is required")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.started {
		return fmt.Errorf("polis is not initialized")
	}
	if _, exists := p.supportModules[name]; exists {
		return fmt.Errorf("support module already registered: %s", name)
	}
	if err := p.startSupportModule(ctx, name, module, restartPolicy); err != nil {
		return fmt.Errorf("start support module %s: %w", name, err)
	}
	p.supportModules[name] = module
	return nil
}

func (p *Polis) RemoveSupportModule(ctx context.Context, name string, reason StopReason) error {
	if name == "" {
		return fmt.Errorf("support module name is required")
	}
	if reason == "" {
		reason = StopReasonNormal
	}
	if !isValidStopReason(reason) {
		return fmt.Errorf("unsupported stop reason: %s", reason)
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.started {
		return fmt.Errorf("polis is not initialized")
	}
	module, ok := p.supportModules[name]
	if !ok {
		return fmt.Errorf("support module not found: %s", name)
	}
	if p.supervisor != nil {
		p.supervisor.Stop(supervisedSupportTaskName(name))
	}
	if err := stopSupportModuleWithReason(ctx, module, reason); err != nil {
		return fmt.Errorf("stop support module %s: %w", name, err)
	}
	delete(p.supportModules, name)
	return nil
}

func (p *Polis) AddPublicScape(ctx context.Context, spec PublicScapeSpec) error {
	if spec.Scape == nil {
		return fmt.Errorf("public scape is nil")
	}
	name := spec.Scape.Name()
	if name == "" {
		return fmt.Errorf("public scape name is required")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.started {
		return fmt.Errorf("polis is not initialized")
	}
	if _, exists := p.scapes[name]; exists {
		return fmt.Errorf("duplicate public scape: %s", name)
	}
	summary := publicScapeSummaryFromSpec(name, spec)
	if err := p.startPublicScape(ctx, name, spec.Scape, summary, summary.RestartPolicy); err != nil {
		return fmt.Errorf("start public scape %s: %w", name, err)
	}
	p.scapes[name] = spec.Scape
	p.publicScapes[name] = summary
	p.publicScapeTypeOrder[summary.Type] = append(p.publicScapeTypeOrder[summary.Type], name)
	if _, exists := p.publicScapeByType[summary.Type]; !exists {
		p.publicScapeByType[summary.Type] = name
	}
	return nil
}

func (p *Polis) RemovePublicScape(ctx context.Context, name string, reason StopReason) error {
	if name == "" {
		return fmt.Errorf("public scape name is required")
	}
	if reason == "" {
		reason = StopReasonNormal
	}
	if !isValidStopReason(reason) {
		return fmt.Errorf("unsupported stop reason: %s", reason)
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.started {
		return fmt.Errorf("polis is not initialized")
	}
	sc, ok := p.scapes[name]
	if !ok {
		return fmt.Errorf("public scape not found: %s", name)
	}
	summary, ok := p.publicScapes[name]
	if !ok {
		return fmt.Errorf("public scape not found: %s", name)
	}
	if p.supervisor != nil {
		p.supervisor.Stop(supervisedScapeTaskName(name))
	}
	if err := stopPublicScapeWithReason(ctx, sc, reason); err != nil {
		return fmt.Errorf("stop public scape %s: %w", name, err)
	}
	delete(p.scapes, name)
	delete(p.publicScapes, name)
	orderedNames := p.publicScapeTypeOrder[summary.Type]
	filtered := make([]string, 0, len(orderedNames))
	for _, candidate := range orderedNames {
		if candidate == name {
			continue
		}
		if _, exists := p.publicScapes[candidate]; !exists {
			continue
		}
		filtered = append(filtered, candidate)
	}
	if len(filtered) == 0 {
		delete(p.publicScapeTypeOrder, summary.Type)
		delete(p.publicScapeByType, summary.Type)
	} else {
		p.publicScapeTypeOrder[summary.Type] = filtered
		p.publicScapeByType[summary.Type] = filtered[0]
	}
	return nil
}

func (p *Polis) GetScape(name string) (scape.Scape, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	s, ok := p.scapes[name]
	return s, ok
}

func (p *Polis) GetScapeByType(scapeType string) (scape.Scape, bool) {
	if scapeType == "" {
		return nil, false
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	orderedNames := p.publicScapeTypeOrder[scapeType]
	if len(orderedNames) == 0 {
		return nil, false
	}
	for _, name := range orderedNames {
		s, ok := p.scapes[name]
		if ok {
			return s, true
		}
	}
	return nil, false
}

func (p *Polis) Call(ctx context.Context, msg CallMessage) (any, error) {
	if msg == nil {
		return nil, fmt.Errorf("call message is required")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	p.mu.RLock()
	active := p.mailboxActive
	callCh := p.mailboxCallCh
	p.mu.RUnlock()
	if !active || callCh == nil {
		value, err, _ := p.handleCallMessage(ctx, msg, false)
		return value, err
	}
	reply := make(chan polisCallResponse, 1)
	envelope := polisCallEnvelope{
		ctx:   ctx,
		msg:   msg,
		reply: reply,
	}
	select {
	case callCh <- envelope:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	select {
	case out := <-reply:
		return out.value, out.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (p *Polis) Cast(ctx context.Context, msg CastMessage) error {
	if msg == nil {
		return fmt.Errorf("cast message is required")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	p.mu.RLock()
	active := p.mailboxActive
	castCh := p.mailboxCastCh
	p.mu.RUnlock()
	if !active || castCh == nil {
		err, _ := p.handleCastMessage(ctx, msg, false)
		return err
	}
	reply := make(chan error, 1)
	envelope := polisCastEnvelope{
		ctx:   ctx,
		msg:   msg,
		reply: reply,
	}
	select {
	case castCh <- envelope:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case err := <-reply:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *Polis) Stop() {
	_ = p.StopWithReason(StopReasonNormal)
}

func (p *Polis) Shutdown() {
	_ = p.StopWithReason(StopReasonShutdown)
}

func (p *Polis) StopWithReason(reason StopReason) error {
	return p.stopWithReason(reason, false)
}

func (p *Polis) stopWithReason(reason StopReason, fromMailbox bool) error {
	if reason == "" {
		reason = StopReasonNormal
	}
	if !isValidStopReason(reason) {
		return fmt.Errorf("unsupported stop reason: %s", reason)
	}

	var mailboxDone chan struct{}
	p.mu.Lock()
	if !fromMailbox {
		mailboxDone = p.stopMailboxLocked()
	}
	p.stopRuntimeLocked(reason)
	p.mu.Unlock()
	if mailboxDone != nil {
		<-mailboxDone
	}
	return nil
}

func (p *Polis) stopRuntimeLocked(reason StopReason) {
	if p.supervisor != nil {
		p.supervisor.StopAll()
	}
	for _, control := range p.runs {
		select {
		case control <- evo.CommandStop:
		default:
		}
	}
	for _, sc := range p.scapes {
		_ = stopPublicScapeWithReason(context.Background(), sc, reason)
	}
	for _, module := range p.supportModules {
		_ = stopSupportModuleWithReason(context.Background(), module, reason)
	}
	p.lastStopReason = reason
	p.resetRuntimeStateLocked()
}

func (p *Polis) RunEvolution(ctx context.Context, cfg EvolutionConfig) (EvolutionResult, error) {
	if len(cfg.Initial) != cfg.PopulationSize {
		return EvolutionResult{}, fmt.Errorf("initial population mismatch: got=%d want=%d", len(cfg.Initial), cfg.PopulationSize)
	}
	if cfg.Mutation == nil {
		return EvolutionResult{}, fmt.Errorf("mutation operator is required")
	}
	if cfg.ScapeName == "" {
		return EvolutionResult{}, fmt.Errorf("scape name is required")
	}
	if cfg.EliteCount <= 0 {
		cfg.EliteCount = 1
	}
	if cfg.Workers <= 0 {
		cfg.Workers = 1
	}

	p.mu.RLock()
	targetScape, ok := p.scapes[cfg.ScapeName]
	started := p.started
	p.mu.RUnlock()

	if !started {
		return EvolutionResult{}, fmt.Errorf("polis is not initialized")
	}
	if !ok {
		return EvolutionResult{}, fmt.Errorf("scape not registered: %s", cfg.ScapeName)
	}

	runID := cfg.RunID
	if runID == "" {
		runID = fmt.Sprintf("evo:%s:%d", cfg.ScapeName, cfg.Seed)
	}
	control := cfg.Control
	if control == nil {
		control = make(chan evo.MonitorCommand, 16)
	}
	if err := p.registerRunControl(runID, control); err != nil {
		return EvolutionResult{}, err
	}
	defer p.unregisterRunControl(runID)

	monitor, err := evo.NewPopulationMonitor(evo.MonitorConfig{
		Scape:                targetScape,
		OpMode:               cfg.OpMode,
		EvolutionType:        cfg.EvolutionType,
		Mutation:             cfg.Mutation,
		PopulationSize:       cfg.PopulationSize,
		EliteCount:           cfg.EliteCount,
		SurvivalPercentage:   cfg.SurvivalPercentage,
		SpecieSizeLimit:      cfg.SpecieSizeLimit,
		Generations:          cfg.Generations,
		GenerationOffset:     cfg.InitialGeneration,
		FitnessGoal:          cfg.FitnessGoal,
		EvaluationsLimit:     cfg.EvaluationsLimit,
		Workers:              cfg.Workers,
		Seed:                 cfg.Seed,
		InputNeuronIDs:       cfg.InputNeuronIDs,
		OutputNeuronIDs:      cfg.OutputNeuronIDs,
		MutationPolicy:       cfg.MutationPolicy,
		Selector:             cfg.Selector,
		Postprocessor:        cfg.Postprocessor,
		TopologicalMutations: cfg.TopologicalMutations,
		Tuner:                cfg.Tuner,
		TuneAttempts:         cfg.TuneAttempts,
		TuneAttemptPolicy:    cfg.TuneAttemptPolicy,
		Control:              control,
	})
	if err != nil {
		return EvolutionResult{}, err
	}

	result, err := monitor.Run(ctx, cfg.Initial)
	if err != nil {
		return EvolutionResult{}, err
	}
	if cfg.InitialGeneration > 0 {
		result, err = p.mergeExistingRunHistory(ctx, persistenceRunID(cfg, runID), result)
		if err != nil {
			return EvolutionResult{}, err
		}
	}
	finalGenomes := make([]model.Genome, 0, len(result.FinalPopulation))
	for _, scored := range result.FinalPopulation {
		finalGenomes = append(finalGenomes, scored.Genome)
	}
	executedGenerations := len(result.BestByGeneration) + cfg.InitialGeneration
	persistenceRunID := persistenceRunID(cfg, runID)
	populationID := persistenceRunID
	if err := genotype.SavePopulationSnapshot(ctx, p.store, populationID, executedGenerations, finalGenomes); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.store.SaveFitnessHistory(ctx, persistenceRunID, result.BestByGeneration); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.store.SaveGenerationDiagnostics(ctx, persistenceRunID, toModelDiagnostics(result.GenerationDiagnostics)); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.store.SaveSpeciesHistory(ctx, persistenceRunID, toModelSpeciesHistory(result.SpeciesHistory)); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.store.SaveLineage(ctx, persistenceRunID, toModelLineage(result.Lineage)); err != nil {
		return EvolutionResult{}, err
	}

	bestFinal := 0.0
	topFinal := []evo.ScoredGenome{}
	if len(result.FinalPopulation) > 0 {
		ranked := append([]evo.ScoredGenome(nil), result.FinalPopulation...)
		sort.Slice(ranked, func(i, j int) bool {
			return ranked[i].Fitness > ranked[j].Fitness
		})
		bestFinal = ranked[0].Fitness
		topCount := 5
		if len(ranked) < topCount {
			topCount = len(ranked)
		}
		topFinal = append(topFinal, ranked[:topCount]...)
	}
	if err := p.store.SaveTopGenomes(ctx, persistenceRunID, toModelTopGenomes(topFinal)); err != nil {
		return EvolutionResult{}, err
	}
	if err := p.updateScapeSummary(ctx, cfg.ScapeName, bestFinal); err != nil {
		return EvolutionResult{}, err
	}

	return EvolutionResult{
		BestByGeneration:      result.BestByGeneration,
		GenerationDiagnostics: toModelDiagnostics(result.GenerationDiagnostics),
		SpeciesHistory:        toModelSpeciesHistory(result.SpeciesHistory),
		BestFinalFitness:      bestFinal,
		TopFinal:              topFinal,
		Lineage:               result.Lineage,
	}, nil
}

func persistenceRunID(cfg EvolutionConfig, fallback string) string {
	if cfg.RunID != "" {
		return cfg.RunID
	}
	return fallback
}

func (p *Polis) mergeExistingRunHistory(ctx context.Context, runID string, current evo.RunResult) (evo.RunResult, error) {
	if runID == "" {
		return current, nil
	}

	if history, ok, err := p.store.GetFitnessHistory(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok {
		current.BestByGeneration = append(append([]float64{}, history...), current.BestByGeneration...)
	}

	if diagnostics, ok, err := p.store.GetGenerationDiagnostics(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok {
		prefix := make([]evo.GenerationDiagnostics, 0, len(diagnostics))
		for _, item := range diagnostics {
			prefix = append(prefix, evo.GenerationDiagnostics{
				Generation:            item.Generation,
				BestFitness:           item.BestFitness,
				MeanFitness:           item.MeanFitness,
				MinFitness:            item.MinFitness,
				SpeciesCount:          item.SpeciesCount,
				FingerprintDiversity:  item.FingerprintDiversity,
				SpeciationThreshold:   item.SpeciationThreshold,
				TargetSpeciesCount:    item.TargetSpeciesCount,
				MeanSpeciesSize:       item.MeanSpeciesSize,
				LargestSpeciesSize:    item.LargestSpeciesSize,
				TuningInvocations:     item.TuningInvocations,
				TuningAttempts:        item.TuningAttempts,
				TuningEvaluations:     item.TuningEvaluations,
				TuningAccepted:        item.TuningAccepted,
				TuningRejected:        item.TuningRejected,
				TuningGoalHits:        item.TuningGoalHits,
				TuningAcceptRate:      item.TuningAcceptRate,
				TuningEvalsPerAttempt: item.TuningEvalsPerAttempt,
			})
		}
		current.GenerationDiagnostics = append(prefix, current.GenerationDiagnostics...)
	}

	if speciesHistory, ok, err := p.store.GetSpeciesHistory(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok {
		prefix := make([]evo.SpeciesGeneration, 0, len(speciesHistory))
		for _, generation := range speciesHistory {
			species := make([]evo.SpeciesMetrics, 0, len(generation.Species))
			for _, metric := range generation.Species {
				species = append(species, evo.SpeciesMetrics{
					Key:         metric.Key,
					Size:        metric.Size,
					MeanFitness: metric.MeanFitness,
					BestFitness: metric.BestFitness,
				})
			}
			prefix = append(prefix, evo.SpeciesGeneration{
				Generation:     generation.Generation,
				Species:        species,
				NewSpecies:     append([]string{}, generation.NewSpecies...),
				ExtinctSpecies: append([]string{}, generation.ExtinctSpecies...),
			})
		}
		current.SpeciesHistory = append(prefix, current.SpeciesHistory...)
	}

	if lineage, ok, err := p.store.GetLineage(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok {
		prefix := make([]evo.LineageRecord, 0, len(lineage))
		for _, rec := range lineage {
			prefix = append(prefix, evo.LineageRecord{
				GenomeID:    rec.GenomeID,
				ParentID:    rec.ParentID,
				Generation:  rec.Generation,
				Operation:   rec.Operation,
				Fingerprint: rec.Fingerprint,
				Summary: evo.TopologySummary{
					TotalNeurons:           rec.Summary.TotalNeurons,
					TotalSynapses:          rec.Summary.TotalSynapses,
					TotalRecurrentSynapses: rec.Summary.TotalRecurrentSynapses,
					TotalSensors:           rec.Summary.TotalSensors,
					TotalActuators:         rec.Summary.TotalActuators,
					ActivationDistribution: rec.Summary.ActivationDistribution,
					AggregatorDistribution: rec.Summary.AggregatorDistribution,
				},
			})
		}
		current.Lineage = append(prefix, current.Lineage...)
	}

	if top, ok, err := p.store.GetTopGenomes(ctx, runID); err != nil {
		return evo.RunResult{}, err
	} else if ok && len(top) > 0 {
		merged := make([]evo.ScoredGenome, 0, len(top)+len(current.FinalPopulation))
		for _, item := range top {
			merged = append(merged, evo.ScoredGenome{
				Genome:  item.Genome,
				Fitness: item.Fitness,
			})
		}
		merged = append(merged, current.FinalPopulation...)
		sort.Slice(merged, func(i, j int) bool {
			return merged[i].Fitness > merged[j].Fitness
		})
		seen := make(map[string]struct{}, len(merged))
		unique := make([]evo.ScoredGenome, 0, len(merged))
		for _, item := range merged {
			if item.Genome.ID != "" {
				if _, exists := seen[item.Genome.ID]; exists {
					continue
				}
				seen[item.Genome.ID] = struct{}{}
			}
			unique = append(unique, item)
		}
		current.FinalPopulation = unique
	}

	return current, nil
}

func toModelLineage(lineage []evo.LineageRecord) []model.LineageRecord {
	out := make([]model.LineageRecord, 0, len(lineage))
	for _, rec := range lineage {
		out = append(out, model.LineageRecord{
			VersionedRecord: model.VersionedRecord{
				SchemaVersion: storage.CurrentSchemaVersion,
				CodecVersion:  storage.CurrentCodecVersion,
			},
			GenomeID:    rec.GenomeID,
			ParentID:    rec.ParentID,
			Generation:  rec.Generation,
			Operation:   rec.Operation,
			Fingerprint: rec.Fingerprint,
			Summary: model.LineageSummary{
				TotalNeurons:           rec.Summary.TotalNeurons,
				TotalSynapses:          rec.Summary.TotalSynapses,
				TotalRecurrentSynapses: rec.Summary.TotalRecurrentSynapses,
				TotalSensors:           rec.Summary.TotalSensors,
				TotalActuators:         rec.Summary.TotalActuators,
				ActivationDistribution: rec.Summary.ActivationDistribution,
				AggregatorDistribution: rec.Summary.AggregatorDistribution,
			},
		})
	}
	return out
}

func toModelDiagnostics(diags []evo.GenerationDiagnostics) []model.GenerationDiagnostics {
	out := make([]model.GenerationDiagnostics, 0, len(diags))
	for _, d := range diags {
		out = append(out, model.GenerationDiagnostics{
			Generation:            d.Generation,
			BestFitness:           d.BestFitness,
			MeanFitness:           d.MeanFitness,
			MinFitness:            d.MinFitness,
			SpeciesCount:          d.SpeciesCount,
			FingerprintDiversity:  d.FingerprintDiversity,
			SpeciationThreshold:   d.SpeciationThreshold,
			TargetSpeciesCount:    d.TargetSpeciesCount,
			MeanSpeciesSize:       d.MeanSpeciesSize,
			LargestSpeciesSize:    d.LargestSpeciesSize,
			TuningInvocations:     d.TuningInvocations,
			TuningAttempts:        d.TuningAttempts,
			TuningEvaluations:     d.TuningEvaluations,
			TuningAccepted:        d.TuningAccepted,
			TuningRejected:        d.TuningRejected,
			TuningGoalHits:        d.TuningGoalHits,
			TuningAcceptRate:      d.TuningAcceptRate,
			TuningEvalsPerAttempt: d.TuningEvalsPerAttempt,
		})
	}
	return out
}

func toModelTopGenomes(top []evo.ScoredGenome) []model.TopGenomeRecord {
	out := make([]model.TopGenomeRecord, 0, len(top))
	for i, item := range top {
		out = append(out, model.TopGenomeRecord{
			Rank:    i + 1,
			Fitness: item.Fitness,
			Genome:  item.Genome,
		})
	}
	return out
}

func toModelSpeciesHistory(history []evo.SpeciesGeneration) []model.SpeciesGeneration {
	out := make([]model.SpeciesGeneration, 0, len(history))
	for _, generation := range history {
		species := make([]model.SpeciesMetrics, 0, len(generation.Species))
		for _, item := range generation.Species {
			species = append(species, model.SpeciesMetrics{
				Key:         item.Key,
				Size:        item.Size,
				MeanFitness: item.MeanFitness,
				BestFitness: item.BestFitness,
			})
		}
		out = append(out, model.SpeciesGeneration{
			Generation:     generation.Generation,
			Species:        species,
			NewSpecies:     append([]string(nil), generation.NewSpecies...),
			ExtinctSpecies: append([]string(nil), generation.ExtinctSpecies...),
		})
	}
	return out
}

func (p *Polis) updateScapeSummary(ctx context.Context, scapeName string, fitness float64) error {
	summary, ok, err := p.store.GetScapeSummary(ctx, scapeName)
	if err != nil {
		return err
	}
	if !ok {
		summary = model.ScapeSummary{
			VersionedRecord: model.VersionedRecord{
				SchemaVersion: storage.CurrentSchemaVersion,
				CodecVersion:  storage.CurrentCodecVersion,
			},
			Name:        scapeName,
			Description: fmt.Sprintf("best observed fitness for scape %s", scapeName),
		}
	}
	if fitness > summary.BestFitness {
		summary.BestFitness = fitness
	}
	return p.store.SaveScapeSummary(ctx, summary)
}

func (p *Polis) PauseRun(runID string) error {
	return p.sendRunCommand(runID, evo.CommandPause)
}

func (p *Polis) ContinueRun(runID string) error {
	return p.sendRunCommand(runID, evo.CommandContinue)
}

func (p *Polis) StopRun(runID string) error {
	return p.sendRunCommand(runID, evo.CommandStop)
}

func (p *Polis) registerRunControl(runID string, control chan evo.MonitorCommand) error {
	if runID == "" {
		return fmt.Errorf("run id is required")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.started {
		return fmt.Errorf("polis is not initialized")
	}
	if _, exists := p.runs[runID]; exists {
		return fmt.Errorf("run already active: %s", runID)
	}
	p.runs[runID] = control
	return nil
}

func (p *Polis) unregisterRunControl(runID string) {
	if runID == "" {
		return
	}
	p.mu.Lock()
	delete(p.runs, runID)
	p.mu.Unlock()
}

func (p *Polis) sendRunCommand(runID string, cmd evo.MonitorCommand) error {
	if runID == "" {
		return fmt.Errorf("run id is required")
	}
	p.mu.RLock()
	control, ok := p.runs[runID]
	p.mu.RUnlock()
	if !ok {
		return fmt.Errorf("run not active: %s", runID)
	}
	select {
	case control <- cmd:
		return nil
	default:
		return fmt.Errorf("run control channel is full: %s", runID)
	}
}

func (p *Polis) RegisteredScapes() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	names := make([]string, 0, len(p.scapes))
	for name := range p.scapes {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (p *Polis) ActiveSupportModules() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	names := make([]string, 0, len(p.supportModules))
	for name := range p.supportModules {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (p *Polis) ActivePublicScapes() []PublicScapeSummary {
	p.mu.RLock()
	defer p.mu.RUnlock()

	names := make([]string, 0, len(p.publicScapes))
	for name := range p.publicScapes {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]PublicScapeSummary, 0, len(names))
	for _, name := range names {
		summary := p.publicScapes[name]
		summary.Parameters = append([]any(nil), summary.Parameters...)
		out = append(out, summary)
	}
	return out
}

func (p *Polis) Started() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.started
}

func (p *Polis) LastStopReason() StopReason {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.lastStopReason
}

func (p *Polis) MailboxActive() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.mailboxActive
}

func (p *Polis) Sync(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	p.mu.RLock()
	started := p.started
	store := p.store
	modules := make([]SupportModule, 0, len(p.supportModules))
	for _, module := range p.supportModules {
		modules = append(modules, module)
	}
	scapes := make([]scape.Scape, 0, len(p.scapes))
	for _, activeScape := range p.scapes {
		scapes = append(scapes, activeScape)
	}
	p.mu.RUnlock()

	if !started {
		return fmt.Errorf("polis is not initialized")
	}
	if store != nil {
		if err := store.Init(ctx); err != nil {
			return err
		}
	}
	for _, module := range modules {
		syncable, ok := module.(syncableRuntime)
		if !ok {
			continue
		}
		if err := syncable.Sync(ctx); err != nil {
			return fmt.Errorf("sync support module %s: %w", module.Name(), err)
		}
	}
	for _, activeScape := range scapes {
		syncable, ok := activeScape.(syncableRuntime)
		if !ok {
			continue
		}
		if err := syncable.Sync(ctx); err != nil {
			return fmt.Errorf("sync scape %s: %w", activeScape.Name(), err)
		}
	}
	return nil
}

func (p *Polis) ActiveSupervisedTasks() []string {
	p.mu.RLock()
	supervisor := p.supervisor
	p.mu.RUnlock()
	if supervisor == nil {
		return nil
	}
	return supervisor.Tasks()
}

func (p *Polis) ActiveSupervisedChildren() []SupervisorChildStatus {
	p.mu.RLock()
	supervisor := p.supervisor
	p.mu.RUnlock()
	if supervisor == nil {
		return nil
	}
	return supervisor.Children()
}

func (p *Polis) SupervisionFailures() []SupervisionFailure {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]SupervisionFailure, len(p.supervisionFailures))
	copy(out, p.supervisionFailures)
	return out
}

type managedScape interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
}

type summaryAwareManagedScape interface {
	managedScape
	StartWithSummary(ctx context.Context, summary PublicScapeSummary) error
}

type reasonAwareManagedScape interface {
	managedScape
	StopWithReason(ctx context.Context, reason StopReason) error
}

type reasonAwareSupportModule interface {
	SupportModule
	StopWithReason(ctx context.Context, reason StopReason) error
}

type supervisedRuntime interface {
	Supervise(ctx context.Context) error
}

type syncableRuntime interface {
	Sync(ctx context.Context) error
}

func isValidStopReason(reason StopReason) bool {
	switch reason {
	case StopReasonNormal, StopReasonShutdown:
		return true
	default:
		return false
	}
}

func (p *Polis) resetRuntimeStateLocked() {
	p.started = false
	p.scapes = make(map[string]scape.Scape)
	p.supportModules = make(map[string]SupportModule)
	p.publicScapes = make(map[string]PublicScapeSummary)
	p.publicScapeByType = make(map[string]string)
	p.publicScapeTypeOrder = make(map[string][]string)
	p.runs = make(map[string]chan evo.MonitorCommand)
	if p.supervisor != nil {
		p.supervisor.StopAll()
	}
	p.supervisor = p.newSupervisor()
}

func (p *Polis) ensureMailboxLocked() {
	if p.mailboxActive {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	callCh := make(chan polisCallEnvelope, 32)
	castCh := make(chan polisCastEnvelope, 32)
	done := make(chan struct{})
	p.mailboxActive = true
	p.mailboxCallCh = callCh
	p.mailboxCastCh = castCh
	p.mailboxCancel = cancel
	p.mailboxDone = done
	go p.mailboxLoop(ctx, callCh, castCh, done)
}

func (p *Polis) stopMailboxLocked() chan struct{} {
	if !p.mailboxActive || p.mailboxCancel == nil {
		return nil
	}
	done := p.mailboxDone
	cancel := p.mailboxCancel
	p.mailboxActive = false
	p.mailboxCallCh = nil
	p.mailboxCastCh = nil
	p.mailboxCancel = nil
	p.mailboxDone = nil
	cancel()
	return done
}

func (p *Polis) mailboxLoop(
	ctx context.Context,
	callCh <-chan polisCallEnvelope,
	castCh <-chan polisCastEnvelope,
	done chan struct{},
) {
	defer func() {
		p.mu.Lock()
		if p.mailboxDone == done {
			p.mailboxActive = false
			p.mailboxCallCh = nil
			p.mailboxCastCh = nil
			p.mailboxCancel = nil
			p.mailboxDone = nil
		}
		p.mu.Unlock()
		close(done)
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case call := <-callCh:
			value, err, terminate := p.handleCallMessage(call.ctx, call.msg, true)
			call.reply <- polisCallResponse{value: value, err: err}
			if terminate {
				return
			}
		case cast := <-castCh:
			err, terminate := p.handleCastMessage(cast.ctx, cast.msg, true)
			cast.reply <- err
			if terminate {
				return
			}
		}
	}
}

func (p *Polis) handleCallMessage(ctx context.Context, msg CallMessage, fromMailbox bool) (any, error, bool) {
	switch req := msg.(type) {
	case GetScapeCall:
		p.mu.RLock()
		started := p.started
		p.mu.RUnlock()
		if !started {
			return GetScapeCallResult{}, fmt.Errorf("polis is not initialized"), false
		}
		sc, ok := p.GetScapeByType(req.Type)
		return GetScapeCallResult{Scape: sc, Found: ok}, nil, false
	case StopCall:
		err := p.stopWithReason(req.Reason, fromMailbox)
		return nil, err, fromMailbox
	default:
		return nil, fmt.Errorf("unsupported call message: %T", msg), false
	}
}

func (p *Polis) handleCastMessage(ctx context.Context, msg CastMessage, fromMailbox bool) (error, bool) {
	switch req := msg.(type) {
	case StopCast:
		return p.stopWithReason(req.Reason, fromMailbox), fromMailbox
	case InitCast:
		if req.State != nil {
			return p.InitWithState(ctx, *req.State), false
		}
		return p.Init(ctx), false
	default:
		return fmt.Errorf("unsupported cast message: %T", msg), false
	}
}

func (p *Polis) handleSupervisionFailure(taskName string, err error, restartCount int) {
	failure := SupervisionFailure{
		TaskName:     taskName,
		ErrorMessage: errString(err),
		RestartCount: restartCount,
		ObservedAt:   time.Now().UTC(),
	}

	p.mu.Lock()
	p.supervisionFailures = append(p.supervisionFailures, failure)
	started := p.started
	escalate := p.config.EscalateOnSupervisorFailure
	reason := p.config.SupervisorFailureReason
	p.mu.Unlock()

	if !started || !escalate {
		return
	}
	if reason == "" || !isValidStopReason(reason) {
		reason = StopReasonShutdown
	}
	_ = p.stopWithReason(reason, false)
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func publicScapeSummaryFromSpec(name string, spec PublicScapeSpec) PublicScapeSummary {
	restart := spec.RestartPolicy
	if restart == "" {
		restart = SupervisorRestartPermanent
	}
	summary := PublicScapeSummary{
		Name:          name,
		Type:          spec.Type,
		Parameters:    append([]any(nil), spec.Parameters...),
		Metabolics:    spec.Metabolics,
		Physics:       spec.Physics,
		RestartPolicy: restart,
	}
	if summary.Type == "" {
		summary.Type = name
	}
	return summary
}

func supervisedSupportTaskName(name string) string {
	return "support:" + name
}

func supervisedScapeTaskName(name string) string {
	return "scape:" + name
}

func (p *Polis) startSupportModule(
	ctx context.Context,
	name string,
	module SupportModule,
	restartPolicy SupervisorRestartPolicy,
) error {
	if err := module.Start(ctx); err != nil {
		return err
	}
	supervised, ok := module.(supervisedRuntime)
	if !ok {
		return nil
	}
	if p.supervisor == nil {
		p.supervisor = p.newSupervisor()
	}
	spec := SupervisorChildSpec{
		Name:    supervisedSupportTaskName(name),
		Group:   "support",
		Restart: restartPolicy,
	}
	if err := p.supervisor.StartSpec(spec, supervised.Supervise); err != nil {
		_ = module.Stop(ctx)
		return err
	}
	return nil
}

func stopSupportModuleWithReason(ctx context.Context, module SupportModule, reason StopReason) error {
	if withReason, ok := module.(reasonAwareSupportModule); ok {
		return withReason.StopWithReason(ctx, reason)
	}
	return module.Stop(ctx)
}

func (p *Polis) startPublicScape(
	ctx context.Context,
	name string,
	sc scape.Scape,
	summary PublicScapeSummary,
	restartPolicy SupervisorRestartPolicy,
) error {
	managed, ok := sc.(managedScape)
	managedStarted := false
	if ok {
		if withSummary, ok := sc.(summaryAwareManagedScape); ok {
			if err := withSummary.StartWithSummary(ctx, summary); err != nil {
				return err
			}
		} else {
			if err := managed.Start(ctx); err != nil {
				return err
			}
		}
		managedStarted = true
	}

	supervised, ok := sc.(supervisedRuntime)
	if !ok {
		return nil
	}
	if p.supervisor == nil {
		p.supervisor = p.newSupervisor()
	}
	spec := SupervisorChildSpec{
		Name:    supervisedScapeTaskName(name),
		Group:   "scape",
		Restart: restartPolicy,
	}
	if err := p.supervisor.StartSpec(spec, supervised.Supervise); err != nil {
		if managedStarted {
			_ = managed.Stop(ctx)
		}
		return err
	}
	return nil
}

func stopPublicScapeWithReason(ctx context.Context, sc scape.Scape, reason StopReason) error {
	managed, ok := sc.(managedScape)
	if !ok {
		return nil
	}
	if withReason, ok := sc.(reasonAwareManagedScape); ok {
		return withReason.StopWithReason(ctx, reason)
	}
	return managed.Stop(ctx)
}

func stopSupportModules(ctx context.Context, modules []SupportModule) {
	for i := len(modules) - 1; i >= 0; i-- {
		_ = modules[i].Stop(ctx)
	}
}

func stopManagedScapes(ctx context.Context, scapes []managedScape) {
	for i := len(scapes) - 1; i >= 0; i-- {
		_ = scapes[i].Stop(ctx)
	}
}
