package platform

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"protogonos/internal/model"
	"protogonos/internal/scape"
	"protogonos/internal/storage"
)

type testScape struct {
	name string
}

func (s testScape) Name() string {
	if s.name == "" {
		return "noop"
	}
	return s.name
}

func (s testScape) Evaluate(context.Context, scape.Agent) (scape.Fitness, scape.Trace, error) {
	return 0, scape.Trace{"status": "ok"}, nil
}

type managedTestScape struct {
	testScape
	startCalls            int
	startWithSummaryCalls int
	stopCalls             int
	startErr              error
	stopErr               error
	stopReason            StopReason
	lastStartWithSummary  PublicScapeSummary
}

func (s *managedTestScape) Start(context.Context) error {
	s.startCalls++
	return s.startErr
}

func (s *managedTestScape) StartWithSummary(ctx context.Context, summary PublicScapeSummary) error {
	s.startWithSummaryCalls++
	s.lastStartWithSummary = summary
	return s.Start(ctx)
}

func (s *managedTestScape) Stop(context.Context) error {
	s.stopCalls++
	return s.stopErr
}

func (s *managedTestScape) StopWithReason(ctx context.Context, reason StopReason) error {
	s.stopReason = reason
	return s.Stop(ctx)
}

type testSupportModule struct {
	name       string
	startCalls int
	stopCalls  int
	startErr   error
	stopErr    error
	stopReason StopReason
}

func (m *testSupportModule) Name() string { return m.name }

func (m *testSupportModule) Start(context.Context) error {
	m.startCalls++
	return m.startErr
}

func (m *testSupportModule) Stop(context.Context) error {
	m.stopCalls++
	return m.stopErr
}

func (m *testSupportModule) StopWithReason(ctx context.Context, reason StopReason) error {
	m.stopReason = reason
	return m.Stop(ctx)
}

type syncableTestSupportModule struct {
	testSupportModule
	syncCalls int
	syncErr   error
}

func (m *syncableTestSupportModule) Sync(context.Context) error {
	m.syncCalls++
	return m.syncErr
}

type syncableManagedTestScape struct {
	managedTestScape
	syncCalls int
	syncErr   error
}

func (s *syncableManagedTestScape) Sync(context.Context) error {
	s.syncCalls++
	return s.syncErr
}

type supervisedTestSupportModule struct {
	testSupportModule
	failRuns      int32
	superviseRuns atomic.Int32
}

func (m *supervisedTestSupportModule) Supervise(ctx context.Context) error {
	run := m.superviseRuns.Add(1)
	if run <= m.failRuns {
		return errors.New("supervise failure")
	}
	<-ctx.Done()
	return ctx.Err()
}

type supervisedManagedTestScape struct {
	managedTestScape
	failRuns      int32
	superviseRuns atomic.Int32
}

func (s *supervisedManagedTestScape) Supervise(ctx context.Context) error {
	run := s.superviseRuns.Add(1)
	if run <= s.failRuns {
		return errors.New("supervise failure")
	}
	<-ctx.Done()
	return ctx.Err()
}

type scriptedSupervisedSupportModule struct {
	testSupportModule
	superviseRuns atomic.Int32
	superviseFn   func(context.Context) error
}

func (m *scriptedSupervisedSupportModule) Supervise(ctx context.Context) error {
	m.superviseRuns.Add(1)
	if m.superviseFn != nil {
		return m.superviseFn(ctx)
	}
	<-ctx.Done()
	return ctx.Err()
}

type scriptedSupervisedManagedScape struct {
	managedTestScape
	superviseRuns atomic.Int32
	superviseFn   func(context.Context) error
}

func (s *scriptedSupervisedManagedScape) Supervise(ctx context.Context) error {
	s.superviseRuns.Add(1)
	if s.superviseFn != nil {
		return s.superviseFn(ctx)
	}
	<-ctx.Done()
	return ctx.Err()
}

func TestPolisInitAndRegisterScape(t *testing.T) {
	p := NewPolis(Config{Store: storage.NewMemoryStore()})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if !p.Started() {
		t.Fatal("polis should be started after init")
	}
	if err := p.RegisterScape(testScape{}); err != nil {
		t.Fatalf("register scape failed: %v", err)
	}
	if len(p.RegisteredScapes()) != 1 {
		t.Fatalf("expected 1 registered scape, got %d", len(p.RegisteredScapes()))
	}
	if _, ok := p.GetScape("noop"); !ok {
		t.Fatal("expected get scape to resolve registered scape")
	}
	if _, ok := p.GetScapeByType("noop"); ok {
		t.Fatal("expected type lookup to skip non-public registered scape")
	}
}

func TestPolisCreateAliasInit(t *testing.T) {
	p := NewPolis(Config{Store: storage.NewMemoryStore()})
	if err := p.Create(context.Background()); err != nil {
		t.Fatalf("create failed: %v", err)
	}
	if !p.Started() {
		t.Fatal("polis should be started after create")
	}
}

func TestPolisLifecycleStopAndReinit(t *testing.T) {
	p := NewPolis(Config{Store: storage.NewMemoryStore()})

	if err := p.RegisterScape(testScape{}); err == nil {
		t.Fatal("expected register scape to fail before init")
	}
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("second init should be idempotent: %v", err)
	}
	if err := p.RegisterScape(testScape{}); err != nil {
		t.Fatalf("register scape failed: %v", err)
	}
	if len(p.RegisteredScapes()) != 1 {
		t.Fatalf("expected 1 registered scape, got %d", len(p.RegisteredScapes()))
	}

	p.Stop()
	if p.Started() {
		t.Fatal("expected polis stopped after stop call")
	}
	if p.LastStopReason() != StopReasonNormal {
		t.Fatalf("expected stop reason %q, got=%q", StopReasonNormal, p.LastStopReason())
	}
	if len(p.RegisteredScapes()) != 0 {
		t.Fatalf("expected scapes cleared after stop, got %d", len(p.RegisteredScapes()))
	}

	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("re-init failed: %v", err)
	}
	if !p.Started() {
		t.Fatal("expected polis started after re-init")
	}
}

func TestPolisInitStartsConfiguredModulesAndPublicScapes(t *testing.T) {
	module := &testSupportModule{name: "metrics"}
	public := &managedTestScape{testScape: testScape{name: "public-xor"}}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{module},
		PublicScapes: []PublicScapeSpec{
			{
				Scape:      public,
				Type:       "flatland",
				Parameters: []any{"seeded"},
				Metabolics: "static",
				Physics:    "default",
			},
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if module.startCalls != 1 {
		t.Fatalf("expected support module start call, got=%d", module.startCalls)
	}
	if public.startCalls != 1 {
		t.Fatalf("expected public scape start call, got=%d", public.startCalls)
	}
	if public.startWithSummaryCalls != 1 {
		t.Fatalf("expected public scape start-with-summary call, got=%d", public.startWithSummaryCalls)
	}
	if public.lastStartWithSummary.Name != "public-xor" ||
		public.lastStartWithSummary.Type != "flatland" ||
		public.lastStartWithSummary.Metabolics != "static" ||
		public.lastStartWithSummary.Physics != "default" {
		t.Fatalf("unexpected public scape start summary: %+v", public.lastStartWithSummary)
	}
	if len(public.lastStartWithSummary.Parameters) != 1 || public.lastStartWithSummary.Parameters[0] != "seeded" {
		t.Fatalf("unexpected public scape start summary parameters: %+v", public.lastStartWithSummary.Parameters)
	}
	if len(p.ActiveSupportModules()) != 1 || p.ActiveSupportModules()[0] != "metrics" {
		t.Fatalf("unexpected active support modules: %+v", p.ActiveSupportModules())
	}
	summaries := p.ActivePublicScapes()
	if len(summaries) != 1 {
		t.Fatalf("expected one active public scape summary, got=%d", len(summaries))
	}
	if summaries[0].Name != "public-xor" || summaries[0].Type != "flatland" {
		t.Fatalf("unexpected public scape summary identity: %+v", summaries[0])
	}
	if summaries[0].Metabolics != "static" || summaries[0].Physics != "default" {
		t.Fatalf("unexpected public scape summary metadata: %+v", summaries[0])
	}
	if len(summaries[0].Parameters) != 1 || summaries[0].Parameters[0] != "seeded" {
		t.Fatalf("unexpected public scape summary parameters: %+v", summaries[0].Parameters)
	}
	gotByType, ok := p.GetScapeByType("flatland")
	if !ok {
		t.Fatal("expected scape lookup by type to resolve public scape")
	}
	if gotByType != public {
		t.Fatal("expected scape lookup by type to resolve configured public scape instance")
	}
	if _, ok := p.GetScapeByType("unknown"); ok {
		t.Fatal("expected unknown scape type lookup to return not found")
	}

	p.Stop()
	if module.stopCalls != 1 {
		t.Fatalf("expected support module stop call, got=%d", module.stopCalls)
	}
	if public.stopCalls != 1 {
		t.Fatalf("expected public scape stop call, got=%d", public.stopCalls)
	}
	if module.stopReason != StopReasonNormal {
		t.Fatalf("expected support module stop reason %q, got=%q", StopReasonNormal, module.stopReason)
	}
	if public.stopReason != StopReasonNormal {
		t.Fatalf("expected public scape stop reason %q, got=%q", StopReasonNormal, public.stopReason)
	}
	if len(p.ActiveSupportModules()) != 0 {
		t.Fatalf("expected cleared active support modules after stop, got=%+v", p.ActiveSupportModules())
	}
	if len(p.ActivePublicScapes()) != 0 {
		t.Fatalf("expected cleared active public scape summaries after stop, got=%+v", p.ActivePublicScapes())
	}
}

func TestPolisInitRollsBackOnSupportModuleStartFailure(t *testing.T) {
	okModule := &testSupportModule{name: "ok"}
	failModule := &testSupportModule{name: "bad", startErr: errors.New("boom")}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{okModule, failModule},
	})
	if err := p.Init(context.Background()); err == nil {
		t.Fatal("expected init failure from support module start error")
	}
	if p.Started() {
		t.Fatal("expected polis to remain stopped after failed init")
	}
	if okModule.startCalls != 1 || okModule.stopCalls != 1 {
		t.Fatalf("expected rollback stop for successfully started module, start=%d stop=%d", okModule.startCalls, okModule.stopCalls)
	}
	if failModule.startCalls != 1 {
		t.Fatalf("expected failing module start to be attempted once, got=%d", failModule.startCalls)
	}
	if len(p.ActiveSupportModules()) != 0 {
		t.Fatalf("expected no active support modules after rollback, got=%+v", p.ActiveSupportModules())
	}
	if len(p.RegisteredScapes()) != 0 {
		t.Fatalf("expected no registered scapes after rollback, got=%+v", p.RegisteredScapes())
	}
}

func TestPolisResetClearsStoreAndRestartsLifecycle(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	module := &testSupportModule{name: "metrics"}
	public := &managedTestScape{testScape: testScape{name: "public-xor"}}
	p := NewPolis(Config{
		Store:          store,
		SupportModules: []SupportModule{module},
		PublicScapes:   []PublicScapeSpec{{Scape: public, Type: "xor"}},
	})
	if err := p.Init(ctx); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	pop := model.Population{ID: "pop-1", AgentIDs: []string{"a1"}, Generation: 1}
	if err := store.SavePopulation(ctx, pop); err != nil {
		t.Fatalf("save population before reset: %v", err)
	}

	if err := p.Reset(ctx); err != nil {
		t.Fatalf("reset failed: %v", err)
	}
	if !p.Started() {
		t.Fatal("expected polis to be started after reset")
	}
	if module.startCalls != 2 || module.stopCalls != 1 {
		t.Fatalf("expected support module restart around reset, start=%d stop=%d", module.startCalls, module.stopCalls)
	}
	if public.startCalls != 2 || public.stopCalls != 1 {
		t.Fatalf("expected public scape restart around reset, start=%d stop=%d", public.startCalls, public.stopCalls)
	}
	if p.LastStopReason() != StopReasonShutdown {
		t.Fatalf("expected reset stop reason %q, got=%q", StopReasonShutdown, p.LastStopReason())
	}
	if module.stopReason != StopReasonShutdown {
		t.Fatalf("expected support module reset stop reason %q, got=%q", StopReasonShutdown, module.stopReason)
	}
	if public.stopReason != StopReasonShutdown {
		t.Fatalf("expected public scape reset stop reason %q, got=%q", StopReasonShutdown, public.stopReason)
	}
	if len(p.ActivePublicScapes()) != 1 || len(p.ActiveSupportModules()) != 1 {
		t.Fatalf("expected public scape and support module active after reset: scapes=%+v mods=%+v", p.ActivePublicScapes(), p.ActiveSupportModules())
	}
	_, ok, err := store.GetPopulation(ctx, pop.ID)
	if err != nil {
		t.Fatalf("get population after reset: %v", err)
	}
	if ok {
		t.Fatal("expected reset to clear persisted population data")
	}
}

func TestPolisAddAndRemoveSupportModule(t *testing.T) {
	ctx := context.Background()
	module := &testSupportModule{name: "dynamic-metrics"}
	p := NewPolis(Config{Store: storage.NewMemoryStore()})

	if err := p.AddSupportModule(ctx, module); err == nil {
		t.Fatal("expected add support module before init to fail")
	}
	if err := p.Init(ctx); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.AddSupportModule(ctx, module); err != nil {
		t.Fatalf("add support module: %v", err)
	}
	if module.startCalls != 1 {
		t.Fatalf("expected support module start call, got=%d", module.startCalls)
	}
	if len(p.ActiveSupportModules()) != 1 || p.ActiveSupportModules()[0] != "dynamic-metrics" {
		t.Fatalf("expected dynamic support module registration, got=%+v", p.ActiveSupportModules())
	}
	if err := p.AddSupportModule(ctx, module); err == nil {
		t.Fatal("expected duplicate support module add to fail")
	}
	if err := p.RemoveSupportModule(ctx, "dynamic-metrics", StopReasonShutdown); err != nil {
		t.Fatalf("remove support module: %v", err)
	}
	if module.stopCalls != 1 {
		t.Fatalf("expected support module stop call, got=%d", module.stopCalls)
	}
	if module.stopReason != StopReasonShutdown {
		t.Fatalf("expected support module stop reason %q, got=%q", StopReasonShutdown, module.stopReason)
	}
	if len(p.ActiveSupportModules()) != 0 {
		t.Fatalf("expected dynamic support module removal, got=%+v", p.ActiveSupportModules())
	}
	if err := p.RemoveSupportModule(ctx, "dynamic-metrics", StopReasonNormal); err == nil {
		t.Fatal("expected removing missing support module to fail")
	}
}

func TestPolisAddAndRemovePublicScape(t *testing.T) {
	ctx := context.Background()
	first := &managedTestScape{testScape: testScape{name: "public-a"}}
	second := &managedTestScape{testScape: testScape{name: "public-b"}}
	p := NewPolis(Config{Store: storage.NewMemoryStore()})

	if err := p.AddPublicScape(ctx, PublicScapeSpec{Scape: first, Type: "flatland"}); err == nil {
		t.Fatal("expected add public scape before init to fail")
	}
	if err := p.Init(ctx); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.AddPublicScape(ctx, PublicScapeSpec{Scape: first, Type: "flatland"}); err != nil {
		t.Fatalf("add first public scape: %v", err)
	}
	if first.startCalls != 1 {
		t.Fatalf("expected first public scape start call, got=%d", first.startCalls)
	}
	if first.startWithSummaryCalls != 1 {
		t.Fatalf("expected first public scape start-with-summary call, got=%d", first.startWithSummaryCalls)
	}
	if first.lastStartWithSummary.Type != "flatland" {
		t.Fatalf("expected first public scape start summary type flatland, got=%q", first.lastStartWithSummary.Type)
	}
	if _, ok := p.GetScape("public-a"); !ok {
		t.Fatal("expected first public scape to be registered by name")
	}
	gotByType, ok := p.GetScapeByType("flatland")
	if !ok || gotByType != first {
		t.Fatal("expected type lookup to resolve first public scape")
	}
	if err := p.AddPublicScape(ctx, PublicScapeSpec{Scape: second, Type: "flatland"}); err != nil {
		t.Fatalf("add second public scape: %v", err)
	}
	if second.startWithSummaryCalls != 1 {
		t.Fatalf("expected second public scape start-with-summary call, got=%d", second.startWithSummaryCalls)
	}
	gotByType, ok = p.GetScapeByType("flatland")
	if !ok || gotByType != first {
		t.Fatal("expected type lookup to stay pinned to first public scape")
	}
	if err := p.RemovePublicScape(ctx, "public-a", StopReasonShutdown); err != nil {
		t.Fatalf("remove first public scape: %v", err)
	}
	if first.stopCalls != 1 || first.stopReason != StopReasonShutdown {
		t.Fatalf("expected first public scape shutdown stop semantics, calls=%d reason=%q", first.stopCalls, first.stopReason)
	}
	if _, ok := p.GetScape("public-a"); ok {
		t.Fatal("expected first public scape name lookup removed")
	}
	gotByType, ok = p.GetScapeByType("flatland")
	if !ok || gotByType != second {
		t.Fatal("expected type lookup to remap to remaining public scape")
	}
	if err := p.RemovePublicScape(ctx, "public-b", ""); err != nil {
		t.Fatalf("remove second public scape: %v", err)
	}
	if second.stopCalls != 1 || second.stopReason != StopReasonNormal {
		t.Fatalf("expected second public scape normal stop semantics, calls=%d reason=%q", second.stopCalls, second.stopReason)
	}
	if len(p.ActivePublicScapes()) != 0 {
		t.Fatalf("expected no active public scapes after removals, got=%+v", p.ActivePublicScapes())
	}
	if err := p.RemovePublicScape(ctx, "public-b", StopReasonNormal); err == nil {
		t.Fatal("expected removing missing public scape to fail")
	}
}

func TestPolisAddPublicScapeNormalizesNameAliases(t *testing.T) {
	ctx := context.Background()
	aliased := &managedTestScape{testScape: testScape{name: "scape_GTSA"}}
	p := NewPolis(Config{Store: storage.NewMemoryStore()})
	if err := p.Init(ctx); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.AddPublicScape(ctx, PublicScapeSpec{Scape: aliased, Type: "timeseries"}); err != nil {
		t.Fatalf("add aliased public scape: %v", err)
	}
	if _, ok := p.GetScape("gtsa"); !ok {
		t.Fatal("expected canonical gtsa lookup to resolve aliased public scape")
	}
	if _, ok := p.GetScape("scape_GTSA"); !ok {
		t.Fatal("expected alias gtsa lookup to resolve aliased public scape")
	}
	if err := p.RemovePublicScape(ctx, "scape_GTSA", StopReasonNormal); err != nil {
		t.Fatalf("remove aliased public scape by alias: %v", err)
	}
	if aliased.stopCalls != 1 || aliased.stopReason != StopReasonNormal {
		t.Fatalf("expected aliased scape normal stop semantics, calls=%d reason=%q", aliased.stopCalls, aliased.stopReason)
	}
}

func TestPolisInitNormalizesConfiguredPublicScapeAliases(t *testing.T) {
	aliased := &managedTestScape{testScape: testScape{name: "scape_LLVMPhaseOrdering"}}
	p := NewPolis(Config{
		Store: storage.NewMemoryStore(),
		PublicScapes: []PublicScapeSpec{
			{Scape: aliased, Type: "llvm"},
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if _, ok := p.GetScape("llvm-phase-ordering"); !ok {
		t.Fatal("expected canonical llvm-phase-ordering lookup to resolve configured aliased scape")
	}
	if _, ok := p.GetScape("scape_LLVMPhaseOrdering"); !ok {
		t.Fatal("expected alias llvm lookup to resolve configured aliased scape")
	}
	summaries := p.ActivePublicScapes()
	if len(summaries) != 1 {
		t.Fatalf("expected one active public scape, got=%d", len(summaries))
	}
	if summaries[0].Name != "llvm-phase-ordering" {
		t.Fatalf("expected normalized summary name llvm-phase-ordering, got=%q", summaries[0].Name)
	}
}

func TestPolisStopWithReasonRejectsInvalidReason(t *testing.T) {
	p := NewPolis(Config{Store: storage.NewMemoryStore()})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.StopWithReason(StopReason("bad")); err == nil {
		t.Fatal("expected invalid stop reason to fail")
	}
	if !p.Started() {
		t.Fatal("expected polis to remain started after invalid stop reason")
	}
}

func TestStartDefaultReusesRunningPolis(t *testing.T) {
	resetDefaultPolisForTest()
	t.Cleanup(resetDefaultPolisForTest)

	ctx := context.Background()
	first, err := StartDefault(ctx, Config{Store: storage.NewMemoryStore()})
	if err != nil {
		t.Fatalf("start default first: %v", err)
	}
	second, err := StartDefault(ctx, Config{Store: storage.NewMemoryStore()})
	if err != nil {
		t.Fatalf("start default second: %v", err)
	}
	if first != second {
		t.Fatal("expected second start to reuse running default polis")
	}
	if _, ok := Default(); !ok {
		t.Fatal("expected default polis to be discoverable while running")
	}
	if err := StopDefault(StopReasonNormal); err != nil {
		t.Fatalf("stop default: %v", err)
	}
	if first.Started() {
		t.Fatal("expected default polis instance to be stopped")
	}
	if first.LastStopReason() != StopReasonNormal {
		t.Fatalf("expected default stop reason %q, got=%q", StopReasonNormal, first.LastStopReason())
	}
	if _, ok := Default(); ok {
		t.Fatal("expected no default polis after stop")
	}

	third, err := StartDefault(ctx, Config{Store: storage.NewMemoryStore()})
	if err != nil {
		t.Fatalf("start default third: %v", err)
	}
	if third == first {
		t.Fatal("expected restarted default polis to allocate a new instance")
	}
}

func TestStopDefaultRejectsInvalidReason(t *testing.T) {
	resetDefaultPolisForTest()
	t.Cleanup(resetDefaultPolisForTest)

	ctx := context.Background()
	if _, err := StartDefault(ctx, Config{Store: storage.NewMemoryStore()}); err != nil {
		t.Fatalf("start default: %v", err)
	}
	if err := StopDefault(StopReason("bad")); err == nil {
		t.Fatal("expected invalid default stop reason to fail")
	}
	if _, ok := Default(); !ok {
		t.Fatal("expected default polis to remain available after invalid stop reason")
	}
	if err := StopDefault(StopReasonShutdown); err != nil {
		t.Fatalf("stop default shutdown: %v", err)
	}
}

func TestPolisGetScapeByTypeUsesFirstConfiguredType(t *testing.T) {
	first := &managedTestScape{testScape: testScape{name: "scape-a"}}
	second := &managedTestScape{testScape: testScape{name: "scape-b"}}
	p := NewPolis(Config{
		Store: storage.NewMemoryStore(),
		PublicScapes: []PublicScapeSpec{
			{Scape: first, Type: "shared"},
			{Scape: second, Type: "shared"},
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	got, ok := p.GetScapeByType("shared")
	if !ok {
		t.Fatal("expected shared type lookup to resolve a scape")
	}
	if got != first {
		t.Fatal("expected shared type lookup to resolve first configured scape")
	}
}

func TestPolisGetScapeByTypeKeepsInsertionOrderAfterRemoval(t *testing.T) {
	ctx := context.Background()
	first := &managedTestScape{testScape: testScape{name: "scape-c"}}
	second := &managedTestScape{testScape: testScape{name: "scape-b"}}
	third := &managedTestScape{testScape: testScape{name: "scape-a"}}
	p := NewPolis(Config{Store: storage.NewMemoryStore()})
	if err := p.Init(ctx); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.AddPublicScape(ctx, PublicScapeSpec{Scape: first, Type: "shared"}); err != nil {
		t.Fatalf("add first public scape: %v", err)
	}
	if err := p.AddPublicScape(ctx, PublicScapeSpec{Scape: second, Type: "shared"}); err != nil {
		t.Fatalf("add second public scape: %v", err)
	}
	if err := p.AddPublicScape(ctx, PublicScapeSpec{Scape: third, Type: "shared"}); err != nil {
		t.Fatalf("add third public scape: %v", err)
	}

	got, ok := p.GetScapeByType("shared")
	if !ok || got != first {
		t.Fatal("expected first-insertion scape lookup before removal")
	}
	if err := p.RemovePublicScape(ctx, "scape-c", StopReasonNormal); err != nil {
		t.Fatalf("remove first-insertion public scape: %v", err)
	}
	got, ok = p.GetScapeByType("shared")
	if !ok || got != second {
		t.Fatal("expected type lookup to advance to next insertion-order scape after removal")
	}
}

func TestPolisCallGetScapeByType(t *testing.T) {
	ctx := context.Background()
	public := &managedTestScape{testScape: testScape{name: "call-scape"}}
	p := NewPolis(Config{
		Store:        storage.NewMemoryStore(),
		PublicScapes: []PublicScapeSpec{{Scape: public, Type: "call-type"}},
	})
	if _, err := p.Call(ctx, GetScapeCall{Type: "call-type"}); err == nil {
		t.Fatal("expected call get_scape before init to fail")
	}
	if err := p.Init(ctx); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	out, err := p.Call(ctx, GetScapeCall{Type: "call-type"})
	if err != nil {
		t.Fatalf("call get_scape: %v", err)
	}
	result, ok := out.(GetScapeCallResult)
	if !ok {
		t.Fatalf("expected GetScapeCallResult, got %T", out)
	}
	if !result.Found || result.Scape != public {
		t.Fatalf("unexpected call get_scape result: %+v", result)
	}
}

func TestPolisCallAndCastStopReason(t *testing.T) {
	ctx := context.Background()
	module := &testSupportModule{name: "call-stop-module"}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{module},
	})
	if err := p.Cast(ctx, InitCast{}); err != nil {
		t.Fatalf("cast init: %v", err)
	}
	if !p.MailboxActive() {
		t.Fatal("expected mailbox active after init cast")
	}
	if _, err := p.Call(ctx, StopCall{Reason: StopReasonShutdown}); err != nil {
		t.Fatalf("call stop: %v", err)
	}
	if p.Started() {
		t.Fatal("expected polis stopped by stop call")
	}
	waitForMailboxState(t, p, false, 100*time.Millisecond)
	if p.LastStopReason() != StopReasonShutdown {
		t.Fatalf("expected call stop reason %q, got=%q", StopReasonShutdown, p.LastStopReason())
	}
	if module.stopReason != StopReasonShutdown {
		t.Fatalf("expected module stop reason %q, got=%q", StopReasonShutdown, module.stopReason)
	}

	if err := p.Cast(ctx, InitCast{}); err != nil {
		t.Fatalf("cast re-init: %v", err)
	}
	if !p.MailboxActive() {
		t.Fatal("expected mailbox active after cast re-init")
	}
	if err := p.Cast(ctx, StopCast{}); err != nil {
		t.Fatalf("cast stop default: %v", err)
	}
	if p.LastStopReason() != StopReasonNormal {
		t.Fatalf("expected cast default stop reason %q, got=%q", StopReasonNormal, p.LastStopReason())
	}
	waitForMailboxState(t, p, false, 100*time.Millisecond)
}

func TestPolisCastInitWithStateReplacesRuntime(t *testing.T) {
	ctx := context.Background()
	oldModule := &testSupportModule{name: "old-module"}
	oldScape := &managedTestScape{testScape: testScape{name: "old-scape"}}
	newModule := &testSupportModule{name: "new-module"}
	newScape := &managedTestScape{testScape: testScape{name: "new-scape"}}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{oldModule},
		PublicScapes:   []PublicScapeSpec{{Scape: oldScape, Type: "old-type"}},
	})
	if err := p.Init(ctx); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.Cast(ctx, InitCast{
		State: &PolisInitState{
			SupportModules: []SupportModule{newModule},
			PublicScapes:   []PublicScapeSpec{{Scape: newScape, Type: "new-type"}},
		},
	}); err != nil {
		t.Fatalf("cast init with state: %v", err)
	}
	if oldModule.stopCalls != 1 || oldScape.stopCalls != 1 {
		t.Fatalf("expected old runtime stop during init-with-state, module_stop=%d scape_stop=%d", oldModule.stopCalls, oldScape.stopCalls)
	}
	if newModule.startCalls != 1 || newScape.startCalls != 1 {
		t.Fatalf("expected new runtime start during init-with-state, module_start=%d scape_start=%d", newModule.startCalls, newScape.startCalls)
	}
	mods := p.ActiveSupportModules()
	if len(mods) != 1 || mods[0] != "new-module" {
		t.Fatalf("expected only new support module active, got=%v", mods)
	}
	if _, ok := p.GetScape("old-scape"); ok {
		t.Fatal("expected old public scape to be removed after init-with-state")
	}
	if _, ok := p.GetScapeByType("old-type"); ok {
		t.Fatal("expected old public scape type lookup to be removed after init-with-state")
	}
	got, ok := p.GetScapeByType("new-type")
	if !ok || got != newScape {
		t.Fatalf("expected new public scape type lookup after init-with-state, got=%v found=%t", got, ok)
	}
	if !p.MailboxActive() {
		t.Fatal("expected mailbox to remain active after init-with-state cast")
	}
}

func TestPolisCastInitWithStateBootstrapsWhenStopped(t *testing.T) {
	ctx := context.Background()
	module := &testSupportModule{name: "boot-module"}
	public := &managedTestScape{testScape: testScape{name: "boot-scape"}}
	p := NewPolis(Config{Store: storage.NewMemoryStore()})

	if err := p.Cast(ctx, InitCast{
		State: &PolisInitState{
			SupportModules: []SupportModule{module},
			PublicScapes:   []PublicScapeSpec{{Scape: public, Type: "boot-type"}},
		},
	}); err != nil {
		t.Fatalf("cast init-with-state before init: %v", err)
	}
	if !p.Started() {
		t.Fatal("expected polis started by init-with-state cast")
	}
	if module.startCalls != 1 || public.startCalls != 1 {
		t.Fatalf("expected init-with-state to start provided runtime, module_start=%d scape_start=%d", module.startCalls, public.startCalls)
	}
	if _, ok := p.GetScapeByType("boot-type"); !ok {
		t.Fatal("expected public scape lookup by cast-provided type")
	}
}

func TestPolisCallCastRejectUnsupportedMessage(t *testing.T) {
	p := NewPolis(Config{Store: storage.NewMemoryStore()})
	if _, err := p.Call(context.Background(), nil); err == nil {
		t.Fatal("expected nil call message to fail")
	}
	if err := p.Cast(context.Background(), nil); err == nil {
		t.Fatal("expected nil cast message to fail")
	}
}

func TestPolisStartWithSummaryDefaultsTypeToScapeName(t *testing.T) {
	public := &managedTestScape{testScape: testScape{name: "fallback-type"}}
	p := NewPolis(Config{
		Store:        storage.NewMemoryStore(),
		PublicScapes: []PublicScapeSpec{{Scape: public}},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if public.startWithSummaryCalls != 1 {
		t.Fatalf("expected start-with-summary call, got=%d", public.startWithSummaryCalls)
	}
	if public.lastStartWithSummary.Type != "fallback-type" {
		t.Fatalf("expected defaulted start-summary type fallback-type, got=%q", public.lastStartWithSummary.Type)
	}
}

func TestPolisSupervisesConfiguredSupportModuleRuntime(t *testing.T) {
	module := &supervisedTestSupportModule{
		testSupportModule: testSupportModule{name: "supervised-metrics"},
		failRuns:          1,
	}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{module},
		SupervisorPolicy: SupervisorPolicy{
			InitialBackoff: time.Millisecond,
			MaxBackoff:     time.Millisecond,
			BackoffFactor:  1,
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	waitForAtLeastRuns(t, &module.superviseRuns, 2, 250*time.Millisecond)
	if len(p.ActiveSupervisedTasks()) == 0 {
		t.Fatal("expected active supervised task for support module")
	}
	children := p.ActiveSupervisedChildren()
	if len(children) == 0 {
		t.Fatal("expected active supervised child metadata")
	}
	if children[0].Group != "support" || children[0].RestartPolicy != SupervisorRestartPermanent {
		t.Fatalf("unexpected supervised child metadata: %+v", children[0])
	}
	p.Stop()
	if len(p.ActiveSupervisedTasks()) != 0 {
		t.Fatalf("expected no supervised tasks after stop, got=%v", p.ActiveSupervisedTasks())
	}
}

func TestPolisSupervisesPublicScapeRuntime(t *testing.T) {
	public := &supervisedManagedTestScape{
		managedTestScape: managedTestScape{testScape: testScape{name: "supervised-scape"}},
		failRuns:         1,
	}
	p := NewPolis(Config{
		Store: storage.NewMemoryStore(),
		PublicScapes: []PublicScapeSpec{
			{Scape: public, Type: "supervised-type"},
		},
		SupervisorPolicy: SupervisorPolicy{
			InitialBackoff: time.Millisecond,
			MaxBackoff:     time.Millisecond,
			BackoffFactor:  1,
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	waitForAtLeastRuns(t, &public.superviseRuns, 2, 250*time.Millisecond)
	if len(p.ActiveSupervisedTasks()) == 0 {
		t.Fatal("expected active supervised task for public scape")
	}
	children := p.ActiveSupervisedChildren()
	if len(children) == 0 {
		t.Fatal("expected active supervised child metadata")
	}
	if children[0].Group != "scape" || children[0].RestartPolicy != SupervisorRestartPermanent {
		t.Fatalf("unexpected supervised child metadata: %+v", children[0])
	}
	if err := p.RemovePublicScape(context.Background(), "supervised-scape", StopReasonNormal); err != nil {
		t.Fatalf("remove public scape: %v", err)
	}
	if len(p.ActiveSupervisedTasks()) != 0 {
		t.Fatalf("expected no supervised tasks after public scape removal, got=%v", p.ActiveSupervisedTasks())
	}
}

func TestPolisEscalatesOnSupervisorPermanentFailure(t *testing.T) {
	module := &supervisedTestSupportModule{
		testSupportModule: testSupportModule{name: "fatal-module"},
		failRuns:          1000,
	}
	p := NewPolis(Config{
		Store:                       storage.NewMemoryStore(),
		SupportModules:              []SupportModule{module},
		EscalateOnSupervisorFailure: true,
		SupervisorPolicy: SupervisorPolicy{
			InitialBackoff: time.Millisecond,
			MaxBackoff:     time.Millisecond,
			BackoffFactor:  1,
			MaxRestarts:    1,
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	waitForPolisStarted(t, p, false, 300*time.Millisecond)
	if p.LastStopReason() != StopReasonShutdown {
		t.Fatalf("expected escalation stop reason %q, got=%q", StopReasonShutdown, p.LastStopReason())
	}
	failures := p.SupervisionFailures()
	if len(failures) == 0 {
		t.Fatal("expected supervision failure history")
	}
	last := failures[len(failures)-1]
	if last.TaskName != supervisedSupportTaskName("fatal-module") {
		t.Fatalf("unexpected supervision failure task name: %s", last.TaskName)
	}
	if last.RestartCount != 1 {
		t.Fatalf("expected supervision restart count=1, got=%d", last.RestartCount)
	}
}

func TestPolisRecordsSupervisorPermanentFailureWithoutEscalation(t *testing.T) {
	module := &supervisedTestSupportModule{
		testSupportModule: testSupportModule{name: "nonfatal-module"},
		failRuns:          1000,
	}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{module},
		SupervisorPolicy: SupervisorPolicy{
			InitialBackoff: time.Millisecond,
			MaxBackoff:     time.Millisecond,
			BackoffFactor:  1,
			MaxRestarts:    1,
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	waitForAtLeastRuns(t, &module.superviseRuns, 2, 300*time.Millisecond)
	time.Sleep(20 * time.Millisecond)
	if !p.Started() {
		t.Fatal("expected polis to remain started without escalation")
	}
	if len(p.ActiveSupervisedTasks()) != 0 {
		t.Fatalf("expected failed supervised task to be removed, got=%v", p.ActiveSupervisedTasks())
	}
	failures := p.SupervisionFailures()
	if len(failures) == 0 {
		t.Fatal("expected supervision failure history")
	}
	last := failures[len(failures)-1]
	if last.TaskName != supervisedSupportTaskName("nonfatal-module") {
		t.Fatalf("unexpected supervision failure task name: %s", last.TaskName)
	}
}

func TestPolisOneForAllSupervisorStrategyStopsSiblingTasks(t *testing.T) {
	stable := &supervisedTestSupportModule{
		testSupportModule: testSupportModule{name: "stable-module"},
	}
	failing := &supervisedTestSupportModule{
		testSupportModule: testSupportModule{name: "failing-module"},
		failRuns:          1000,
	}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{stable, failing},
		SupervisorPolicy: SupervisorPolicy{
			InitialBackoff: time.Millisecond,
			MaxBackoff:     time.Millisecond,
			BackoffFactor:  1,
			MaxRestarts:    1,
			Strategy:       SupervisorStrategyOneForAll,
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	waitForActiveSupervisedTaskCount(t, p, 0, 300*time.Millisecond)
	if !p.Started() {
		t.Fatal("expected polis to remain started without escalation")
	}
	if stable.superviseRuns.Load() == 0 {
		t.Fatal("expected stable sibling module to have started before one_for_all stop")
	}
	failures := p.SupervisionFailures()
	if len(failures) == 0 {
		t.Fatal("expected supervision failure history")
	}
	last := failures[len(failures)-1]
	if last.TaskName != supervisedSupportTaskName("failing-module") {
		t.Fatalf("unexpected supervision failure task name: %s", last.TaskName)
	}
}

func TestPolisOneForAllSupervisorStrategyRestartsSiblingTasks(t *testing.T) {
	stable := &supervisedTestSupportModule{
		testSupportModule: testSupportModule{name: "stable-module"},
	}
	failing := &supervisedTestSupportModule{
		testSupportModule: testSupportModule{name: "failing-once-module"},
		failRuns:          1,
	}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{stable, failing},
		SupervisorPolicy: SupervisorPolicy{
			InitialBackoff: time.Millisecond,
			MaxBackoff:     time.Millisecond,
			BackoffFactor:  1,
			MaxRestarts:    2,
			Strategy:       SupervisorStrategyOneForAll,
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	defer p.Stop()

	waitForAtLeastRuns(t, &stable.superviseRuns, 2, 300*time.Millisecond)
	waitForAtLeastRuns(t, &failing.superviseRuns, 2, 300*time.Millisecond)

	active := p.ActiveSupervisedTasks()
	if len(active) != 2 {
		t.Fatalf("expected both modules supervised after one_for_all restart, got=%v", active)
	}
	expected := map[string]struct{}{
		supervisedSupportTaskName("stable-module"):       {},
		supervisedSupportTaskName("failing-once-module"): {},
	}
	for _, name := range active {
		delete(expected, name)
	}
	if len(expected) != 0 {
		t.Fatalf("expected supervised task names missing after restart: missing=%v active=%v", expected, active)
	}
}

func TestPolisSupportModuleTemporaryPolicyNoRestartOnError(t *testing.T) {
	module := &scriptedSupervisedSupportModule{
		testSupportModule: testSupportModule{name: "temporary-module"},
		superviseFn: func(context.Context) error {
			return errors.New("boom")
		},
	}
	p := NewPolis(Config{
		Store: storage.NewMemoryStore(),
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.AddSupportModuleWithPolicy(context.Background(), module, SupervisorRestartTemporary); err != nil {
		t.Fatalf("add support module with temporary policy: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
	if module.superviseRuns.Load() != 1 {
		t.Fatalf("expected temporary support module supervise runs=1, got=%d", module.superviseRuns.Load())
	}
	if len(p.ActiveSupervisedTasks()) != 0 {
		t.Fatalf("expected temporary support module not supervised after error exit, got=%v", p.ActiveSupervisedTasks())
	}
	if !p.Started() {
		t.Fatal("expected polis to remain started")
	}
	if len(p.SupervisionFailures()) != 0 {
		t.Fatalf("expected no supervision failures for temporary policy, got=%+v", p.SupervisionFailures())
	}
}

func TestPolisPublicScapeTransientPolicyNoRestartOnNormalExit(t *testing.T) {
	public := &scriptedSupervisedManagedScape{
		managedTestScape: managedTestScape{testScape: testScape{name: "transient-scape"}},
		superviseFn: func(context.Context) error {
			return nil
		},
	}
	p := NewPolis(Config{
		Store: storage.NewMemoryStore(),
		PublicScapes: []PublicScapeSpec{
			{
				Scape:         public,
				Type:          "transient-type",
				RestartPolicy: SupervisorRestartTransient,
			},
		},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
	if public.superviseRuns.Load() != 1 {
		t.Fatalf("expected transient public scape supervise runs=1, got=%d", public.superviseRuns.Load())
	}
	if len(p.ActiveSupervisedTasks()) != 0 {
		t.Fatalf("expected no active supervised tasks for transient normal exit, got=%v", p.ActiveSupervisedTasks())
	}
	if _, ok := p.GetScapeByType("transient-type"); !ok {
		t.Fatal("expected transient public scape registration to remain available")
	}
	if len(p.SupervisionFailures()) != 0 {
		t.Fatalf("expected no supervision failures for transient normal exit, got=%+v", p.SupervisionFailures())
	}
}

func TestPolisSyncInvokesSyncableRuntime(t *testing.T) {
	module := &syncableTestSupportModule{
		testSupportModule: testSupportModule{name: "sync-module"},
	}
	public := &syncableManagedTestScape{
		managedTestScape: managedTestScape{testScape: testScape{name: "sync-scape"}},
	}
	p := NewPolis(Config{
		Store:          storage.NewMemoryStore(),
		SupportModules: []SupportModule{module},
		PublicScapes:   []PublicScapeSpec{{Scape: public, Type: "sync-type"}},
	})
	if err := p.Init(context.Background()); err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := p.Sync(context.Background()); err != nil {
		t.Fatalf("sync failed: %v", err)
	}
	if module.syncCalls != 1 {
		t.Fatalf("expected one sync call for support module, got=%d", module.syncCalls)
	}
	if public.syncCalls != 1 {
		t.Fatalf("expected one sync call for public scape, got=%d", public.syncCalls)
	}
}

func TestPolisSyncRejectsUninitialized(t *testing.T) {
	p := NewPolis(Config{Store: storage.NewMemoryStore()})
	if err := p.Sync(context.Background()); err == nil {
		t.Fatal("expected sync before init to fail")
	}
}

func TestSyncDefaultRequiresRunningPolis(t *testing.T) {
	resetDefaultPolisForTest()
	t.Cleanup(resetDefaultPolisForTest)

	if err := SyncDefault(context.Background()); err == nil {
		t.Fatal("expected sync default without running polis to fail")
	}
	if _, err := StartDefault(context.Background(), Config{Store: storage.NewMemoryStore()}); err != nil {
		t.Fatalf("start default: %v", err)
	}
	if err := SyncDefault(context.Background()); err != nil {
		t.Fatalf("sync default: %v", err)
	}
}

func waitForAtLeastRuns(t *testing.T, counter *atomic.Int32, min int32, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if counter.Load() >= min {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("expected counter >= %d, got=%d", min, counter.Load())
}

func waitForPolisStarted(t *testing.T, p *Polis, started bool, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p.Started() == started {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("expected polis started=%t, got=%t", started, p.Started())
}

func waitForMailboxState(t *testing.T, p *Polis, active bool, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p.MailboxActive() == active {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("expected mailbox active=%t, got=%t", active, p.MailboxActive())
}

func waitForActiveSupervisedTaskCount(t *testing.T, p *Polis, want int, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if len(p.ActiveSupervisedTasks()) == want {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("expected active supervised task count=%d, got=%d (%v)", want, len(p.ActiveSupervisedTasks()), p.ActiveSupervisedTasks())
}

func resetDefaultPolisForTest() {
	defaultPolisMu.Lock()
	p := defaultPolis
	defaultPolis = nil
	defaultPolisMu.Unlock()
	if p != nil {
		p.Stop()
	}
}
