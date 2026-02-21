package platform

import (
	"context"
	"errors"
	"testing"

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

func resetDefaultPolisForTest() {
	defaultPolisMu.Lock()
	p := defaultPolis
	defaultPolis = nil
	defaultPolisMu.Unlock()
	if p != nil {
		p.Stop()
	}
}
