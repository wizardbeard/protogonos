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
	startCalls int
	stopCalls  int
	startErr   error
	stopErr    error
}

func (s *managedTestScape) Start(context.Context) error {
	s.startCalls++
	return s.startErr
}

func (s *managedTestScape) Stop(context.Context) error {
	s.stopCalls++
	return s.stopErr
}

type testSupportModule struct {
	name       string
	startCalls int
	stopCalls  int
	startErr   error
	stopErr    error
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

	p.Stop()
	if module.stopCalls != 1 {
		t.Fatalf("expected support module stop call, got=%d", module.stopCalls)
	}
	if public.stopCalls != 1 {
		t.Fatalf("expected public scape stop call, got=%d", public.stopCalls)
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
