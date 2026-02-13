package platform

import (
	"context"
	"testing"

	"protogonos/internal/scape"
	"protogonos/internal/storage"
)

type testScape struct{}

func (testScape) Name() string { return "noop" }

func (testScape) Evaluate(context.Context, scape.Agent) (scape.Fitness, scape.Trace, error) {
	return 0, scape.Trace{"status": "ok"}, nil
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
