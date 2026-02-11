package tuning

import (
	"testing"

	"protogonos/internal/model"
)

func TestFixedAttemptPolicy(t *testing.T) {
	p := FixedAttemptPolicy{}
	if got := p.Attempts(4, 1, 10, model.Genome{}); got != 4 {
		t.Fatalf("expected fixed attempts=4, got=%d", got)
	}
}

func TestLinearDecayAttemptPolicy(t *testing.T) {
	p := LinearDecayAttemptPolicy{MinAttempts: 1}
	if got := p.Attempts(4, 0, 4, model.Genome{}); got != 4 {
		t.Fatalf("expected gen0 attempts=4, got=%d", got)
	}
	if got := p.Attempts(4, 2, 4, model.Genome{}); got != 2 {
		t.Fatalf("expected gen2 attempts=2, got=%d", got)
	}
	if got := p.Attempts(4, 9, 4, model.Genome{}); got != 1 {
		t.Fatalf("expected clamped attempts=1, got=%d", got)
	}
}

func TestTopologyScaledAttemptPolicy(t *testing.T) {
	p := TopologyScaledAttemptPolicy{Scale: 1.0, MinAttempts: 1}
	genome := model.Genome{
		Synapses: make([]model.Synapse, 10),
	}
	if got := p.Attempts(4, 0, 1, genome); got != 8 {
		t.Fatalf("expected scaled attempts=8, got=%d", got)
	}
}

func TestAttemptPolicyFromConfig(t *testing.T) {
	if _, err := AttemptPolicyFromConfig("fixed", 0); err != nil {
		t.Fatalf("fixed policy: %v", err)
	}
	if _, err := AttemptPolicyFromConfig("linear_decay", 2); err != nil {
		t.Fatalf("linear_decay policy: %v", err)
	}
	if _, err := AttemptPolicyFromConfig("topology_scaled", 1.2); err != nil {
		t.Fatalf("topology_scaled policy: %v", err)
	}
	if _, err := AttemptPolicyFromConfig("unknown", 1); err == nil {
		t.Fatal("expected unknown policy error")
	}
}
