package nn

import (
	"testing"

	"protogonos/internal/model"
)

func TestApplyPlasticityHebbian(t *testing.T) {
	g := model.Genome{
		Synapses: []model.Synapse{
			{ID: "s1", From: "a", To: "b", Weight: 1.0, Enabled: true},
		},
	}
	values := map[string]float64{"a": 2, "b": 3}

	err := ApplyPlasticity(&g, values, model.PlasticityConfig{
		Rule: PlasticityHebbian,
		Rate: 0.1,
	})
	if err != nil {
		t.Fatalf("apply hebbian: %v", err)
	}
	// w += rate * pre * post = 1 + 0.1*2*3 = 1.6
	if g.Synapses[0].Weight != 1.6 {
		t.Fatalf("unexpected weight after hebbian: %f", g.Synapses[0].Weight)
	}
}

func TestApplyPlasticityOja(t *testing.T) {
	g := model.Genome{
		Synapses: []model.Synapse{
			{ID: "s1", From: "a", To: "b", Weight: 0.5, Enabled: true},
		},
	}
	values := map[string]float64{"a": 1, "b": 2}

	err := ApplyPlasticity(&g, values, model.PlasticityConfig{
		Rule: PlasticityOja,
		Rate: 0.1,
	})
	if err != nil {
		t.Fatalf("apply oja: %v", err)
	}
	// w += rate * post * (pre - post*w) = 0.5 + 0.1*2*(1 - 2*0.5) = 0.5
	if g.Synapses[0].Weight != 0.5 {
		t.Fatalf("unexpected weight after oja: %f", g.Synapses[0].Weight)
	}
}

func TestApplyPlasticityValidation(t *testing.T) {
	g := model.Genome{}
	err := ApplyPlasticity(&g, map[string]float64{}, model.PlasticityConfig{
		Rule: "bad",
		Rate: 0.1,
	})
	if err == nil {
		t.Fatal("expected unsupported rule error")
	}
}

func TestApplyPlasticityAcceptsReferenceRuleAliases(t *testing.T) {
	g1 := model.Genome{
		Synapses: []model.Synapse{
			{ID: "s1", From: "a", To: "b", Weight: 1.0, Enabled: true},
		},
	}
	values1 := map[string]float64{"a": 2, "b": 3}
	err := ApplyPlasticity(&g1, values1, model.PlasticityConfig{
		Rule: "hebbian_w",
		Rate: 0.1,
	})
	if err != nil {
		t.Fatalf("apply hebbian_w: %v", err)
	}
	if g1.Synapses[0].Weight != 1.6 {
		t.Fatalf("unexpected weight after hebbian_w: %f", g1.Synapses[0].Weight)
	}

	g2 := model.Genome{
		Synapses: []model.Synapse{
			{ID: "s1", From: "a", To: "b", Weight: 0.5, Enabled: true},
		},
	}
	values2 := map[string]float64{"a": 1, "b": 2}
	err = ApplyPlasticity(&g2, values2, model.PlasticityConfig{
		Rule: "ojas_w",
		Rate: 0.1,
	})
	if err != nil {
		t.Fatalf("apply ojas_w: %v", err)
	}
	if g2.Synapses[0].Weight != 0.5 {
		t.Fatalf("unexpected weight after ojas_w: %f", g2.Synapses[0].Weight)
	}
}

func TestApplyPlasticityUsesPerNeuronRuleAndRateOverrides(t *testing.T) {
	g := model.Genome{
		Neurons: []model.Neuron{
			{ID: "h-hebb", PlasticityRule: PlasticityHebbian, PlasticityRate: 0.1},
			{ID: "h-oja", PlasticityRule: PlasticityOja, PlasticityRate: 0.2},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "in", To: "h-hebb", Weight: 1.0, Enabled: true},
			{ID: "s2", From: "in", To: "h-oja", Weight: 0.5, Enabled: true},
		},
	}
	values := map[string]float64{"in": 1.0, "h-hebb": 2.0, "h-oja": 2.0}

	err := ApplyPlasticity(&g, values, model.PlasticityConfig{
		Rule: PlasticityHebbian,
		Rate: 0.01,
	})
	if err != nil {
		t.Fatalf("apply plasticity: %v", err)
	}
	// s1 (hebbian, 0.1): 1.0 + 0.1*1*2 = 1.2
	if g.Synapses[0].Weight != 1.2 {
		t.Fatalf("unexpected weight for hebbian override: %f", g.Synapses[0].Weight)
	}
	// s2 (oja, 0.2): 0.5 + 0.2*2*(1 - 2*0.5) = 0.5
	if g.Synapses[1].Weight != 0.5 {
		t.Fatalf("unexpected weight for oja override: %f", g.Synapses[1].Weight)
	}
}

func TestApplyPlasticityFallsBackToGenomeRuleWhenNeuronRuleMissing(t *testing.T) {
	g := model.Genome{
		Neurons: []model.Neuron{
			{ID: "h"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "in", To: "h", Weight: 1.0, Enabled: true},
		},
	}
	values := map[string]float64{"in": 2.0, "h": 3.0}

	err := ApplyPlasticity(&g, values, model.PlasticityConfig{
		Rule: PlasticityHebbian,
		Rate: 0.1,
	})
	if err != nil {
		t.Fatalf("apply plasticity: %v", err)
	}
	if g.Synapses[0].Weight != 1.6 {
		t.Fatalf("unexpected weight with genome fallback rule/rate: %f", g.Synapses[0].Weight)
	}
}

func TestApplyPlasticityUsesNeuronRateWithGenomeRuleFallback(t *testing.T) {
	g := model.Genome{
		Neurons: []model.Neuron{
			{ID: "h", PlasticityRate: 0.2},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "in", To: "h", Weight: 1.0, Enabled: true},
		},
	}
	values := map[string]float64{"in": 2.0, "h": 3.0}

	err := ApplyPlasticity(&g, values, model.PlasticityConfig{
		Rule: PlasticityHebbian,
		Rate: 0.1,
	})
	if err != nil {
		t.Fatalf("apply plasticity: %v", err)
	}
	// neuron rate override: 1.0 + 0.2*2*3 = 2.2
	if g.Synapses[0].Weight != 2.2 {
		t.Fatalf("unexpected weight with neuron rate override: %f", g.Synapses[0].Weight)
	}
}

func TestApplyPlasticityRejectsUnsupportedNeuronRuleOverride(t *testing.T) {
	g := model.Genome{
		Neurons: []model.Neuron{
			{ID: "h", PlasticityRule: "custom-rule", PlasticityRate: 0.2},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "in", To: "h", Weight: 1.0, Enabled: true},
		},
	}
	values := map[string]float64{"in": 2.0, "h": 3.0}

	err := ApplyPlasticity(&g, values, model.PlasticityConfig{
		Rule: PlasticityHebbian,
		Rate: 0.1,
	})
	if err == nil {
		t.Fatal("expected unsupported neuron plasticity rule error")
	}
}

func TestNormalizePlasticityRuleName(t *testing.T) {
	cases := map[string]string{
		"":          "none",
		"none":      "none",
		"hebbian":   "hebbian",
		"hebbian_w": "hebbian",
		"oja":       "oja",
		"ojas":      "oja",
		"ojas_w":    "oja",
		"custom":    "custom",
	}
	for in, want := range cases {
		if got := NormalizePlasticityRuleName(in); got != want {
			t.Fatalf("NormalizePlasticityRuleName(%q)=%q want=%q", in, got, want)
		}
	}
}
