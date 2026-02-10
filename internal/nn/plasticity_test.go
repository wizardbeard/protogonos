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
