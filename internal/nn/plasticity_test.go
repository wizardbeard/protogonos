package nn

import (
	"math"
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

func TestApplyPlasticitySelfModulationUsesGeneralizedHebbianCoefficients(t *testing.T) {
	g := model.Genome{
		Synapses: []model.Synapse{
			{ID: "s1", From: "in", To: "h", Weight: 1.0, Enabled: true},
		},
	}
	values := map[string]float64{"in": 2.0, "h": 3.0}

	err := ApplyPlasticity(&g, values, model.PlasticityConfig{
		Rule:   "self_modulationV3",
		Rate:   0.5,
		CoeffA: 0.2,
		CoeffB: 0.1,
		CoeffC: -0.05,
		CoeffD: 0.4,
	})
	if err != nil {
		t.Fatalf("apply self-modulation: %v", err)
	}
	// w += rate*(A*pre*post + B*pre + C*post + D)
	const want = 1.825
	if math.Abs(g.Synapses[0].Weight-want) > 1e-12 {
		t.Fatalf("unexpected self-modulation weight: got=%f want=%f", g.Synapses[0].Weight, want)
	}
}

func TestApplyPlasticitySelfModulationUsesNeuronCoefficientOverrides(t *testing.T) {
	g := model.Genome{
		Neurons: []model.Neuron{
			{
				ID:             "h",
				PlasticityRule: "self_modulationV1",
				PlasticityRate: 0.1,
				PlasticityA:    0.5,
				PlasticityB:    0.1,
			},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "in", To: "h", Weight: 1.0, Enabled: true},
		},
	}
	values := map[string]float64{"in": 1.0, "h": 2.0}

	err := ApplyPlasticity(&g, values, model.PlasticityConfig{
		Rule:   "self_modulationV1",
		Rate:   0.2,
		CoeffA: 0.2,
	})
	if err != nil {
		t.Fatalf("apply self-modulation with neuron overrides: %v", err)
	}
	// neuron overrides A=0.5, B=0.1 and rate=0.1:
	// w += 0.1*(0.5*1*2 + 0.1*1) = 0.11
	const want = 1.11
	if math.Abs(g.Synapses[0].Weight-want) > 1e-12 {
		t.Fatalf("unexpected neuron override weight: got=%f want=%f", g.Synapses[0].Weight, want)
	}
}

func TestApplyPlasticityNeuromodulationUsesDeadzoneScaling(t *testing.T) {
	g := model.Genome{
		Synapses: []model.Synapse{
			{ID: "s1", From: "in", To: "h", Weight: 1.0, Enabled: true},
		},
	}
	cfg := model.PlasticityConfig{
		Rule:   PlasticityNeuromodulation,
		Rate:   0.5,
		CoeffA: 1.0,
	}

	values := map[string]float64{"in": 1.0, "h": 0.2}
	if err := ApplyPlasticity(&g, values, cfg); err != nil {
		t.Fatalf("apply neuromodulation (deadzone): %v", err)
	}
	if g.Synapses[0].Weight != 1.0 {
		t.Fatalf("expected no update in deadzone, got=%f", g.Synapses[0].Weight)
	}

	values["h"] = 0.8
	if err := ApplyPlasticity(&g, values, cfg); err != nil {
		t.Fatalf("apply neuromodulation (outside deadzone): %v", err)
	}
	modulator := scaleDeadzone(values["h"], 0.33, math.Pi*2)
	want := 1.0 + modulator*0.5*(1.0*values["in"]*values["h"])
	if math.Abs(g.Synapses[0].Weight-want) > 1e-12 {
		t.Fatalf("unexpected neuromodulation weight: got=%f want=%f", g.Synapses[0].Weight, want)
	}
}

func TestApplyPlasticitySelfModulationV1UsesDynamicHFromSynapseParameters(t *testing.T) {
	g := model.Genome{
		Synapses: []model.Synapse{
			{ID: "s1", From: "in1", To: "h", Weight: 1.0, Enabled: true, PlasticityParams: []float64{0.5}},
			{ID: "s2", From: "in2", To: "h", Weight: 1.0, Enabled: true, PlasticityParams: []float64{0.25}},
		},
	}
	values := map[string]float64{
		"in1": 1.0,
		"in2": 2.0,
		"h":   0.4,
	}
	cfg := model.PlasticityConfig{
		Rule:   PlasticitySelfModulationV1,
		Rate:   1.0,
		CoeffA: 1.0,
	}
	if err := ApplyPlasticity(&g, values, cfg); err != nil {
		t.Fatalf("apply self_modulationV1: %v", err)
	}

	h := math.Tanh(values["in1"]*0.5 + values["in2"]*0.25)
	want1 := 1.0 + h*(values["in1"]*values["h"])
	want2 := 1.0 + h*(values["in2"]*values["h"])
	if math.Abs(g.Synapses[0].Weight-want1) > 1e-12 {
		t.Fatalf("unexpected self_modulationV1 weight s1: got=%f want=%f", g.Synapses[0].Weight, want1)
	}
	if math.Abs(g.Synapses[1].Weight-want2) > 1e-12 {
		t.Fatalf("unexpected self_modulationV1 weight s2: got=%f want=%f", g.Synapses[1].Weight, want2)
	}
}

func TestApplyPlasticitySelfModulationV4UsesDynamicAFromSynapseParameters(t *testing.T) {
	g := model.Genome{
		Synapses: []model.Synapse{
			{ID: "s1", From: "in1", To: "h", Weight: 0.0, Enabled: true, PlasticityParams: []float64{0.5, 0.2}},
			{ID: "s2", From: "in2", To: "h", Weight: 0.0, Enabled: true, PlasticityParams: []float64{0.5, -0.1}},
		},
	}
	values := map[string]float64{
		"in1": 1.0,
		"in2": 1.0,
		"h":   0.5,
	}
	cfg := model.PlasticityConfig{
		Rule: PlasticitySelfModulationV4,
		Rate: 1.0,
	}
	if err := ApplyPlasticity(&g, values, cfg); err != nil {
		t.Fatalf("apply self_modulationV4: %v", err)
	}

	h := math.Tanh(values["in1"]*0.5 + values["in2"]*0.5)
	a := math.Tanh(values["in1"]*0.2 + values["in2"]*-0.1)
	want := h * (a * values["in1"] * values["h"])
	if math.Abs(g.Synapses[0].Weight-want) > 1e-12 {
		t.Fatalf("unexpected self_modulationV4 weight s1: got=%f want=%f", g.Synapses[0].Weight, want)
	}
	if math.Abs(g.Synapses[1].Weight-want) > 1e-12 {
		t.Fatalf("unexpected self_modulationV4 weight s2: got=%f want=%f", g.Synapses[1].Weight, want)
	}
}

func TestApplyPlasticitySelfModulationV6UsesDynamicABCDFromSynapseParameters(t *testing.T) {
	g := model.Genome{
		Synapses: []model.Synapse{
			{
				ID:               "s1",
				From:             "in",
				To:               "h",
				Weight:           0.0,
				Enabled:          true,
				PlasticityParams: []float64{1.0, 0.5, -0.25, 0.125, 0.05},
			},
		},
	}
	values := map[string]float64{
		"in": 2.0,
		"h":  0.25,
	}
	cfg := model.PlasticityConfig{
		Rule: PlasticitySelfModulationV6,
		Rate: 1.0,
	}
	if err := ApplyPlasticity(&g, values, cfg); err != nil {
		t.Fatalf("apply self_modulationV6: %v", err)
	}

	h := math.Tanh(values["in"] * 1.0)
	a := math.Tanh(values["in"] * 0.5)
	b := math.Tanh(values["in"] * -0.25)
	c := math.Tanh(values["in"] * 0.125)
	d := math.Tanh(values["in"] * 0.05)
	want := h * (a*values["in"]*values["h"] + b*values["in"] + c*values["h"] + d)
	if math.Abs(g.Synapses[0].Weight-want) > 1e-12 {
		t.Fatalf("unexpected self_modulationV6 weight: got=%f want=%f", g.Synapses[0].Weight, want)
	}
}

func TestNormalizePlasticityRuleName(t *testing.T) {
	cases := map[string]string{
		"":                   "none",
		"none":               "none",
		"hebbian":            "hebbian",
		"hebbian_w":          "hebbian",
		"oja":                "oja",
		"ojas":               "oja",
		"ojas_w":             "oja",
		"neuromodulation":    "neuromodulation",
		"self_modulationV1":  "self_modulationv1",
		"self_modulation_v2": "self_modulationv2",
		"self_modulationV3":  "self_modulationv3",
		"self_modulationV4":  "self_modulationv4",
		"self_modulationV5":  "self_modulationv5",
		"self_modulationV6":  "self_modulationv6",
		"custom":             "custom",
	}
	for in, want := range cases {
		if got := NormalizePlasticityRuleName(in); got != want {
			t.Fatalf("NormalizePlasticityRuleName(%q)=%q want=%q", in, got, want)
		}
	}
}
