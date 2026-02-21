package nn

import (
	"math"
	"testing"

	"protogonos/internal/model"
)

func TestForwardSimpleFeedForward(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0.5},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "o", Weight: 2, Enabled: true},
			{From: "i2", To: "o", Weight: -1, Enabled: true},
		},
	}

	values, err := Forward(genome, map[string]float64{"i1": 1.0, "i2": 0.25})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	want := 1.0
	if math.Abs(values["o"]-want) > 1e-9 {
		t.Fatalf("unexpected output: got=%f want=%f", values["o"], want)
	}
}

func TestForwardUnsupportedActivation(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{{ID: "o", Activation: "unknown"}},
	}

	_, err := Forward(genome, map[string]float64{})
	if err == nil {
		t.Fatal("expected unsupported activation error")
	}
}

func TestApplyActivation(t *testing.T) {
	tests := []struct {
		name   string
		act    string
		x      float64
		want   float64
		delta  float64
		hasErr bool
	}{
		{name: "identity", act: "identity", x: 2.5, want: 2.5, delta: 1e-9},
		{name: "relu-negative", act: "relu", x: -1, want: 0, delta: 1e-9},
		{name: "relu-positive", act: "relu", x: 3, want: 3, delta: 1e-9},
		{name: "tanh", act: "tanh", x: 0, want: 0, delta: 1e-9},
		{name: "sigmoid", act: "sigmoid", x: 0, want: 0.5, delta: 1e-9},
		{name: "unknown", act: "none", hasErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := applyActivation(tc.act, tc.x)
			if tc.hasErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if math.Abs(got-tc.want) > tc.delta {
				t.Fatalf("unexpected value: got=%f want=%f", got, tc.want)
			}
		})
	}
}

func TestForwardAggregatorModes(t *testing.T) {
	base := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 1.0},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i1", To: "o", Weight: 2, Enabled: true},
			{ID: "s2", From: "i2", To: "o", Weight: 3, Enabled: true},
		},
	}

	inputs := map[string]float64{"i1": 2, "i2": 4}

	dot := base
	dot.Neurons[2].Aggregator = "dot_product"
	values, err := Forward(dot, inputs)
	if err != nil {
		t.Fatalf("forward dot: %v", err)
	}
	if values["o"] != 1 {
		t.Fatalf("unexpected dot output: %f", values["o"])
	}

	mult := base
	mult.Neurons[2].Aggregator = "mult_product"
	values, err = Forward(mult, inputs)
	if err != nil {
		t.Fatalf("forward mult: %v", err)
	}
	if values["o"] != 1 {
		t.Fatalf("unexpected mult output: %f", values["o"])
	}

	diff := base
	diff.Neurons[2].Aggregator = "diff_product"
	values, err = Forward(diff, inputs)
	if err != nil {
		t.Fatalf("forward diff: %v", err)
	}
	if values["o"] != 1 {
		t.Fatalf("unexpected diff output: %f", values["o"])
	}
}

func TestForwardUnsupportedAggregator(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Aggregator: "unknown"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}
	_, err := Forward(genome, map[string]float64{"i": 1})
	if err == nil {
		t.Fatal("expected unsupported aggregator error")
	}
}

func TestMultProductUsesMultiplicativeBiasParity(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "o", Activation: "identity", Aggregator: "mult_product", Bias: 3.0},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i1", To: "o", Weight: 1, Enabled: true},
			{ID: "s2", From: "i2", To: "o", Weight: 2, Enabled: true},
		},
	}

	values, err := Forward(genome, map[string]float64{"i1": 0.1, "i2": 0.2})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	// (0.1*1)*(0.2*2)*3 = 0.12
	if math.Abs(values["o"]-0.12) > 1e-9 {
		t.Fatalf("unexpected mult_product output with multiplicative bias: got=%f want=0.12", values["o"])
	}
}

func TestDiffProductUsesPreviousInputsWhenStateProvided(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "o", Activation: "identity", Aggregator: "diff_product", Bias: 0},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i1", To: "o", Weight: 1, Enabled: true},
			{ID: "s2", From: "i2", To: "o", Weight: 1, Enabled: true},
		},
	}

	state := NewForwardState()

	values, err := ForwardWithState(genome, map[string]float64{"i1": 0.6, "i2": 0.2}, state)
	if err != nil {
		t.Fatalf("first forward: %v", err)
	}
	// First call has no previous input, so this is equivalent to dot_product.
	if math.Abs(values["o"]-0.8) > 1e-9 {
		t.Fatalf("unexpected first diff_product output: got=%f want=0.8", values["o"])
	}

	values, err = ForwardWithState(genome, map[string]float64{"i1": 0.7, "i2": 0.4}, state)
	if err != nil {
		t.Fatalf("second forward: %v", err)
	}
	// Second call uses input deltas: (0.7-0.6)+(0.4-0.2) = 0.3.
	if math.Abs(values["o"]-0.3) > 1e-9 {
		t.Fatalf("unexpected second diff_product output: got=%f want=0.3", values["o"])
	}
}

func TestForwardRecurrentSynapseUsesPreviousStepOutputWhenStateProvided(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Aggregator: "dot_product"},
		},
		Synapses: []model.Synapse{
			{ID: "s_in", From: "i", To: "o", Weight: 1, Enabled: true},
			{ID: "s_rec", From: "o", To: "o", Weight: 0.5, Enabled: true, Recurrent: true},
		},
	}
	state := NewForwardState()

	values, err := ForwardWithState(genome, map[string]float64{"i": 1.0}, state)
	if err != nil {
		t.Fatalf("first forward: %v", err)
	}
	if math.Abs(values["o"]-1.0) > 1e-9 {
		t.Fatalf("unexpected first recurrent output: got=%f want=1.0", values["o"])
	}

	values, err = ForwardWithState(genome, map[string]float64{"i": 0.0}, state)
	if err != nil {
		t.Fatalf("second forward: %v", err)
	}
	// Recurrent input should use previous-step output: 0 + 0.5*1.0 = 0.5.
	if math.Abs(values["o"]-0.5) > 1e-9 {
		t.Fatalf("unexpected second recurrent output: got=%f want=0.5", values["o"])
	}
}

func TestForwardRecurrentSynapseWithoutStateFallsBackToCurrentValues(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Aggregator: "dot_product"},
		},
		Synapses: []model.Synapse{
			{ID: "s_in", From: "i", To: "o", Weight: 1, Enabled: true},
			{ID: "s_rec", From: "o", To: "o", Weight: 0.5, Enabled: true, Recurrent: true},
		},
	}

	values, err := ForwardWithState(genome, map[string]float64{"i": 0.0}, nil)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	// No state means no previous-step memory: recurrent term resolves to current map default 0.
	if math.Abs(values["o"]-0.0) > 1e-9 {
		t.Fatalf("unexpected recurrent output without state: got=%f want=0.0", values["o"])
	}
}
