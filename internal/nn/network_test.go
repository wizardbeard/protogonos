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
	if values["o"] != -1 {
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
