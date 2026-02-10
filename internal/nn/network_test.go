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

	want := 2*1.0 + (-1)*0.25 + 0.5
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
