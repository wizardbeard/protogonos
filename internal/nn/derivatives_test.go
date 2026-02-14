package nn

import (
	"math"
	"testing"
)

func TestDerivativeBasics(t *testing.T) {
	cases := []struct {
		name string
		x    float64
	}{
		{"identity", 1},
		{"linear", 1},
		{"relu", 1},
		{"tanh", 0.5},
		{"sigmoid", 0.5},
		{"sigmoid1", 0.5},
		{"sin", 0.25},
		{"cos", 0.25},
		{"multiquadric", 0.25},
		{"absolute", -1},
		{"quadratic", 2},
		{"gaussian", 0.1},
		{"sqrt", 4},
		{"log", 2},
	}
	for _, c := range cases {
		if _, err := Derivative(c.name, c.x); err != nil {
			t.Fatalf("derivative %s failed: %v", c.name, err)
		}
	}
}

func TestDerivativeReferenceParityNumerics(t *testing.T) {
	cases := []struct {
		name string
		x    float64
		want float64
	}{
		{name: "sigmoid1", x: 2, want: 1.0 / 9.0},
		{name: "sqrt", x: 4, want: 0.25},
		{name: "log", x: -2, want: 0.5},
		{name: "absolute", x: 0, want: -1},
	}
	for _, tc := range cases {
		got, err := Derivative(tc.name, tc.x)
		if err != nil {
			t.Fatalf("derivative %s failed: %v", tc.name, err)
		}
		if math.Abs(got-tc.want) > 1e-9 {
			t.Fatalf("derivative %s(%f): got=%f want=%f", tc.name, tc.x, got, tc.want)
		}
	}
}

func TestDerivativeInputClippingParity(t *testing.T) {
	gotA, err := Derivative("gaussian", 1000)
	if err != nil {
		t.Fatalf("gaussian derivative: %v", err)
	}
	gotB, err := Derivative("gaussian", 10)
	if err != nil {
		t.Fatalf("gaussian derivative: %v", err)
	}
	if gotA != gotB {
		t.Fatalf("expected gaussian clipping parity: high=%f clipped=%f", gotA, gotB)
	}

	gotSigA, err := Derivative("sigmoid", -1000)
	if err != nil {
		t.Fatalf("sigmoid derivative: %v", err)
	}
	gotSigB, err := Derivative("sigmoid", -10)
	if err != nil {
		t.Fatalf("sigmoid derivative: %v", err)
	}
	if gotSigA != gotSigB {
		t.Fatalf("expected sigmoid clipping parity: low=%f clipped=%f", gotSigA, gotSigB)
	}
}

func TestDerivativeUnsupported(t *testing.T) {
	if _, err := Derivative("unknown", 1); err == nil {
		t.Fatal("expected unsupported derivative error")
	}
}
