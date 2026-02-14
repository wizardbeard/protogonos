package nn

import (
	"errors"
	"math"
	"testing"
)

func TestRegisterAndGetActivation(t *testing.T) {
	resetActivationRegistryForTests()
	t.Cleanup(resetActivationRegistryForTests)

	if err := RegisterActivation("quad", func(x float64) float64 { return x * x }); err != nil {
		t.Fatalf("register activation: %v", err)
	}
	fn, err := GetActivation("quad")
	if err != nil {
		t.Fatalf("get activation: %v", err)
	}
	if got := fn(3); got != 9 {
		t.Fatalf("unexpected activation result: got=%f want=9", got)
	}
}

func TestRegisterActivationValidation(t *testing.T) {
	resetActivationRegistryForTests()
	t.Cleanup(resetActivationRegistryForTests)

	if err := RegisterActivation("", func(x float64) float64 { return x }); err == nil {
		t.Fatal("expected empty name error")
	}
	if err := RegisterActivation("nil", nil); err == nil {
		t.Fatal("expected nil function error")
	}
	if err := RegisterActivationWithSpec(ActivationSpec{
		Name:          "bad-version",
		Func:          func(x float64) float64 { return x },
		SchemaVersion: 99,
		CodecVersion:  1,
	}); !errors.Is(err, ErrActivationVersion) {
		t.Fatalf("expected ErrActivationVersion, got: %v", err)
	}
}

func TestRegisterActivationDuplicate(t *testing.T) {
	resetActivationRegistryForTests()
	t.Cleanup(resetActivationRegistryForTests)

	if err := RegisterActivation("dup", func(x float64) float64 { return x }); err != nil {
		t.Fatalf("first register: %v", err)
	}
	if err := RegisterActivation("dup", func(x float64) float64 { return x }); !errors.Is(err, ErrActivationExists) {
		t.Fatalf("expected ErrActivationExists, got: %v", err)
	}
}

func TestGetActivationNotFound(t *testing.T) {
	resetActivationRegistryForTests()
	t.Cleanup(resetActivationRegistryForTests)

	_, err := GetActivation("missing")
	if !errors.Is(err, ErrActivationNotFound) {
		t.Fatalf("expected ErrActivationNotFound, got: %v", err)
	}
}

func TestListActivationsSorted(t *testing.T) {
	resetActivationRegistryForTests()
	t.Cleanup(resetActivationRegistryForTests)

	if err := RegisterActivation("b", func(x float64) float64 { return x }); err != nil {
		t.Fatalf("register b: %v", err)
	}
	if err := RegisterActivation("a", func(x float64) float64 { return x }); err != nil {
		t.Fatalf("register a: %v", err)
	}

	names := ListActivations()
	if len(names) < 6 {
		t.Fatalf("expected built-ins plus custom activations, got: %+v", names)
	}
	foundA := false
	foundB := false
	for i := 1; i < len(names); i++ {
		if names[i-1] > names[i] {
			t.Fatalf("expected sorted activation list, got: %+v", names)
		}
	}
	for _, name := range names {
		if name == "a" {
			foundA = true
		}
		if name == "b" {
			foundB = true
		}
	}
	if !foundA || !foundB {
		t.Fatalf("expected custom activations in list, got: %+v", names)
	}
}

func TestBuiltinsAvailable(t *testing.T) {
	// Built-ins are registered during init and should remain available in regular runtime.
	for _, name := range []string{
		"identity", "linear", "relu", "tanh", "cos", "sin", "sgn", "bin", "bip",
		"trinary", "multiquadric", "absolute", "quadratic", "gaussian", "sqrt",
		"log", "sigmoid", "sigmoid1",
	} {
		fn, err := GetActivation(name)
		if err != nil {
			t.Fatalf("get builtin activation %s: %v", name, err)
		}
		_ = fn(1.0)
	}
}

func TestExtendedBuiltinsBehavior(t *testing.T) {
	cases := []struct {
		name  string
		x     float64
		want  float64
		delta float64
	}{
		{name: "sgn", x: -1.2, want: -1, delta: 1e-9},
		{name: "bin", x: -0.1, want: 0, delta: 1e-9},
		{name: "bip", x: -0.1, want: -1, delta: 1e-9},
		{name: "trinary", x: 0.0, want: 0, delta: 1e-9},
		{name: "absolute", x: -2.5, want: 2.5, delta: 1e-9},
		{name: "quadratic", x: -2.0, want: -4.0, delta: 1e-9},
		{name: "gaussian", x: 0.0, want: 1.0, delta: 1e-9},
		{name: "sqrt", x: -4.0, want: -2.0, delta: 1e-9},
		{name: "log", x: -math.E, want: -1.0, delta: 1e-9},
		{name: "sigmoid1", x: 2.0, want: 2.0 / 3.0, delta: 1e-9},
	}

	for _, tc := range cases {
		fn, err := GetActivation(tc.name)
		if err != nil {
			t.Fatalf("get activation %s: %v", tc.name, err)
		}
		got := fn(tc.x)
		if math.Abs(got-tc.want) > tc.delta {
			t.Fatalf("activation %s(%f): got=%f want=%f", tc.name, tc.x, got, tc.want)
		}
	}
}
