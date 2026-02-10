package nn

import (
	"errors"
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
	if names[0] != "a" || names[1] != "b" {
		t.Fatalf("unexpected activation list: %+v", names)
	}
}

func TestBuiltinsAvailable(t *testing.T) {
	// Built-ins are registered during init and should remain available in regular runtime.
	for _, name := range []string{"identity", "relu", "tanh", "sigmoid"} {
		fn, err := GetActivation(name)
		if err != nil {
			t.Fatalf("get builtin activation %s: %v", name, err)
		}
		_ = fn(1.0)
	}
}
