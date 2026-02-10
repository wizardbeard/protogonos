package substrate

import (
	"context"
	"errors"
	"testing"
)

type testCPP struct{}

func (testCPP) Name() string { return "test-cpp" }
func (testCPP) Compute(context.Context, []float64, map[string]float64) (float64, error) {
	return 1, nil
}

type testCEP struct{}

func (testCEP) Name() string { return "test-cep" }
func (testCEP) Apply(_ context.Context, current float64, delta float64, _ map[string]float64) (float64, error) {
	return current + delta, nil
}

func TestRegisterAndResolveCPP(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("x", func() CPP { return testCPP{} }); err != nil {
		t.Fatalf("register cpp: %v", err)
	}
	cpp, err := ResolveCPP("x")
	if err != nil {
		t.Fatalf("resolve cpp: %v", err)
	}
	if cpp.Name() != "test-cpp" {
		t.Fatalf("unexpected cpp: %s", cpp.Name())
	}
}

func TestRegisterAndResolveCEP(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCEP("x", func() CEP { return testCEP{} }); err != nil {
		t.Fatalf("register cep: %v", err)
	}
	cep, err := ResolveCEP("x")
	if err != nil {
		t.Fatalf("resolve cep: %v", err)
	}
	if cep.Name() != "test-cep" {
		t.Fatalf("unexpected cep: %s", cep.Name())
	}
}

func TestRegistryValidationAndDuplicates(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("", func() CPP { return testCPP{} }); err == nil {
		t.Fatal("expected cpp name validation")
	}
	if err := RegisterCEP("", func() CEP { return testCEP{} }); err == nil {
		t.Fatal("expected cep name validation")
	}
	if err := RegisterCPP("dup", func() CPP { return testCPP{} }); err != nil {
		t.Fatalf("register cpp dup: %v", err)
	}
	if err := RegisterCPP("dup", func() CPP { return testCPP{} }); !errors.Is(err, ErrCPPExists) {
		t.Fatalf("expected ErrCPPExists, got: %v", err)
	}
	if err := RegisterCEP("dup", func() CEP { return testCEP{} }); err != nil {
		t.Fatalf("register cep dup: %v", err)
	}
	if err := RegisterCEP("dup", func() CEP { return testCEP{} }); !errors.Is(err, ErrCEPExists) {
		t.Fatalf("expected ErrCEPExists, got: %v", err)
	}
}
