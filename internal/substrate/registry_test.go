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

func TestRegisterAndResolveCPPTrimsNames(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("  trimmed-cpp  ", func() CPP { return testCPP{} }); err != nil {
		t.Fatalf("register cpp with spaces: %v", err)
	}
	cpp, err := ResolveCPP(" trimmed-cpp ")
	if err != nil {
		t.Fatalf("resolve trimmed cpp: %v", err)
	}
	if cpp.Name() != "test-cpp" {
		t.Fatalf("unexpected trimmed cpp: %s", cpp.Name())
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

func TestRegisterAndResolveCEPTrimsNames(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCEP("  trimmed-cep  ", func() CEP { return testCEP{} }); err != nil {
		t.Fatalf("register cep with spaces: %v", err)
	}
	cep, err := ResolveCEP(" trimmed-cep ")
	if err != nil {
		t.Fatalf("resolve trimmed cep: %v", err)
	}
	if cep.Name() != "test-cep" {
		t.Fatalf("unexpected trimmed cep: %s", cep.Name())
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
	if err := RegisterCPP(" dup ", func() CPP { return testCPP{} }); !errors.Is(err, ErrCPPExists) {
		t.Fatalf("expected ErrCPPExists for spaced cpp duplicate, got: %v", err)
	}
	if err := RegisterCEP("dup", func() CEP { return testCEP{} }); err != nil {
		t.Fatalf("register cep dup: %v", err)
	}
	if err := RegisterCEP("dup", func() CEP { return testCEP{} }); !errors.Is(err, ErrCEPExists) {
		t.Fatalf("expected ErrCEPExists, got: %v", err)
	}
	if err := RegisterCEP(" dup ", func() CEP { return testCEP{} }); !errors.Is(err, ErrCEPExists) {
		t.Fatalf("expected ErrCEPExists for spaced cep duplicate, got: %v", err)
	}
}

func TestResolveSetIterativeCEPAlias(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	cep, err := ResolveCEP(SetIterativeCEPName)
	if err != nil {
		t.Fatalf("resolve set_iterative cep alias: %v", err)
	}
	if cep.Name() != DefaultCEPName {
		t.Fatalf("expected set_iterative alias to resolve delta_weight behavior, got=%s", cep.Name())
	}
}

func TestResolveWeightExpressionCEP(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	cep, err := ResolveCEP(WeightExpressionCEPName)
	if err != nil {
		t.Fatalf("resolve weight_expression cep: %v", err)
	}
	if cep.Name() != WeightExpressionCEPName {
		t.Fatalf("expected weight_expression CEP name, got=%s", cep.Name())
	}
}
