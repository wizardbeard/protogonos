package substrate

import (
	"context"
	"errors"
	"testing"
)

func TestSimpleRuntimeStep(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
		Parameters: map[string]float64{
			"scale": 0.5,
		},
	}, 3)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1, 3})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	// mean(inputs)=2, scale=0.5 => delta=1
	for i, v := range w {
		if v != 1 {
			t.Fatalf("unexpected weight[%d]=%f", i, v)
		}
	}

	w, err = rt.Step(context.Background(), []float64{2, 2})
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	// mean(inputs)=2, delta=1, so each weight should now be 2.
	for i, v := range w {
		if v != 2 {
			t.Fatalf("unexpected weight[%d]=%f", i, v)
		}
	}
}

func TestSimpleRuntimeValidation(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if _, err := NewSimpleRuntime(Spec{}, 0); err == nil {
		t.Fatal("expected weight count validation error")
	}
	if _, err := NewSimpleRuntime(Spec{CPPName: "missing"}, 1); err == nil {
		t.Fatal("expected missing cpp error")
	}
	if _, err := NewSimpleRuntime(Spec{CEPName: "missing"}, 1); err == nil {
		t.Fatal("expected missing cep error")
	}
}

func TestSimpleRuntimeBackupRestoreReset(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
		Parameters: map[string]float64{
			"scale": 1.0,
		},
	}, 2)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	first, err := rt.Step(context.Background(), []float64{1, 3})
	if err != nil {
		t.Fatalf("step first: %v", err)
	}
	rt.Backup()

	second, err := rt.Step(context.Background(), []float64{1, 3})
	if err != nil {
		t.Fatalf("step second: %v", err)
	}
	if second[0] <= first[0] || second[1] <= first[1] {
		t.Fatalf("expected second step to accumulate weights, first=%v second=%v", first, second)
	}

	if err := rt.Restore(); err != nil {
		t.Fatalf("restore: %v", err)
	}
	restored := rt.Weights()
	if restored[0] != first[0] || restored[1] != first[1] {
		t.Fatalf("expected restored weights=%v, got=%v", first, restored)
	}

	rt.Reset()
	resetWeights := rt.Weights()
	if resetWeights[0] != 0 || resetWeights[1] != 0 {
		t.Fatalf("expected reset weights to be zeroed, got=%v", resetWeights)
	}
}

func TestSimpleRuntimeRestoreRequiresBackup(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if err := rt.Restore(); !errors.Is(err, ErrNoSubstrateBackup) {
		t.Fatalf("expected ErrNoSubstrateBackup, got %v", err)
	}
}
