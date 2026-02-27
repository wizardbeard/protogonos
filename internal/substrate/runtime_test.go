package substrate

import (
	"context"
	"errors"
	"math"
	"testing"
)

type customRuntimeCEP struct{}

func (customRuntimeCEP) Name() string { return "custom_runtime_cep" }

func (customRuntimeCEP) Apply(_ context.Context, current float64, _ float64, _ map[string]float64) (float64, error) {
	return current + 0.25, nil
}

type vectorRuntimeCPP struct {
	signals []float64
}

func (c vectorRuntimeCPP) Name() string { return "vector_runtime_cpp" }

func (c vectorRuntimeCPP) Compute(_ context.Context, _ []float64, _ map[string]float64) (float64, error) {
	if len(c.signals) == 0 {
		return 0, nil
	}
	return c.signals[0], nil
}

func (c vectorRuntimeCPP) ComputeVector(_ context.Context, _ []float64, _ map[string]float64) ([]float64, error) {
	return append([]float64(nil), c.signals...), nil
}

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
	// mean(inputs)=2 -> control saturates to 1, then scale=0.5 => delta=0.5.
	for i, v := range w {
		if v != 0.5 {
			t.Fatalf("unexpected weight[%d]=%f", i, v)
		}
	}

	w, err = rt.Step(context.Background(), []float64{2, 2})
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	// mean(inputs)=2 -> control saturates to 1, then scale=0.5 => +0.5 each step.
	for i, v := range w {
		if v != 1 {
			t.Fatalf("unexpected weight[%d]=%f", i, v)
		}
	}
}

func TestSimpleRuntimeSetWeightCEP(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: SetWeightCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || w[0] != 1 {
		t.Fatalf("expected set_weight to set exact bounded weight to 1, got=%v", w)
	}

	// set_weight semantics set the value directly instead of accumulating.
	w, err = rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	if len(w) != 1 || w[0] != 1 {
		t.Fatalf("expected set_weight to keep bounded weight at 1, got=%v", w)
	}
}

func TestSimpleRuntimeSetIterativeCEPAlias(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: SetIterativeCEPName,
		Parameters: map[string]float64{
			"scale": 0.5,
		},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1, 3})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || w[0] != 0.5 {
		t.Fatalf("expected iterative alias first step to be 0.5, got=%v", w)
	}

	w, err = rt.Step(context.Background(), []float64{1, 3})
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	if len(w) != 1 || w[0] != 1.0 {
		t.Fatalf("expected iterative alias second step to be 1.0, got=%v", w)
	}
}

func TestSimpleRuntimeSetWeightCEPSaturatesReferenceLimit(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: SetWeightCEPName,
		Parameters: map[string]float64{
			"scale": 10,
		},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step +1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-referenceSubstrateWeightLimit) > 1e-9 {
		t.Fatalf("expected set_weight saturation to +%v, got=%v", referenceSubstrateWeightLimit, w)
	}

	w, err = rt.Step(context.Background(), []float64{-1})
	if err != nil {
		t.Fatalf("step -1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]+referenceSubstrateWeightLimit) > 1e-9 {
		t.Fatalf("expected set_weight saturation to -%v, got=%v", referenceSubstrateWeightLimit, w)
	}
}

func TestSimpleRuntimeSetABCNCEP(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: SetABCNCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	first, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	second, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	if len(first) != 1 || len(second) != 1 || second[0] <= first[0] {
		t.Fatalf("expected iterative set_abcn surrogate behavior, first=%v second=%v", first, second)
	}
}

func TestSimpleRuntimeSetABCNCEPUsesCoefficientParameters(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: SetABCNCEPName,
		Parameters: map[string]float64{
			"A": 0.2,
			"B": 0.5,
			"C": -0.1,
			"N": 0.8,
		},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	first, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(first) != 1 || math.Abs(first[0]-0.4) > 1e-9 {
		t.Fatalf("unexpected first abcn update, got=%v want=0.4", first)
	}

	second, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	if len(second) != 1 || math.Abs(second[0]-0.832) > 1e-9 {
		t.Fatalf("unexpected second abcn update, got=%v want=0.832", second)
	}
}

func TestSimpleRuntimeSetABCNCEPSupportsVectorFanInSignals(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("vector_runtime_cpp", func() CPP {
		return vectorRuntimeCPP{signals: []float64{1, 0.2, 0.5, -0.1, 0.8}}
	}); err != nil {
		t.Fatalf("register vector cpp: %v", err)
	}

	rt, err := NewSimpleRuntime(Spec{
		CPPName:      "vector_runtime_cpp",
		CEPName:      SetABCNCEPName,
		CEPFaninPIDs: []string{"n1", "n2", "n3", "n4", "n5"},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{0})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.4) > 1e-9 {
		t.Fatalf("unexpected vector fan-in set_abcn update, got=%v want=0.4", w)
	}
}

func TestSimpleRuntimeSetABCNCEPUsesInputFanInSignalsWithoutVectorCPP(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName:      DefaultCPPName,
		CEPName:      SetABCNCEPName,
		CEPFaninPIDs: []string{"n1", "n2", "n3", "n4", "n5"},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1, 0.2, 0.5, -0.1, 0.8})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.4) > 1e-9 {
		t.Fatalf("unexpected input fan-in set_abcn update, got=%v want=0.4", w)
	}
}

func TestSimpleRuntimeSetABCNCEPSaturatesReferenceLimit(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: SetABCNCEPName,
		Parameters: map[string]float64{
			"scale": 10,
		},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-referenceSubstrateWeightLimit) > 1e-9 {
		t.Fatalf("expected set_abcn surrogate saturation at +%v, got=%v", referenceSubstrateWeightLimit, w)
	}
}

func TestSimpleRuntimeDeltaWeightCEPSaturatesReferenceLimit(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
		Parameters: map[string]float64{
			"scale": 10,
		},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-referenceSubstrateWeightLimit) > 1e-9 {
		t.Fatalf("expected delta_weight saturation at +%v, got=%v", referenceSubstrateWeightLimit, w)
	}

	w, err = rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-referenceSubstrateWeightLimit) > 1e-9 {
		t.Fatalf("expected saturated value to remain capped at +%v, got=%v", referenceSubstrateWeightLimit, w)
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
	if _, err := NewSimpleRuntime(Spec{CEPNames: []string{"missing"}}, 1); err == nil {
		t.Fatal("expected missing cep error in chain")
	}
}

func TestSimpleRuntimeCEPChainAppliesInOrder(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	setThenDelta, err := NewSimpleRuntime(Spec{
		CPPName:  DefaultCPPName,
		CEPNames: []string{SetWeightCEPName, DefaultCEPName},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime set->delta: %v", err)
	}
	w1, err := setThenDelta.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step set->delta: %v", err)
	}
	if len(w1) != 1 || w1[0] != 2 {
		t.Fatalf("expected set->delta chain to produce 2, got=%v", w1)
	}

	deltaThenSet, err := NewSimpleRuntime(Spec{
		CPPName:  DefaultCPPName,
		CEPNames: []string{DefaultCEPName, SetWeightCEPName},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime delta->set: %v", err)
	}
	w2, err := deltaThenSet.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step delta->set: %v", err)
	}
	if len(w2) != 1 || w2[0] != 1 {
		t.Fatalf("expected delta->set chain to produce 1, got=%v", w2)
	}
}

func TestSimpleRuntimeFallbackForCustomCEP(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCEP("custom_runtime_cep", func() CEP { return customRuntimeCEP{} }); err != nil {
		t.Fatalf("register custom cep: %v", err)
	}

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: "custom_runtime_cep",
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.25) > 1e-9 {
		t.Fatalf("unexpected custom cep fallback update after step 1: %v", w)
	}

	w, err = rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step 2: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.5) > 1e-9 {
		t.Fatalf("unexpected custom cep fallback update after step 2: %v", w)
	}
}

func TestSimpleRuntimeScalarCEPDoesNotUseInputFanInSignals(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName:      DefaultCPPName,
		CEPName:      SetWeightCEPName,
		CEPFaninPIDs: []string{"n1", "n2"},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{1, -1})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]) > 1e-9 {
		t.Fatalf("expected scalar set_weight control from cpp mean, got=%v", w)
	}
}

func TestSimpleRuntimeCEPFanInSignalMismatch(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("vector_runtime_cpp", func() CPP {
		return vectorRuntimeCPP{signals: []float64{0.1, 0.2}}
	}); err != nil {
		t.Fatalf("register vector cpp: %v", err)
	}

	rt, err := NewSimpleRuntime(Spec{
		CPPName:      "vector_runtime_cpp",
		CEPName:      SetWeightCEPName,
		CEPFaninPIDs: []string{"n1"},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	if _, err := rt.Step(context.Background(), []float64{0}); !errors.Is(err, ErrInvalidCEPOutputWidth) {
		t.Fatalf("expected ErrInvalidCEPOutputWidth, got %v", err)
	}
}

func TestSimpleRuntimeCEPChainUsesPerCEPFanInConfig(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("vector_runtime_cpp", func() CPP {
		return vectorRuntimeCPP{signals: []float64{1, 0.2, 0.5, -0.1, 0.8}}
	}); err != nil {
		t.Fatalf("register vector cpp: %v", err)
	}

	rt, err := NewSimpleRuntime(Spec{
		CPPName: "vector_runtime_cpp",
		CEPNames: []string{
			SetABCNCEPName,
			SetABCNCEPName,
		},
		CEPFaninPIDsByCEP: [][]string{
			{"n1", "n2", "n3", "n4", "n5"},
			{"n1"},
		},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.Step(context.Background(), []float64{0})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-1.4) > 1e-9 {
		t.Fatalf("unexpected per-cep fan-in chain update, got=%v want=1.4", w)
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

func TestSimpleRuntimeTerminateBlocksStep(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	rt.Terminate()
	if _, err := rt.Step(context.Background(), []float64{1}); !errors.Is(err, ErrSubstrateRuntimeTerminated) {
		t.Fatalf("expected ErrSubstrateRuntimeTerminated, got %v", err)
	}

	// Terminate should be idempotent.
	rt.Terminate()
}
