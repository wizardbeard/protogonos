package substrate

import (
	"context"
	"errors"
	"math"
	"reflect"
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

func TestSimpleRuntimeSetABCNCEPUsesNamedFanInSignals(t *testing.T) {
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

	w, err := rt.StepWithFanin(context.Background(), []float64{0, 0, 0, 0, 0}, map[string]float64{
		"n1": 1,
		"n2": 0.2,
		"n3": 0.5,
		"n4": -0.1,
		"n5": 0.8,
	})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.4) > 1e-9 {
		t.Fatalf("unexpected named fan-in set_abcn update, got=%v want=0.4", w)
	}
}

func TestSimpleRuntimeSetABCNCEPTrimsNamedFanInSignalKeys(t *testing.T) {
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

	w, err := rt.StepWithFanin(context.Background(), []float64{0, 0, 0, 0, 0}, map[string]float64{
		" n1 ": 1,
		" n2 ": 0.2,
		" n3 ": 0.5,
		" n4 ": -0.1,
		" n5 ": 0.8,
	})
	if err != nil {
		t.Fatalf("step with trimmed named fan-in keys: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.4) > 1e-9 {
		t.Fatalf("unexpected trimmed-key named fan-in set_abcn update, got=%v want=0.4", w)
	}
}

func TestSimpleRuntimeSetABCNCEPPersistsSignalCoefficientsAcrossCEPStages(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
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

	w, err := rt.StepWithFanin(context.Background(), []float64{0, 0, 0, 0, 0}, map[string]float64{
		"n1": 1,
		"n2": 0.2,
		"n3": 0.5,
		"n4": -0.1,
		"n5": 0.8,
	})
	if err != nil {
		t.Fatalf("step 1: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.832) > 1e-9 {
		t.Fatalf("unexpected persisted-coefficient set_abcn stage update, got=%v want=0.832", w)
	}
}

func TestSimpleRuntimeWeightExpressionCEPUsesNamedFanInSignals(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName:      DefaultCPPName,
		CEPName:      WeightExpressionCEPName,
		CEPFaninPIDs: []string{"n1", "n2"},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.StepWithFanin(context.Background(), []float64{0, 0}, map[string]float64{
		"n1": 0.75,
		"n2": 1,
	})
	if err != nil {
		t.Fatalf("step positive expression: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.75) > 1e-9 {
		t.Fatalf("unexpected weight_expression result for positive expression, got=%v want=0.75", w)
	}

	w, err = rt.StepWithFanin(context.Background(), []float64{0, 0}, map[string]float64{
		"n1": 0.75,
		"n2": -1,
	})
	if err != nil {
		t.Fatalf("step nonpositive expression: %v", err)
	}
	if len(w) != 1 || w[0] != 0 {
		t.Fatalf("unexpected weight_expression result for nonpositive expression, got=%v want=0", w)
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

func TestResolveGlobalCEPFaninPIDsUsesCPPIDsFallback(t *testing.T) {
	got := resolveGlobalCEPFaninPIDs(nil, nil, []string{"", "cpp_endpoint_1", "cpp_endpoint_2"}, []CEP{DeltaWeightCEP{}})
	if len(got) != 1 || got[0] != "cpp_endpoint_1" {
		t.Fatalf("expected cpp-id fallback fan-in pid, got=%v", got)
	}
}

func TestResolveGlobalCEPFaninPIDsFlattensPerCEPUnion(t *testing.T) {
	got := resolveGlobalCEPFaninPIDs(nil, [][]string{
		{"n2", "n1"},
		{"n1", "o1"},
		{"", "o2"},
		nil,
	}, nil, nil)
	want := []string{"n2", "n1", "o1", "o2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected flattened global fan-in ids: got=%v want=%v", got, want)
	}
}

func TestResolveGlobalCEPFaninPIDsUsesAllCPPIDsForMultiSignalCEP(t *testing.T) {
	got := resolveGlobalCEPFaninPIDs(nil, nil, []string{"", "cpp_endpoint_1", "cpp_endpoint_2"}, []CEP{WeightExpressionCEP{}})
	want := []string{"cpp_endpoint_1", "cpp_endpoint_2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected cpp-id multi-signal fallback fan-in ids: got=%v want=%v", got, want)
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

func TestSimpleRuntimeUsesConfiguredCPPIDAsDefaultFaninPID(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CPPIDs:  []string{"cpp_endpoint_1"},
		CEPName: DefaultCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.cepFaninPIDs) != 1 || rt.cepFaninPIDs[0] != "cpp_endpoint_1" {
		t.Fatalf("expected runtime fan-in pid from cpp ids, got=%v", rt.cepFaninPIDs)
	}

	w, err := rt.StepWithFanin(context.Background(), []float64{0}, map[string]float64{
		"cpp_endpoint_1": 1,
	})
	if err != nil {
		t.Fatalf("step with named cpp fan-in pid: %v", err)
	}
	if len(w) != 1 || w[0] != 1 {
		t.Fatalf("expected named cpp fan-in pid to drive update to 1, got=%v", w)
	}
}

func TestSimpleRuntimeUsesAllConfiguredCPPIDsForMultiSignalCEP(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CPPIDs:  []string{"cpp_endpoint_1", "cpp_endpoint_2"},
		CEPName: WeightExpressionCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	wantFanin := []string{"cpp_endpoint_1", "cpp_endpoint_2"}
	if !reflect.DeepEqual(rt.cepFaninPIDs, wantFanin) {
		t.Fatalf("expected runtime multi-signal fan-in ids from cpp ids, got=%v want=%v", rt.cepFaninPIDs, wantFanin)
	}

	w, err := rt.StepWithFanin(context.Background(), []float64{0, 0}, map[string]float64{
		"cpp_endpoint_1": 0.75,
		"cpp_endpoint_2": 1.0,
	})
	if err != nil {
		t.Fatalf("step with named cpp fan-in ids: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.75) > 1e-9 {
		t.Fatalf("expected multi-signal cpp fan-in ids to drive update to 0.75, got=%v", w)
	}
}

func TestResolveCEPChainExpandsPrimaryCEPByCEPIDs(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	chain, err := resolveCEPChain(Spec{
		CEPName: DefaultCEPName,
		CEPIDs:  []string{"cep_a", "cep_b", ""},
	})
	if err != nil {
		t.Fatalf("resolve cep chain: %v", err)
	}
	if len(chain) != 2 {
		t.Fatalf("expected cep chain length from non-empty cep ids, got=%d", len(chain))
	}
	if chain[0].Name() != DefaultCEPName || chain[1].Name() != DefaultCEPName {
		t.Fatalf("unexpected expanded cep chain names: %q %q", chain[0].Name(), chain[1].Name())
	}
}

func TestResolveCEPChainExpandsSingularCEPNamesByCEPIDs(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	chain, err := resolveCEPChain(Spec{
		CEPName:  DefaultCEPName,
		CEPNames: []string{SetWeightCEPName},
		CEPIDs:   []string{"cep_a", "cep_b", "cep_c"},
	})
	if err != nil {
		t.Fatalf("resolve cep chain: %v", err)
	}
	if len(chain) != 3 {
		t.Fatalf("expected expanded cep chain length from cep ids, got=%d", len(chain))
	}
	for i, cep := range chain {
		if cep.Name() != SetWeightCEPName {
			t.Fatalf("unexpected cep chain name at idx=%d, got=%q want=%q", i, cep.Name(), SetWeightCEPName)
		}
	}
}

func TestSimpleRuntimeExpandsPrimaryCEPByCEPIDs(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
		CEPIDs:  []string{"cep_1", "cep_2"},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.ceps) != 2 {
		t.Fatalf("expected expanded cep chain length 2, got=%d", len(rt.ceps))
	}
	if len(rt.cepActorInits) != 2 {
		t.Fatalf("expected actor init count to follow expanded cep chain, got=%d", len(rt.cepActorInits))
	}
	if rt.cepActorInits[0].id != "cep_1" || rt.cepActorInits[1].id != "cep_2" {
		t.Fatalf("unexpected expanded cep actor ids: %q %q", rt.cepActorInits[0].id, rt.cepActorInits[1].id)
	}

	w, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if len(w) != 1 || w[0] != 2 {
		t.Fatalf("expected two-stage delta chain from cep ids to produce 2, got=%v", w)
	}
}

func TestSimpleRuntimeExpandsSingularCEPNamesByCEPIDs(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName:  DefaultCPPName,
		CEPName:  DefaultCEPName,
		CEPNames: []string{SetWeightCEPName},
		CEPIDs:   []string{"cep_1", "cep_2"},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.ceps) != 2 {
		t.Fatalf("expected expanded cep chain length 2 from cep ids, got=%d", len(rt.ceps))
	}
	if rt.ceps[0].Name() != SetWeightCEPName || rt.ceps[1].Name() != SetWeightCEPName {
		t.Fatalf("unexpected expanded singular cep-names chain: %q %q", rt.ceps[0].Name(), rt.ceps[1].Name())
	}

	w, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if len(w) != 1 || w[0] != 1 {
		t.Fatalf("expected two-stage set-weight chain to remain 1, got=%v", w)
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
	if len(w) != 1 || math.Abs(w[0]-0.832) > 1e-9 {
		t.Fatalf("unexpected per-cep fan-in chain update, got=%v want=0.832", w)
	}
}

func TestSimpleRuntimeStepWithFaninUsesFlattenedPerCEPFanInUnion(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPNames: []string{
			WeightExpressionCEPName,
			WeightExpressionCEPName,
		},
		CEPFaninPIDsByCEP: [][]string{
			{"n1", "n2"},
			{"n3", "n4"},
		},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	w, err := rt.StepWithFanin(context.Background(), []float64{0, 0, 0, 0}, map[string]float64{
		"n1": 0.5,
		"n2": 1.0,
		"n3": 0.75,
		"n4": 1.0,
	})
	if err != nil {
		t.Fatalf("step with flattened per-cep fan-in union: %v", err)
	}
	if len(w) != 1 || math.Abs(w[0]-0.75) > 1e-9 {
		t.Fatalf("unexpected flattened per-cep fan-in union result, got=%v want=0.75", w)
	}
}

func TestSimpleRuntimeStepRequiresCEPActor(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.cepActors) == 0 {
		t.Fatal("expected cep actor to be initialized")
	}
	rt.cepActors[0] = nil

	if _, err := rt.Step(context.Background(), []float64{1}); !errors.Is(err, ErrMissingCEPActor) {
		t.Fatalf("expected ErrMissingCEPActor, got %v", err)
	}
}

func TestSimpleRuntimeStepValidatesCEPCommandSenderEnvelope(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.cepActorsByWeight) == 0 || len(rt.cepActorsByWeight[0]) == 0 {
		t.Fatal("expected cep actor pool to be initialized")
	}
	_ = rt.cepActorsByWeight[0][0].TerminateFrom(runtimeExoSelfProcessID)

	process, err := NewCEPProcessWithOwner("cep_unexpected_sender", runtimeExoSelfProcessID, DefaultCEPName, nil, []string{runtimeCPPProcessID})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	process.substratePID = "substrate_w1"
	actor := NewCEPActor(process)
	t.Cleanup(func() {
		_ = actor.TerminateFrom(runtimeExoSelfProcessID)
	})
	rt.cepActorsByWeight[0][0] = actor
	rt.cepActors[0] = actor
	if len(rt.cepFaninRelays) == 0 || len(rt.cepFaninRelays[0]) == 0 || len(rt.cepFaninRelays[0][0]) == 0 {
		t.Fatal("expected cep fan-in relay topology to be initialized")
	}
	relayID := rt.cepFaninRelays[0][0][0].ID()
	relayFrom := rt.cepFaninRelays[0][0][0].FromPID()
	rt.cepFaninRelays[0][0][0] = NewCEPFaninRelay(relayID, relayFrom, actor)

	if _, err := rt.Step(context.Background(), []float64{1}); !errors.Is(err, ErrUnexpectedCEPCommandSender) {
		t.Fatalf("expected ErrUnexpectedCEPCommandSender, got %v", err)
	}
}

func TestSimpleRuntimeStepValidatesCEPCommandTargetEnvelope(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.cepActorsByWeight) == 0 || len(rt.cepActorsByWeight[0]) == 0 {
		t.Fatal("expected cep actor pool to be initialized")
	}
	_ = rt.cepActorsByWeight[0][0].TerminateFrom(runtimeExoSelfProcessID)

	expectedInits := scopeCEPActorInitsForWeight(rt.cepActorInits, 0)
	if len(expectedInits) == 0 {
		t.Fatal("expected scoped actor init metadata")
	}

	process, err := NewCEPProcessWithOwner(expectedInits[0].id, runtimeExoSelfProcessID, DefaultCEPName, nil, []string{runtimeCPPProcessID})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	process.substratePID = "substrate_wrong_target"
	actor := NewCEPActor(process)
	t.Cleanup(func() {
		_ = actor.TerminateFrom(runtimeExoSelfProcessID)
	})
	rt.cepActorsByWeight[0][0] = actor
	rt.cepActors[0] = actor
	if len(rt.cepFaninRelays) == 0 || len(rt.cepFaninRelays[0]) == 0 || len(rt.cepFaninRelays[0][0]) == 0 {
		t.Fatal("expected cep fan-in relay topology to be initialized")
	}
	relayID := rt.cepFaninRelays[0][0][0].ID()
	relayFrom := rt.cepFaninRelays[0][0][0].FromPID()
	rt.cepFaninRelays[0][0][0] = NewCEPFaninRelay(relayID, relayFrom, actor)

	if _, err := rt.Step(context.Background(), []float64{1}); !errors.Is(err, ErrUnexpectedCEPCommandTarget) {
		t.Fatalf("expected ErrUnexpectedCEPCommandTarget, got %v", err)
	}
}

func TestSimpleRuntimeBuildsPerWeightCEPActorPool(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
	}, 3)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.cepActorsByWeight) != 3 {
		t.Fatalf("expected actor pool per weight, got=%d", len(rt.cepActorsByWeight))
	}
	if len(rt.cepActorsByWeight[0]) == 0 || rt.cepActorsByWeight[0][0] == nil {
		t.Fatal("expected first weight actor set initialized")
	}
	if len(rt.cepActorsByWeight[1]) == 0 || rt.cepActorsByWeight[1][0] == nil {
		t.Fatal("expected second weight actor set initialized")
	}
	if rt.cepActorsByWeight[0][0] == rt.cepActorsByWeight[1][0] {
		t.Fatal("expected distinct actor instances per weight")
	}
	if len(rt.cepActors) == 0 || rt.cepActors[0] != rt.cepActorsByWeight[0][0] {
		t.Fatal("expected compatibility actor view to mirror first weight actor set")
	}
	if len(rt.cepFaninRelays) != 3 {
		t.Fatalf("expected fan-in relay pool per weight, got=%d", len(rt.cepFaninRelays))
	}
	if len(rt.cepFaninRelays[0]) == 0 || len(rt.cepFaninRelays[0][0]) == 0 || rt.cepFaninRelays[0][0][0] == nil {
		t.Fatal("expected first weight fan-in relay set initialized")
	}
	if len(rt.cepFaninRelays[1]) == 0 || len(rt.cepFaninRelays[1][0]) == 0 || rt.cepFaninRelays[1][0][0] == nil {
		t.Fatal("expected second weight fan-in relay set initialized")
	}
	if rt.cepFaninRelays[0][0][0] == rt.cepFaninRelays[1][0][0] {
		t.Fatal("expected distinct fan-in relay instances per weight")
	}
	if rt.cepFaninRelays[0][0][0].FromPID() != runtimeCPPProcessID {
		t.Fatalf("unexpected fan-in relay sender pid: got=%q", rt.cepFaninRelays[0][0][0].FromPID())
	}
	if len(rt.substrateMailboxes) != 3 {
		t.Fatalf("expected substrate mailbox pool per weight, got=%d", len(rt.substrateMailboxes))
	}
	if rt.substrateMailboxes[0] == nil || rt.substrateMailboxes[1] == nil {
		t.Fatal("expected substrate mailboxes initialized")
	}
	if rt.substrateMailboxes[0].ID() == rt.substrateMailboxes[1].ID() {
		t.Fatalf("expected distinct substrate mailbox IDs per weight, got=%q", rt.substrateMailboxes[0].ID())
	}
	if len(rt.cepCommandRelays) != 3 {
		t.Fatalf("expected cep command relay pool per weight, got=%d", len(rt.cepCommandRelays))
	}
	if len(rt.cepCommandRelays[0]) == 0 || rt.cepCommandRelays[0][0] == nil {
		t.Fatal("expected first weight cep command relay initialized")
	}
	if len(rt.cepCommandRelays[1]) == 0 || rt.cepCommandRelays[1][0] == nil {
		t.Fatal("expected second weight cep command relay initialized")
	}
	if rt.cepCommandRelays[0][0] == rt.cepCommandRelays[1][0] {
		t.Fatal("expected distinct cep command relay instances per weight")
	}
	if rt.cepCommandRelays[0][0].ToPID() != rt.substrateMailboxes[0].ID() {
		t.Fatalf("unexpected first command relay target: got=%q want=%q", rt.cepCommandRelays[0][0].ToPID(), rt.substrateMailboxes[0].ID())
	}
	if rt.cepCommandRelays[1][0].ToPID() != rt.substrateMailboxes[1].ID() {
		t.Fatalf("unexpected second command relay target: got=%q want=%q", rt.cepCommandRelays[1][0].ToPID(), rt.substrateMailboxes[1].ID())
	}
}

func TestSimpleRuntimeUsesConfiguredCEPIDs(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
		CEPIDs:  []string{"cep_custom_runtime"},
		Parameters: map[string]float64{
			"scale": 1,
		},
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	if len(rt.cepActorInits) == 0 {
		t.Fatal("expected cep actor init metadata")
	}
	if rt.cepActorInits[0].id != "cep_custom_runtime" {
		t.Fatalf("expected configured cep id, got=%q", rt.cepActorInits[0].id)
	}

	w, err := rt.Step(context.Background(), []float64{1})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if len(w) != 1 || w[0] != 1 {
		t.Fatalf("unexpected runtime step output with configured cep id: %v", w)
	}
}

func TestBuildCEPActorInitsDeduplicatesConfiguredCEPIDs(t *testing.T) {
	inits, fanin, err := buildCEPActorInits(
		[]CEP{DeltaWeightCEP{}, SetWeightCEP{}, SetABCNCEP{}},
		nil,
		[]string{"n1"},
		nil,
		[]string{"cep_dup", "cep_dup", ""},
	)
	if err != nil {
		t.Fatalf("build cep actor inits: %v", err)
	}
	if len(inits) != 3 || len(fanin) != 3 {
		t.Fatalf("unexpected init shape: inits=%d fanin=%d", len(inits), len(fanin))
	}
	if inits[0].id != "cep_dup" {
		t.Fatalf("unexpected first configured id: %q", inits[0].id)
	}
	if inits[1].id != "cep_dup_2" {
		t.Fatalf("expected deduped second configured id, got=%q", inits[1].id)
	}
	if inits[2].id != "cep_3" {
		t.Fatalf("expected fallback generated id for empty configured entry, got=%q", inits[2].id)
	}
}

func TestBuildCEPActorPoolScopesProcessIDsPerWeight(t *testing.T) {
	inits := []cepActorInit{{
		id:           "cep_scope",
		substratePID: "substrate_scope",
		cepName:      DefaultCEPName,
		faninPIDs:    []string{"n1"},
	}}
	pool, err := buildCEPActorPool(inits, 2)
	if err != nil {
		t.Fatalf("build cep actor pool: %v", err)
	}
	if len(pool) != 2 || len(pool[0]) == 0 || len(pool[1]) == 0 {
		t.Fatalf("unexpected actor pool shape: %+v", pool)
	}

	actorW1 := pool[0][0]
	actorW2 := pool[1][0]
	t.Cleanup(func() {
		_ = actorW1.TerminateFrom(runtimeExoSelfProcessID)
		_ = actorW2.TerminateFrom(runtimeExoSelfProcessID)
	})

	if err := actorW1.Post(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); err != nil {
		t.Fatalf("post w1: %v", err)
	}
	syncID, err := actorW1.PostSync()
	if err != nil {
		t.Fatalf("post sync w1: %v", err)
	}
	if err := actorW1.AwaitSync(syncID); err != nil {
		t.Fatalf("await sync w1: %v", err)
	}
	commandW1, err := actorW1.NextCommand()
	if err != nil {
		t.Fatalf("next command w1: %v", err)
	}

	if err := actorW2.Post(CEPForwardMessage{FromPID: "n1", Input: []float64{1}}); err != nil {
		t.Fatalf("post w2: %v", err)
	}
	syncID, err = actorW2.PostSync()
	if err != nil {
		t.Fatalf("post sync w2: %v", err)
	}
	if err := actorW2.AwaitSync(syncID); err != nil {
		t.Fatalf("await sync w2: %v", err)
	}
	commandW2, err := actorW2.NextCommand()
	if err != nil {
		t.Fatalf("next command w2: %v", err)
	}

	if commandW1.FromPID == commandW2.FromPID {
		t.Fatalf("expected distinct scoped CEP process IDs per weight, got same=%q", commandW1.FromPID)
	}
	if commandW1.FromPID != "cep_scope_w1" || commandW2.FromPID != "cep_scope_w2" {
		t.Fatalf("unexpected scoped CEP process IDs: w1=%q w2=%q", commandW1.FromPID, commandW2.FromPID)
	}
	if commandW1.ToPID == commandW2.ToPID {
		t.Fatalf("expected distinct scoped substrate target PIDs per weight, got same=%q", commandW1.ToPID)
	}
	if commandW1.ToPID != "substrate_scope_w1" || commandW2.ToPID != "substrate_scope_w2" {
		t.Fatalf("unexpected scoped substrate target PIDs: w1=%q w2=%q", commandW1.ToPID, commandW2.ToPID)
	}
}

func TestBuildCEPActorsInitializesFromPayloadState(t *testing.T) {
	actors, err := buildCEPActors([]cepActorInit{{
		id:           "cep_payload_bootstrap",
		substratePID: "substrate_payload",
		cepName:      WeightExpressionCEPName,
		faninPIDs:    []string{"n1", "n2"},
	}})
	if err != nil {
		t.Fatalf("build cep actors: %v", err)
	}
	if len(actors) != 1 || actors[0] == nil {
		t.Fatalf("expected one initialized actor, got=%d", len(actors))
	}
	actor := actors[0]
	t.Cleanup(func() {
		_ = actor.TerminateFrom(runtimeExoSelfProcessID)
	})

	if err := actor.Post(CEPForwardMessage{FromPID: "n1", Input: []float64{0.2}}); err != nil {
		t.Fatalf("post n1: %v", err)
	}
	if err := actor.Post(CEPForwardMessage{FromPID: "n2", Input: []float64{0.8}}); err != nil {
		t.Fatalf("post n2: %v", err)
	}
	syncID, err := actor.PostSync()
	if err != nil {
		t.Fatalf("post sync: %v", err)
	}
	if err := actor.AwaitSync(syncID); err != nil {
		t.Fatalf("await sync: %v", err)
	}

	command, err := actor.NextCommand()
	if err != nil {
		t.Fatalf("next command: %v", err)
	}
	if command.FromPID != "cep_payload_bootstrap" || command.Command != WeightExpressionCEPName {
		t.Fatalf("unexpected command envelope: %+v", command)
	}
	if command.ToPID != "substrate_payload" {
		t.Fatalf("unexpected command target pid: got=%q want=%q", command.ToPID, "substrate_payload")
	}
}

func TestSimpleRuntimeStepRequiresSubstrateMailbox(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.substrateMailboxes) == 0 {
		t.Fatal("expected substrate mailbox to be initialized")
	}
	rt.substrateMailboxes[0] = nil

	if _, err := rt.Step(context.Background(), []float64{1}); !errors.Is(err, ErrMissingSubstrateMailbox) {
		t.Fatalf("expected ErrMissingSubstrateMailbox, got %v", err)
	}
}

func TestSubstrateCommandMailboxActorSyncAndTerminate(t *testing.T) {
	mailbox := newSubstrateCommandMailbox("substrate_w1")
	commandA := CEPCommand{
		FromPID: "cep_a",
		ToPID:   "substrate_w1",
		Command: SetIterativeCEPName,
		Signal:  []float64{0.1},
	}
	commandB := CEPCommand{
		FromPID: "cep_b",
		ToPID:   "substrate_w1",
		Command: SetIterativeCEPName,
		Signal:  []float64{0.2},
	}

	if err := mailbox.Post(commandA); err != nil {
		t.Fatalf("post commandA: %v", err)
	}
	if err := mailbox.Post(commandB); err != nil {
		t.Fatalf("post commandB: %v", err)
	}
	syncID, err := mailbox.PostSync()
	if err != nil {
		t.Fatalf("post sync: %v", err)
	}
	if err := mailbox.AwaitSync(syncID); err != nil {
		t.Fatalf("await sync: %v", err)
	}

	commands := mailbox.Drain()
	if len(commands) != 2 {
		t.Fatalf("expected 2 commands in mailbox, got=%d", len(commands))
	}
	if commands[0].FromPID != "cep_a" || commands[1].FromPID != "cep_b" {
		t.Fatalf("unexpected command ordering from mailbox actor: %+v", commands)
	}

	mailbox.Terminate()
	if !mailbox.IsTerminated() {
		t.Fatal("expected mailbox terminated state")
	}
	if err := mailbox.Post(commandA); !errors.Is(err, ErrSubstrateMailboxTerminated) {
		t.Fatalf("expected ErrSubstrateMailboxTerminated from post after terminate, got %v", err)
	}
	if _, err := mailbox.PostSync(); !errors.Is(err, ErrSubstrateMailboxTerminated) {
		t.Fatalf("expected ErrSubstrateMailboxTerminated from post sync after terminate, got %v", err)
	}
}

func TestSimpleRuntimeStepRequiresCEPFaninRelay(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.cepFaninRelays) == 0 || len(rt.cepFaninRelays[0]) == 0 || len(rt.cepFaninRelays[0][0]) == 0 {
		t.Fatal("expected cep fan-in relay topology initialized")
	}
	rt.cepFaninRelays[0][0][0] = nil

	if _, err := rt.Step(context.Background(), []float64{1}); !errors.Is(err, ErrMissingCEPFaninRelay) {
		t.Fatalf("expected ErrMissingCEPFaninRelay, got %v", err)
	}
}

func TestSimpleRuntimeStepRequiresCEPCommandRelay(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	rt, err := NewSimpleRuntime(Spec{
		CPPName: DefaultCPPName,
		CEPName: DefaultCEPName,
	}, 1)
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}
	if len(rt.cepCommandRelays) == 0 || len(rt.cepCommandRelays[0]) == 0 {
		t.Fatal("expected cep command relay topology initialized")
	}
	rt.cepCommandRelays[0][0] = nil

	if _, err := rt.Step(context.Background(), []float64{1}); !errors.Is(err, ErrMissingCEPCommandRelay) {
		t.Fatalf("expected ErrMissingCEPCommandRelay, got %v", err)
	}
}

func TestCEPFaninRelayMailboxForwardAndTerminate(t *testing.T) {
	process, err := NewCEPProcessWithID("cep_fanin_relay", DefaultCEPName, nil, []string{"n1"})
	if err != nil {
		t.Fatalf("new cep process: %v", err)
	}
	actor := NewCEPActor(process)
	t.Cleanup(func() {
		_ = actor.TerminateFrom(runtimeExoSelfProcessID)
	})

	relay := NewCEPFaninRelay("fanin_1", "n1", actor)
	if err := relay.Post([]float64{1}); err != nil {
		t.Fatalf("relay post: %v", err)
	}
	syncID, err := relay.PostSync()
	if err != nil {
		t.Fatalf("relay post sync: %v", err)
	}
	if err := relay.AwaitSync(syncID); err != nil {
		t.Fatalf("relay await sync: %v", err)
	}
	syncID, err = actor.PostSync()
	if err != nil {
		t.Fatalf("post sync: %v", err)
	}
	if err := actor.AwaitSync(syncID); err != nil {
		t.Fatalf("await sync: %v", err)
	}
	command, err := actor.NextCommand()
	if err != nil {
		t.Fatalf("next command: %v", err)
	}
	if command.FromPID != "cep_fanin_relay" || command.Command != SetIterativeCEPName {
		t.Fatalf("unexpected command envelope from relay-forwarded post: %+v", command)
	}

	relay.Terminate()
	if err := relay.Post([]float64{1}); !errors.Is(err, ErrCEPFaninRelayTerminated) {
		t.Fatalf("expected ErrCEPFaninRelayTerminated after relay stop, got %v", err)
	}
	if _, err := relay.PostSync(); !errors.Is(err, ErrCEPFaninRelayTerminated) {
		t.Fatalf("expected ErrCEPFaninRelayTerminated from relay post sync after stop, got %v", err)
	}
}

func TestCEPCommandRelayMailboxForwardAndTerminate(t *testing.T) {
	mailbox := newSubstrateCommandMailbox("substrate_w1")
	relay := NewCEPCommandRelay("command_cep_1_w1", "substrate_w1", mailbox)

	command := CEPCommand{
		FromPID: "cep_1_w1",
		ToPID:   "substrate_w1",
		Command: SetIterativeCEPName,
		Signal:  []float64{0.5},
	}
	if err := relay.Post(command); err != nil {
		t.Fatalf("relay post: %v", err)
	}
	syncID, err := relay.PostSync()
	if err != nil {
		t.Fatalf("relay post sync: %v", err)
	}
	if err := relay.AwaitSync(syncID); err != nil {
		t.Fatalf("relay await sync: %v", err)
	}
	if err := relay.NextError(); !errors.Is(err, ErrCEPCommandRelayNoError) {
		t.Fatalf("expected no relay forwarding errors, got %v", err)
	}

	mailboxSyncID, err := mailbox.PostSync()
	if err != nil {
		t.Fatalf("mailbox post sync: %v", err)
	}
	if err := mailbox.AwaitSync(mailboxSyncID); err != nil {
		t.Fatalf("mailbox await sync: %v", err)
	}
	commands := mailbox.Drain()
	if len(commands) != 1 {
		t.Fatalf("expected one command forwarded through relay, got=%d", len(commands))
	}
	if commands[0].FromPID != "cep_1_w1" || commands[0].ToPID != "substrate_w1" {
		t.Fatalf("unexpected forwarded command envelope: %+v", commands[0])
	}

	if err := relay.Post(CEPCommand{
		FromPID: "cep_1_w1",
		ToPID:   "substrate_wrong",
		Command: SetIterativeCEPName,
		Signal:  []float64{0.1},
	}); err != nil {
		t.Fatalf("relay post wrong target command: %v", err)
	}
	syncID, err = relay.PostSync()
	if err != nil {
		t.Fatalf("relay post sync wrong target: %v", err)
	}
	if err := relay.AwaitSync(syncID); err != nil {
		t.Fatalf("relay await sync wrong target: %v", err)
	}
	if err := relay.NextError(); !errors.Is(err, ErrUnexpectedCEPCommandTarget) {
		t.Fatalf("expected ErrUnexpectedCEPCommandTarget relay error, got %v", err)
	}

	relay.Terminate()
	if err := relay.Post(command); !errors.Is(err, ErrCEPCommandRelayTerminated) {
		t.Fatalf("expected ErrCEPCommandRelayTerminated after relay stop, got %v", err)
	}
	if _, err := relay.PostSync(); !errors.Is(err, ErrCEPCommandRelayTerminated) {
		t.Fatalf("expected ErrCEPCommandRelayTerminated from relay post sync after stop, got %v", err)
	}
	mailbox.Terminate()
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

func TestSimpleRuntimeBackupRestoreIncludesPersistentWeightParameters(t *testing.T) {
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

	first, err := rt.StepWithFanin(context.Background(), []float64{0, 0, 0, 0, 0}, map[string]float64{
		"n1": 1,
		"n2": 0.2,
		"n3": 0.5,
		"n4": -0.1,
		"n5": 0.8,
	})
	if err != nil {
		t.Fatalf("step with initial coefficients: %v", err)
	}
	if len(first) != 1 || math.Abs(first[0]-0.4) > 1e-9 {
		t.Fatalf("unexpected initial weight update: got=%v want=0.4", first)
	}

	rt.Backup()

	second, err := rt.StepWithFanin(context.Background(), []float64{0, 0, 0, 0, 0}, map[string]float64{
		"n1": 1,
		"n2": 1.0,
		"n3": 0.0,
		"n4": 0.0,
		"n5": 1.0,
	})
	if err != nil {
		t.Fatalf("step with replacement coefficients: %v", err)
	}
	if len(second) != 1 || math.Abs(second[0]-0.8) > 1e-9 {
		t.Fatalf("unexpected replacement-coefficient update: got=%v want=0.8", second)
	}

	if err := rt.Restore(); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if len(rt.weightParams) != 1 {
		t.Fatalf("expected one restored weight-parameter set, got=%d", len(rt.weightParams))
	}
	restored := rt.weightParams[0]
	if restored["A"] != 0.2 || restored["B"] != 0.5 || restored["C"] != -0.1 || restored["N"] != 0.8 {
		t.Fatalf("expected restore to recover persisted coefficients, got=%v", restored)
	}
}

func TestSimpleRuntimeResetClearsPersistentWeightParametersToBaseConfig(t *testing.T) {
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

	updated, err := rt.StepWithFanin(context.Background(), []float64{0, 0, 0, 0, 0}, map[string]float64{
		"n1": 1,
		"n2": 0.2,
		"n3": 0.5,
		"n4": -0.1,
		"n5": 0.8,
	})
	if err != nil {
		t.Fatalf("step with coefficient update: %v", err)
	}
	if len(updated) != 1 || math.Abs(updated[0]-0.4) > 1e-9 {
		t.Fatalf("unexpected coefficient update result: got=%v want=0.4", updated)
	}

	rt.Reset()

	resetWeights := rt.Weights()
	if len(resetWeights) != 1 || resetWeights[0] != 0 {
		t.Fatalf("expected reset weights to be zeroed, got=%v", resetWeights)
	}
	if len(rt.weightParams) != 1 {
		t.Fatalf("expected one reset weight-parameter set, got=%d", len(rt.weightParams))
	}
	if rt.weightParams[0] != nil {
		t.Fatalf("expected reset to clear persisted coefficients back to base config, got=%v", rt.weightParams[0])
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
