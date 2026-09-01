package substrate

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestLayerRuntimeStaticResetThenHold(t *testing.T) {
	rt, err := NewLayerRuntime(LayerRuntimeSpec{
		Plasticity: SubstratePlasticityNone,
		LinkForm:   LinkFormL2LFeedforward,
		Substrate: []CoordinateHyperlayer{
			{
				{Coords: []float64{0}},
				{Coords: []float64{1}},
			},
			{
				{Coords: []float64{2}, Weights: []float64{99, 99}},
			},
		},
		StaticCPP: coordinateSumCPP{},
		CEPs:      []CEP{rawSignalCEP{}},
	})
	if err != nil {
		t.Fatalf("new layer runtime: %v", err)
	}

	got, err := rt.Step(context.Background(), []float64{0.5, 0.25})
	if err != nil {
		t.Fatalf("step reset: %v", err)
	}
	want := []float64{math.Tanh(0.5*2 + 0.25*3)}
	if !nearlyEqualSlice(got, want, 1e-12) {
		t.Fatalf("unexpected reset output: got=%v want=%v", got, want)
	}
	if !reflect.DeepEqual(rt.Weights(), []float64{2, 3}) {
		t.Fatalf("unexpected populated weights: %v", rt.Weights())
	}

	got, err = rt.Step(context.Background(), []float64{1, -1})
	if err != nil {
		t.Fatalf("step hold: %v", err)
	}
	want = []float64{math.Tanh(1*2 + -1*3)}
	if !nearlyEqualSlice(got, want, 1e-12) {
		t.Fatalf("unexpected hold output: got=%v want=%v", got, want)
	}
	if !reflect.DeepEqual(rt.Weights(), []float64{2, 3}) {
		t.Fatalf("expected hold to reuse weights, got=%v", rt.Weights())
	}
}

func TestLayerRuntimeIterativeRepopulatesEachStep(t *testing.T) {
	rt, err := NewLayerRuntime(LayerRuntimeSpec{
		Plasticity: SubstratePlasticityIterative,
		LinkForm:   LinkFormL2LFeedforward,
		Substrate: []CoordinateHyperlayer{
			{{Coords: []float64{0}, Output: 0.2}},
			{{Coords: []float64{0.3}, Output: 0.3, Weights: []float64{0.4}}},
		},
		IterativeCPP: iowDeltaCPP{},
		CEPs:         []CEP{rawSignalCEP{}},
	})
	if err != nil {
		t.Fatalf("new layer runtime: %v", err)
	}

	got, err := rt.Step(context.Background(), []float64{0.8})
	if err != nil {
		t.Fatalf("step iterative reset: %v", err)
	}
	firstWeight := 0.2 + 0.3
	want := []float64{math.Tanh(0.8 * firstWeight)}
	if !nearlyEqualSlice(got, want, 1e-12) {
		t.Fatalf("unexpected first output: got=%v want=%v", got, want)
	}
	if !nearlyEqualSlice(rt.Weights(), []float64{firstWeight}, 1e-12) {
		t.Fatalf("unexpected first weight: %v", rt.Weights())
	}

	got, err = rt.Step(context.Background(), []float64{0.8})
	if err != nil {
		t.Fatalf("step iterative: %v", err)
	}
	secondWeight := 0.2 + want[0]
	want = []float64{math.Tanh(0.8 * secondWeight)}
	if !nearlyEqualSlice(got, want, 1e-12) {
		t.Fatalf("unexpected second output: got=%v want=%v", got, want)
	}
	if !nearlyEqualSlice(rt.Weights(), []float64{secondWeight}, 1e-12) {
		t.Fatalf("unexpected second weight: %v", rt.Weights())
	}
}

func TestLayerRuntimeABCNUsesTypedState(t *testing.T) {
	rt, err := NewLayerRuntime(LayerRuntimeSpec{
		Plasticity: SubstratePlasticityABCN,
		LinkForm:   LinkFormL2LFeedforward,
		ABCN: ABCNSubstrate{
			InputLayer: CoordinateHyperlayer{{Coords: []float64{0}, Output: 0.2}},
			Layers: []ABCNCoordinateHyperlayer{
				{{Coords: []float64{0}, Output: 0.3, Weights: []ABCNWeight{{Weight: 0.4}}}},
			},
		},
		IterativeCPP: abcnSignalCPP{},
		CEPs:         []CEP{rawSignalCEP{}},
	})
	if err != nil {
		t.Fatalf("new layer runtime: %v", err)
	}

	got, err := rt.Step(context.Background(), []float64{0.8})
	if err != nil {
		t.Fatalf("step abcn reset: %v", err)
	}
	want := []float64{0}
	if !nearlyEqualSlice(got, want, 1e-12) {
		t.Fatalf("unexpected abcn output: got=%v want=%v", got, want)
	}
	if !nearlyEqualSlice(rt.Weights(), []float64{0.12}, 1e-12) {
		t.Fatalf("unexpected abcn weight update: %v", rt.Weights())
	}
}

func TestLayerRuntimeBackupRestoreResetAndTerminate(t *testing.T) {
	rt, err := NewLayerRuntime(LayerRuntimeSpec{
		Plasticity: SubstratePlasticityIterative,
		LinkForm:   LinkFormL2LFeedforward,
		Substrate: []CoordinateHyperlayer{
			{{Coords: []float64{0}, Output: 0.2}},
			{{Coords: []float64{0.3}, Output: 0.3, Weights: []float64{0.4}}},
		},
		IterativeCPP: iowDeltaCPP{},
		CEPs:         []CEP{rawSignalCEP{}},
	})
	if err != nil {
		t.Fatalf("new layer runtime: %v", err)
	}
	if err := rt.Restore(); !errors.Is(err, ErrNoSubstrateBackup) {
		t.Fatalf("expected missing backup error, got %v", err)
	}

	if _, err := rt.Step(context.Background(), []float64{0.8}); err != nil {
		t.Fatalf("first step: %v", err)
	}
	rt.Backup()
	backupWeights := rt.Weights()

	if _, err := rt.Step(context.Background(), []float64{0.8}); err != nil {
		t.Fatalf("second step: %v", err)
	}
	if reflect.DeepEqual(rt.Weights(), backupWeights) {
		t.Fatalf("expected second step to update weights")
	}

	rt.Terminate()
	if _, err := rt.Step(context.Background(), []float64{0.8}); !errors.Is(err, ErrSubstrateRuntimeTerminated) {
		t.Fatalf("expected terminated error, got %v", err)
	}
	if err := rt.Restore(); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if !nearlyEqualSlice(rt.Weights(), backupWeights, 1e-12) {
		t.Fatalf("unexpected restored weights: got=%v want=%v", rt.Weights(), backupWeights)
	}

	rt.Terminate()
	rt.Reset()
	if !nearlyEqualSlice(rt.Weights(), []float64{0.4}, 1e-12) {
		t.Fatalf("unexpected reset weights: %v", rt.Weights())
	}
	if _, err := rt.Step(context.Background(), []float64{0.8}); err != nil {
		t.Fatalf("step after reset revives runtime: %v", err)
	}
}

func TestLayerRuntimeSnapshotReportsScalarStateAndCopiesLayers(t *testing.T) {
	rt, err := NewLayerRuntime(LayerRuntimeSpec{
		Plasticity: SubstratePlasticityNone,
		LinkForm:   LinkFormL2LFeedforward,
		Substrate: []CoordinateHyperlayer{
			{{Coords: []float64{0}}},
			{{Coords: []float64{1}, Weights: []float64{2}}},
		},
		StaticCPP: coordinateSumCPP{},
		CEPs:      []CEP{rawSignalCEP{}},
	})
	if err != nil {
		t.Fatalf("new layer runtime: %v", err)
	}

	initial := rt.Snapshot()
	if initial.Plasticity != SubstratePlasticityNone || initial.LinkForm != LinkFormL2LFeedforward || initial.StateMode != SubstrateStateReset {
		t.Fatalf("unexpected initial snapshot modes: %+v", initial)
	}
	if initial.Terminated {
		t.Fatalf("expected active runtime")
	}
	if len(initial.Substrate) != 2 || len(initial.ABCN.Layers) != 0 || !reflect.DeepEqual(initial.Weights, []float64{2}) {
		t.Fatalf("unexpected initial snapshot state: %+v", initial)
	}

	initial.Substrate[1][0].Weights[0] = 99
	initial.Weights[0] = 88
	if !reflect.DeepEqual(rt.Weights(), []float64{2}) {
		t.Fatalf("expected snapshot copy, got runtime weights=%v", rt.Weights())
	}

	if _, err := rt.Step(context.Background(), []float64{0.5}); err != nil {
		t.Fatalf("step: %v", err)
	}
	updated := rt.Snapshot()
	if updated.StateMode != SubstrateStateHold {
		t.Fatalf("expected hold state after non-plastic reset, got %+v", updated)
	}
	if len(updated.Substrate) != 2 || len(updated.Substrate[1][0].Weights) != 1 {
		t.Fatalf("unexpected updated scalar layer shape: %+v", updated.Substrate)
	}

	rt.Terminate()
	if !rt.Snapshot().Terminated {
		t.Fatalf("expected terminated snapshot state")
	}
}

func TestLayerRuntimeFromSnapshotReplaysScalarHoldState(t *testing.T) {
	original, err := NewLayerRuntime(LayerRuntimeSpec{
		Plasticity: SubstratePlasticityNone,
		LinkForm:   LinkFormL2LFeedforward,
		Substrate: []CoordinateHyperlayer{
			{{Coords: []float64{0}}},
			{{Coords: []float64{1}, Weights: []float64{99}}},
		},
		StaticCPP: coordinateSumCPP{},
		CEPs:      []CEP{rawSignalCEP{}},
	})
	if err != nil {
		t.Fatalf("new original runtime: %v", err)
	}
	if _, err := original.Step(context.Background(), []float64{0.5}); err != nil {
		t.Fatalf("prime original runtime: %v", err)
	}
	snapshot := original.Snapshot()

	replayed, err := NewLayerRuntimeFromSnapshot(snapshot, LayerRuntimeSpec{})
	if err != nil {
		t.Fatalf("replay runtime from snapshot: %v", err)
	}
	inputs := []float64{0.25}
	want, err := original.Step(context.Background(), inputs)
	if err != nil {
		t.Fatalf("continue original runtime: %v", err)
	}
	got, err := replayed.Step(context.Background(), inputs)
	if err != nil {
		t.Fatalf("step replayed runtime: %v", err)
	}
	if !nearlyEqualSlice(got, want, 1e-12) {
		t.Fatalf("replayed scalar output mismatch: got=%v want=%v", got, want)
	}
	if !reflect.DeepEqual(replayed.Weights(), snapshot.Weights) {
		t.Fatalf("expected replayed weights to match snapshot weights, got=%v want=%v", replayed.Weights(), snapshot.Weights)
	}
}

func TestLayerRuntimeFromSnapshotReplaysABCNHoldState(t *testing.T) {
	snapshot := LayerRuntimeSnapshot{
		Plasticity: SubstratePlasticityABCN,
		LinkForm:   LinkFormL2LFeedforward,
		StateMode:  SubstrateStateHold,
		ABCN: ABCNSubstrate{
			InputLayer: CoordinateHyperlayer{{Coords: []float64{0}}},
			Layers: []ABCNCoordinateHyperlayer{
				{{Coords: []float64{1}, Weights: []ABCNWeight{{Weight: 0.5, A: 0.1, B: 0.2, C: 0.3, N: 0.4}}}},
			},
		},
		Weights: []float64{0.5},
	}
	replayed, err := NewLayerRuntimeFromSnapshot(snapshot, LayerRuntimeSpec{})
	if err != nil {
		t.Fatalf("replay runtime from snapshot: %v", err)
	}

	got, err := replayed.Step(context.Background(), []float64{0.25})
	if err != nil {
		t.Fatalf("step replayed runtime: %v", err)
	}
	want := []float64{math.Tanh(0.25 * 0.5)}
	if !nearlyEqualSlice(got, want, 1e-12) {
		t.Fatalf("replayed abcn output mismatch: got=%v want=%v", got, want)
	}
	next := replayed.Snapshot()
	gotWeight := next.ABCN.Layers[0][0].Weights[0]
	wantWeight := snapshot.ABCN.Layers[0][0].Weights[0]
	if gotWeight.A != wantWeight.A || gotWeight.B != wantWeight.B || gotWeight.C != wantWeight.C || gotWeight.N != wantWeight.N {
		t.Fatalf("expected abcn coefficients to survive replay, got=%+v", next.ABCN.Layers[0][0].Weights[0])
	}
	if gotWeight.Weight == wantWeight.Weight {
		t.Fatalf("expected abcn weight value to update during replay, got=%+v", gotWeight)
	}
}

func TestLayerRuntimeFromSnapshotReplaysABCNNonHoldStatesWithComponents(t *testing.T) {
	tests := []struct {
		name          string
		stateMode     string
		nextStateMode string
	}{
		{name: "reset", stateMode: SubstrateStateReset, nextStateMode: SubstrateStateHold},
		{name: "iterative", stateMode: SubstrateStateIterative, nextStateMode: SubstrateStateIterative},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := LayerRuntimeSnapshot{
				Plasticity: SubstratePlasticityABCN,
				LinkForm:   LinkFormL2LFeedforward,
				StateMode:  tt.stateMode,
				ABCN: ABCNSubstrate{
					InputLayer: CoordinateHyperlayer{{Coords: []float64{0}, Output: 0.2}},
					Layers: []ABCNCoordinateHyperlayer{
						{{Coords: []float64{1}, Output: 0.3, Weights: []ABCNWeight{{Weight: 0.4, A: 0.1, B: 0.2, C: 0.3, N: 0.4}}}},
					},
				},
				Weights: []float64{0.4},
			}
			replayed, err := NewLayerRuntimeFromSnapshot(snapshot, LayerRuntimeSpec{
				IterativeCPP: abcnSignalCPP{},
				CEPs:         []CEP{rawSignalCEP{}},
			})
			if err != nil {
				t.Fatalf("replay runtime from snapshot: %v", err)
			}

			got, err := replayed.Step(context.Background(), []float64{0.8})
			if err != nil {
				t.Fatalf("step replayed runtime: %v", err)
			}
			if len(got) != 1 {
				t.Fatalf("expected one replay output, got=%v", got)
			}
			next := replayed.Snapshot()
			if next.StateMode != tt.nextStateMode {
				t.Fatalf("unexpected next state: got=%q want=%q snapshot=%+v", next.StateMode, tt.nextStateMode, next)
			}
			updated := next.ABCN.Layers[0][0].Weights[0]
			if updated.A == 0 || updated.B == 0 || updated.C == 0 || updated.N == 0 {
				t.Fatalf("expected replayed abcn coefficients from component path, got=%+v", updated)
			}
			if updated.Weight == snapshot.ABCN.Layers[0][0].Weights[0].Weight {
				t.Fatalf("expected abcn replay to update weight, got=%+v", updated)
			}
		})
	}
}

func TestLayerRuntimeFromSnapshotPreservesTerminatedState(t *testing.T) {
	replayed, err := NewLayerRuntimeFromSnapshot(LayerRuntimeSnapshot{
		Plasticity: SubstratePlasticityNone,
		LinkForm:   LinkFormL2LFeedforward,
		StateMode:  SubstrateStateHold,
		Terminated: true,
		Substrate: []CoordinateHyperlayer{
			{{Coords: []float64{0}}},
			{{Coords: []float64{1}, Weights: []float64{0.5}}},
		},
		Weights: []float64{0.5},
	}, LayerRuntimeSpec{})
	if err != nil {
		t.Fatalf("replay runtime from terminated snapshot: %v", err)
	}
	if _, err := replayed.Step(context.Background(), []float64{0.25}); !errors.Is(err, ErrSubstrateRuntimeTerminated) {
		t.Fatalf("expected terminated replay runtime error, got %v", err)
	}
}

func TestLayerRuntimeSnapshotReportsABCNCoefficientsAndCopiesState(t *testing.T) {
	rt, err := NewLayerRuntime(LayerRuntimeSpec{
		Plasticity: SubstratePlasticityABCN,
		LinkForm:   LinkFormL2LFeedforward,
		ABCN: ABCNSubstrate{
			InputLayer: CoordinateHyperlayer{{Coords: []float64{0}}},
			Layers: []ABCNCoordinateHyperlayer{
				{{Coords: []float64{1}, Weights: []ABCNWeight{{Weight: 0.5, A: 0.1, B: 0.2, C: 0.3, N: 0.4}}}},
			},
		},
	})
	if err != nil {
		t.Fatalf("new layer runtime: %v", err)
	}

	snapshot := rt.Snapshot()
	got := snapshot.ABCN.Layers[0][0].Weights[0]
	want := ABCNWeight{Weight: 0.5, A: 0.1, B: 0.2, C: 0.3, N: 0.4}
	if got != want {
		t.Fatalf("unexpected abcn coefficients: got=%+v want=%+v", got, want)
	}
	if !reflect.DeepEqual(snapshot.Weights, []float64{0.5}) {
		t.Fatalf("unexpected abcn snapshot weights: %v", snapshot.Weights)
	}

	snapshot.ABCN.Layers[0][0].Weights[0].A = 99
	snapshot.Weights[0] = 88
	next := rt.Snapshot()
	if next.ABCN.Layers[0][0].Weights[0] != want || !reflect.DeepEqual(next.Weights, []float64{0.5}) {
		t.Fatalf("expected abcn snapshot copy, got %+v", next)
	}
}

func TestLayerRuntimeConstructorCopiesInputs(t *testing.T) {
	layers := []CoordinateHyperlayer{
		{{Coords: []float64{0}}},
		{{Coords: []float64{1}, Weights: []float64{2}}},
	}
	rt, err := NewLayerRuntime(LayerRuntimeSpec{
		Substrate: layers,
	})
	if err != nil {
		t.Fatalf("new layer runtime: %v", err)
	}

	layers[1][0].Weights[0] = 99
	if !reflect.DeepEqual(rt.Weights(), []float64{2}) {
		t.Fatalf("expected constructor copy, got weights=%v", rt.Weights())
	}

	got := rt.Weights()
	got[0] = 42
	if !reflect.DeepEqual(rt.Weights(), []float64{2}) {
		t.Fatalf("expected weights copy, got weights=%v", rt.Weights())
	}
}

func TestNewLayerRuntimeValidatesTypedState(t *testing.T) {
	if _, err := NewLayerRuntime(LayerRuntimeSpec{Plasticity: "unknown"}); err == nil {
		t.Fatalf("expected unknown plasticity error")
	}
	if _, err := NewLayerRuntime(LayerRuntimeSpec{StateMode: "unknown"}); err == nil {
		t.Fatalf("expected unknown state error")
	}
	if _, err := NewLayerRuntime(LayerRuntimeSpec{Plasticity: SubstratePlasticityABCN}); err == nil {
		t.Fatalf("expected missing abcn state error")
	}
}

func nearlyEqualSlice(got, want []float64, tolerance float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tolerance {
			return false
		}
	}
	return true
}
