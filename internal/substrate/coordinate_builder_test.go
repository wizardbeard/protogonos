package substrate

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type sumCoordinateBuilderCPP struct{}

func (sumCoordinateBuilderCPP) Name() string { return "sum_coordinate_builder_cpp" }

func (sumCoordinateBuilderCPP) Compute(_ context.Context, _ []float64, _ map[string]float64) (float64, error) {
	return 0, nil
}

func (sumCoordinateBuilderCPP) ComputeCoordinates(_ context.Context, presynaptic []float64, postsynaptic []float64, _ map[string]float64) ([]float64, error) {
	return []float64{presynaptic[0] + postsynaptic[0], 1}, nil
}

func TestBuildL2LFeedforwardCoordinatePairsOrdersCurrentThenPreviousLayer(t *testing.T) {
	pairs, err := BuildL2LFeedforwardCoordinatePairs(
		[][]float64{{0}, {1}},
		[][]float64{{10}, {20}},
	)
	if err != nil {
		t.Fatalf("build coordinate pairs: %v", err)
	}

	want := []CoordinatePair{
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{1}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{20}},
		{PresynapticCoords: []float64{1}, PostsynapticCoords: []float64{20}},
	}
	if !reflect.DeepEqual(pairs, want) {
		t.Fatalf("unexpected l2l coordinate order: got=%v want=%v", pairs, want)
	}
}

func TestBuildL2LFeedforwardCoordinatePairsCopiesCoordinates(t *testing.T) {
	presynaptic := [][]float64{{0.25, 0.75}}
	postsynaptic := [][]float64{{1.5, -0.5}}

	pairs, err := BuildL2LFeedforwardCoordinatePairs(presynaptic, postsynaptic)
	if err != nil {
		t.Fatalf("build coordinate pairs: %v", err)
	}

	presynaptic[0][0] = 99
	postsynaptic[0][0] = 88

	if got := pairs[0].PresynapticCoords[0]; got != 0.25 {
		t.Fatalf("presynaptic coordinate was not copied: got=%v", got)
	}
	if got := pairs[0].PostsynapticCoords[0]; got != 1.5 {
		t.Fatalf("postsynaptic coordinate was not copied: got=%v", got)
	}
}

func TestBuildL2LFeedforwardCoordinatePairsValidatesInput(t *testing.T) {
	if _, err := BuildL2LFeedforwardCoordinatePairs(nil, [][]float64{{1}}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing presynaptic coordinates, got %v", err)
	}
	if _, err := BuildL2LFeedforwardCoordinatePairs([][]float64{{1}}, nil); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing postsynaptic coordinates, got %v", err)
	}
}

func TestBuildL2LFeedforwardCoordinatePairsPopulatesCoordinateBatch(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("sum_coordinate_builder_cpp", func() CPP {
		return sumCoordinateBuilderCPP{}
	}); err != nil {
		t.Fatalf("register coordinate cpp: %v", err)
	}

	pairs, err := BuildL2LFeedforwardCoordinatePairs(
		[][]float64{{0.25}, {0.5}},
		[][]float64{{1}, {2}},
	)
	if err != nil {
		t.Fatalf("build coordinate pairs: %v", err)
	}

	rt, err := NewSimpleRuntime(Spec{
		CPPName:      "sum_coordinate_builder_cpp",
		CEPName:      WeightExpressionCEPName,
		CEPFaninPIDs: []string{"cpp1", "cpp2"},
	}, len(pairs))
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	weights, err := rt.StepCoordinateBatch(context.Background(), pairs)
	if err != nil {
		t.Fatalf("step coordinate batch: %v", err)
	}

	want := []float64{1.25, 1.5, 2.25, 2.5}
	if len(weights) != len(want) {
		t.Fatalf("unexpected weight count: got=%d want=%d", len(weights), len(want))
	}
	for i := range want {
		if math.Abs(weights[i]-want[i]) > 1e-9 {
			t.Fatalf("unexpected weight[%d]: got=%v want=%v all=%v", i, weights[i], want[i], weights)
		}
	}
}

func TestBuildNeuronSelfRecurrentCoordinatePairsOrdersSelfThenPreviousLayer(t *testing.T) {
	pairs, err := BuildNeuronSelfRecurrentCoordinatePairs(
		[][]float64{{0}, {1}},
		[][]float64{{10}, {20}},
	)
	if err != nil {
		t.Fatalf("build coordinate pairs: %v", err)
	}

	want := []CoordinatePair{
		{PresynapticCoords: []float64{10}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{1}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{20}, PostsynapticCoords: []float64{20}},
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{20}},
		{PresynapticCoords: []float64{1}, PostsynapticCoords: []float64{20}},
	}
	if !reflect.DeepEqual(pairs, want) {
		t.Fatalf("unexpected neuron-self coordinate order: got=%v want=%v", pairs, want)
	}
}

func TestBuildNeuronSelfRecurrentCoordinatePairsCopiesCoordinates(t *testing.T) {
	previous := [][]float64{{0.25}}
	current := [][]float64{{1.5}}

	pairs, err := BuildNeuronSelfRecurrentCoordinatePairs(previous, current)
	if err != nil {
		t.Fatalf("build coordinate pairs: %v", err)
	}

	previous[0][0] = 99
	current[0][0] = 88

	if got := pairs[0].PresynapticCoords[0]; got != 1.5 {
		t.Fatalf("self presynaptic coordinate was not copied: got=%v", got)
	}
	if got := pairs[0].PostsynapticCoords[0]; got != 1.5 {
		t.Fatalf("self postsynaptic coordinate was not copied: got=%v", got)
	}
	if got := pairs[1].PresynapticCoords[0]; got != 0.25 {
		t.Fatalf("previous presynaptic coordinate was not copied: got=%v", got)
	}
}

func TestBuildNeuronSelfRecurrentCoordinatePairsValidatesInput(t *testing.T) {
	if _, err := BuildNeuronSelfRecurrentCoordinatePairs(nil, [][]float64{{1}}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing previous-layer coordinates, got %v", err)
	}
	if _, err := BuildNeuronSelfRecurrentCoordinatePairs([][]float64{{1}}, nil); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing current-layer coordinates, got %v", err)
	}
}

func TestBuildNeuronSelfRecurrentCoordinatePairsPopulatesCoordinateBatch(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("sum_coordinate_builder_cpp", func() CPP {
		return sumCoordinateBuilderCPP{}
	}); err != nil {
		t.Fatalf("register coordinate cpp: %v", err)
	}

	pairs, err := BuildNeuronSelfRecurrentCoordinatePairs(
		[][]float64{{0.25}},
		[][]float64{{0.5}, {1}},
	)
	if err != nil {
		t.Fatalf("build coordinate pairs: %v", err)
	}

	rt, err := NewSimpleRuntime(Spec{
		CPPName:      "sum_coordinate_builder_cpp",
		CEPName:      WeightExpressionCEPName,
		CEPFaninPIDs: []string{"cpp1", "cpp2"},
	}, len(pairs))
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	weights, err := rt.StepCoordinateBatch(context.Background(), pairs)
	if err != nil {
		t.Fatalf("step coordinate batch: %v", err)
	}

	want := []float64{1, 0.75, 2, 1.25}
	if len(weights) != len(want) {
		t.Fatalf("unexpected weight count: got=%d want=%d", len(weights), len(want))
	}
	for i := range want {
		if math.Abs(weights[i]-want[i]) > 1e-9 {
			t.Fatalf("unexpected weight[%d]: got=%v want=%v all=%v", i, weights[i], want[i], weights)
		}
	}
}

func TestBuildJordanRecurrentCoordinatePairsOrdersInputThenOutputSources(t *testing.T) {
	pairs, err := BuildJordanRecurrentCoordinatePairs(
		[][]float64{{0}, {1}},
		[][]float64{{90}},
		[][]float64{{10}, {20}},
	)
	if err != nil {
		t.Fatalf("build coordinate pairs: %v", err)
	}

	want := []CoordinatePair{
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{1}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{90}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{20}},
		{PresynapticCoords: []float64{1}, PostsynapticCoords: []float64{20}},
		{PresynapticCoords: []float64{90}, PostsynapticCoords: []float64{20}},
	}
	if !reflect.DeepEqual(pairs, want) {
		t.Fatalf("unexpected jordan coordinate order: got=%v want=%v", pairs, want)
	}
}

func TestBuildJordanRecurrentCoordinatePairsValidatesInput(t *testing.T) {
	if _, err := BuildJordanRecurrentCoordinatePairs(nil, [][]float64{{2}}, [][]float64{{3}}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing input-layer coordinates, got %v", err)
	}
	if _, err := BuildJordanRecurrentCoordinatePairs([][]float64{{1}}, nil, [][]float64{{3}}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing output-layer coordinates, got %v", err)
	}
	if _, err := BuildJordanRecurrentCoordinatePairs([][]float64{{1}}, [][]float64{{2}}, nil); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing current-layer coordinates, got %v", err)
	}
}

func TestBuildJordanRecurrentCoordinatePairsPopulatesCoordinateBatch(t *testing.T) {
	resetRegistriesForTests()
	t.Cleanup(resetRegistriesForTests)

	if err := RegisterCPP("sum_coordinate_builder_cpp", func() CPP {
		return sumCoordinateBuilderCPP{}
	}); err != nil {
		t.Fatalf("register coordinate cpp: %v", err)
	}

	pairs, err := BuildJordanRecurrentCoordinatePairs(
		[][]float64{{0.25}},
		[][]float64{{0.75}},
		[][]float64{{1}, {2}},
	)
	if err != nil {
		t.Fatalf("build coordinate pairs: %v", err)
	}

	rt, err := NewSimpleRuntime(Spec{
		CPPName:      "sum_coordinate_builder_cpp",
		CEPName:      WeightExpressionCEPName,
		CEPFaninPIDs: []string{"cpp1", "cpp2"},
	}, len(pairs))
	if err != nil {
		t.Fatalf("new runtime: %v", err)
	}

	weights, err := rt.StepCoordinateBatch(context.Background(), pairs)
	if err != nil {
		t.Fatalf("step coordinate batch: %v", err)
	}

	want := []float64{1.25, 1.75, 2.25, 2.75}
	if len(weights) != len(want) {
		t.Fatalf("unexpected weight count: got=%d want=%d", len(weights), len(want))
	}
	for i := range want {
		if math.Abs(weights[i]-want[i]) > 1e-9 {
			t.Fatalf("unexpected weight[%d]: got=%v want=%v all=%v", i, weights[i], want[i], weights)
		}
	}
}
