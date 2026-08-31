package substrate

import (
	"errors"
	"reflect"
	"testing"
)

func TestCoordinateHyperlayerCoordinatesCopiesInOrder(t *testing.T) {
	layer := CoordinateHyperlayer{
		{Coords: []float64{0, 1}, Output: 0.5, Weights: []float64{1}},
		{Coords: []float64{2, 3}, Output: -0.5, Weights: []float64{2}},
	}

	got := layer.Coordinates()
	want := [][]float64{{0, 1}, {2, 3}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected coordinates: got=%v want=%v", got, want)
	}

	layer[0].Coords[0] = 99
	if got[0][0] != 0 {
		t.Fatalf("expected coordinate copy, got=%v", got)
	}
}

func TestFlattenCoordinateHyperlayersCopiesInLayerOrder(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{0}}, {Coords: []float64{1}}}
	hidden := CoordinateHyperlayer{{Coords: []float64{10}}}
	output := CoordinateHyperlayer{{Coords: []float64{90}}}

	got := FlattenCoordinateHyperlayers(input, hidden, output)
	want := [][]float64{{0}, {1}, {10}, {90}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected flat coordinates: got=%v want=%v", got, want)
	}

	input[0].Coords[0] = 99
	if got[0][0] != 0 {
		t.Fatalf("expected flattened coordinate copy, got=%v", got)
	}
}

func TestBuildCoordinatePairsForLinkFormLayersDispatchesL2L(t *testing.T) {
	pairs, err := BuildCoordinatePairsForLinkFormLayers(CoordinateLayerPairBuildRequest{
		LinkForm: LinkFormL2LFeedforward,
		PreviousLayer: CoordinateHyperlayer{
			{Coords: []float64{0}},
			{Coords: []float64{1}},
		},
		CurrentLayer: CoordinateHyperlayer{
			{Coords: []float64{10}},
		},
	})
	if err != nil {
		t.Fatalf("build layer coordinate pairs: %v", err)
	}

	want := []CoordinatePair{
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{1}, PostsynapticCoords: []float64{10}},
	}
	if !reflect.DeepEqual(pairs, want) {
		t.Fatalf("unexpected l2l layer pairs: got=%v want=%v", pairs, want)
	}
}

func TestBuildCoordinatePairsForLinkFormLayersDispatchesFullyInterconnected(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{0}}, {Coords: []float64{1}}}
	hidden := CoordinateHyperlayer{{Coords: []float64{10}}}
	output := CoordinateHyperlayer{{Coords: []float64{90}}}

	pairs, err := BuildCoordinatePairsForLinkFormLayers(CoordinateLayerPairBuildRequest{
		LinkForm:      LinkFormFullyInterconnected,
		FlatSubstrate: []CoordinateHyperlayer{input, hidden, output},
		CurrentLayer:  hidden,
	})
	if err != nil {
		t.Fatalf("build layer coordinate pairs: %v", err)
	}

	want := []CoordinatePair{
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{1}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{10}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{90}, PostsynapticCoords: []float64{10}},
	}
	if !reflect.DeepEqual(pairs, want) {
		t.Fatalf("unexpected fully-interconnected layer pairs: got=%v want=%v", pairs, want)
	}
}

func TestBuildCoordinatePairsForLinkFormLayersDispatchesJordanRecurrent(t *testing.T) {
	pairs, err := BuildCoordinatePairsForLinkFormLayers(CoordinateLayerPairBuildRequest{
		LinkForm: LinkFormJordanRecurrent,
		InputLayer: CoordinateHyperlayer{
			{Coords: []float64{0}},
		},
		OutputLayer: CoordinateHyperlayer{
			{Coords: []float64{90}},
		},
		CurrentLayer: CoordinateHyperlayer{
			{Coords: []float64{10}},
		},
	})
	if err != nil {
		t.Fatalf("build layer coordinate pairs: %v", err)
	}

	want := []CoordinatePair{
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{90}, PostsynapticCoords: []float64{10}},
	}
	if !reflect.DeepEqual(pairs, want) {
		t.Fatalf("unexpected jordan layer pairs: got=%v want=%v", pairs, want)
	}
}

func TestBuildCoordinatePairsForLinkFormLayersDispatchesNeuronSelfRecurrent(t *testing.T) {
	pairs, err := BuildCoordinatePairsForLinkFormLayers(CoordinateLayerPairBuildRequest{
		LinkForm: LinkFormNeuronSelfRecurrent,
		PreviousLayer: CoordinateHyperlayer{
			{Coords: []float64{0}},
		},
		CurrentLayer: CoordinateHyperlayer{
			{Coords: []float64{10}},
		},
	})
	if err != nil {
		t.Fatalf("build layer coordinate pairs: %v", err)
	}

	want := []CoordinatePair{
		{PresynapticCoords: []float64{10}, PostsynapticCoords: []float64{10}},
		{PresynapticCoords: []float64{0}, PostsynapticCoords: []float64{10}},
	}
	if !reflect.DeepEqual(pairs, want) {
		t.Fatalf("unexpected neuron-self layer pairs: got=%v want=%v", pairs, want)
	}
}

func TestBuildCoordListMatchesReferenceOrder(t *testing.T) {
	tests := []struct {
		name    string
		density int
		want    []float64
	}{
		{name: "one", density: 1, want: []float64{0}},
		{name: "two", density: 2, want: []float64{-1, 1}},
		{name: "three", density: 3, want: []float64{-1, 0, 1}},
		{name: "five", density: 5, want: []float64{-1, -0.5, 0, 0.5, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildCoordList(tt.density)
			if err != nil {
				t.Fatalf("build coord list: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("unexpected coord list: got=%v want=%v", got, tt.want)
			}
		})
	}
}

func TestBuildCoordListValidatesDensity(t *testing.T) {
	if _, err := BuildCoordList(0); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates, got %v", err)
	}
}

func TestCreateCoordListsPrependsDimensionsInReferenceOrder(t *testing.T) {
	got, err := CreateCoordLists([]int{3, 2})
	if err != nil {
		t.Fatalf("create coord lists: %v", err)
	}

	want := [][]float64{
		{-1, -1},
		{-1, 0},
		{-1, 1},
		{1, -1},
		{1, 0},
		{1, 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected coord lists: got=%v want=%v", got, want)
	}
}

func TestCreateCoordListsValidatesDensities(t *testing.T) {
	if _, err := CreateCoordLists(nil); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing densities, got %v", err)
	}
	if _, err := CreateCoordLists([]int{2, 0}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for invalid density, got %v", err)
	}
}

func TestBuildCoordinateHypercubeMirrorsCS(t *testing.T) {
	layer, err := BuildCoordinateHypercube([]int{2, 3}, []float64{0.1, 0.2})
	if err != nil {
		t.Fatalf("build coordinate hypercube: %v", err)
	}

	wantCoords := [][]float64{
		{-1, -1},
		{-1, 0},
		{-1, 1},
		{1, -1},
		{1, 0},
		{1, 1},
	}
	if got := layer.Coordinates(); !reflect.DeepEqual(got, wantCoords) {
		t.Fatalf("unexpected hypercube coords: got=%v want=%v", got, wantCoords)
	}
	for i, neurode := range layer {
		if neurode.Output != 0 {
			t.Fatalf("expected zero output at neurode %d, got=%v", i, neurode.Output)
		}
		if !reflect.DeepEqual(neurode.Weights, []float64{0.1, 0.2}) {
			t.Fatalf("unexpected weights at neurode %d: %v", i, neurode.Weights)
		}
	}
}

func TestBuildCoordinateHypercubeCopiesWeights(t *testing.T) {
	weights := []float64{0.1}
	layer, err := BuildCoordinateHypercube([]int{1}, weights)
	if err != nil {
		t.Fatalf("build coordinate hypercube: %v", err)
	}

	weights[0] = 99
	if got := layer[0].Weights[0]; got != 0.1 {
		t.Fatalf("expected weight copy, got=%v", got)
	}
	layer[0].Weights[0] = 88
	other, err := BuildCoordinateHypercube([]int{2}, []float64{0.2})
	if err != nil {
		t.Fatalf("build other coordinate hypercube: %v", err)
	}
	if got := other[0].Weights[0]; got != 0.2 {
		t.Fatalf("expected independent hypercube weights, got=%v", got)
	}
}

func TestBuildCoordinateHypercubeValidatesDensities(t *testing.T) {
	if _, err := BuildCoordinateHypercube(nil, nil); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing densities, got %v", err)
	}
}

func TestExtrudeCoordinateHyperlayerPrependsDimensionAndCopiesState(t *testing.T) {
	layer := CoordinateHyperlayer{
		{Coords: []float64{-1, 1}, Output: 0.25, Weights: []float64{0.1}},
		{Coords: []float64{1, -1}, Output: -0.25, Weights: []float64{0.2}},
	}

	got := ExtrudeCoordinateHyperlayer(0.5, layer)
	wantCoords := [][]float64{{0.5, -1, 1}, {0.5, 1, -1}}
	if coords := got.Coordinates(); !reflect.DeepEqual(coords, wantCoords) {
		t.Fatalf("unexpected extruded coords: got=%v want=%v", coords, wantCoords)
	}
	if got[0].Output != 0.25 || got[1].Output != -0.25 {
		t.Fatalf("unexpected extruded outputs: %+v", got)
	}

	layer[0].Coords[0] = 99
	layer[0].Weights[0] = 88
	if got[0].Coords[1] != -1 || got[0].Weights[0] != 0.1 {
		t.Fatalf("expected extruded layer copies, got=%+v", got[0])
	}
}
