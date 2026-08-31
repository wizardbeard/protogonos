package substrate

import (
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
