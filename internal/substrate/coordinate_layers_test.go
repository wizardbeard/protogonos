package substrate

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type coordinateSumCPP struct{}

func (coordinateSumCPP) ComputeCoordinates(_ context.Context, presynaptic []float64, postsynaptic []float64, _ map[string]float64) ([]float64, error) {
	sum := 0.0
	for _, value := range presynaptic {
		sum += value
	}
	for _, value := range postsynaptic {
		sum += value
	}
	return []float64{sum}, nil
}

type rawSignalCEP struct{}

func (rawSignalCEP) Name() string { return "raw_signal" }

func (rawSignalCEP) Apply(_ context.Context, _ float64, delta float64, _ map[string]float64) (float64, error) {
	return delta, nil
}

type rawAddCEP struct{}

func (rawAddCEP) Name() string { return "raw_add" }

func (rawAddCEP) Apply(_ context.Context, current float64, delta float64, _ map[string]float64) (float64, error) {
	return current + delta, nil
}

type iowDeltaCPP struct{}

func (iowDeltaCPP) ComputeCoordinatesIOW(_ context.Context, _ []float64, _ []float64, iow []float64, _ map[string]float64) ([]float64, error) {
	if len(iow) != 3 {
		return nil, ErrInvalidSubstrateCoordinates
	}
	return []float64{iow[0] + iow[1]}, nil
}

func abcnTestWeight(weight float64) ABCNWeight {
	return ABCNWeight{Weight: weight}
}

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

func TestCountIONeurodesMatchesCoveredReferenceFormats(t *testing.T) {
	got, err := CountIONeurodes([]IOCoordinateSpec{
		{Format: CoordinateFormatUndefined, VL: 2},
		{Format: CoordinateFormatNoGeo, VL: 3},
		{Format: CoordinateFormatSymmetric, Resolutions: []int{2, 3}},
		{Format: CoordinateFormatCoorded, Dim: 2, Neurodes: CoordinateHyperlayer{
			{Coords: []float64{-1, 1}},
			{Coords: []float64{1, -1}},
		}},
	})
	if err != nil {
		t.Fatalf("count io neurodes: %v", err)
	}
	if got != 13 {
		t.Fatalf("unexpected io neurode count: got=%d want=13", got)
	}
}

func TestCountIONeurodesDefaultsEmptyFormatToUndefined(t *testing.T) {
	got, err := CountIONeurodes([]IOCoordinateSpec{{VL: 4}})
	if err != nil {
		t.Fatalf("count io neurodes: %v", err)
	}
	if got != 4 {
		t.Fatalf("unexpected default-format io neurode count: got=%d want=4", got)
	}
}

func TestCountIONeurodesValidatesSpecs(t *testing.T) {
	tests := []struct {
		name  string
		specs []IOCoordinateSpec
	}{
		{name: "missing", specs: nil},
		{name: "invalid vl", specs: []IOCoordinateSpec{{Format: CoordinateFormatNoGeo, VL: 0}}},
		{name: "invalid symmetric resolution", specs: []IOCoordinateSpec{{Format: CoordinateFormatSymmetric, Resolutions: []int{2, 0}}}},
		{name: "invalid coorded dim", specs: []IOCoordinateSpec{{Format: CoordinateFormatCoorded, Neurodes: CoordinateHyperlayer{{Coords: []float64{0}}}}}},
		{name: "missing coorded neurodes", specs: []IOCoordinateSpec{{Format: CoordinateFormatCoorded, Dim: 1}}},
		{name: "mismatched coorded dim", specs: []IOCoordinateSpec{{Format: CoordinateFormatCoorded, Dim: 2, Neurodes: CoordinateHyperlayer{{Coords: []float64{0}}}}}},
		{name: "unsupported", specs: []IOCoordinateSpec{{Format: "asymmetric", VL: 1}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := CountIONeurodes(tt.specs); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
				t.Fatalf("expected ErrInvalidSubstrateCoordinates, got %v", err)
			}
		})
	}
}

func TestFlattenInputValuesPreservesBatchOrder(t *testing.T) {
	got := FlattenInputValues([]float64{1, 2}, nil, []float64{3})
	want := []float64{1, 2, 3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected flattened input: got=%v want=%v", got, want)
	}
}

func TestPopulateInputHyperlayerAssignsInputOutputsInOrder(t *testing.T) {
	layer := CoordinateHyperlayer{
		{Coords: []float64{-1, 0}, Output: 9, Weights: []float64{0.1}},
		{Coords: []float64{1, 0}, Output: 8},
	}
	got, err := PopulateInputHyperlayer(layer, FlattenInputValues([]float64{0.25}, []float64{-0.75}))
	if err != nil {
		t.Fatalf("populate input hyperlayer: %v", err)
	}

	if coords := got.Coordinates(); !reflect.DeepEqual(coords, [][]float64{{-1, 0}, {1, 0}}) {
		t.Fatalf("unexpected populated input coords: got=%v", coords)
	}
	if got[0].Output != 0.25 || got[1].Output != -0.75 {
		t.Fatalf("unexpected populated input outputs: %v, %v", got[0].Output, got[1].Output)
	}
	if !reflect.DeepEqual(got[0].Weights, []float64{0.1}) || got[1].Weights != nil {
		t.Fatalf("unexpected populated input weights: %v", got)
	}

	layer[0].Coords[0] = 99
	layer[0].Weights[0] = 99
	if got[0].Coords[0] != -1 || got[0].Weights[0] != 0.1 {
		t.Fatalf("expected populated input layer to be copied, got=%v", got)
	}
}

func TestPopulateInputHyperlayerValidatesInputLength(t *testing.T) {
	if _, err := PopulateInputHyperlayer(CoordinateHyperlayer{{Coords: []float64{0}}}, nil); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates, got %v", err)
	}
}

func TestCalculateNeurodeOutputStdAppliesWeightedTanh(t *testing.T) {
	previous := CoordinateHyperlayer{
		{Coords: []float64{-1}, Output: 0.5},
		{Coords: []float64{1}, Output: -0.25},
	}
	neurode := NeurodeCoordinate{
		Coords:  []float64{0},
		Output:  99,
		Weights: []float64{0.8, -0.4},
	}

	got, err := CalculateNeurodeOutputStd(previous, neurode)
	if err != nil {
		t.Fatalf("calculate neurode output: %v", err)
	}
	want := math.Tanh(0.5*0.8 + -0.25*-0.4)
	if math.Abs(got.Output-want) > 1e-12 {
		t.Fatalf("unexpected neurode output: got=%v want=%v", got.Output, want)
	}
	if !reflect.DeepEqual(got.Coords, []float64{0}) {
		t.Fatalf("unexpected neurode coords: %v", got.Coords)
	}
	if !reflect.DeepEqual(got.Weights, []float64{0.8, -0.4}) {
		t.Fatalf("unexpected neurode weights: %v", got.Weights)
	}

	neurode.Coords[0] = 99
	neurode.Weights[0] = 99
	if got.Coords[0] != 0 || got.Weights[0] != 0.8 {
		t.Fatalf("expected calculated neurode to be copied, got=%v", got)
	}
}

func TestCalculateNeurodeOutputStdValidatesWeightCount(t *testing.T) {
	_, err := CalculateNeurodeOutputStd(
		CoordinateHyperlayer{{Coords: []float64{0}, Output: 1}},
		NeurodeCoordinate{Coords: []float64{1}, Weights: []float64{0.1, 0.2}},
	)
	if !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates, got %v", err)
	}
}

func TestCalculateOutputStdPropagatesL2LLayers(t *testing.T) {
	input := CoordinateHyperlayer{
		{Coords: []float64{-1}, Output: 0.5},
		{Coords: []float64{1}, Output: -0.25},
	}
	hidden := CoordinateHyperlayer{
		{Coords: []float64{0}, Output: 9, Weights: []float64{0.8, -0.4}},
	}
	output := CoordinateHyperlayer{
		{Coords: []float64{1}, Output: 8, Weights: []float64{0.3}},
		{Coords: []float64{-1}, Output: 7, Weights: []float64{-0.2}},
	}

	outputs, updated, err := CalculateOutputStd(input, []CoordinateHyperlayer{hidden, output})
	if err != nil {
		t.Fatalf("calculate output std: %v", err)
	}
	hiddenOut := math.Tanh(0.5*0.8 + -0.25*-0.4)
	wantOutputs := []float64{math.Tanh(hiddenOut * 0.3), math.Tanh(hiddenOut * -0.2)}
	if !reflect.DeepEqual(outputs, wantOutputs) {
		t.Fatalf("unexpected outputs: got=%v want=%v", outputs, wantOutputs)
	}
	if len(updated) != 2 {
		t.Fatalf("unexpected updated layer count: got=%d want=2", len(updated))
	}
	if math.Abs(updated[0][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected hidden output: got=%v want=%v", updated[0][0].Output, hiddenOut)
	}
	if !reflect.DeepEqual(updated[1].Coordinates(), [][]float64{{1}, {-1}}) {
		t.Fatalf("unexpected updated output coords: %v", updated[1].Coordinates())
	}

	hidden[0].Weights[0] = 99
	if updated[0][0].Weights[0] != 0.8 {
		t.Fatalf("expected updated layers to be copied, got=%v", updated[0][0].Weights)
	}
}

func TestCalculateOutputStdValidatesInputs(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{0}, Output: 1}}
	if _, _, err := CalculateOutputStd(input, nil); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing layers, got %v", err)
	}
	if _, _, err := CalculateOutputStd(input, []CoordinateHyperlayer{{{Coords: []float64{1}, Weights: []float64{0.1, 0.2}}}}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for weight mismatch, got %v", err)
	}
}

func TestABCNWeightUpdateMatchesReferenceRule(t *testing.T) {
	weight := ABCNWeight{Weight: 0.5, A: 0.2, B: -0.3, C: 0.4, N: 0.1}
	got := ABCNWeightUpdate(0.25, -0.5, weight)
	want := 0.5 + 0.1*(0.2*0.25*-0.5+-0.3*0.25+0.4*-0.5)
	if math.Abs(got.Weight-want) > 1e-12 {
		t.Fatalf("unexpected abcn weight update: got=%v want=%v", got.Weight, want)
	}
	if got.A != 0.2 || got.B != -0.3 || got.C != 0.4 || got.N != 0.1 {
		t.Fatalf("expected abcn parameters to be preserved, got=%+v", got)
	}
}

func TestCalculateNeurodeOutputABCNUpdatesWeights(t *testing.T) {
	previous := CoordinateHyperlayer{
		{Coords: []float64{-1}, Output: 0.5},
		{Coords: []float64{1}, Output: -0.25},
	}
	neurode := ABCNNeurodeCoordinate{
		Coords: []float64{0},
		Output: 99,
		Weights: []ABCNWeight{
			{Weight: 0.8, A: 0.1, B: 0.2, C: 0.3, N: 0.4},
			{Weight: -0.4, A: -0.2, B: 0.1, C: -0.3, N: 0.5},
		},
	}

	got, err := CalculateNeurodeOutputABCN(previous, neurode)
	if err != nil {
		t.Fatalf("calculate abcn neurode output: %v", err)
	}
	output := math.Tanh(0.5*0.8 + -0.25*-0.4)
	if math.Abs(got.Output-output) > 1e-12 {
		t.Fatalf("unexpected abcn neurode output: got=%v want=%v", got.Output, output)
	}
	wantFirst := ABCNWeightUpdate(0.5, output, neurode.Weights[0])
	wantSecond := ABCNWeightUpdate(-0.25, output, neurode.Weights[1])
	if got.Weights[0] != wantFirst || got.Weights[1] != wantSecond {
		t.Fatalf("unexpected abcn updated weights: got=%+v want=%+v", got.Weights, []ABCNWeight{wantFirst, wantSecond})
	}

	neurode.Coords[0] = 99
	neurode.Weights[0].Weight = 99
	if got.Coords[0] != 0 || math.Abs(got.Weights[0].Weight-wantFirst.Weight) > 1e-12 {
		t.Fatalf("expected abcn neurode result to be copied, got=%+v", got)
	}
}

func TestCalculateNeurodeOutputABCNValidatesWeightCount(t *testing.T) {
	_, err := CalculateNeurodeOutputABCN(
		CoordinateHyperlayer{{Coords: []float64{0}, Output: 1}},
		ABCNNeurodeCoordinate{Coords: []float64{1}, Weights: []ABCNWeight{{}, {}}},
	)
	if !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates, got %v", err)
	}
}

func TestCalculateOutputABCNStdPropagatesAndUpdatesLayers(t *testing.T) {
	input := CoordinateHyperlayer{
		{Coords: []float64{-1}, Output: 0.5},
		{Coords: []float64{1}, Output: -0.25},
	}
	hidden := ABCNCoordinateHyperlayer{
		{Coords: []float64{0}, Output: 9, Weights: []ABCNWeight{
			{Weight: 0.8, A: 0.1, B: 0.2, C: 0.3, N: 0.4},
			{Weight: -0.4, A: -0.2, B: 0.1, C: -0.3, N: 0.5},
		}},
	}

	outputs, updated, err := CalculateOutputABCNStd(input, []ABCNCoordinateHyperlayer{hidden})
	if err != nil {
		t.Fatalf("calculate abcn output std: %v", err)
	}
	wantOutput := math.Tanh(0.5*0.8 + -0.25*-0.4)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if len(updated) != 1 || len(updated[0]) != 1 {
		t.Fatalf("unexpected abcn updated layer shape: %v", updated)
	}
	if math.Abs(updated[0][0].Output-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn updated output: got=%v want=%v", updated[0][0].Output, wantOutput)
	}
	wantWeight := ABCNWeightUpdate(0.5, wantOutput, hidden[0].Weights[0])
	if updated[0][0].Weights[0] != wantWeight {
		t.Fatalf("unexpected abcn updated layer weight: got=%+v want=%+v", updated[0][0].Weights[0], wantWeight)
	}

	hidden[0].Weights[0].Weight = 99
	if math.Abs(updated[0][0].Weights[0].Weight-wantWeight.Weight) > 1e-12 {
		t.Fatalf("expected abcn updated layers to be copied, got=%+v", updated[0][0].Weights[0])
	}
}

func TestCalculateOutputABCNStdValidatesInputs(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{0}, Output: 1}}
	if _, _, err := CalculateOutputABCNStd(input, nil); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing abcn layers, got %v", err)
	}
	if _, _, err := CalculateOutputABCNStd(input, []ABCNCoordinateHyperlayer{{{Coords: []float64{1}, Weights: []ABCNWeight{{}, {}}}}}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for abcn weight mismatch, got %v", err)
	}
}

func TestCalculateOutputABCNFullyInterconnectedUsesFlattenedSource(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}, Output: 0.5}}
	hidden := ABCNCoordinateHyperlayer{{Coords: []float64{0}, Output: 0.1, Weights: []ABCNWeight{
		abcnTestWeight(1), abcnTestWeight(2), abcnTestWeight(3),
	}}}
	output := ABCNCoordinateHyperlayer{{Coords: []float64{1}, Output: -0.2, Weights: []ABCNWeight{
		abcnTestWeight(0.4), abcnTestWeight(0.5), abcnTestWeight(-0.25),
	}}}

	outputs, updated, err := CalculateOutputABCNFullyInterconnected(input, []ABCNCoordinateHyperlayer{hidden, output})
	if err != nil {
		t.Fatalf("calculate abcn fully-interconnected output: %v", err)
	}
	hiddenOut := math.Tanh(0.5*1 + 0.1*2 + -0.2*3)
	wantOutput := math.Tanh(0.5*0.4 + hiddenOut*0.5 + -0.2*-0.25)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn fully-interconnected outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if math.Abs(updated[0][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected abcn fully-interconnected hidden output: got=%v want=%v", updated[0][0].Output, hiddenOut)
	}
}

func TestCalculateOutputABCNJordanRecurrentUsesPreviousOutputForFirstLayer(t *testing.T) {
	input := CoordinateHyperlayer{
		{Coords: []float64{-1}, Output: 0.5},
		{Coords: []float64{1}, Output: -0.25},
	}
	hidden := ABCNCoordinateHyperlayer{{Coords: []float64{0}, Output: 0, Weights: []ABCNWeight{
		abcnTestWeight(0.2), abcnTestWeight(0.3), abcnTestWeight(0.5),
	}}}
	output := ABCNCoordinateHyperlayer{{Coords: []float64{1}, Output: 0.4, Weights: []ABCNWeight{
		abcnTestWeight(-0.7),
	}}}

	outputs, updated, err := CalculateOutputABCNJordanRecurrent(input, []ABCNCoordinateHyperlayer{hidden, output})
	if err != nil {
		t.Fatalf("calculate abcn jordan output: %v", err)
	}
	hiddenOut := math.Tanh(0.5*0.2 + -0.25*0.3 + 0.4*0.5)
	wantOutput := math.Tanh(hiddenOut * -0.7)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn jordan outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if math.Abs(updated[0][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected abcn jordan hidden output: got=%v want=%v", updated[0][0].Output, hiddenOut)
	}
}

func TestCalculateOutputABCNNeuronSelfRecurrentPrependsPreviousNeurodeState(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}, Output: 0.5}}
	hidden := ABCNCoordinateHyperlayer{{Coords: []float64{0}, Output: 0.3, Weights: []ABCNWeight{
		abcnTestWeight(0.4), abcnTestWeight(0.8),
	}}}
	output := ABCNCoordinateHyperlayer{{Coords: []float64{1}, Output: 0.2, Weights: []ABCNWeight{
		abcnTestWeight(-0.5), abcnTestWeight(0.6),
	}}}

	outputs, updated, err := CalculateOutputABCNNeuronSelfRecurrent(input, []ABCNCoordinateHyperlayer{hidden, output})
	if err != nil {
		t.Fatalf("calculate abcn neuron-self recurrent output: %v", err)
	}
	hiddenOut := math.Tanh(0.3*0.4 + 0.5*0.8)
	wantOutput := math.Tanh(0.2*-0.5 + hiddenOut*0.6)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn neuron-self recurrent outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if math.Abs(updated[0][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected abcn neuron-self recurrent hidden output: got=%v want=%v", updated[0][0].Output, hiddenOut)
	}
}

func TestCalculateOutputABCNForLinkFormDispatchesActiveForms(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}, Output: 1}}

	if _, _, err := CalculateOutputABCNForLinkForm(LinkFormL2LFeedforward, input, []ABCNCoordinateHyperlayer{{{Coords: []float64{1}, Weights: []ABCNWeight{abcnTestWeight(0.5)}}}}); err != nil {
		t.Fatalf("calculate abcn output for %s: %v", LinkFormL2LFeedforward, err)
	}
	if _, _, err := CalculateOutputABCNForLinkForm(LinkFormFullyInterconnected, input, []ABCNCoordinateHyperlayer{{{Coords: []float64{1}, Weights: []ABCNWeight{abcnTestWeight(0.5), abcnTestWeight(0.1)}}}}); err != nil {
		t.Fatalf("calculate abcn output for %s: %v", LinkFormFullyInterconnected, err)
	}
	if _, _, err := CalculateOutputABCNForLinkForm(LinkFormJordanRecurrent, input, []ABCNCoordinateHyperlayer{{{Coords: []float64{1}, Weights: []ABCNWeight{abcnTestWeight(0.5), abcnTestWeight(0.1)}}}}); err != nil {
		t.Fatalf("calculate abcn output for %s: %v", LinkFormJordanRecurrent, err)
	}
	if _, _, err := CalculateOutputABCNForLinkForm(LinkFormNeuronSelfRecurrent, input, []ABCNCoordinateHyperlayer{{{Coords: []float64{1}, Weights: []ABCNWeight{abcnTestWeight(0.1), abcnTestWeight(0.5)}}}}); err != nil {
		t.Fatalf("calculate abcn output for %s: %v", LinkFormNeuronSelfRecurrent, err)
	}
	if _, _, err := CalculateOutputABCNForLinkForm("planeself_recurrent", input, []ABCNCoordinateHyperlayer{{{Coords: []float64{1}, Weights: []ABCNWeight{abcnTestWeight(0.5)}}}}); !errors.Is(err, ErrUnsupportedSubstrateLink) {
		t.Fatalf("expected ErrUnsupportedSubstrateLink, got %v", err)
	}
}

func TestCalculateOutputFullyInterconnectedUsesFlattenedSubstrateSource(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}, Output: 0.5}}
	hidden := CoordinateHyperlayer{{Coords: []float64{0}, Output: 0.1, Weights: []float64{1, 2, 3}}}
	output := CoordinateHyperlayer{{Coords: []float64{1}, Output: -0.2, Weights: []float64{0.4, 0.5, -0.25}}}

	outputs, updated, err := CalculateOutputFullyInterconnected(input, []CoordinateHyperlayer{hidden, output})
	if err != nil {
		t.Fatalf("calculate fully-interconnected output: %v", err)
	}
	hiddenOut := math.Tanh(0.5*1 + 0.1*2 + -0.2*3)
	wantOutput := math.Tanh(0.5*0.4 + hiddenOut*0.5 + -0.2*-0.25)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected fully-interconnected outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if math.Abs(updated[0][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected fully-interconnected hidden output: got=%v want=%v", updated[0][0].Output, hiddenOut)
	}
}

func TestCalculateOutputJordanRecurrentUsesPreviousOutputForFirstLayer(t *testing.T) {
	input := CoordinateHyperlayer{
		{Coords: []float64{-1}, Output: 0.5},
		{Coords: []float64{1}, Output: -0.25},
	}
	hidden := CoordinateHyperlayer{{Coords: []float64{0}, Output: 0, Weights: []float64{0.2, 0.3, 0.5}}}
	output := CoordinateHyperlayer{{Coords: []float64{1}, Output: 0.4, Weights: []float64{-0.7}}}

	outputs, updated, err := CalculateOutputJordanRecurrent(input, []CoordinateHyperlayer{hidden, output})
	if err != nil {
		t.Fatalf("calculate jordan output: %v", err)
	}
	hiddenOut := math.Tanh(0.5*0.2 + -0.25*0.3 + 0.4*0.5)
	wantOutput := math.Tanh(hiddenOut * -0.7)
	if !reflect.DeepEqual(outputs, []float64{wantOutput}) {
		t.Fatalf("unexpected jordan outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if math.Abs(updated[0][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected jordan hidden output: got=%v want=%v", updated[0][0].Output, hiddenOut)
	}
}

func TestCalculateOutputNeuronSelfRecurrentPrependsPreviousNeurodeState(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}, Output: 0.5}}
	hidden := CoordinateHyperlayer{{Coords: []float64{0}, Output: 0.3, Weights: []float64{0.4, 0.8}}}
	output := CoordinateHyperlayer{{Coords: []float64{1}, Output: 0.2, Weights: []float64{-0.5, 0.6}}}

	outputs, updated, err := CalculateOutputNeuronSelfRecurrent(input, []CoordinateHyperlayer{hidden, output})
	if err != nil {
		t.Fatalf("calculate neuron-self recurrent output: %v", err)
	}
	hiddenOut := math.Tanh(0.3*0.4 + 0.5*0.8)
	wantOutput := math.Tanh(0.2*-0.5 + hiddenOut*0.6)
	if !reflect.DeepEqual(outputs, []float64{wantOutput}) {
		t.Fatalf("unexpected neuron-self recurrent outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if math.Abs(updated[0][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected neuron-self recurrent hidden output: got=%v want=%v", updated[0][0].Output, hiddenOut)
	}
}

func TestCalculateOutputForLinkFormDispatchesActiveForms(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}, Output: 1}}

	if _, _, err := CalculateOutputForLinkForm(LinkFormL2LFeedforward, input, []CoordinateHyperlayer{{{Coords: []float64{1}, Weights: []float64{0.5}}}}); err != nil {
		t.Fatalf("calculate output for %s: %v", LinkFormL2LFeedforward, err)
	}
	if _, _, err := CalculateOutputForLinkForm(LinkFormFullyInterconnected, input, []CoordinateHyperlayer{{{Coords: []float64{1}, Weights: []float64{0.5, 0.1}}}}); err != nil {
		t.Fatalf("calculate output for %s: %v", LinkFormFullyInterconnected, err)
	}
	if _, _, err := CalculateOutputForLinkForm(LinkFormJordanRecurrent, input, []CoordinateHyperlayer{{{Coords: []float64{1}, Weights: []float64{0.5, 0.1}}}}); err != nil {
		t.Fatalf("calculate output for %s: %v", LinkFormJordanRecurrent, err)
	}

	nsrLayers := []CoordinateHyperlayer{{{Coords: []float64{1}, Weights: []float64{0.1, 0.5}}}}
	if _, _, err := CalculateOutputForLinkForm(LinkFormNeuronSelfRecurrent, input, nsrLayers); err != nil {
		t.Fatalf("calculate output for %s: %v", LinkFormNeuronSelfRecurrent, err)
	}
	if _, _, err := CalculateOutputForLinkForm("planeself_recurrent", input, []CoordinateHyperlayer{{{Coords: []float64{1}, Weights: []float64{0.5}}}}); !errors.Is(err, ErrUnsupportedSubstrateLink) {
		t.Fatalf("expected ErrUnsupportedSubstrateLink, got %v", err)
	}
}

func TestCalculateHoldOutputPopulatesInputAndReturnsUpdatedSubstrate(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 9}, {Coords: []float64{1}, Output: 8}},
		{{Coords: []float64{0}, Output: 7, Weights: []float64{0.5, -0.25}}},
		{{Coords: []float64{1}, Output: 6, Weights: []float64{0.75}}},
	}

	outputs, updated, err := CalculateHoldOutput(substrateLayers, [][]float64{{0.4}, {-0.2}}, LinkFormL2LFeedforward)
	if err != nil {
		t.Fatalf("calculate hold output: %v", err)
	}
	hiddenOut := math.Tanh(0.4*0.5 + -0.2*-0.25)
	wantOutput := math.Tanh(hiddenOut * 0.75)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected hold outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if len(updated) != 3 {
		t.Fatalf("unexpected updated substrate layer count: got=%d want=3", len(updated))
	}
	if updated[0][0].Output != 9 || updated[0][1].Output != 8 {
		t.Fatalf("expected original input layer to be retained, got=%v", updated[0])
	}
	if math.Abs(updated[1][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected updated hidden output: got=%v want=%v", updated[1][0].Output, hiddenOut)
	}
	if math.Abs(updated[2][0].Output-wantOutput) > 1e-12 {
		t.Fatalf("unexpected updated output neurode output: got=%v want=%v", updated[2][0].Output, wantOutput)
	}

	substrateLayers[1][0].Weights[0] = 99
	if updated[1][0].Weights[0] != 0.5 {
		t.Fatalf("expected updated substrate to be copied, got=%v", updated[1][0].Weights)
	}
}

func TestCalculateResetOutputUsesSameTypedNonPlasticLifecycle(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0}},
		{{Coords: []float64{1}, Output: 0, Weights: []float64{-0.5}}},
	}

	outputs, updated, err := CalculateResetOutput(substrateLayers, [][]float64{{0.8}}, LinkFormL2LFeedforward)
	if err != nil {
		t.Fatalf("calculate reset output: %v", err)
	}
	wantOutput := math.Tanh(0.8 * -0.5)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected reset outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if len(updated) != 2 || math.Abs(updated[1][0].Output-wantOutput) > 1e-12 {
		t.Fatalf("unexpected reset updated substrate: %v", updated)
	}
}

func TestCalculateTypedOutputLifecycleValidatesInputs(t *testing.T) {
	if _, _, err := CalculateHoldOutput(nil, [][]float64{{1}}, LinkFormL2LFeedforward); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing substrate, got %v", err)
	}
	if _, _, err := CalculateHoldOutput([]CoordinateHyperlayer{{{Coords: []float64{0}}}}, [][]float64{{1}}, LinkFormL2LFeedforward); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for shallow substrate, got %v", err)
	}
	if _, _, err := CalculateHoldOutput([]CoordinateHyperlayer{
		{{Coords: []float64{0}}},
		{{Coords: []float64{1}, Weights: []float64{0.1}}},
	}, nil, LinkFormL2LFeedforward); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for input mismatch, got %v", err)
	}
	if _, _, err := CalculateHoldOutput([]CoordinateHyperlayer{
		{{Coords: []float64{0}}},
		{{Coords: []float64{1}, Weights: []float64{0.1}}},
	}, [][]float64{{1}}, "planeself_recurrent"); !errors.Is(err, ErrUnsupportedSubstrateLink) {
		t.Fatalf("expected ErrUnsupportedSubstrateLink, got %v", err)
	}
}

func TestCalculateHoldOutputABCNPopulatesInputAndReturnsUpdatedSubstrate(t *testing.T) {
	substrateLayers := ABCNSubstrate{
		InputLayer: CoordinateHyperlayer{
			{Coords: []float64{-1}, Output: 9},
			{Coords: []float64{1}, Output: 8},
		},
		Layers: []ABCNCoordinateHyperlayer{
			{{Coords: []float64{0}, Output: 7, Weights: []ABCNWeight{
				{Weight: 0.5, A: 0.1, B: 0.2, C: 0.3, N: 0.4},
				{Weight: -0.25, A: -0.1, B: 0.05, C: 0.2, N: 0.3},
			}}},
			{{Coords: []float64{1}, Output: 6, Weights: []ABCNWeight{
				{Weight: 0.75, A: 0.2, B: 0.1, C: -0.1, N: 0.5},
			}}},
		},
	}

	outputs, updated, err := CalculateHoldOutputABCN(substrateLayers, [][]float64{{0.4}, {-0.2}}, LinkFormL2LFeedforward)
	if err != nil {
		t.Fatalf("calculate abcn hold output: %v", err)
	}
	hiddenOut := math.Tanh(0.4*0.5 + -0.2*-0.25)
	wantOutput := math.Tanh(hiddenOut * 0.75)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn hold outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if updated.InputLayer[0].Output != 9 || updated.InputLayer[1].Output != 8 {
		t.Fatalf("expected original input layer to be retained, got=%v", updated.InputLayer)
	}
	if math.Abs(updated.Layers[0][0].Output-hiddenOut) > 1e-12 {
		t.Fatalf("unexpected abcn hidden output: got=%v want=%v", updated.Layers[0][0].Output, hiddenOut)
	}
	if math.Abs(updated.Layers[1][0].Output-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn output neurode output: got=%v want=%v", updated.Layers[1][0].Output, wantOutput)
	}
	wantHiddenWeight := ABCNWeightUpdate(0.4, hiddenOut, substrateLayers.Layers[0][0].Weights[0])
	wantOutputWeight := ABCNWeightUpdate(hiddenOut, wantOutput, substrateLayers.Layers[1][0].Weights[0])
	if updated.Layers[0][0].Weights[0] != wantHiddenWeight {
		t.Fatalf("unexpected abcn hidden weight: got=%+v want=%+v", updated.Layers[0][0].Weights[0], wantHiddenWeight)
	}
	if updated.Layers[1][0].Weights[0] != wantOutputWeight {
		t.Fatalf("unexpected abcn output weight: got=%+v want=%+v", updated.Layers[1][0].Weights[0], wantOutputWeight)
	}

	substrateLayers.InputLayer[0].Coords[0] = 99
	substrateLayers.Layers[0][0].Weights[0].Weight = 99
	if updated.InputLayer[0].Coords[0] != -1 || math.Abs(updated.Layers[0][0].Weights[0].Weight-wantHiddenWeight.Weight) > 1e-12 {
		t.Fatalf("expected abcn updated substrate to be copied, got=%+v", updated)
	}
}

func TestCalculateResetOutputABCNUsesSameTypedLifecycle(t *testing.T) {
	substrateLayers := ABCNSubstrate{
		InputLayer: CoordinateHyperlayer{{Coords: []float64{-1}, Output: 0}},
		Layers: []ABCNCoordinateHyperlayer{
			{{Coords: []float64{1}, Output: 0, Weights: []ABCNWeight{{Weight: -0.5, A: 0.1, B: 0.2, C: 0.3, N: 0.4}}}},
		},
	}

	outputs, updated, err := CalculateResetOutputABCN(substrateLayers, [][]float64{{0.8}}, LinkFormL2LFeedforward)
	if err != nil {
		t.Fatalf("calculate abcn reset output: %v", err)
	}
	wantOutput := math.Tanh(0.8 * -0.5)
	if len(outputs) != 1 || math.Abs(outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn reset outputs: got=%v want=%v", outputs, []float64{wantOutput})
	}
	if len(updated.Layers) != 1 || math.Abs(updated.Layers[0][0].Output-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn reset updated substrate: %+v", updated)
	}
}

func TestCalculateABCNOutputLifecycleValidatesInputs(t *testing.T) {
	if _, _, err := CalculateHoldOutputABCN(ABCNSubstrate{}, [][]float64{{1}}, LinkFormL2LFeedforward); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing input layer, got %v", err)
	}
	if _, _, err := CalculateHoldOutputABCN(ABCNSubstrate{InputLayer: CoordinateHyperlayer{{Coords: []float64{0}}}}, [][]float64{{1}}, LinkFormL2LFeedforward); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing abcn layers, got %v", err)
	}
	if _, _, err := CalculateHoldOutputABCN(ABCNSubstrate{
		InputLayer: CoordinateHyperlayer{{Coords: []float64{0}}},
		Layers:     []ABCNCoordinateHyperlayer{{{Coords: []float64{1}, Weights: []ABCNWeight{{Weight: 0.1}}}}},
	}, nil, LinkFormL2LFeedforward); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for input mismatch, got %v", err)
	}
	if _, _, err := CalculateHoldOutputABCN(ABCNSubstrate{
		InputLayer: CoordinateHyperlayer{{Coords: []float64{0}}},
		Layers:     []ABCNCoordinateHyperlayer{{{Coords: []float64{1}, Weights: []ABCNWeight{{Weight: 0.1}}}}},
	}, [][]float64{{1}}, "planeself_recurrent"); !errors.Is(err, ErrUnsupportedSubstrateLink) {
		t.Fatalf("expected ErrUnsupportedSubstrateLink, got %v", err)
	}
}

func TestCalculateOutputLifecycleResetNonePopulatesThenHolds(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0}, {Coords: []float64{1}, Output: 0}},
		{{Coords: []float64{0}, Output: 0, Weights: []float64{99}}},
	}

	result, err := CalculateOutputLifecycle(OutputLifecycleRequest{
		StateMode:   SubstrateStateReset,
		Plasticity:  SubstratePlasticityNone,
		LinkForm:    LinkFormL2LFeedforward,
		InputValues: [][]float64{{0.5}, {0.25}},
		Substrate:   substrateLayers,
		StaticCPP:   coordinateSumCPP{},
		CEPs:        []CEP{rawSignalCEP{}},
		Context:     context.Background(),
	})
	if err != nil {
		t.Fatalf("calculate output lifecycle reset none: %v", err)
	}
	wantOutput := math.Tanh(0.5*-1 + 0.25*1)
	if result.StateMode != SubstrateStateHold {
		t.Fatalf("unexpected next state: got=%q want=%q", result.StateMode, SubstrateStateHold)
	}
	if len(result.Outputs) != 1 || math.Abs(result.Outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected lifecycle outputs: got=%v want=%v", result.Outputs, []float64{wantOutput})
	}
	if !reflect.DeepEqual(result.Substrate[1][0].Weights, []float64{-1, 1}) {
		t.Fatalf("expected reset to repopulate static weights, got=%v", result.Substrate[1][0].Weights)
	}
}

func TestCalculateOutputLifecycleHoldNoneReusesWeights(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0}},
		{{Coords: []float64{1}, Output: 0, Weights: []float64{-0.5}}},
	}

	result, err := CalculateOutputLifecycle(OutputLifecycleRequest{
		StateMode:   SubstrateStateHold,
		Plasticity:  SubstratePlasticityNone,
		LinkForm:    LinkFormL2LFeedforward,
		InputValues: [][]float64{{0.8}},
		Substrate:   substrateLayers,
	})
	if err != nil {
		t.Fatalf("calculate output lifecycle hold none: %v", err)
	}
	wantOutput := math.Tanh(0.8 * -0.5)
	if result.StateMode != SubstrateStateHold {
		t.Fatalf("unexpected next state: got=%q want=%q", result.StateMode, SubstrateStateHold)
	}
	if len(result.Outputs) != 1 || math.Abs(result.Outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected lifecycle hold outputs: got=%v want=%v", result.Outputs, []float64{wantOutput})
	}
	if result.Substrate[1][0].Weights[0] != -0.5 {
		t.Fatalf("expected hold to reuse weights, got=%v", result.Substrate[1][0].Weights)
	}
}

func TestCalculateOutputLifecycleIterativeStaysIterative(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0.2}},
		{{Coords: []float64{1}, Output: 0.3, Weights: []float64{0.4}}},
	}

	result, err := CalculateOutputLifecycle(OutputLifecycleRequest{
		StateMode:    SubstrateStateIterative,
		Plasticity:   SubstratePlasticityIterative,
		LinkForm:     LinkFormL2LFeedforward,
		InputValues:  [][]float64{{0.8}},
		Substrate:    substrateLayers,
		IterativeCPP: iowDeltaCPP{},
		CEPs:         []CEP{rawSignalCEP{}},
		Context:      context.Background(),
	})
	if err != nil {
		t.Fatalf("calculate output lifecycle iterative: %v", err)
	}
	wantWeight := 0.2 + 0.3
	wantOutput := math.Tanh(0.8 * wantWeight)
	if result.StateMode != SubstrateStateIterative {
		t.Fatalf("unexpected next state: got=%q want=%q", result.StateMode, SubstrateStateIterative)
	}
	if len(result.Outputs) != 1 || math.Abs(result.Outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected iterative lifecycle outputs: got=%v want=%v", result.Outputs, []float64{wantOutput})
	}
	if math.Abs(result.Substrate[1][0].Weights[0]-wantWeight) > 1e-12 {
		t.Fatalf("expected iterative weight update, got=%v want=%v", result.Substrate[1][0].Weights[0], wantWeight)
	}
}

func TestCalculateOutputLifecycleABCNSelectsTypedABCNState(t *testing.T) {
	abcnSubstrate := ABCNSubstrate{
		InputLayer: CoordinateHyperlayer{{Coords: []float64{-1}, Output: 0}},
		Layers: []ABCNCoordinateHyperlayer{
			{{Coords: []float64{1}, Output: 0, Weights: []ABCNWeight{{Weight: 0.5, A: 0.1, B: 0.2, C: 0.3, N: 0.4}}}},
		},
	}

	result, err := CalculateOutputLifecycle(OutputLifecycleRequest{
		StateMode:   SubstrateStateReset,
		Plasticity:  SubstratePlasticityABCN,
		LinkForm:    LinkFormL2LFeedforward,
		InputValues: [][]float64{{0.6}},
		ABCN:        abcnSubstrate,
	})
	if err != nil {
		t.Fatalf("calculate output lifecycle abcn: %v", err)
	}
	wantOutput := math.Tanh(0.6 * 0.5)
	if result.StateMode != SubstrateStateHold {
		t.Fatalf("unexpected next state: got=%q want=%q", result.StateMode, SubstrateStateHold)
	}
	if len(result.Outputs) != 1 || math.Abs(result.Outputs[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected abcn lifecycle outputs: got=%v want=%v", result.Outputs, []float64{wantOutput})
	}
	wantWeight := ABCNWeightUpdate(0.6, wantOutput, abcnSubstrate.Layers[0][0].Weights[0])
	if result.ABCNSubstrate.Layers[0][0].Weights[0] != wantWeight {
		t.Fatalf("unexpected abcn lifecycle weight: got=%+v want=%+v", result.ABCNSubstrate.Layers[0][0].Weights[0], wantWeight)
	}
}

func TestCalculateOutputLifecycleValidatesStateAndPlasticity(t *testing.T) {
	if _, err := CalculateOutputLifecycle(OutputLifecycleRequest{StateMode: "unknown"}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for unknown state, got %v", err)
	}
	if _, err := CalculateOutputLifecycle(OutputLifecycleRequest{Plasticity: "hebbian"}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for unknown plasticity, got %v", err)
	}
	if _, err := CalculateOutputLifecycle(OutputLifecycleRequest{
		StateMode:  SubstrateStateHold,
		Plasticity: SubstratePlasticityIterative,
	}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for iterative hold state, got %v", err)
	}
	if _, err := CalculateOutputLifecycle(OutputLifecycleRequest{
		StateMode:  SubstrateStateIterative,
		Plasticity: SubstratePlasticityABCN,
	}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for abcn iterative state, got %v", err)
	}
}

func TestPopulateProcessHyperlayersStaticL2L(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0.5}, {Coords: []float64{1}, Output: -0.25}},
		{{Coords: []float64{0}, Output: 0.1, Weights: []float64{99}}},
		{{Coords: []float64{2}, Output: 0.2}},
	}

	got, err := PopulateProcessHyperlayersStatic(context.Background(), substrateLayers, LinkFormL2LFeedforward, coordinateSumCPP{}, []CEP{rawSignalCEP{}}, nil)
	if err != nil {
		t.Fatalf("populate process hyperlayers static: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("unexpected populated layer count: got=%d want=2", len(got))
	}
	if !reflect.DeepEqual(got[0][0].Weights, []float64{-1, 1}) {
		t.Fatalf("unexpected l2l hidden weights: %v", got[0][0].Weights)
	}
	if !reflect.DeepEqual(got[1][0].Weights, []float64{2}) {
		t.Fatalf("unexpected l2l output weights: %v", got[1][0].Weights)
	}
	if got[0][0].Output != 0.1 {
		t.Fatalf("expected l2l output state to be preserved, got=%v", got[0][0].Output)
	}

	substrateLayers[0][0].Coords[0] = 99
	if got[0][0].Coords[0] != 0 {
		t.Fatalf("expected populated l2l layers to be copied, got=%v", got[0][0].Coords)
	}
}

func TestPopulateProcessHyperlayersStaticFullyInterconnected(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}}},
		{{Coords: []float64{0}, Output: 0.1}},
		{{Coords: []float64{1}, Output: 0.2}},
	}

	got, err := PopulateProcessHyperlayersStatic(context.Background(), substrateLayers, LinkFormFullyInterconnected, coordinateSumCPP{}, []CEP{rawSignalCEP{}}, nil)
	if err != nil {
		t.Fatalf("populate process hyperlayers static: %v", err)
	}
	if !reflect.DeepEqual(got[0][0].Weights, []float64{-1, 0, 1}) {
		t.Fatalf("unexpected fully-interconnected hidden weights: %v", got[0][0].Weights)
	}
	if !reflect.DeepEqual(got[1][0].Weights, []float64{0, 1, 2}) {
		t.Fatalf("unexpected fully-interconnected output weights: %v", got[1][0].Weights)
	}
}

func TestPopulateProcessHyperlayersStaticJordanRecurrent(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}}, {Coords: []float64{1}}},
		{{Coords: []float64{0}}},
		{{Coords: []float64{2}, Output: 0.4}},
	}

	got, err := PopulateProcessHyperlayersStatic(context.Background(), substrateLayers, LinkFormJordanRecurrent, coordinateSumCPP{}, []CEP{rawSignalCEP{}}, nil)
	if err != nil {
		t.Fatalf("populate process hyperlayers static: %v", err)
	}
	if !reflect.DeepEqual(got[0][0].Weights, []float64{-1, 1, 2}) {
		t.Fatalf("unexpected jordan hidden weights: %v", got[0][0].Weights)
	}
	if !reflect.DeepEqual(got[1][0].Weights, []float64{2}) {
		t.Fatalf("unexpected jordan output weights: %v", got[1][0].Weights)
	}
}

func TestPopulateProcessHyperlayersStaticNeuronSelfRecurrent(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}}},
		{{Coords: []float64{0}, Output: 0.3}},
		{{Coords: []float64{2}, Output: 0.4}},
	}

	got, err := PopulateProcessHyperlayersStatic(context.Background(), substrateLayers, LinkFormNeuronSelfRecurrent, coordinateSumCPP{}, []CEP{rawSignalCEP{}}, nil)
	if err != nil {
		t.Fatalf("populate process hyperlayers static: %v", err)
	}
	if !reflect.DeepEqual(got[0][0].Weights, []float64{0, -1}) {
		t.Fatalf("unexpected neuron-self hidden weights: %v", got[0][0].Weights)
	}
	if !reflect.DeepEqual(got[1][0].Weights, []float64{4, 2}) {
		t.Fatalf("unexpected neuron-self output weights: %v", got[1][0].Weights)
	}
}

func TestPopulateProcessHyperlayersStaticValidatesInputs(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}}},
		{{Coords: []float64{1}}},
	}
	tests := []struct {
		name     string
		layers   []CoordinateHyperlayer
		linkForm string
		cpp      CoordinateCPP
		ceps     []CEP
		want     error
	}{
		{name: "missing substrate", layers: nil, linkForm: LinkFormL2LFeedforward, cpp: coordinateSumCPP{}, ceps: []CEP{rawSignalCEP{}}, want: ErrInvalidSubstrateCoordinates},
		{name: "missing cpp", layers: substrateLayers, linkForm: LinkFormL2LFeedforward, ceps: []CEP{rawSignalCEP{}}, want: ErrInvalidSubstrateCoordinates},
		{name: "missing cep", layers: substrateLayers, linkForm: LinkFormL2LFeedforward, cpp: coordinateSumCPP{}, want: ErrInvalidSubstrateCoordinates},
		{name: "unsupported link", layers: substrateLayers, linkForm: "planeself_recurrent", cpp: coordinateSumCPP{}, ceps: []CEP{rawSignalCEP{}}, want: ErrUnsupportedSubstrateLink},
		{name: "empty target", layers: []CoordinateHyperlayer{{{Coords: []float64{0}}}, nil}, linkForm: LinkFormL2LFeedforward, cpp: coordinateSumCPP{}, ceps: []CEP{rawSignalCEP{}}, want: ErrInvalidSubstrateCoordinates},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PopulateProcessHyperlayersStatic(context.Background(), tt.layers, tt.linkForm, tt.cpp, tt.ceps, nil)
			if !errors.Is(err, tt.want) {
				t.Fatalf("expected %v, got %v", tt.want, err)
			}
		})
	}
}

func TestPopulateProcessHyperlayersIterativeL2LUsesIOWAndPreviousWeights(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0.5}, {Coords: []float64{1}, Output: -0.25}},
		{{Coords: []float64{0}, Output: 0.1, Weights: []float64{0.2, -0.3}}},
		{{Coords: []float64{2}, Output: 0.2, Weights: []float64{0.4}}},
	}

	got, err := PopulateProcessHyperlayersIterative(context.Background(), substrateLayers, LinkFormL2LFeedforward, iowDeltaCPP{}, []CEP{rawAddCEP{}}, nil)
	if err != nil {
		t.Fatalf("populate process hyperlayers iterative: %v", err)
	}
	wantHidden := []float64{0.2 + 0.5 + 0.1, -0.3 + -0.25 + 0.1}
	for i := range wantHidden {
		if math.Abs(got[0][0].Weights[i]-wantHidden[i]) > 1e-12 {
			t.Fatalf("unexpected iterative hidden weight[%d]: got=%v want=%v", i, got[0][0].Weights[i], wantHidden[i])
		}
	}
	hiddenOut := got[0][0].Output
	wantOutput := 0.4 + hiddenOut + 0.2
	if math.Abs(got[1][0].Weights[0]-wantOutput) > 1e-12 {
		t.Fatalf("unexpected iterative output weight: got=%v want=%v", got[1][0].Weights[0], wantOutput)
	}
}

func TestPopulateProcessHyperlayersIterativeFullyInterconnected(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0.5}},
		{{Coords: []float64{0}, Output: 0.1, Weights: []float64{0.2, 0.3, 0.4}}},
		{{Coords: []float64{1}, Output: -0.2, Weights: []float64{-0.3, -0.4, -0.5}}},
	}

	got, err := PopulateProcessHyperlayersIterative(context.Background(), substrateLayers, LinkFormFullyInterconnected, iowDeltaCPP{}, []CEP{rawAddCEP{}}, nil)
	if err != nil {
		t.Fatalf("populate process hyperlayers iterative: %v", err)
	}
	wantHidden := []float64{0.2 + 0.5 + 0.1, 0.3 + 0.1 + 0.1, 0.4 + -0.2 + 0.1}
	for i := range wantHidden {
		if math.Abs(got[0][0].Weights[i]-wantHidden[i]) > 1e-12 {
			t.Fatalf("unexpected iterative fi hidden weight[%d]: got=%v want=%v", i, got[0][0].Weights[i], wantHidden[i])
		}
	}
	wantOutput := []float64{-0.3 + 0.5 - 0.2, -0.4 + 0.1 - 0.2, -0.5 - 0.2 - 0.2}
	for i := range wantOutput {
		if math.Abs(got[1][0].Weights[i]-wantOutput[i]) > 1e-12 {
			t.Fatalf("unexpected iterative fi output weight[%d]: got=%v want=%v", i, got[1][0].Weights[i], wantOutput[i])
		}
	}
}

func TestPopulateProcessHyperlayersIterativeJordanRecurrent(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0.5}, {Coords: []float64{1}, Output: -0.25}},
		{{Coords: []float64{0}, Output: 0.1, Weights: []float64{0.2, -0.3, 0.4}}},
		{{Coords: []float64{2}, Output: 0.2, Weights: []float64{0.6}}},
	}

	got, err := PopulateProcessHyperlayersIterative(context.Background(), substrateLayers, LinkFormJordanRecurrent, iowDeltaCPP{}, []CEP{rawAddCEP{}}, nil)
	if err != nil {
		t.Fatalf("populate process hyperlayers iterative: %v", err)
	}
	wantHidden := []float64{0.2 + 0.5 + 0.1, -0.3 + -0.25 + 0.1, 0.4 + 0.2 + 0.1}
	for i := range wantHidden {
		if math.Abs(got[0][0].Weights[i]-wantHidden[i]) > 1e-12 {
			t.Fatalf("unexpected iterative jordan hidden weight[%d]: got=%v want=%v", i, got[0][0].Weights[i], wantHidden[i])
		}
	}
}

func TestPopulateProcessHyperlayersIterativeNeuronSelfRecurrent(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0.5}},
		{{Coords: []float64{0}, Output: 0.1, Weights: []float64{0.2, -0.3}}},
		{{Coords: []float64{2}, Output: 0.2, Weights: []float64{0.4, 0.6}}},
	}

	got, err := PopulateProcessHyperlayersIterative(context.Background(), substrateLayers, LinkFormNeuronSelfRecurrent, iowDeltaCPP{}, []CEP{rawAddCEP{}}, nil)
	if err != nil {
		t.Fatalf("populate process hyperlayers iterative: %v", err)
	}
	wantHidden := []float64{0.2 + 0.1 + 0.1, -0.3 + 0.5 + 0.1}
	for i := range wantHidden {
		if math.Abs(got[0][0].Weights[i]-wantHidden[i]) > 1e-12 {
			t.Fatalf("unexpected iterative nsr hidden weight[%d]: got=%v want=%v", i, got[0][0].Weights[i], wantHidden[i])
		}
	}
	wantOutput := []float64{0.4 + 0.2 + 0.2, 0.6 + 0.1 + 0.2}
	for i := range wantOutput {
		if math.Abs(got[1][0].Weights[i]-wantOutput[i]) > 1e-12 {
			t.Fatalf("unexpected iterative nsr output weight[%d]: got=%v want=%v", i, got[1][0].Weights[i], wantOutput[i])
		}
	}
}

func TestPopulateProcessHyperlayersIterativeValidatesInputs(t *testing.T) {
	substrateLayers := []CoordinateHyperlayer{
		{{Coords: []float64{-1}, Output: 0.5}},
		{{Coords: []float64{1}, Output: 0.1, Weights: []float64{0.2}}},
	}
	tests := []struct {
		name     string
		layers   []CoordinateHyperlayer
		linkForm string
		cpp      CoordinateIOWCPP
		ceps     []CEP
		want     error
	}{
		{name: "missing substrate", layers: nil, linkForm: LinkFormL2LFeedforward, cpp: iowDeltaCPP{}, ceps: []CEP{rawAddCEP{}}, want: ErrInvalidSubstrateCoordinates},
		{name: "missing cpp", layers: substrateLayers, linkForm: LinkFormL2LFeedforward, ceps: []CEP{rawAddCEP{}}, want: ErrInvalidSubstrateCoordinates},
		{name: "missing cep", layers: substrateLayers, linkForm: LinkFormL2LFeedforward, cpp: iowDeltaCPP{}, want: ErrInvalidSubstrateCoordinates},
		{name: "unsupported link", layers: substrateLayers, linkForm: "planeself_recurrent", cpp: iowDeltaCPP{}, ceps: []CEP{rawAddCEP{}}, want: ErrUnsupportedSubstrateLink},
		{name: "weight count mismatch", layers: []CoordinateHyperlayer{{{Coords: []float64{-1}, Output: 0.5}}, {{Coords: []float64{1}, Output: 0.1, Weights: []float64{0.2, 0.3}}}}, linkForm: LinkFormL2LFeedforward, cpp: iowDeltaCPP{}, ceps: []CEP{rawAddCEP{}}, want: ErrInvalidSubstrateCoordinates},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PopulateProcessHyperlayersIterative(context.Background(), tt.layers, tt.linkForm, tt.cpp, tt.ceps, nil)
			if !errors.Is(err, tt.want) {
				t.Fatalf("expected %v, got %v", tt.want, err)
			}
		})
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

func TestCreateSubstrateBuildsDimensionedDepthZeroLayers(t *testing.T) {
	layers, err := CreateSubstrate(CreateSubstrateRequest{
		InputSpecs: []IOCoordinateSpec{
			{Format: CoordinateFormatNoGeo, VL: 2},
		},
		Densities: []int{0, 2, 2},
		OutputSpecs: []IOCoordinateSpec{
			{Format: CoordinateFormatNoGeo, VL: 1},
		},
		LinkForm: LinkFormL2LFeedforward,
	})
	if err != nil {
		t.Fatalf("create substrate: %v", err)
	}
	if len(layers) != 2 {
		t.Fatalf("unexpected layer count: got=%d want=2", len(layers))
	}
	if got := layers[0].Coordinates(); !reflect.DeepEqual(got, [][]float64{{-1, 0, -1}, {-1, 0, 1}}) {
		t.Fatalf("unexpected create-substrate input coords: got=%v", got)
	}
	if got := layers[1].Coordinates(); !reflect.DeepEqual(got, [][]float64{{1, 0, 0}}) {
		t.Fatalf("unexpected create-substrate output coords: got=%v", got)
	}
	if got := layers[1][0].Weights; !reflect.DeepEqual(got, []float64{0, 0}) {
		t.Fatalf("unexpected create-substrate output weights: got=%v", got)
	}
}

func TestCreateSubstrateBuildsRecurrentLayersWithDimensionedIO(t *testing.T) {
	layers, err := CreateSubstrate(CreateSubstrateRequest{
		InputSpecs: []IOCoordinateSpec{
			{Format: CoordinateFormatNoGeo, VL: 2},
		},
		Densities: []int{2, 2, 2},
		OutputSpecs: []IOCoordinateSpec{
			{Format: CoordinateFormatCoorded, Dim: 1, Neurodes: CoordinateHyperlayer{{Coords: []float64{0.5}, Output: 0.75}}},
		},
		LinkForm: LinkFormJordanRecurrent,
	})
	if err != nil {
		t.Fatalf("create substrate: %v", err)
	}
	if len(layers) != 3 {
		t.Fatalf("unexpected layer count: got=%d want=3", len(layers))
	}
	wantInput := [][]float64{{-1, 0, -1}, {-1, 0, 1}}
	if got := layers[0].Coordinates(); !reflect.DeepEqual(got, wantInput) {
		t.Fatalf("unexpected create-substrate recurrent input coords: got=%v want=%v", got, wantInput)
	}
	wantRecurrent := [][]float64{
		{0, -1, -1},
		{0, -1, 1},
		{0, 1, -1},
		{0, 1, 1},
	}
	if got := layers[1].Coordinates(); !reflect.DeepEqual(got, wantRecurrent) {
		t.Fatalf("unexpected create-substrate recurrent coords: got=%v", got)
	}
	if got := layers[1][0].Weights; !reflect.DeepEqual(got, []float64{0, 0, 0}) {
		t.Fatalf("unexpected create-substrate recurrent weights: got=%v", got)
	}
	if got := layers[2].Coordinates(); !reflect.DeepEqual(got, [][]float64{{1, 0, 0.5}}) {
		t.Fatalf("unexpected create-substrate output coords: got=%v", got)
	}
	if layers[2][0].Output != 0.75 {
		t.Fatalf("unexpected create-substrate output value: got=%v", layers[2][0].Output)
	}
	if got := layers[2][0].Weights; !reflect.DeepEqual(got, []float64{0, 0, 0, 0}) {
		t.Fatalf("unexpected create-substrate output weights: got=%v", got)
	}
}

func TestCreateSubstrateValidatesInputs(t *testing.T) {
	_, err := CreateSubstrate(CreateSubstrateRequest{
		InputSpecs:  []IOCoordinateSpec{{Format: CoordinateFormatNoGeo, VL: 1}},
		OutputSpecs: []IOCoordinateSpec{{Format: CoordinateFormatNoGeo, VL: 1}},
		LinkForm:    LinkFormL2LFeedforward,
	})
	if !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for missing densities, got %v", err)
	}
}

func TestBuildSubstrateLayersDepthZeroAttachesInputWeightsToOutput(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}}, {Coords: []float64{1}}}
	output := CoordinateHyperlayer{{Coords: []float64{0.5}}}

	layers, err := BuildSubstrateLayers(SubstrateLayerBuildRequest{
		InputLayer:  input,
		Densities:   []int{0, 2},
		OutputLayer: output,
		LinkForm:    LinkFormL2LFeedforward,
	})
	if err != nil {
		t.Fatalf("build substrate layers: %v", err)
	}
	if len(layers) != 2 {
		t.Fatalf("unexpected layer count: got=%d want=2", len(layers))
	}
	if got := layers[0].Coordinates(); !reflect.DeepEqual(got, [][]float64{{-1}, {1}}) {
		t.Fatalf("unexpected input coordinates: got=%v", got)
	}
	if got := layers[1][0].Weights; !reflect.DeepEqual(got, []float64{0, 0}) {
		t.Fatalf("unexpected output weights: got=%v want=[0 0]", got)
	}
}

func TestBuildSubstrateLayersDepthOneBuildsExtrudedRecurrentLayer(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}}, {Coords: []float64{1}}}
	output := CoordinateHyperlayer{{Coords: []float64{0.5}}}

	layers, err := BuildSubstrateLayers(SubstrateLayerBuildRequest{
		InputLayer:  input,
		Densities:   []int{1, 2, 3},
		OutputLayer: output,
		LinkForm:    LinkFormL2LFeedforward,
	})
	if err != nil {
		t.Fatalf("build substrate layers: %v", err)
	}
	if len(layers) != 3 {
		t.Fatalf("unexpected layer count: got=%d want=3", len(layers))
	}
	wantHidden := [][]float64{
		{0, -1, -1},
		{0, -1, 0},
		{0, -1, 1},
		{0, 1, -1},
		{0, 1, 0},
		{0, 1, 1},
	}
	if got := layers[1].Coordinates(); !reflect.DeepEqual(got, wantHidden) {
		t.Fatalf("unexpected recurrent coordinates: got=%v want=%v", got, wantHidden)
	}
	if got := layers[1][0].Weights; !reflect.DeepEqual(got, []float64{0, 0}) {
		t.Fatalf("unexpected recurrent input weights: got=%v want=[0 0]", got)
	}
	if got := layers[2][0].Weights; !reflect.DeepEqual(got, []float64{0, 0, 0, 0, 0, 0}) {
		t.Fatalf("unexpected output hidden weights: got=%v", got)
	}
}

func TestBuildSubstrateLayersDepthThreeBuildsReferenceDepthCoordinates(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{-1}}}
	output := CoordinateHyperlayer{{Coords: []float64{1}}}

	layers, err := BuildSubstrateLayers(SubstrateLayerBuildRequest{
		InputLayer:  input,
		Densities:   []int{3, 2},
		OutputLayer: output,
		LinkForm:    LinkFormL2LFeedforward,
	})
	if err != nil {
		t.Fatalf("build substrate layers: %v", err)
	}
	if len(layers) != 4 {
		t.Fatalf("unexpected layer count: got=%d want=4", len(layers))
	}
	recurrentCoord := -1 + 2.0/3.0
	hiddenCoord := -1 + 4.0/3.0
	if got := layers[1].Coordinates(); !coordsAlmostEqual(got, [][]float64{{recurrentCoord, -1}, {recurrentCoord, 1}}) {
		t.Fatalf("unexpected recurrent depth coordinates: got=%v", got)
	}
	if got := layers[2].Coordinates(); !coordsAlmostEqual(got, [][]float64{{hiddenCoord, -1}, {hiddenCoord, 1}}) {
		t.Fatalf("unexpected first hidden depth coordinates: got=%v", got)
	}
}

func TestBuildSubstrateLayersSetsLinkFormWeightCounts(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{0}}, {Coords: []float64{1}}}
	output := CoordinateHyperlayer{{Coords: []float64{9}}, {Coords: []float64{10}}}

	tests := []struct {
		name          string
		linkForm      string
		wantFirst     int
		wantFollowing int
	}{
		{name: "l2l", linkForm: LinkFormL2LFeedforward, wantFirst: 2, wantFollowing: 6},
		{name: "fully", linkForm: LinkFormFullyInterconnected, wantFirst: 10, wantFollowing: 10},
		{name: "jordan", linkForm: LinkFormJordanRecurrent, wantFirst: 4, wantFollowing: 6},
		{name: "neuronself", linkForm: LinkFormNeuronSelfRecurrent, wantFirst: 3, wantFollowing: 7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layers, err := BuildSubstrateLayers(SubstrateLayerBuildRequest{
				InputLayer:  input,
				Densities:   []int{2, 2, 3},
				OutputLayer: output,
				LinkForm:    tt.linkForm,
			})
			if err != nil {
				t.Fatalf("build substrate layers: %v", err)
			}
			if got := len(layers[1][0].Weights); got != tt.wantFirst {
				t.Fatalf("unexpected first hidden weight count: got=%d want=%d", got, tt.wantFirst)
			}
			if got := len(layers[2][0].Weights); got != tt.wantFollowing {
				t.Fatalf("unexpected output weight count: got=%d want=%d", got, tt.wantFollowing)
			}
		})
	}
}

func TestBuildSubstrateLayersValidatesInput(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{0}}}
	output := CoordinateHyperlayer{{Coords: []float64{1}}}

	tests := []struct {
		name string
		req  SubstrateLayerBuildRequest
		err  error
	}{
		{name: "missing densities", req: SubstrateLayerBuildRequest{InputLayer: input, OutputLayer: output, LinkForm: LinkFormL2LFeedforward}, err: ErrInvalidSubstrateCoordinates},
		{name: "missing input", req: SubstrateLayerBuildRequest{Densities: []int{0}, OutputLayer: output, LinkForm: LinkFormL2LFeedforward}, err: ErrInvalidSubstrateCoordinates},
		{name: "missing output", req: SubstrateLayerBuildRequest{InputLayer: input, Densities: []int{0}, LinkForm: LinkFormL2LFeedforward}, err: ErrInvalidSubstrateCoordinates},
		{name: "negative depth", req: SubstrateLayerBuildRequest{InputLayer: input, Densities: []int{-1, 2}, OutputLayer: output, LinkForm: LinkFormL2LFeedforward}, err: ErrInvalidSubstrateCoordinates},
		{name: "missing hidden densities", req: SubstrateLayerBuildRequest{InputLayer: input, Densities: []int{1}, OutputLayer: output, LinkForm: LinkFormL2LFeedforward}, err: ErrInvalidSubstrateCoordinates},
		{name: "unsupported link", req: SubstrateLayerBuildRequest{InputLayer: input, Densities: []int{0, 2}, OutputLayer: output, LinkForm: "planeself_recurrent"}, err: ErrUnsupportedSubstrateLink},
		{name: "fully zero depth", req: SubstrateLayerBuildRequest{InputLayer: input, Densities: []int{0, 2}, OutputLayer: output, LinkForm: LinkFormFullyInterconnected}, err: ErrInvalidSubstrateCoordinates},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := BuildSubstrateLayers(tt.req); !errors.Is(err, tt.err) {
				t.Fatalf("expected %v, got %v", tt.err, err)
			}
		})
	}
}

func TestBuildSubstrateLayersCopiesInputsOutputsAndWeights(t *testing.T) {
	input := CoordinateHyperlayer{{Coords: []float64{0}, Weights: []float64{7}}}
	output := CoordinateHyperlayer{{Coords: []float64{1}, Weights: []float64{8}}}

	layers, err := BuildSubstrateLayers(SubstrateLayerBuildRequest{
		InputLayer:  input,
		Densities:   []int{0, 2},
		OutputLayer: output,
		LinkForm:    LinkFormL2LFeedforward,
	})
	if err != nil {
		t.Fatalf("build substrate layers: %v", err)
	}

	input[0].Coords[0] = 99
	input[0].Weights[0] = 88
	output[0].Coords[0] = 77
	output[0].Weights[0] = 66

	if got := layers[0][0].Coords[0]; got != 0 {
		t.Fatalf("expected input coord copy, got=%v", got)
	}
	if got := layers[0][0].Weights[0]; got != 7 {
		t.Fatalf("expected input weight copy, got=%v", got)
	}
	if got := layers[1][0].Coords[0]; got != 1 {
		t.Fatalf("expected output coord copy, got=%v", got)
	}
	if got := layers[1][0].Weights; !reflect.DeepEqual(got, []float64{0}) {
		t.Fatalf("expected output weights attached independently, got=%v", got)
	}
}

func TestComposeInputSubstrateSupportsUndefinedAndNoGeo(t *testing.T) {
	layer, err := ComposeInputSubstrate([]IOCoordinateSpec{
		{Format: CoordinateFormatUndefined, VL: 2},
		{Format: CoordinateFormatNoGeo, VL: 3},
	})
	if err != nil {
		t.Fatalf("compose input substrate: %v", err)
	}

	want := [][]float64{{-1}, {0}, {1}, {-1}, {1}}
	if got := layer.Coordinates(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected input substrate coords: got=%v want=%v", got, want)
	}
	for i, neurode := range layer {
		if neurode.Output != 0 {
			t.Fatalf("expected zero input output at neurode %d, got=%v", i, neurode.Output)
		}
		if neurode.Weights != nil {
			t.Fatalf("expected nil input weights at neurode %d, got=%v", i, neurode.Weights)
		}
	}
}

func TestComposeInputSubstrateDefaultsEmptyFormatToUndefined(t *testing.T) {
	layer, err := ComposeInputSubstrate([]IOCoordinateSpec{{VL: 1}})
	if err != nil {
		t.Fatalf("compose input substrate: %v", err)
	}

	if got := layer.Coordinates(); !reflect.DeepEqual(got, [][]float64{{0}}) {
		t.Fatalf("unexpected default-format input coords: got=%v", got)
	}
}

func TestComposeOutputSubstrateAttachesWeights(t *testing.T) {
	weights := []float64{0.1, 0.2}
	layer, err := ComposeOutputSubstrate([]IOCoordinateSpec{
		{Format: CoordinateFormatNoGeo, VL: 2},
	}, weights)
	if err != nil {
		t.Fatalf("compose output substrate: %v", err)
	}

	if got := layer.Coordinates(); !reflect.DeepEqual(got, [][]float64{{-1}, {1}}) {
		t.Fatalf("unexpected output substrate coords: got=%v", got)
	}
	for i, neurode := range layer {
		if !reflect.DeepEqual(neurode.Weights, []float64{0.1, 0.2}) {
			t.Fatalf("unexpected output weights at neurode %d: %v", i, neurode.Weights)
		}
	}

	weights[0] = 99
	if layer[0].Weights[0] != 0.1 {
		t.Fatalf("expected output weights to be copied, got=%v", layer[0].Weights)
	}
}

func TestComposeInputSubstrateSupportsSymmetric(t *testing.T) {
	layer, err := ComposeInputSubstrate([]IOCoordinateSpec{
		{Format: CoordinateFormatSymmetric, Resolutions: []int{2, 3}},
	})
	if err != nil {
		t.Fatalf("compose input substrate: %v", err)
	}

	want := [][]float64{
		{-1, -1},
		{-1, 1},
		{0, -1},
		{0, 1},
		{1, -1},
		{1, 1},
	}
	if got := layer.Coordinates(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected symmetric input coords: got=%v want=%v", got, want)
	}
	for i, neurode := range layer {
		if neurode.Output != 0 {
			t.Fatalf("expected zero symmetric input output at neurode %d, got=%v", i, neurode.Output)
		}
		if neurode.Weights != nil {
			t.Fatalf("expected nil symmetric input weights at neurode %d, got=%v", i, neurode.Weights)
		}
	}
}

func TestComposeOutputSubstrateSupportsSymmetric(t *testing.T) {
	weights := []float64{0.1, 0.2}
	layer, err := ComposeOutputSubstrate([]IOCoordinateSpec{
		{Format: CoordinateFormatSymmetric, Resolutions: []int{2, 2}},
	}, weights)
	if err != nil {
		t.Fatalf("compose output substrate: %v", err)
	}

	want := [][]float64{
		{-1, -1},
		{-1, 1},
		{1, -1},
		{1, 1},
	}
	if got := layer.Coordinates(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected symmetric output coords: got=%v want=%v", got, want)
	}
	for i, neurode := range layer {
		if !reflect.DeepEqual(neurode.Weights, []float64{0.1, 0.2}) {
			t.Fatalf("unexpected symmetric output weights at neurode %d: %v", i, neurode.Weights)
		}
	}

	weights[0] = 99
	if layer[0].Weights[0] != 0.1 {
		t.Fatalf("expected symmetric output weights to be copied, got=%v", layer[0].Weights)
	}
}

func TestComposeInputSubstrateSupportsCoorded(t *testing.T) {
	input := CoordinateHyperlayer{
		{Coords: []float64{-0.5, 0.25}, Output: 1.5},
		{Coords: []float64{0.75, -0.25}, Output: -0.5, Weights: []float64{0.9}},
	}
	layer, err := ComposeInputSubstrate([]IOCoordinateSpec{
		{Format: CoordinateFormatCoorded, Dim: 2, Neurodes: input},
	})
	if err != nil {
		t.Fatalf("compose input substrate: %v", err)
	}

	wantCoords := [][]float64{{-0.5, 0.25}, {0.75, -0.25}}
	if got := layer.Coordinates(); !reflect.DeepEqual(got, wantCoords) {
		t.Fatalf("unexpected coorded input coords: got=%v want=%v", got, wantCoords)
	}
	if layer[0].Output != 1.5 || layer[1].Output != -0.5 {
		t.Fatalf("unexpected coorded input outputs: %v, %v", layer[0].Output, layer[1].Output)
	}
	if layer[0].Weights != nil || !reflect.DeepEqual(layer[1].Weights, []float64{0.9}) {
		t.Fatalf("unexpected coorded input weights: %v", layer)
	}

	input[0].Coords[0] = 99
	input[1].Weights[0] = 99
	if layer[0].Coords[0] != -0.5 || layer[1].Weights[0] != 0.9 {
		t.Fatalf("expected coorded input neurodes to be copied, got=%v", layer)
	}
}

func TestComposeOutputSubstrateSupportsCoorded(t *testing.T) {
	input := CoordinateHyperlayer{
		{Coords: []float64{-1, 0}, Output: 0.25, Weights: []float64{99}},
		{Coords: []float64{1, 0}, Output: 0.75},
	}
	weights := []float64{0.1, 0.2}
	layer, err := ComposeOutputSubstrate([]IOCoordinateSpec{
		{Format: CoordinateFormatCoorded, Dim: 2, Neurodes: input},
	}, weights)
	if err != nil {
		t.Fatalf("compose output substrate: %v", err)
	}

	wantCoords := [][]float64{{-1, 0}, {1, 0}}
	if got := layer.Coordinates(); !reflect.DeepEqual(got, wantCoords) {
		t.Fatalf("unexpected coorded output coords: got=%v want=%v", got, wantCoords)
	}
	if layer[0].Output != 0.25 || layer[1].Output != 0.75 {
		t.Fatalf("unexpected coorded output outputs: %v, %v", layer[0].Output, layer[1].Output)
	}
	for i, neurode := range layer {
		if !reflect.DeepEqual(neurode.Weights, []float64{0.1, 0.2}) {
			t.Fatalf("unexpected coorded output weights at neurode %d: %v", i, neurode.Weights)
		}
	}

	input[0].Coords[0] = 99
	weights[0] = 99
	if layer[0].Coords[0] != -1 || layer[0].Weights[0] != 0.1 {
		t.Fatalf("expected coorded output neurodes and weights to be copied, got=%v", layer)
	}
}

func TestComposeInputSubstrateForDimensionAdvExtrudesParts(t *testing.T) {
	layer, err := ComposeInputSubstrateForDimension([]IOCoordinateSpec{
		{Format: CoordinateFormatNoGeo, VL: 2},
		{Format: CoordinateFormatSymmetric, Resolutions: []int{2, 2}},
	}, 4)
	if err != nil {
		t.Fatalf("compose input substrate for dimension: %v", err)
	}

	want := [][]float64{
		{-1, -1, 0, -1},
		{-1, -1, 0, 1},
		{-1, 1, -1, -1},
		{-1, 1, -1, 1},
		{-1, 1, 1, -1},
		{-1, 1, 1, 1},
	}
	if got := layer.Coordinates(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected dimensioned input coords: got=%v want=%v", got, want)
	}
	for i, neurode := range layer {
		if neurode.Output != 0 {
			t.Fatalf("expected zero dimensioned input output at neurode %d, got=%v", i, neurode.Output)
		}
		if neurode.Weights != nil {
			t.Fatalf("expected nil dimensioned input weights at neurode %d, got=%v", i, neurode.Weights)
		}
	}
}

func TestComposeOutputSubstrateForDimensionAdvExtrudesParts(t *testing.T) {
	weights := []float64{0.1, 0.2}
	layer, err := ComposeOutputSubstrateForDimension([]IOCoordinateSpec{
		{Format: CoordinateFormatCoorded, Dim: 2, Neurodes: CoordinateHyperlayer{
			{Coords: []float64{-0.5, 0.5}, Output: 0.75, Weights: []float64{99}},
		}},
		{Format: CoordinateFormatNoGeo, VL: 2},
	}, 5, weights)
	if err != nil {
		t.Fatalf("compose output substrate for dimension: %v", err)
	}

	want := [][]float64{
		{1, -1, 0, -0.5, 0.5},
		{1, 1, 0, 0, -1},
		{1, 1, 0, 0, 1},
	}
	if got := layer.Coordinates(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected dimensioned output coords: got=%v want=%v", got, want)
	}
	if layer[0].Output != 0.75 || layer[1].Output != 0 || layer[2].Output != 0 {
		t.Fatalf("unexpected dimensioned output values: %v", layer)
	}
	for i, neurode := range layer {
		if !reflect.DeepEqual(neurode.Weights, []float64{0.1, 0.2}) {
			t.Fatalf("unexpected dimensioned output weights at neurode %d: %v", i, neurode.Weights)
		}
	}

	weights[0] = 99
	if layer[0].Weights[0] != 0.1 {
		t.Fatalf("expected dimensioned output weights to be copied, got=%v", layer[0].Weights)
	}
}

func TestComposeIOSubstrateForDimensionValidatesRequiredDimension(t *testing.T) {
	if _, err := ComposeInputSubstrateForDimension([]IOCoordinateSpec{
		{Format: CoordinateFormatSymmetric, Resolutions: []int{2, 2}},
	}, 3); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for shallow input dimension, got %v", err)
	}

	if _, err := ComposeOutputSubstrateForDimension([]IOCoordinateSpec{
		{Format: CoordinateFormatCoorded, Dim: 2, Neurodes: CoordinateHyperlayer{{Coords: []float64{-1, 1}}}},
	}, 1, []float64{0}); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected ErrInvalidSubstrateCoordinates for invalid substrate dimension, got %v", err)
	}
}

func TestComposeBaseIOSubstrateValidatesSpecs(t *testing.T) {
	tests := []struct {
		name string
		fn   func() (CoordinateHyperlayer, error)
	}{
		{name: "missing input specs", fn: func() (CoordinateHyperlayer, error) {
			return ComposeInputSubstrate(nil)
		}},
		{name: "invalid input vl", fn: func() (CoordinateHyperlayer, error) {
			return ComposeInputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatNoGeo, VL: 0}})
		}},
		{name: "missing symmetric input resolutions", fn: func() (CoordinateHyperlayer, error) {
			return ComposeInputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatSymmetric}})
		}},
		{name: "invalid symmetric input resolutions", fn: func() (CoordinateHyperlayer, error) {
			return ComposeInputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatSymmetric, Resolutions: []int{2, 0}}})
		}},
		{name: "invalid coorded input dim", fn: func() (CoordinateHyperlayer, error) {
			return ComposeInputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatCoorded, Neurodes: CoordinateHyperlayer{{Coords: []float64{0}}}}})
		}},
		{name: "missing coorded input neurodes", fn: func() (CoordinateHyperlayer, error) {
			return ComposeInputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatCoorded, Dim: 1}})
		}},
		{name: "mismatched coorded input neurode dim", fn: func() (CoordinateHyperlayer, error) {
			return ComposeInputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatCoorded, Dim: 2, Neurodes: CoordinateHyperlayer{{Coords: []float64{0}}}}})
		}},
		{name: "unsupported input format", fn: func() (CoordinateHyperlayer, error) {
			return ComposeInputSubstrate([]IOCoordinateSpec{{Format: "asymmetric", VL: 2}})
		}},
		{name: "missing output specs", fn: func() (CoordinateHyperlayer, error) {
			return ComposeOutputSubstrate(nil, []float64{0})
		}},
		{name: "invalid output vl", fn: func() (CoordinateHyperlayer, error) {
			return ComposeOutputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatNoGeo, VL: -1}}, []float64{0})
		}},
		{name: "missing symmetric output resolutions", fn: func() (CoordinateHyperlayer, error) {
			return ComposeOutputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatSymmetric}}, []float64{0})
		}},
		{name: "invalid symmetric output resolutions", fn: func() (CoordinateHyperlayer, error) {
			return ComposeOutputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatSymmetric, Resolutions: []int{0}}}, []float64{0})
		}},
		{name: "invalid coorded output dim", fn: func() (CoordinateHyperlayer, error) {
			return ComposeOutputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatCoorded, Neurodes: CoordinateHyperlayer{{Coords: []float64{0}}}}}, []float64{0})
		}},
		{name: "missing coorded output neurodes", fn: func() (CoordinateHyperlayer, error) {
			return ComposeOutputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatCoorded, Dim: 1}}, []float64{0})
		}},
		{name: "mismatched coorded output neurode dim", fn: func() (CoordinateHyperlayer, error) {
			return ComposeOutputSubstrate([]IOCoordinateSpec{{Format: CoordinateFormatCoorded, Dim: 2, Neurodes: CoordinateHyperlayer{{Coords: []float64{0}}}}}, []float64{0})
		}},
		{name: "unsupported output format", fn: func() (CoordinateHyperlayer, error) {
			return ComposeOutputSubstrate([]IOCoordinateSpec{{Format: "asymmetric", VL: 2}}, []float64{0})
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.fn(); !errors.Is(err, ErrInvalidSubstrateCoordinates) {
				t.Fatalf("expected ErrInvalidSubstrateCoordinates, got %v", err)
			}
		})
	}
}

func coordsAlmostEqual(got [][]float64, want [][]float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			return false
		}
		for j := range want[i] {
			if math.Abs(got[i][j]-want[i][j]) > 1e-12 {
				return false
			}
		}
	}
	return true
}
