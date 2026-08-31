package genotype

import (
	"errors"
	"reflect"
	"testing"

	"protogonos/internal/model"
	"protogonos/internal/substrate"
)

func TestMaterializeSubstrateCreatesABCNFromExplicitPlasticity(t *testing.T) {
	genome := model.Genome{
		ID: "agent-abcn",
		Substrate: &model.SubstrateConfig{
			Dimensions: []int{0, 2, 2},
			Plasticity: substrate.SubstratePlasticityABCN,
			LinkForm:   substrate.LinkFormL2LFeedforward,
			Parameters: map[string]float64{
				"abcn_a": 0.1,
				"abcn_b": 0.2,
				"abcn_c": 0.3,
				"abcn_n": 0.4,
			},
		},
	}

	got, err := MaterializeSubstrate(SubstrateMaterializationRequest{
		Genome:      genome,
		InputSpecs:  []substrate.IOCoordinateSpec{{Format: substrate.CoordinateFormatNoGeo, VL: 2}},
		OutputSpecs: []substrate.IOCoordinateSpec{{Format: substrate.CoordinateFormatNoGeo, VL: 1}},
	})
	if err != nil {
		t.Fatalf("materialize abcn substrate: %v", err)
	}
	if got.Plasticity != substrate.SubstratePlasticityABCN || got.LinkForm != substrate.LinkFormL2LFeedforward {
		t.Fatalf("unexpected materialized mode: %+v", got)
	}
	if len(got.Scalar) != 0 {
		t.Fatalf("expected scalar substrate to be empty for abcn materialization, got=%+v", got.Scalar)
	}
	if len(got.ABCN.InputLayer) != 2 || len(got.ABCN.Layers) != 1 {
		t.Fatalf("unexpected abcn materialized shape: %+v", got.ABCN)
	}
	wantWeight := substrate.ABCNWeight{Weight: 0, A: 0.1, B: 0.2, C: 0.3, N: 0.4}
	if got.ABCN.Layers[0][0].Weights[0] != wantWeight {
		t.Fatalf("unexpected materialized abcn weight: got=%+v want=%+v", got.ABCN.Layers[0][0].Weights[0], wantWeight)
	}
}

func TestMaterializeSubstrateCreatesScalarForNonABCN(t *testing.T) {
	genome := model.Genome{
		ID: "agent-scalar",
		Substrate: &model.SubstrateConfig{
			Dimensions: []int{0, 2, 2},
			Plasticity: substrate.SubstratePlasticityNone,
			LinkForm:   substrate.LinkFormL2LFeedforward,
		},
	}

	got, err := MaterializeSubstrate(SubstrateMaterializationRequest{
		Genome:      genome,
		InputSpecs:  []substrate.IOCoordinateSpec{{Format: substrate.CoordinateFormatNoGeo, VL: 2}},
		OutputSpecs: []substrate.IOCoordinateSpec{{Format: substrate.CoordinateFormatNoGeo, VL: 1}},
	})
	if err != nil {
		t.Fatalf("materialize scalar substrate: %v", err)
	}
	if got.Plasticity != substrate.SubstratePlasticityNone || got.LinkForm != substrate.LinkFormL2LFeedforward {
		t.Fatalf("unexpected materialized scalar mode: %+v", got)
	}
	if len(got.Scalar) != 2 {
		t.Fatalf("expected scalar substrate layers, got=%+v", got.Scalar)
	}
	if len(got.ABCN.InputLayer) != 0 || len(got.ABCN.Layers) != 0 {
		t.Fatalf("expected abcn substrate to be empty for scalar materialization, got=%+v", got.ABCN)
	}
	if got.Scalar[1][0].Weights[0] != 0 {
		t.Fatalf("unexpected scalar weight: got=%v", got.Scalar[1][0].Weights)
	}
}

func TestMaterializeSubstrateFallsBackToLegacyFields(t *testing.T) {
	genome := model.Genome{
		ID: "agent-legacy",
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.SubstratePlasticityABCN,
			CEPName:    substrate.LinkFormL2LFeedforward,
			Dimensions: []int{0, 2, 2},
			Parameters: map[string]float64{
				"a": 0.5,
				"b": 0.6,
				"c": 0.7,
				"n": 0.8,
			},
		},
	}

	got, err := MaterializeSubstrate(SubstrateMaterializationRequest{
		Genome:      genome,
		InputSpecs:  []substrate.IOCoordinateSpec{{Format: substrate.CoordinateFormatNoGeo, VL: 2}},
		OutputSpecs: []substrate.IOCoordinateSpec{{Format: substrate.CoordinateFormatNoGeo, VL: 1}},
	})
	if err != nil {
		t.Fatalf("materialize legacy substrate: %v", err)
	}
	want := substrate.ABCNWeight{Weight: 0, A: 0.5, B: 0.6, C: 0.7, N: 0.8}
	if got.Plasticity != substrate.SubstratePlasticityABCN || got.LinkForm != substrate.LinkFormL2LFeedforward {
		t.Fatalf("unexpected legacy materialized mode: %+v", got)
	}
	if got.ABCN.Layers[0][0].Weights[0] != want {
		t.Fatalf("unexpected legacy materialized abcn weight: got=%+v want=%+v", got.ABCN.Layers[0][0].Weights[0], want)
	}
}

func TestMaterializeSubstrateCopiesSpecs(t *testing.T) {
	genome := model.Genome{
		ID: "agent-copy",
		Substrate: &model.SubstrateConfig{
			Dimensions: []int{0, 2, 2},
			Plasticity: substrate.SubstratePlasticityNone,
			LinkForm:   substrate.LinkFormL2LFeedforward,
		},
	}
	inputSpecs := []substrate.IOCoordinateSpec{{Format: substrate.CoordinateFormatCoorded, Dim: 1, Neurodes: substrate.CoordinateHyperlayer{{Coords: []float64{0.25}}}}}
	outputSpecs := []substrate.IOCoordinateSpec{{Format: substrate.CoordinateFormatCoorded, Dim: 1, Neurodes: substrate.CoordinateHyperlayer{{Coords: []float64{0.75}}}}}

	got, err := MaterializeSubstrate(SubstrateMaterializationRequest{
		Genome:      genome,
		InputSpecs:  inputSpecs,
		OutputSpecs: outputSpecs,
	})
	if err != nil {
		t.Fatalf("materialize copied substrate: %v", err)
	}
	inputSpecs[0].Neurodes[0].Coords[0] = 99
	outputSpecs[0].Neurodes[0].Coords[0] = 88
	if coords := got.Scalar[0].Coordinates(); !reflect.DeepEqual(coords, [][]float64{{-1, 0, 0.25}}) {
		t.Fatalf("expected materialized input coords to be copied, got=%v", coords)
	}
	if coords := got.Scalar[1].Coordinates(); !reflect.DeepEqual(coords, [][]float64{{1, 0, 0.75}}) {
		t.Fatalf("expected materialized output coords to be copied, got=%v", coords)
	}
}

func TestMaterializeSubstrateRequiresConfig(t *testing.T) {
	_, err := MaterializeSubstrate(SubstrateMaterializationRequest{
		Genome: model.Genome{ID: "agent-no-substrate"},
	})
	if err == nil || errors.Is(err, substrate.ErrInvalidSubstrateCoordinates) {
		t.Fatalf("expected explicit missing config error, got %v", err)
	}
}
