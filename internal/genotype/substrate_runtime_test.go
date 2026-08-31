package genotype

import (
	"context"
	"testing"

	"protogonos/internal/model"
	"protogonos/internal/substrate"
)

func TestBuildSubstrateLayerRuntimeUsesExplicitTypedConfig(t *testing.T) {
	rt, handled, err := BuildSubstrateLayerRuntime(SubstrateLayerRuntimeBuildRequest{
		Genome: model.Genome{
			ID: "typed-substrate-runtime",
			Substrate: &model.SubstrateConfig{
				Dimensions: []int{0, 2, 2},
				CEPName:    substrate.WeightExpressionCEPName,
				Plasticity: substrate.SubstratePlasticityNone,
				LinkForm:   substrate.LinkFormL2LFeedforward,
			},
		},
		InputWidth:  2,
		OutputWidth: 1,
	})
	if err != nil {
		t.Fatalf("build typed substrate runtime: %v", err)
	}
	if !handled {
		t.Fatalf("expected explicit typed config to be handled")
	}
	if _, ok := rt.(*substrate.LayerRuntime); !ok {
		t.Fatalf("expected LayerRuntime, got %T", rt)
	}
	if got, err := rt.Step(context.Background(), []float64{1, 1}); err != nil || len(got) != 1 {
		t.Fatalf("step typed substrate runtime: got=%v err=%v", got, err)
	}
	weights := rt.Weights()
	if len(weights) != 2 || weights[0] == 0 || weights[1] == 0 {
		t.Fatalf("expected default coordinate population, got weights=%v", weights)
	}
}

func TestBuildSubstrateLayerRuntimeUsesABCNConfig(t *testing.T) {
	rt, handled, err := BuildSubstrateLayerRuntime(SubstrateLayerRuntimeBuildRequest{
		Genome: model.Genome{
			ID: "typed-abcn-substrate-runtime",
			Substrate: &model.SubstrateConfig{
				Dimensions: []int{0, 2, 2},
				CEPName:    substrate.WeightExpressionCEPName,
				Plasticity: substrate.SubstratePlasticityABCN,
				LinkForm:   substrate.LinkFormL2LFeedforward,
				Parameters: map[string]float64{
					"abcn_a": 0.1,
					"abcn_b": 0.2,
					"abcn_c": 0.3,
					"abcn_n": 0.4,
				},
			},
		},
		InputWidth:  2,
		OutputWidth: 1,
	})
	if err != nil {
		t.Fatalf("build typed abcn substrate runtime: %v", err)
	}
	if !handled {
		t.Fatalf("expected explicit abcn config to be handled")
	}
	if _, ok := rt.(*substrate.LayerRuntime); !ok {
		t.Fatalf("expected LayerRuntime, got %T", rt)
	}
	if got, err := rt.Step(context.Background(), []float64{1, 1}); err != nil || len(got) != 1 {
		t.Fatalf("step typed abcn substrate runtime: got=%v err=%v", got, err)
	}
	weights := rt.Weights()
	if len(weights) != 2 || weights[0] == 0 || weights[1] == 0 {
		t.Fatalf("expected abcn coordinate population, got weights=%v", weights)
	}
}

func TestBuildSubstrateLayerRuntimeLeavesLegacyComponentConfigUnhandled(t *testing.T) {
	rt, handled, err := BuildSubstrateLayerRuntime(SubstrateLayerRuntimeBuildRequest{
		Genome: model.Genome{
			ID: "legacy-substrate-runtime",
			Substrate: &model.SubstrateConfig{
				CPPName: substrate.DefaultCPPName,
				CEPName: substrate.DefaultCEPName,
			},
		},
		InputWidth:  1,
		OutputWidth: 1,
	})
	if err != nil {
		t.Fatalf("build substrate layer runtime: %v", err)
	}
	if handled || rt != nil {
		t.Fatalf("expected legacy component config to fall back, handled=%v rt=%T", handled, rt)
	}
}
