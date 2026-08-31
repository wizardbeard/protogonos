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
				Plasticity: substrate.SubstratePlasticityNone,
				LinkForm:   substrate.LinkFormL2LFeedforward,
			},
		},
		InputWidth:  1,
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
	if got, err := rt.Step(context.Background(), []float64{1}); err != nil || len(got) != 1 {
		t.Fatalf("step typed substrate runtime: got=%v err=%v", got, err)
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
