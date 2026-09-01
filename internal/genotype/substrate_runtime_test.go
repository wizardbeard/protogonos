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

func TestBuildSubstrateLayerRuntimeUsesIterativeConfig(t *testing.T) {
	rt, handled, err := BuildSubstrateLayerRuntime(SubstrateLayerRuntimeBuildRequest{
		Genome: model.Genome{
			ID: "typed-iterative-substrate-runtime",
			Substrate: &model.SubstrateConfig{
				Dimensions: []int{0, 2, 2},
				CEPName:    substrate.WeightExpressionCEPName,
				Plasticity: substrate.SubstratePlasticityIterative,
				LinkForm:   substrate.LinkFormL2LFeedforward,
			},
		},
		InputWidth:  2,
		OutputWidth: 1,
	})
	if err != nil {
		t.Fatalf("build typed iterative substrate runtime: %v", err)
	}
	if !handled {
		t.Fatalf("expected explicit iterative config to be handled")
	}
	if _, ok := rt.(*substrate.LayerRuntime); !ok {
		t.Fatalf("expected LayerRuntime, got %T", rt)
	}

	if got, err := rt.Step(context.Background(), []float64{1, 1}); err != nil || len(got) != 1 {
		t.Fatalf("step typed iterative substrate runtime: got=%v err=%v", got, err)
	}
	firstWeights := rt.Weights()
	if len(firstWeights) != 2 || firstWeights[0] == 0 || firstWeights[1] == 0 {
		t.Fatalf("expected first iterative population, got weights=%v", firstWeights)
	}
	if got, err := rt.Step(context.Background(), []float64{1, 1}); err != nil || len(got) != 1 {
		t.Fatalf("step typed iterative substrate runtime again: got=%v err=%v", got, err)
	}
	secondWeights := rt.Weights()
	if len(secondWeights) != len(firstWeights) || secondWeights[0] == firstWeights[0] || secondWeights[1] == firstWeights[1] {
		t.Fatalf("expected repeated iterative population to update weights, first=%v second=%v", firstWeights, secondWeights)
	}
}

func TestLayerRuntimeSpecForSnapshotResolvesGenomeComponents(t *testing.T) {
	genome := model.Genome{
		ID: "typed-iterative-snapshot-runtime",
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
			CEPName:    substrate.WeightExpressionCEPName,
			Plasticity: substrate.SubstratePlasticityIterative,
			LinkForm:   substrate.LinkFormL2LFeedforward,
		},
	}
	spec, err := LayerRuntimeSpecForSnapshot(genome, substrate.LayerRuntimeSnapshot{
		Plasticity: substrate.SubstratePlasticityIterative,
	})
	if err != nil {
		t.Fatalf("snapshot runtime spec: %v", err)
	}
	if spec.IterativeCPP == nil || len(spec.CEPs) != 1 {
		t.Fatalf("expected iterative CPP and CEP components, got %+v", spec)
	}
}

func TestLayerRuntimeSpecForSnapshotResolvesABCNGenomeComponents(t *testing.T) {
	genome := model.Genome{
		ID: "typed-abcn-snapshot-runtime",
		Substrate: &model.SubstrateConfig{
			CPPName:    substrate.DefaultCPPName,
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
	}
	spec, err := LayerRuntimeSpecForSnapshot(genome, substrate.LayerRuntimeSnapshot{
		Plasticity: substrate.SubstratePlasticityABCN,
	})
	if err != nil {
		t.Fatalf("snapshot runtime spec: %v", err)
	}
	if spec.IterativeCPP == nil || len(spec.CEPs) != 1 {
		t.Fatalf("expected abcn CPP and CEP components, got %+v", spec)
	}
	if spec.Parameters["abcn_a"] != 0.1 || spec.Parameters["abcn_n"] != 0.4 {
		t.Fatalf("expected abcn parameters to be forwarded, got %+v", spec.Parameters)
	}
}

func TestBuildSubstrateLayerRuntimeSupportsActiveLinkForms(t *testing.T) {
	tests := []struct {
		name       string
		linkForm   string
		dimensions []int
	}{
		{name: "fully interconnected", linkForm: substrate.LinkFormFullyInterconnected, dimensions: []int{2, 2, 3}},
		{name: "jordan recurrent", linkForm: substrate.LinkFormJordanRecurrent, dimensions: []int{0, 2, 2}},
		{name: "neuron self recurrent", linkForm: substrate.LinkFormNeuronSelfRecurrent, dimensions: []int{0, 2, 2}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rt, handled, err := BuildSubstrateLayerRuntime(SubstrateLayerRuntimeBuildRequest{
				Genome: model.Genome{
					ID: "typed-link-form-runtime",
					Substrate: &model.SubstrateConfig{
						Dimensions: tt.dimensions,
						CEPName:    substrate.WeightExpressionCEPName,
						Plasticity: substrate.SubstratePlasticityNone,
						LinkForm:   tt.linkForm,
					},
				},
				InputWidth:  2,
				OutputWidth: 1,
			})
			if err != nil {
				t.Fatalf("build typed substrate runtime: %v", err)
			}
			if !handled {
				t.Fatalf("expected typed link-form config to be handled")
			}
			if _, ok := rt.(*substrate.LayerRuntime); !ok {
				t.Fatalf("expected LayerRuntime, got %T", rt)
			}
			if got, err := rt.Step(context.Background(), []float64{1, 1}); err != nil || len(got) != 1 {
				t.Fatalf("step typed link-form runtime: got=%v err=%v", got, err)
			}
			if len(rt.Weights()) == 0 {
				t.Fatalf("expected populated weight surface")
			}
		})
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
