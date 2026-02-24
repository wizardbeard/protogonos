package scape

import (
	"context"
	"testing"
)

func TestScapeBenchmarkModeAlias(t *testing.T) {
	agent := scriptedStepAgent{
		id: "benchmark-mode-agent",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}

	scapes := []ModeAwareScape{
		XORScape{},
		RegressionMimicScape{},
		CartPoleLiteScape{},
		Pole2BalancingScape{},
		DTMScape{},
		FlatlandScape{},
		GTSAScape{},
		FXScape{},
		EpitopesScape{},
		LLVMPhaseOrderingScape{},
	}

	for _, sc := range scapes {
		t.Run(sc.Name(), func(t *testing.T) {
			_, trace, err := sc.EvaluateMode(context.Background(), agent, "benchmark")
			if err != nil {
				t.Fatalf("evaluate benchmark mode: %v", err)
			}
			if mode, _ := trace["mode"].(string); mode != "benchmark" {
				t.Fatalf("expected benchmark mode marker, got %+v", trace)
			}
		})
	}
}
