package scape

import (
	"context"
	"math"
	"strings"
	"testing"
)

func TestWithFlatlandOverridesAppliesScannerAndLayoutSettings(t *testing.T) {
	scape := FlatlandScape{}
	spread := 0.24
	offset := -0.2
	randomize := false
	variants := 5
	forced := 3

	ctx, err := WithFlatlandOverrides(context.Background(), FlatlandOverrides{
		ScannerProfile:     "forward",
		ScannerSpread:      &spread,
		ScannerOffset:      &offset,
		RandomizeLayout:    &randomize,
		LayoutVariants:     &variants,
		ForceLayoutVariant: &forced,
	})
	if err != nil {
		t.Fatalf("with flatland overrides: %v", err)
	}

	agentA := scriptedStepAgent{
		id: "flatland-override-agent-a",
		fn: flatlandGreedyForager,
	}
	agentB := scriptedStepAgent{
		id: "flatland-override-agent-b",
		fn: flatlandGreedyForager,
	}

	_, traceA, err := scape.EvaluateMode(ctx, agentA, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark with overrides agent A: %v", err)
	}
	_, traceB, err := scape.EvaluateMode(ctx, agentB, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark with overrides agent B: %v", err)
	}

	wantVariant := positiveMod(forced, variants)
	wantShift := wrapFlatlandPosition(wantVariant * flatlandRespawnStride)
	wantHeading := flatlandLayoutHeading(wantVariant)

	if profile, _ := traceA["scanner_profile"].(string); profile != flatlandScannerProfileForward {
		t.Fatalf("expected override scanner profile %q, got trace=%+v", flatlandScannerProfileForward, traceA)
	}
	if gotSpread, _ := traceA["scanner_spread"].(float64); gotSpread != spread {
		t.Fatalf("expected override scanner spread=%f, got %f", spread, gotSpread)
	}
	if gotOffset, _ := traceA["scanner_offset"].(float64); gotOffset != offset {
		t.Fatalf("expected override scanner offset=%f, got %f", offset, gotOffset)
	}
	if gotVariant, _ := traceA["layout_variant"].(int); gotVariant != wantVariant {
		t.Fatalf("expected forced layout variant=%d, got trace=%+v", wantVariant, traceA)
	}
	if gotShift, _ := traceA["layout_shift"].(int); gotShift != wantShift {
		t.Fatalf("expected forced layout shift=%d, got trace=%+v", wantShift, traceA)
	}
	if gotHeading, _ := traceA["initial_heading"].(int); gotHeading != wantHeading {
		t.Fatalf("expected forced initial heading=%d, got trace=%+v", wantHeading, traceA)
	}
	if forcedFlag, _ := traceA["layout_forced"].(bool); !forcedFlag {
		t.Fatalf("expected layout_forced=true, got trace=%+v", traceA)
	}
	if gotVariants, _ := traceA["layout_variants"].(int); gotVariants != variants {
		t.Fatalf("expected layout_variants=%d, got trace=%+v", variants, traceA)
	}

	if gotVariant, _ := traceB["layout_variant"].(int); gotVariant != wantVariant {
		t.Fatalf("expected forced variant to ignore agent id, got traceB=%+v", traceB)
	}
	if gotShift, _ := traceB["layout_shift"].(int); gotShift != wantShift {
		t.Fatalf("expected forced shift to ignore agent id, got traceB=%+v", traceB)
	}
}

func TestWithFlatlandOverridesAreScopedToContext(t *testing.T) {
	scape := FlatlandScape{}
	variants := 7
	forced := 6
	ctx, err := WithFlatlandOverrides(context.Background(), FlatlandOverrides{
		ScannerProfile:     "balanced5",
		LayoutVariants:     &variants,
		ForceLayoutVariant: &forced,
	})
	if err != nil {
		t.Fatalf("with flatland overrides: %v", err)
	}

	forager := scriptedStepAgent{
		id: "flatland-override-scope-agent",
		fn: flatlandGreedyForager,
	}

	_, scopedTrace, err := scape.EvaluateMode(ctx, forager, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark with scoped overrides: %v", err)
	}
	_, defaultTrace, err := scape.EvaluateMode(context.Background(), forager, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark with default config: %v", err)
	}

	if profile, _ := scopedTrace["scanner_profile"].(string); profile != flatlandScannerProfileBalanced {
		t.Fatalf("expected scoped scanner profile %q, got trace=%+v", flatlandScannerProfileBalanced, scopedTrace)
	}
	if forcedFlag, _ := scopedTrace["layout_forced"].(bool); !forcedFlag {
		t.Fatalf("expected scoped layout_forced=true, got trace=%+v", scopedTrace)
	}
	if profile, _ := defaultTrace["scanner_profile"].(string); profile != flatlandScannerProfileCore {
		t.Fatalf("expected default benchmark scanner profile %q, got trace=%+v", flatlandScannerProfileCore, defaultTrace)
	}
	if forcedFlag, _ := defaultTrace["layout_forced"].(bool); forcedFlag {
		t.Fatalf("expected default layout_forced=false, got trace=%+v", defaultTrace)
	}
}

func TestWithFlatlandOverridesRejectsInvalidValues(t *testing.T) {
	tests := []struct {
		name    string
		input   FlatlandOverrides
		wantErr string
	}{
		{
			name:    "invalid scanner profile",
			input:   FlatlandOverrides{ScannerProfile: "bad-profile"},
			wantErr: "unsupported flatland scanner profile",
		},
		{
			name: "invalid scanner spread",
			input: FlatlandOverrides{
				ScannerSpread: flatlandFloatPtr(0.01),
			},
			wantErr: "scanner spread",
		},
		{
			name: "invalid scanner offset",
			input: FlatlandOverrides{
				ScannerOffset: flatlandFloatPtr(1.4),
			},
			wantErr: "scanner offset",
		},
		{
			name: "invalid layout variants",
			input: FlatlandOverrides{
				LayoutVariants: flatlandIntPtr(0),
			},
			wantErr: "layout variants",
		},
		{
			name: "invalid benchmark trials",
			input: FlatlandOverrides{
				BenchmarkTrials: flatlandIntPtr(0),
			},
			wantErr: "benchmark trials",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := WithFlatlandOverrides(context.Background(), tt.input)
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestWithFlatlandOverridesAggregatesBenchmarkTrials(t *testing.T) {
	scape := FlatlandScape{}
	trials := 4
	ctx, err := WithFlatlandOverrides(context.Background(), FlatlandOverrides{
		BenchmarkTrials: &trials,
	})
	if err != nil {
		t.Fatalf("with flatland overrides: %v", err)
	}

	forager := scriptedStepAgent{
		id: "flatland-benchmark-trials-agent",
		fn: flatlandGreedyForager,
	}

	fitness, trace, err := scape.EvaluateMode(ctx, forager, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark with trial aggregation: %v", err)
	}
	if aggregated, _ := trace["benchmark_aggregated"].(bool); !aggregated {
		t.Fatalf("expected benchmark_aggregated=true, trace=%+v", trace)
	}
	if gotTrials, _ := trace["benchmark_trials"].(int); gotTrials != trials {
		t.Fatalf("expected benchmark_trials=%d, trace=%+v", trials, trace)
	}

	fitnesses, ok := trace["benchmark_trial_fitnesses"].([]float64)
	if !ok || len(fitnesses) != trials {
		t.Fatalf("expected benchmark_trial_fitnesses len=%d, trace=%+v", trials, trace)
	}
	layoutVariants, ok := trace["benchmark_layout_variants"].([]int)
	if !ok || len(layoutVariants) != trials {
		t.Fatalf("expected benchmark_layout_variants len=%d, trace=%+v", trials, trace)
	}
	layoutShifts, ok := trace["benchmark_layout_shifts"].([]int)
	if !ok || len(layoutShifts) != trials {
		t.Fatalf("expected benchmark_layout_shifts len=%d, trace=%+v", trials, trace)
	}

	mean, ok := trace["benchmark_fitness_mean"].(float64)
	if !ok {
		t.Fatalf("expected benchmark_fitness_mean, trace=%+v", trace)
	}
	stddev, ok := trace["benchmark_fitness_stddev"].(float64)
	if !ok {
		t.Fatalf("expected benchmark_fitness_stddev, trace=%+v", trace)
	}
	minFitness, ok := trace["benchmark_fitness_min"].(float64)
	if !ok {
		t.Fatalf("expected benchmark_fitness_min, trace=%+v", trace)
	}
	maxFitness, ok := trace["benchmark_fitness_max"].(float64)
	if !ok {
		t.Fatalf("expected benchmark_fitness_max, trace=%+v", trace)
	}
	if math.Abs(float64(fitness)-mean) > 1e-9 {
		t.Fatalf("expected returned fitness=%f to match benchmark mean=%f", fitness, mean)
	}
	if minFitness > maxFitness {
		t.Fatalf("expected benchmark min <= max, got min=%f max=%f", minFitness, maxFitness)
	}
	if stddev < 0 {
		t.Fatalf("expected non-negative stddev, got=%f", stddev)
	}
}

func flatlandFloatPtr(v float64) *float64 {
	return &v
}

func flatlandIntPtr(v int) *int {
	return &v
}
