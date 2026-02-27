package scape

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWithDataSourcesScopesGTSATableOverridesToContext(t *testing.T) {
	ResetGTSATableSource()
	t.Cleanup(ResetGTSATableSource)

	path := filepath.Join(t.TempDir(), "gtsa_ctx.csv")
	var b strings.Builder
	b.WriteString("t,value\n")
	for i := 0; i < 96; i++ {
		fmt.Fprintf(&b, "%d,%0.8f\n", i, gtsaSeries(i)+0.05)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write gtsa csv: %v", err)
	}

	overrideCtx, err := WithDataSources(context.Background(), DataSources{
		GTSA: GTSADataSource{
			CSVPath: path,
			Bounds: GTSATableBounds{
				TrainEnd:      24,
				ValidationEnd: 48,
				TestEnd:       96,
			},
		},
	})
	if err != nil {
		t.Fatalf("with data sources: %v", err)
	}

	scape := GTSAScape{}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	_, scopedTrace, err := scape.EvaluateMode(overrideCtx, copyInput, "test")
	if err != nil {
		t.Fatalf("evaluate scoped gtsa: %v", err)
	}
	_, defaultTrace, err := scape.EvaluateMode(context.Background(), copyInput, "test")
	if err != nil {
		t.Fatalf("evaluate default gtsa: %v", err)
	}

	if table, _ := scopedTrace["table_name"].(string); !strings.Contains(table, "gtsa_ctx.csv") {
		t.Fatalf("expected scoped gtsa table name, got %+v", scopedTrace)
	}
	if table, _ := defaultTrace["table_name"].(string); table != "gtsa.synthetic.v2" {
		t.Fatalf("expected default gtsa table name, got %+v", defaultTrace)
	}
}

func TestWithDataSourcesScopesFXSeriesOverridesToContext(t *testing.T) {
	ResetFXSeriesSource()
	t.Cleanup(ResetFXSeriesSource)

	path := filepath.Join(t.TempDir(), "fx_ctx.csv")
	var b strings.Builder
	b.WriteString("t,close\n")
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&b, "%d,%0.6f\n", i, 1.03+0.00025*float64(i))
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write fx csv: %v", err)
	}

	overrideCtx, err := WithDataSources(context.Background(), DataSources{
		FX: FXDataSource{CSVPath: path},
	})
	if err != nil {
		t.Fatalf("with data sources: %v", err)
	}

	scape := FXScape{}
	follow := scriptedStepAgent{
		id: "follow",
		fn: fxFollowSignalAction,
	}

	_, scopedTrace, err := scape.EvaluateMode(overrideCtx, follow, "test")
	if err != nil {
		t.Fatalf("evaluate scoped fx: %v", err)
	}
	_, defaultTrace, err := scape.EvaluateMode(context.Background(), follow, "test")
	if err != nil {
		t.Fatalf("evaluate default fx: %v", err)
	}

	if series, _ := scopedTrace["series_name"].(string); !strings.Contains(series, "fx_ctx.csv") {
		t.Fatalf("expected scoped fx series name, got %+v", scopedTrace)
	}
	if series, _ := defaultTrace["series_name"].(string); series != "fx.synthetic.v2" {
		t.Fatalf("expected default fx series name, got %+v", defaultTrace)
	}
}

func TestWithDataSourcesScopesEpitopesTableOverridesToContext(t *testing.T) {
	ResetEpitopesTableSource()
	t.Cleanup(ResetEpitopesTableSource)

	path := filepath.Join(t.TempDir(), "epitopes_ctx.csv")
	var b strings.Builder
	b.WriteString("signal,memory,class\n")
	for i := 0; i < 220; i++ {
		signal := 0.7 * math.Sin(float64(i)*0.11)
		memory := 0.5 * math.Sin(float64(i-1)*0.11)
		classification := 0
		if signal+0.7*memory >= 0 {
			classification = 1
		}
		fmt.Fprintf(&b, "%0.6f,%0.6f,%d\n", signal, memory, classification)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write epitopes csv: %v", err)
	}

	overrideCtx, err := WithDataSources(context.Background(), DataSources{
		Epitopes: EpitopesDataSource{
			CSVPath: path,
			Bounds: EpitopesTableBounds{
				GTStart:         1,
				GTEnd:           64,
				ValidationStart: 65,
				ValidationEnd:   96,
				TestStart:       97,
				TestEnd:         128,
				BenchmarkStart:  129,
				BenchmarkEnd:    220,
			},
		},
	})
	if err != nil {
		t.Fatalf("with data sources: %v", err)
	}

	scape := EpitopesScape{}
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	_, scopedTrace, err := scape.EvaluateMode(overrideCtx, memoryAware, "benchmark")
	if err != nil {
		t.Fatalf("evaluate scoped epitopes: %v", err)
	}
	_, defaultTrace, err := scape.EvaluateMode(context.Background(), memoryAware, "benchmark")
	if err != nil {
		t.Fatalf("evaluate default epitopes: %v", err)
	}

	if table, _ := scopedTrace["table_name"].(string); !strings.Contains(table, "epitopes_ctx.csv") {
		t.Fatalf("expected scoped epitopes table name, got %+v", scopedTrace)
	}
	if table, _ := defaultTrace["table_name"].(string); table != "abc_pred16" {
		t.Fatalf("expected default epitopes table name, got %+v", defaultTrace)
	}
}

func TestWithDataSourcesScopesEpitopesBuiltInTableSelectionToContext(t *testing.T) {
	ResetEpitopesTableSource()
	t.Cleanup(ResetEpitopesTableSource)

	overrideCtx, err := WithDataSources(context.Background(), DataSources{
		Epitopes: EpitopesDataSource{
			TableName: "abc_pred20",
		},
	})
	if err != nil {
		t.Fatalf("with data sources: %v", err)
	}

	scape := EpitopesScape{}
	copyInput := scriptedStepAgent{
		id: "copy-input",
		fn: func(in []float64) []float64 {
			if len(in) == 0 {
				return []float64{0}
			}
			return []float64{in[0]}
		},
	}

	_, scopedTrace, err := scape.EvaluateMode(overrideCtx, copyInput, "benchmark")
	if err != nil {
		t.Fatalf("evaluate scoped epitopes: %v", err)
	}
	_, defaultTrace, err := scape.EvaluateMode(context.Background(), copyInput, "benchmark")
	if err != nil {
		t.Fatalf("evaluate default epitopes: %v", err)
	}
	if table, _ := scopedTrace["table_name"].(string); table != "abc_pred20" {
		t.Fatalf("expected scoped epitopes table abc_pred20, got %+v", scopedTrace)
	}
	if seqLen, _ := scopedTrace["sequence_length"].(int); seqLen != 20 {
		t.Fatalf("expected scoped sequence_length=20, got %+v", scopedTrace)
	}
	if table, _ := defaultTrace["table_name"].(string); table != "abc_pred16" {
		t.Fatalf("expected default epitopes table abc_pred16, got %+v", defaultTrace)
	}
}

func TestWithDataSourcesRejectsUnknownEpitopesTableSelection(t *testing.T) {
	_, err := WithDataSources(context.Background(), DataSources{
		Epitopes: EpitopesDataSource{
			TableName: "abc_pred999",
		},
	})
	if err == nil {
		t.Fatal("expected unknown epitopes table selection error")
	}
}
