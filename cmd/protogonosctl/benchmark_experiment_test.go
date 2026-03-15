package main

import (
	"reflect"
	"testing"

	"protogonos/internal/stats"
)

func TestBenchmarkExperimentMorphologiesFromItemsSupportsTypedSummaries(t *testing.T) {
	items := []any{
		stats.BenchmarkSummary{RunID: "a", Scape: "fx", Morphology: "fx[market]"},
		&stats.BenchmarkSummary{RunID: "b", Scape: "fx", Morphology: "fx[market]"},
		stats.RunIndexEntry{RunID: "c", Scape: "gtsa", Morphology: "gtsa[core]"},
		map[string]any{"run_id": "d", "scape": "flatland", "morphology": "flatland[core3]"},
	}

	got := benchmarkExperimentMorphologiesFromItems(items)
	want := []string{"flatland[core3]", "fx[market]", "gtsa[core]"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected morphologies: got=%v want=%v", got, want)
	}
}

func TestBenchmarkExperimentMorphologiesFromItemsFallsBackToScape(t *testing.T) {
	items := []any{
		stats.BenchmarkSummary{RunID: "a", Scape: "xor"},
		&stats.RunIndexEntry{RunID: "b", Scape: "fx"},
		map[string]any{"run_id": "c", "scape": "gtsa"},
	}

	got := benchmarkExperimentMorphologiesFromItems(items)
	want := []string{"fx", "gtsa", "xor"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected fallback morphologies: got=%v want=%v", got, want)
	}
}
