package main

import (
	"math"
	"testing"
)

func TestBestSeriesStats(t *testing.T) {
	mean, std, max, min := bestSeriesStats([]float64{1, 2, 3, 4})
	if mean != 2.5 {
		t.Fatalf("unexpected mean: got=%f want=2.5", mean)
	}
	if math.Abs(std-math.Sqrt(1.25)) > 1e-12 {
		t.Fatalf("unexpected std: got=%f want=%f", std, math.Sqrt(1.25))
	}
	if max != 4 || min != 1 {
		t.Fatalf("unexpected extrema: max=%f min=%f", max, min)
	}
}

func TestBestSeriesStatsEmpty(t *testing.T) {
	mean, std, max, min := bestSeriesStats(nil)
	if mean != 0 || std != 0 || max != 0 || min != 0 {
		t.Fatalf("expected zero stats for empty input, got mean=%f std=%f max=%f min=%f", mean, std, max, min)
	}
}
