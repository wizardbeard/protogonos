package scape

import (
	"math"
	"testing"
)

func TestDistanceMatchesReferenceEuclideanHelper(t *testing.T) {
	got, err := Distance([]float64{1, 2, 3}, []float64{4, 6, 3})
	if err != nil {
		t.Fatalf("distance: %v", err)
	}
	if math.Abs(got-5) > 1e-12 {
		t.Fatalf("expected distance 5, got %f", got)
	}
}

func TestDistanceRejectsWidthMismatch(t *testing.T) {
	if _, err := Distance([]float64{1, 2}, []float64{1}); err == nil {
		t.Fatal("expected width mismatch error")
	}
}
