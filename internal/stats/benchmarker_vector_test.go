package stats

import "testing"

func TestBenchmarkerVectorComparisons(t *testing.T) {
	if !BenchmarkerVectorGT([]float64{2, 1}, []float64{1, 1}) {
		t.Fatal("expected vector gt to be true")
	}
	if BenchmarkerVectorGT([]float64{1, 1}, []float64{1, 1}) {
		t.Fatal("expected vector gt to be false for equal vectors")
	}
	if BenchmarkerVectorGT([]float64{1, 0}, []float64{1, 1}) {
		t.Fatal("expected vector gt to be false when one dimension is lower")
	}

	if !BenchmarkerVectorLT([]float64{1, 0}, []float64{1, 1}) {
		t.Fatal("expected vector lt to be true")
	}
	if BenchmarkerVectorLT([]float64{1, 1}, []float64{1, 1}) {
		t.Fatal("expected vector lt to be false for equal vectors")
	}
	if BenchmarkerVectorLT([]float64{2, 1}, []float64{1, 1}) {
		t.Fatal("expected vector lt to be false when one dimension is higher")
	}

	if !BenchmarkerVectorEQ([]float64{1, 2}, []float64{1, 2}) {
		t.Fatal("expected vector eq to be true")
	}
	if BenchmarkerVectorEQ([]float64{1, 2}, []float64{2, 1}) {
		t.Fatal("expected vector eq to be false")
	}
	if BenchmarkerVectorEQ([]float64{1}, nil) {
		t.Fatal("expected vector eq to be false with undefined second vector")
	}
}
