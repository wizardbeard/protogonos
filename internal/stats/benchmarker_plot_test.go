package stats

import "testing"

func TestBuildBenchmarkerAveragePlot(t *testing.T) {
	lists := [][]float64{
		{1, 2, 3},
		{2, 4},
		{3},
	}
	points := BuildBenchmarkerAveragePlot(lists, 500, 500)
	if len(points) != 3 {
		t.Fatalf("expected 3 points, got %d (%+v)", len(points), points)
	}
	if points[0].Index != 500 || points[1].Index != 1000 || points[2].Index != 1500 {
		t.Fatalf("unexpected indices: %+v", points)
	}
	if points[0].Value != 2 {
		t.Fatalf("unexpected first average value: %+v", points)
	}
}

func TestBuildBenchmarkerMaxPlot(t *testing.T) {
	lists := [][]float64{
		{1, 2, 3},
		{4, 1},
		{},
		{0.5, 0.7},
	}
	points := BuildBenchmarkerMaxPlot(lists, 0, 500)
	if len(points) != 3 {
		t.Fatalf("expected 3 points, got %d (%+v)", len(points), points)
	}
	if points[0].Index != 0 || points[1].Index != 500 || points[2].Index != 1000 {
		t.Fatalf("unexpected indices: %+v", points)
	}
	if points[0].Value != 3 || points[1].Value != 4 || points[2].Value != 0.7 {
		t.Fatalf("unexpected max values: %+v", points)
	}
}
