package stats

import (
	"protogonos/internal/nn"
)

type BenchmarkerPlotPoint struct {
	Index int     `json:"index"`
	Value float64 `json:"value"`
}

func BuildBenchmarkerAveragePlot(lists [][]float64, startIndex, step int) []BenchmarkerPlotPoint {
	if step <= 0 {
		step = 500
	}
	if startIndex < 0 {
		startIndex = 500
	}
	points := make([]BenchmarkerPlotPoint, 0, 128)
	index := startIndex
	current := clonePlotLists(lists)
	for {
		values := make([]float64, 0, len(current))
		next := make([][]float64, 0, len(current))
		for _, list := range current {
			if len(list) == 0 {
				continue
			}
			values = append(values, list[0])
			if len(list) > 1 {
				tail := append([]float64(nil), list[1:]...)
				next = append(next, tail)
			}
		}
		if len(values) == 0 {
			break
		}
		avg, _ := nn.Avg(values)
		points = append(points, BenchmarkerPlotPoint{Index: index, Value: avg})
		index += step
		current = next
	}
	return points
}

func BuildBenchmarkerMaxPlot(lists [][]float64, startIndex, step int) []BenchmarkerPlotPoint {
	if step <= 0 {
		step = 500
	}
	if startIndex < 0 {
		startIndex = 0
	}
	points := make([]BenchmarkerPlotPoint, 0, len(lists))
	index := startIndex
	for _, list := range lists {
		if len(list) == 0 {
			continue
		}
		points = append(points, BenchmarkerPlotPoint{
			Index: index,
			Value: maxFloat(list),
		})
		index += step
	}
	return points
}

func clonePlotLists(lists [][]float64) [][]float64 {
	cloned := make([][]float64, 0, len(lists))
	for _, list := range lists {
		cloned = append(cloned, append([]float64(nil), list...))
	}
	return cloned
}
