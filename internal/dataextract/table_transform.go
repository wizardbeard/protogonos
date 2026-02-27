package dataextract

import (
	"fmt"
	"math"
)

type ColumnStats struct {
	Min float64 `json:"min"`
	Avg float64 `json:"avg"`
	Max float64 `json:"max"`
}

func InputColumnStats(table TableFile) ([]ColumnStats, error) {
	if len(table.Rows) == 0 {
		return nil, nil
	}
	width := len(table.Rows[0].Inputs)
	if width == 0 {
		return nil, fmt.Errorf("table has no numeric input columns")
	}

	stats := make([]ColumnStats, width)
	for i := 0; i < width; i++ {
		value := table.Rows[0].Inputs[i]
		stats[i] = ColumnStats{Min: value, Avg: value, Max: value}
	}
	for rowIdx := 1; rowIdx < len(table.Rows); rowIdx++ {
		row := table.Rows[rowIdx]
		if len(row.Inputs) != width {
			return nil, fmt.Errorf(
				"inconsistent input width at row %d: got=%d want=%d",
				row.Index,
				len(row.Inputs),
				width,
			)
		}
		for i, value := range row.Inputs {
			if value < stats[i].Min {
				stats[i].Min = value
			}
			if value > stats[i].Max {
				stats[i].Max = value
			}
			stats[i].Avg += value
		}
	}
	count := float64(len(table.Rows))
	for i := range stats {
		stats[i].Avg /= count
	}
	return stats, nil
}

func ScaleInputsByColumnMax(table *TableFile, stats []ColumnStats) error {
	if table == nil {
		return fmt.Errorf("table is required")
	}
	for rowIdx := range table.Rows {
		row := &table.Rows[rowIdx]
		if len(row.Inputs) != len(stats) {
			return fmt.Errorf(
				"inconsistent input width at row %d: got=%d want=%d",
				row.Index,
				len(row.Inputs),
				len(stats),
			)
		}
		for i, value := range row.Inputs {
			maxVal := stats[i].Max
			if maxVal == 0 {
				row.Inputs[i] = 0
				continue
			}
			row.Inputs[i] = value / maxVal
		}
	}
	return nil
}

func ScaleInputsAsinh(table *TableFile) {
	if table == nil {
		return
	}
	for rowIdx := range table.Rows {
		row := &table.Rows[rowIdx]
		for i, value := range row.Inputs {
			row.Inputs[i] = math.Log(value + math.Sqrt(value*value+1))
		}
	}
}

func BinarizeInputs(table *TableFile) {
	if table == nil {
		return
	}
	for rowIdx := range table.Rows {
		row := &table.Rows[rowIdx]
		for i, value := range row.Inputs {
			if value == 0 {
				row.Inputs[i] = 0
			} else {
				row.Inputs[i] = 1
			}
		}
	}
}

func CleanZeroInputRows(table *TableFile) {
	if table == nil {
		return
	}
	filtered := make([]TableRow, 0, len(table.Rows))
	for _, row := range table.Rows {
		total := 0.0
		for _, value := range row.Inputs {
			total += value
		}
		if total == 0 {
			continue
		}
		row.Index = len(filtered) + 1
		filtered = append(filtered, row)
	}
	table.Rows = filtered
	table.Info.TrnEnd = len(filtered)
	if table.Info.ValEnd > 0 {
		table.Info.ValEnd = minInt(table.Info.ValEnd, len(filtered))
	}
	if table.Info.TstEnd > 0 {
		table.Info.TstEnd = minInt(table.Info.TstEnd, len(filtered))
	}
}

func CountZeroInputs(table TableFile) (zeroes, nonZeroes int, ratio float64) {
	for _, row := range table.Rows {
		for _, value := range row.Inputs {
			if value == 0 {
				zeroes++
			} else {
				nonZeroes++
			}
		}
	}
	if nonZeroes > 0 {
		ratio = float64(zeroes) / float64(nonZeroes)
	}
	return zeroes, nonZeroes, ratio
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
