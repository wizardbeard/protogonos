package dataextract

import (
	"math"
	"testing"
)

func TestInputColumnStatsAndScaleByMax(t *testing.T) {
	table := TableFile{
		Rows: []TableRow{
			{Index: 1, Inputs: []float64{2, 4}},
			{Index: 2, Inputs: []float64{4, 8}},
		},
	}
	stats, err := InputColumnStats(table)
	if err != nil {
		t.Fatalf("input stats: %v", err)
	}
	if len(stats) != 2 || stats[0].Max != 4 || stats[1].Max != 8 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if err := ScaleInputsByColumnMax(&table, stats); err != nil {
		t.Fatalf("scale by max: %v", err)
	}
	if got := table.Rows[0].Inputs[0]; math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("unexpected scaled value: %f", got)
	}
	if got := table.Rows[1].Inputs[1]; math.Abs(got-1.0) > 1e-9 {
		t.Fatalf("unexpected scaled value: %f", got)
	}
}

func TestScaleAsinhBinarizeAndZeroCounts(t *testing.T) {
	table := TableFile{
		Rows: []TableRow{
			{Index: 1, Inputs: []float64{0, 2}},
			{Index: 2, Inputs: []float64{-2, 0}},
		},
	}
	ScaleInputsAsinh(&table)
	if table.Rows[0].Inputs[1] <= 0 {
		t.Fatalf("expected positive asinh-scaled value, got %+v", table.Rows[0].Inputs)
	}
	BinarizeInputs(&table)
	if got := table.Rows[0].Inputs[0]; got != 0 {
		t.Fatalf("expected zero to stay zero, got %f", got)
	}
	if got := table.Rows[0].Inputs[1]; got != 1 {
		t.Fatalf("expected non-zero to binarize to one, got %f", got)
	}
	zeroes, nonZeroes, ratio := CountZeroInputs(table)
	if zeroes != 2 || nonZeroes != 2 || math.Abs(ratio-1.0) > 1e-9 {
		t.Fatalf("unexpected zero counts: zeroes=%d non_zeroes=%d ratio=%f", zeroes, nonZeroes, ratio)
	}
}

func TestCleanZeroInputRowsReindexes(t *testing.T) {
	table := TableFile{
		Info: TableInfo{
			Name:   "clean_test",
			TrnEnd: 3,
			ValEnd: 3,
			TstEnd: 3,
		},
		Rows: []TableRow{
			{Index: 1, Inputs: []float64{0, 0}},
			{Index: 2, Inputs: []float64{1, 0}},
			{Index: 3, Inputs: []float64{0, 0}},
		},
	}
	CleanZeroInputRows(&table)
	if len(table.Rows) != 1 {
		t.Fatalf("expected one row after cleaning, got %d", len(table.Rows))
	}
	if table.Rows[0].Index != 1 {
		t.Fatalf("expected reindexed row=1, got %+v", table.Rows[0])
	}
	if table.Info.TrnEnd != 1 || table.Info.ValEnd != 1 || table.Info.TstEnd != 1 {
		t.Fatalf("unexpected split bounds after clean: %+v", table.Info)
	}
}

func TestResolutionateInputsWithZeroRunDropAndAsinh(t *testing.T) {
	table := TableFile{
		Info: TableInfo{
			Name:   "resolution_test",
			TrnEnd: 5,
			ValEnd: 5,
			TstEnd: 5,
		},
		Rows: []TableRow{
			{Index: 1, Inputs: []float64{0, 0}},
			{Index: 2, Inputs: []float64{0, 0}},
			{Index: 3, Inputs: []float64{1, 3}},
			{Index: 4, Inputs: []float64{3, 5}},
			{Index: 5, Inputs: []float64{10, 10}},
		},
	}

	if err := ResolutionateInputs(&table, 2, 1, true); err != nil {
		t.Fatalf("resolutionate inputs: %v", err)
	}
	if len(table.Rows) != 1 {
		t.Fatalf("expected one resolved row, got %d", len(table.Rows))
	}
	if table.Rows[0].Index != 1 {
		t.Fatalf("expected reindexed resolved row, got %d", table.Rows[0].Index)
	}
	if got := table.Rows[0].Inputs[0]; math.Abs(got-math.Asinh(2)) > 1e-9 {
		t.Fatalf("unexpected resolved col0: %f", got)
	}
	if got := table.Rows[0].Inputs[1]; math.Abs(got-math.Asinh(4)) > 1e-9 {
		t.Fatalf("unexpected resolved col1: %f", got)
	}
	if table.Info.TrnEnd != 1 || table.Info.ValEnd != 1 || table.Info.TstEnd != 1 {
		t.Fatalf("unexpected split bounds after resolutionate: %+v", table.Info)
	}
}
