package dataextract

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"sort"
)

func GenerateCircuitTestTables(seed int64) map[string]TableFile {
	rng := rand.New(rand.NewSource(seed))
	tables := map[string]TableFile{
		"i10o20":           buildRandomTable("i10o20", 10, 20, 500, 100, 100, rng, 0, 1, 0, 1),
		"i50o20":           buildRandomTable("i50o20", 50, 20, 500, 100, 100, rng, 0, 1, 0, 1),
		"i100o20":          buildRandomTable("i100o20", 100, 20, 500, 100, 100, rng, 0, 1, 0, 1),
		"i100o50":          buildRandomTable("i100o50", 100, 50, 500, 100, 100, rng, -0.5, 0.5, -0.5, 0.5),
		"i100o200":         buildRandomTable("i100o200", 100, 200, 500, 100, 100, rng, -0.5, 0.5, -0.5, 0.5),
		"i200o100not_test": buildRandomTable("i200o100not_test", 200, 0, 500, 0, 0, rng, -0.5, 0.5, -0.5, 0.5),
		"i200o100short":    buildRandomTable("i200o100short", 200, 100, 10, 0, 0, rng, -0.5, 0.5, 0, 1),
		"xor_bip":          buildXORBipTable(),
	}
	return tables
}

func GenerateCompetitiveTestTable(seed int64) TableFile {
	rng := rand.New(rand.NewSource(seed))
	rows := make([]TableRow, 0, 700)
	for idx := 1; idx <= 250; idx++ {
		rows = append(rows, TableRow{
			Index: idx,
			Inputs: []float64{
				rng.Float64()*5 - 5,
				(rng.Float64() - 0.5) * 2,
			},
		})
	}
	for idx := 251; idx <= 500; idx++ {
		rows = append(rows, TableRow{
			Index: idx,
			Inputs: []float64{
				rng.Float64()*5 + 5,
				(rng.Float64() - 0.5) * 2,
			},
		})
	}
	for idx := 1; idx <= 100; idx++ {
		source := rows[idx-1]
		rows = append(rows, TableRow{
			Index:  500 + idx,
			Inputs: append([]float64(nil), source.Inputs...),
		})
	}
	for idx := 1; idx <= 100; idx++ {
		source := rows[500+idx-1]
		rows = append(rows, TableRow{
			Index:  600 + idx,
			Inputs: append([]float64(nil), source.Inputs...),
		})
	}

	return TableFile{
		Info: TableInfo{
			Name:   "i2o0C",
			IVL:    2,
			OVL:    0,
			TrnEnd: 500,
			ValEnd: 600,
			TstEnd: 700,
		},
		Rows: rows,
	}
}

func WriteNamedTableFiles(dir string, tables map[string]TableFile) ([]string, error) {
	if dir == "" {
		return nil, fmt.Errorf("output directory is required")
	}
	names := make([]string, 0, len(tables))
	for name := range tables {
		names = append(names, name)
	}
	sort.Strings(names)

	paths := make([]string, 0, len(names))
	for _, name := range names {
		path := filepath.Join(dir, name+".table.json")
		if err := WriteTableFile(path, tables[name]); err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	return paths, nil
}

func buildRandomTable(
	name string,
	ivl int,
	ovl int,
	trainCount int,
	valCount int,
	testCount int,
	rng *rand.Rand,
	inputMin float64,
	inputMax float64,
	targetMin float64,
	targetMax float64,
) TableFile {
	total := trainCount + valCount + testCount
	rows := make([]TableRow, 0, total)
	for idx := 1; idx <= total; idx++ {
		inputs := randomVector(rng, ivl, inputMin, inputMax)
		var targets []float64
		if ovl > 0 {
			targets = randomVector(rng, ovl, targetMin, targetMax)
		}
		rows = append(rows, TableRow{
			Index:   idx,
			Inputs:  inputs,
			Targets: targets,
		})
	}
	return TableFile{
		Info: TableInfo{
			Name:   name,
			IVL:    ivl,
			OVL:    ovl,
			TrnEnd: trainCount,
			ValEnd: trainCount + valCount,
			TstEnd: total,
		},
		Rows: rows,
	}
}

func randomVector(rng *rand.Rand, length int, min float64, max float64) []float64 {
	if length <= 0 {
		return nil
	}
	if max < min {
		min, max = max, min
	}
	span := max - min
	out := make([]float64, length)
	for i := range out {
		if span == 0 {
			out[i] = min
			continue
		}
		out[i] = min + rng.Float64()*span
	}
	return out
}

func buildXORBipTable() TableFile {
	return TableFile{
		Info: TableInfo{
			Name:   "xor_bip",
			IVL:    2,
			OVL:    1,
			TrnEnd: 4,
			ValEnd: 0,
			TstEnd: 0,
		},
		Rows: []TableRow{
			{Index: 1, Inputs: []float64{-1, -1}, Targets: []float64{-1}},
			{Index: 2, Inputs: []float64{1, 1}, Targets: []float64{-1}},
			{Index: 3, Inputs: []float64{-1, 1}, Targets: []float64{1}},
			{Index: 4, Inputs: []float64{1, -1}, Targets: []float64{1}},
		},
	}
}
