package dataextract

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type TableInfo struct {
	Name   string `json:"name"`
	IVL    int    `json:"ivl,omitempty"`
	OVL    int    `json:"ovl,omitempty"`
	TrnEnd int    `json:"trn_end,omitempty"`
	ValEnd int    `json:"val_end,omitempty"`
	TstEnd int    `json:"tst_end,omitempty"`
}

type TableRow struct {
	Index   int       `json:"index"`
	Inputs  []float64 `json:"inputs,omitempty"`
	Targets []float64 `json:"targets,omitempty"`
	Fields  []string  `json:"fields,omitempty"`
}

type TableFile struct {
	Info TableInfo  `json:"info"`
	Rows []TableRow `json:"rows"`
}

type BuildTableOptions struct {
	Scape string
	Name  string
}

func BuildTableFromExtractedCSV(in io.Reader, opts BuildTableOptions) (TableFile, error) {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1

	header, err := reader.Read()
	if err == io.EOF {
		return TableFile{
			Info: TableInfo{Name: strings.TrimSpace(opts.Name)},
			Rows: nil,
		}, nil
	}
	if err != nil {
		return TableFile{}, fmt.Errorf("read table csv header: %w", err)
	}

	scape := strings.TrimSpace(strings.ToLower(opts.Scape))
	name := strings.TrimSpace(opts.Name)
	if name == "" {
		name = "table"
	}

	rows := make([]TableRow, 0, 1024)
	rowIndex := 1
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return TableFile{}, fmt.Errorf("read table csv row %d: %w", rowIndex, err)
		}
		if blankRecord(record) {
			continue
		}

		row, err := buildTableRowFromRecord(scape, header, record, rowIndex)
		if err != nil {
			return TableFile{}, err
		}
		rows = append(rows, row)
		rowIndex++
	}

	info := TableInfo{Name: name}
	if len(rows) > 0 {
		info.IVL = len(rows[0].Inputs)
		info.OVL = len(rows[0].Targets)
		info.TrnEnd = len(rows)
		info.ValEnd = len(rows)
		info.TstEnd = len(rows)
	}
	return TableFile{Info: info, Rows: rows}, nil
}

func buildTableRowFromRecord(scape string, header, record []string, index int) (TableRow, error) {
	switch scape {
	case "chr-hmm", "chr_hmm":
		return TableRow{
			Index:  index,
			Fields: append([]string(nil), record...),
		}, nil
	case "chrom-hmm-expanded", "chrom_hmm_expanded":
		return TableRow{
			Index:  index,
			Fields: append([]string(nil), record...),
		}, nil
	case "abc-pred1", "abc_pred1":
		return TableRow{
			Index:  index,
			Fields: append([]string(nil), record...),
		}, nil
	case "hedge-fund", "hedge_fund":
		return TableRow{
			Index:  index,
			Fields: append([]string(nil), record...),
		}, nil
	case "simple":
		return TableRow{
			Index:  index,
			Fields: append([]string(nil), record...),
		}, nil
	}

	inputs := make([]float64, 0, len(record))
	targets := make([]float64, 0, len(record))
	for i, raw := range record {
		key := ""
		if i < len(header) {
			key = strings.ToLower(strings.TrimSpace(header[i]))
		}
		if key == "t" {
			continue
		}
		value, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
		if err != nil {
			return TableRow{}, fmt.Errorf("parse table row %d column %d: %w", index, i, err)
		}
		if strings.HasPrefix(key, "class") {
			targets = append(targets, value)
		} else {
			inputs = append(inputs, value)
		}
	}
	return TableRow{
		Index:   index,
		Inputs:  inputs,
		Targets: targets,
	}, nil
}

func WriteTableFile(path string, table TableFile) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("table file path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(table, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func ReadTableFile(path string) (TableFile, error) {
	if strings.TrimSpace(path) == "" {
		return TableFile{}, fmt.Errorf("table file path is required")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return TableFile{}, err
	}
	var table TableFile
	if err := json.Unmarshal(data, &table); err != nil {
		return TableFile{}, err
	}
	return table, nil
}

func DumpTable(table TableFile, limit int) []TableRow {
	if limit <= 0 || limit > len(table.Rows) {
		limit = len(table.Rows)
	}
	out := make([]TableRow, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, TableRow{
			Index:   table.Rows[i].Index,
			Inputs:  append([]float64(nil), table.Rows[i].Inputs...),
			Targets: append([]float64(nil), table.Rows[i].Targets...),
			Fields:  append([]string(nil), table.Rows[i].Fields...),
		})
	}
	return out
}
