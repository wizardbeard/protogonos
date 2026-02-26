package dataextract

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

type SeriesOptions struct {
	HasHeader         bool
	ValueColumnName   string
	ValueColumnIndex  int
	OutputValueHeader string
	Normalize         string
}

type EpitopesOptions struct {
	HasHeader             bool
	SignalColumnName      string
	SignalColumnIndex     int
	MemoryColumnName      string
	MemoryColumnIndex     int
	ClassColumnName       string
	ClassColumnIndex      int
	SequenceColumnNames   []string
	SequenceColumnIndexes []int
}

func ExtractSeriesCSV(in io.Reader, out io.Writer, opts SeriesOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	valueHeader := strings.TrimSpace(opts.OutputValueHeader)
	if valueHeader == "" {
		valueHeader = "value"
	}
	valueIdx := opts.ValueColumnIndex
	row := 0
	if opts.HasHeader {
		header, err := reader.Read()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read series header: %w", err)
		}
		row++
		if strings.TrimSpace(opts.ValueColumnName) != "" {
			idx, err := columnIndexByName(header, opts.ValueColumnName)
			if err != nil {
				return err
			}
			valueIdx = idx
		} else if valueIdx < 0 {
			valueIdx = lastNonEmptyColumn(header)
		}
	}
	if valueIdx < 0 {
		valueIdx = 0
	}

	values := make([]float64, 0)
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read series row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if valueIdx >= len(record) {
			return fmt.Errorf("series row %d missing value column index %d", row, valueIdx)
		}
		value, err := strconv.ParseFloat(strings.TrimSpace(record[valueIdx]), 64)
		if err != nil {
			return fmt.Errorf("parse series value row %d: %w", row, err)
		}
		values = append(values, value)
	}

	normalized, err := normalizeSeriesValues(values, opts.Normalize)
	if err != nil {
		return err
	}

	if err := writer.Write([]string{"t", valueHeader}); err != nil {
		return fmt.Errorf("write series header: %w", err)
	}
	for i, value := range normalized {
		if err := writer.Write([]string{
			strconv.Itoa(i),
			strconv.FormatFloat(value, 'f', -1, 64),
		}); err != nil {
			return fmt.Errorf("write series row %d: %w", i+1, err)
		}
	}

	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush series csv: %w", err)
	}
	return nil
}

func ExtractEpitopesCSV(in io.Reader, out io.Writer, opts EpitopesOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	signalIdx := opts.SignalColumnIndex
	memoryIdx := opts.MemoryColumnIndex
	classIdx := opts.ClassColumnIndex
	sequenceIdx := append([]int(nil), opts.SequenceColumnIndexes...)

	row := 0
	if opts.HasHeader {
		header, err := reader.Read()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read epitopes header: %w", err)
		}
		row++

		if strings.TrimSpace(opts.SignalColumnName) != "" {
			idx, err := columnIndexByName(header, opts.SignalColumnName)
			if err != nil {
				return err
			}
			signalIdx = idx
		}
		if strings.TrimSpace(opts.MemoryColumnName) != "" {
			idx, err := columnIndexByName(header, opts.MemoryColumnName)
			if err != nil {
				return err
			}
			memoryIdx = idx
		}
		if strings.TrimSpace(opts.ClassColumnName) != "" {
			idx, err := columnIndexByName(header, opts.ClassColumnName)
			if err != nil {
				return err
			}
			classIdx = idx
		}
		if len(opts.SequenceColumnNames) > 0 {
			sequenceIdx = make([]int, 0, len(opts.SequenceColumnNames))
			for _, name := range opts.SequenceColumnNames {
				idx, err := columnIndexByName(header, name)
				if err != nil {
					return err
				}
				sequenceIdx = append(sequenceIdx, idx)
			}
		}
	}

	if signalIdx < 0 {
		signalIdx = 0
	}
	if memoryIdx < 0 {
		memoryIdx = 1
	}
	if classIdx < 0 {
		classIdx = 2
	}
	if len(sequenceIdx) == 0 {
		maxBase := maxInt(maxInt(signalIdx, memoryIdx), classIdx)
		sequenceIdx = make([]int, 0)
		for idx := maxBase + 1; idx < maxBase+17; idx++ {
			sequenceIdx = append(sequenceIdx, idx)
		}
	}

	header := []string{"signal", "memory", "class"}
	for i := range sequenceIdx {
		header = append(header, fmt.Sprintf("seq%d", i))
	}
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("write epitopes header: %w", err)
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read epitopes row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}

		signal, err := parseFloatField(record, signalIdx, row, "signal")
		if err != nil {
			return err
		}
		memory, err := parseFloatField(record, memoryIdx, row, "memory")
		if err != nil {
			return err
		}
		classValue, err := parseClassField(record, classIdx, row)
		if err != nil {
			return err
		}

		outRow := []string{
			strconv.FormatFloat(signal, 'f', -1, 64),
			strconv.FormatFloat(memory, 'f', -1, 64),
			strconv.Itoa(classValue),
		}
		for _, idx := range sequenceIdx {
			if idx < 0 || idx >= len(record) {
				return fmt.Errorf("epitopes row %d missing sequence column index %d", row, idx)
			}
			residue, err := parseResidue(strings.TrimSpace(record[idx]), row, idx)
			if err != nil {
				return err
			}
			outRow = append(outRow, strconv.Itoa(residue))
		}
		if err := writer.Write(outRow); err != nil {
			return fmt.Errorf("write epitopes row %d: %w", row, err)
		}
	}

	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush epitopes csv: %w", err)
	}
	return nil
}

func parseFloatField(record []string, idx, row int, field string) (float64, error) {
	if idx < 0 || idx >= len(record) {
		return 0, fmt.Errorf("epitopes row %d missing %s column index %d", row, field, idx)
	}
	value, err := strconv.ParseFloat(strings.TrimSpace(record[idx]), 64)
	if err != nil {
		return 0, fmt.Errorf("parse epitopes %s row %d: %w", field, row, err)
	}
	return value, nil
}

func parseClassField(record []string, idx, row int) (int, error) {
	if idx < 0 || idx >= len(record) {
		return 0, fmt.Errorf("epitopes row %d missing class column index %d", row, idx)
	}
	raw := strings.ToLower(strings.TrimSpace(record[idx]))
	switch raw {
	case "0", "false", "f", "neg", "negative", "n", "no":
		return 0, nil
	case "1", "true", "t", "pos", "positive", "y", "yes":
		return 1, nil
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("parse epitopes class row %d: %w", row, err)
	}
	if value > 0 {
		return 1, nil
	}
	return 0, nil
}

func parseResidue(raw string, row, idx int) (int, error) {
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, fmt.Errorf("parse epitopes residue row %d column %d: %w", row, idx, err)
	}
	residue := int(value)
	if residue < 0 {
		return 0, fmt.Errorf("epitopes residue row %d column %d must be >= 0", row, idx)
	}
	return residue, nil
}

func columnIndexByName(header []string, name string) (int, error) {
	want := strings.TrimSpace(strings.ToLower(name))
	for i, field := range header {
		if strings.ToLower(strings.TrimSpace(field)) == want {
			return i, nil
		}
	}
	return -1, fmt.Errorf("csv column not found: %s", name)
}

func blankRecord(record []string) bool {
	for _, field := range record {
		if strings.TrimSpace(field) != "" {
			return false
		}
	}
	return true
}

func lastNonEmptyColumn(record []string) int {
	for i := len(record) - 1; i >= 0; i-- {
		if strings.TrimSpace(record[i]) != "" {
			return i
		}
	}
	return 0
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func normalizeSeriesValues(values []float64, mode string) ([]float64, error) {
	out := append([]float64(nil), values...)
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "none":
		return out, nil
	case "minmax":
		return normalizeSeriesMinMax(out), nil
	case "zscore":
		return normalizeSeriesZScore(out), nil
	default:
		return nil, fmt.Errorf("unsupported series normalization mode: %s", mode)
	}
}

func normalizeSeriesMinMax(values []float64) []float64 {
	if len(values) == 0 {
		return values
	}
	minValue := values[0]
	maxValue := values[0]
	for _, value := range values[1:] {
		if value < minValue {
			minValue = value
		}
		if value > maxValue {
			maxValue = value
		}
	}
	rangeValue := maxValue - minValue
	if rangeValue == 0 {
		for i := range values {
			values[i] = 0
		}
		return values
	}
	for i := range values {
		values[i] = (values[i] - minValue) / rangeValue
	}
	return values
}

func normalizeSeriesZScore(values []float64) []float64 {
	if len(values) == 0 {
		return values
	}
	mean := 0.0
	for _, value := range values {
		mean += value
	}
	mean /= float64(len(values))

	sumSq := 0.0
	for _, value := range values {
		diff := value - mean
		sumSq += diff * diff
	}
	std := math.Sqrt(sumSq / float64(len(values)))
	if std == 0 {
		for i := range values {
			values[i] = 0
		}
		return values
	}
	for i := range values {
		values[i] = (values[i] - mean) / std
	}
	return values
}
