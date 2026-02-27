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

type MNISTOptions struct {
	HasHeader            bool
	LabelColumnName      string
	LabelColumnIndex     int
	FeatureColumnNames   []string
	FeatureColumnIndexes []int
	OneHotClassification bool
}

type WineOptions struct {
	HasHeader            bool
	LabelColumnName      string
	LabelColumnIndex     int
	FeatureColumnNames   []string
	FeatureColumnIndexes []int
	OneHotClassification bool
}

type ChrHMMOptions struct {
	HasHeader       bool
	FromColumnName  string
	FromColumnIndex int
	ToColumnName    string
	ToColumnIndex   int
	TagColumnName   string
	TagColumnIndex  int
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

func ExtractMNISTCSV(in io.Reader, out io.Writer, opts MNISTOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	labelIdx := opts.LabelColumnIndex
	featureIdx := append([]int(nil), opts.FeatureColumnIndexes...)
	row := 0
	var header []string
	if opts.HasHeader {
		record, err := reader.Read()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read mnist header: %w", err)
		}
		row++
		header = record
		if strings.TrimSpace(opts.LabelColumnName) != "" {
			idx, err := columnIndexByName(record, opts.LabelColumnName)
			if err != nil {
				return err
			}
			labelIdx = idx
		} else if labelIdx < 0 {
			labelIdx = lastNonEmptyColumn(record)
		}
		if len(opts.FeatureColumnNames) > 0 {
			featureIdx = make([]int, 0, len(opts.FeatureColumnNames))
			for _, name := range opts.FeatureColumnNames {
				idx, err := columnIndexByName(record, name)
				if err != nil {
					return err
				}
				featureIdx = append(featureIdx, idx)
			}
		}
	}

	if labelIdx < 0 {
		labelIdx = 0
	}
	if len(featureIdx) == 0 && len(header) > 0 {
		featureIdx = defaultFeatureIndexes(len(header), labelIdx)
	}
	headerWritten := false
	if len(featureIdx) > 0 {
		outHeader := featureHeader(header, featureIdx, "px")
		if opts.OneHotClassification {
			outHeader = append(outHeader, "class9", "class8", "class7", "class6", "class5", "class4", "class3", "class2", "class1", "class0")
		} else {
			outHeader = append(outHeader, "class")
		}
		if err := writer.Write(outHeader); err != nil {
			return fmt.Errorf("write mnist header: %w", err)
		}
		headerWritten = true
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read mnist row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if labelIdx >= len(record) {
			return fmt.Errorf("mnist row %d missing label column index %d", row, labelIdx)
		}
		if len(featureIdx) == 0 {
			featureIdx = defaultFeatureIndexes(len(record), labelIdx)
		}
		if len(featureIdx) == 0 {
			return fmt.Errorf("mnist row %d has no feature columns", row)
		}
		if !headerWritten {
			outHeader := featureHeader(nil, featureIdx, "px")
			if opts.OneHotClassification {
				outHeader = append(outHeader, "class9", "class8", "class7", "class6", "class5", "class4", "class3", "class2", "class1", "class0")
			} else {
				outHeader = append(outHeader, "class")
			}
			if err := writer.Write(outHeader); err != nil {
				return fmt.Errorf("write mnist header: %w", err)
			}
			headerWritten = true
		}
		label, err := parseLabelInt(record[labelIdx], row, "mnist")
		if err != nil {
			return err
		}

		outRow := make([]string, 0, len(featureIdx)+10)
		for _, idx := range featureIdx {
			value, err := parseFloatField(record, idx, row, "feature")
			if err != nil {
				return err
			}
			outRow = append(outRow, strconv.FormatFloat(value, 'f', -1, 64))
		}
		if opts.OneHotClassification {
			if label < 0 || label > 9 {
				return fmt.Errorf("mnist row %d class must be in [0,9], got %d", row, label)
			}
			// Reference data_extractor:update/0 stores mnist class one-hot in reverse class order.
			classVec := [10]int{}
			classVec[9-label] = 1
			for _, bit := range classVec {
				outRow = append(outRow, strconv.Itoa(bit))
			}
		} else {
			outRow = append(outRow, strconv.Itoa(label))
		}
		if err := writer.Write(outRow); err != nil {
			return fmt.Errorf("write mnist row %d: %w", row, err)
		}
	}

	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush mnist csv: %w", err)
	}
	return nil
}

func ExtractWineCSV(in io.Reader, out io.Writer, opts WineOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	labelIdx := opts.LabelColumnIndex
	featureIdx := append([]int(nil), opts.FeatureColumnIndexes...)
	row := 0
	var header []string
	if opts.HasHeader {
		record, err := reader.Read()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read wine header: %w", err)
		}
		row++
		header = record
		if strings.TrimSpace(opts.LabelColumnName) != "" {
			idx, err := columnIndexByName(record, opts.LabelColumnName)
			if err != nil {
				return err
			}
			labelIdx = idx
		}
		if len(opts.FeatureColumnNames) > 0 {
			featureIdx = make([]int, 0, len(opts.FeatureColumnNames))
			for _, name := range opts.FeatureColumnNames {
				idx, err := columnIndexByName(record, name)
				if err != nil {
					return err
				}
				featureIdx = append(featureIdx, idx)
			}
		}
	}
	if labelIdx < 0 {
		labelIdx = 0
	}
	if len(featureIdx) == 0 && len(header) > 0 {
		featureIdx = defaultFeatureIndexes(len(header), labelIdx)
	}
	headerWritten := false
	if len(featureIdx) > 0 {
		outHeader := featureHeader(header, featureIdx, "feature")
		if opts.OneHotClassification {
			outHeader = append(outHeader, "class3", "class2", "class1")
		} else {
			outHeader = append(outHeader, "class")
		}
		if err := writer.Write(outHeader); err != nil {
			return fmt.Errorf("write wine header: %w", err)
		}
		headerWritten = true
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read wine row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if labelIdx >= len(record) {
			return fmt.Errorf("wine row %d missing label column index %d", row, labelIdx)
		}
		if len(featureIdx) == 0 {
			featureIdx = defaultFeatureIndexes(len(record), labelIdx)
		}
		if len(featureIdx) == 0 {
			return fmt.Errorf("wine row %d has no feature columns", row)
		}
		if !headerWritten {
			outHeader := featureHeader(nil, featureIdx, "feature")
			if opts.OneHotClassification {
				outHeader = append(outHeader, "class3", "class2", "class1")
			} else {
				outHeader = append(outHeader, "class")
			}
			if err := writer.Write(outHeader); err != nil {
				return fmt.Errorf("write wine header: %w", err)
			}
			headerWritten = true
		}
		label, err := parseLabelInt(record[labelIdx], row, "wine")
		if err != nil {
			return err
		}

		outRow := make([]string, 0, len(featureIdx)+3)
		for _, idx := range featureIdx {
			value, err := parseFloatField(record, idx, row, "feature")
			if err != nil {
				return err
			}
			outRow = append(outRow, strconv.FormatFloat(value, 'f', -1, 64))
		}
		if opts.OneHotClassification {
			classVec, err := wineOneHot(label, row)
			if err != nil {
				return err
			}
			for _, bit := range classVec {
				outRow = append(outRow, strconv.Itoa(bit))
			}
		} else {
			outRow = append(outRow, strconv.Itoa(label))
		}
		if err := writer.Write(outRow); err != nil {
			return fmt.Errorf("write wine row %d: %w", row, err)
		}
	}

	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush wine csv: %w", err)
	}
	return nil
}

func ExtractChrHMMCSV(in io.Reader, out io.Writer, opts ChrHMMOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	fromIdx := opts.FromColumnIndex
	toIdx := opts.ToColumnIndex
	tagIdx := opts.TagColumnIndex
	if strings.TrimSpace(opts.FromColumnName) == "" && fromIdx == 0 {
		fromIdx = -1
	}
	if strings.TrimSpace(opts.ToColumnName) == "" && toIdx == 0 {
		toIdx = -1
	}
	if strings.TrimSpace(opts.TagColumnName) == "" && tagIdx == 0 {
		tagIdx = -1
	}
	row := 0
	if opts.HasHeader {
		header, err := reader.Read()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read chr_hmm header: %w", err)
		}
		row++
		if strings.TrimSpace(opts.FromColumnName) != "" {
			idx, err := columnIndexByName(header, opts.FromColumnName)
			if err != nil {
				return err
			}
			fromIdx = idx
		}
		if strings.TrimSpace(opts.ToColumnName) != "" {
			idx, err := columnIndexByName(header, opts.ToColumnName)
			if err != nil {
				return err
			}
			toIdx = idx
		}
		if strings.TrimSpace(opts.TagColumnName) != "" {
			idx, err := columnIndexByName(header, opts.TagColumnName)
			if err != nil {
				return err
			}
			tagIdx = idx
		}
	}
	if fromIdx < 0 {
		fromIdx = 1
	}
	if toIdx < 0 {
		toIdx = 2
	}
	if tagIdx < 0 {
		tagIdx = 3
	}

	headerWritten := false
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read chr_hmm row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if fromIdx >= len(record) || toIdx >= len(record) || tagIdx >= len(record) {
			return fmt.Errorf("chr_hmm row %d missing from/to/tag columns", row)
		}

		maxCore := maxInt(fromIdx, maxInt(toIdx, tagIdx))
		extras := make([]string, 0, maxInt(0, len(record)-maxCore-1))
		for idx := maxCore + 1; idx < len(record); idx++ {
			extras = append(extras, strings.TrimSpace(record[idx]))
		}

		if !headerWritten {
			outHeader := []string{"from", "to", "tag"}
			for i := range extras {
				outHeader = append(outHeader, fmt.Sprintf("extra%d", i))
			}
			if err := writer.Write(outHeader); err != nil {
				return fmt.Errorf("write chr_hmm header: %w", err)
			}
			headerWritten = true
		}

		outRow := []string{
			strings.TrimSpace(record[fromIdx]),
			strings.TrimSpace(record[toIdx]),
			strings.TrimSpace(record[tagIdx]),
		}
		outRow = append(outRow, extras...)
		if err := writer.Write(outRow); err != nil {
			return fmt.Errorf("write chr_hmm row %d: %w", row, err)
		}
	}
	if !headerWritten {
		if err := writer.Write([]string{"from", "to", "tag"}); err != nil {
			return fmt.Errorf("write chr_hmm empty header: %w", err)
		}
	}
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush chr_hmm csv: %w", err)
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

func parseLabelInt(raw string, row int, dataset string) (int, error) {
	value, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil {
		return 0, fmt.Errorf("parse %s class row %d: %w", dataset, row, err)
	}
	return int(math.Round(value)), nil
}

func wineOneHot(class int, row int) ([3]int, error) {
	switch class {
	case 1:
		return [3]int{0, 0, 1}, nil
	case 2:
		return [3]int{0, 1, 0}, nil
	case 3:
		return [3]int{1, 0, 0}, nil
	default:
		return [3]int{}, fmt.Errorf("wine row %d class must be in [1,3], got %d", row, class)
	}
}

func featureHeader(header []string, indexes []int, fallbackPrefix string) []string {
	out := make([]string, 0, len(indexes))
	for i, idx := range indexes {
		switch {
		case len(header) > idx && idx >= 0 && strings.TrimSpace(header[idx]) != "":
			out = append(out, strings.TrimSpace(header[idx]))
		default:
			out = append(out, fmt.Sprintf("%s%d", fallbackPrefix, i))
		}
	}
	return out
}

func defaultFeatureIndexes(recordLen, labelIdx int) []int {
	out := make([]int, 0, recordLen)
	for idx := 0; idx < recordLen; idx++ {
		if idx == labelIdx {
			continue
		}
		out = append(out, idx)
	}
	return out
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
