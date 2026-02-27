package dataextract

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

type SimpleOptions struct {
	HasHeader bool
}

type VowelRecognitionOptions struct {
	HasHeader bool
}

type ABCPred1Options struct {
	HasHeader bool
}

type HedgeFundOptions struct {
	HasHeader bool
}

type MinesVsRocksOptions struct {
	HasHeader bool
}

type ChromHMMExpandedOptions struct {
	HasHeader      bool
	ChromColumn    string
	ChromIndex     int
	FromColumn     string
	FromIndex      int
	ToColumn       string
	ToIndex        int
	TagColumn      string
	TagIndex       int
	Step           int
	UseKnownTagSet bool
}

func ExtractSimpleCSV(in io.Reader, out io.Writer, opts SimpleOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	row := 0
	headerWritten := false
	var header []string
	if opts.HasHeader {
		record, err := reader.Read()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read simple header: %w", err)
		}
		row++
		header = append([]string(nil), record...)
		if err := writer.Write(header); err != nil {
			return fmt.Errorf("write simple header: %w", err)
		}
		headerWritten = true
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read simple row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if !headerWritten {
			header = make([]string, 0, len(record))
			for i := range record {
				header = append(header, fmt.Sprintf("col%d", i))
			}
			if err := writer.Write(header); err != nil {
				return fmt.Errorf("write simple header: %w", err)
			}
			headerWritten = true
		}
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("write simple row %d: %w", row, err)
		}
	}
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush simple csv: %w", err)
	}
	return nil
}

func ExtractVowelRecognitionCSV(in io.Reader, out io.Writer, opts VowelRecognitionOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	row := 0
	if opts.HasHeader {
		if _, err := reader.Read(); err != nil && err != io.EOF {
			return fmt.Errorf("read vowel header: %w", err)
		}
		if err := writeVowelHeader(writer, 1); err != nil {
			return err
		}
		row++
	} else if err := writeVowelHeader(writer, 1); err != nil {
		return err
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read vowel row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if len(record) < 14 {
			return fmt.Errorf("vowel row %d requires at least 14 columns, got %d", row, len(record))
		}
		outRow := make([]string, 0, len(record))
		for i := 0; i < 3; i++ {
			v, err := parseFloatField(record, i, row, "type")
			if err != nil {
				return err
			}
			outRow = append(outRow, strconv.FormatFloat(v, 'f', -1, 64))
		}
		for i := 3; i < 13; i++ {
			v, err := parseFloatField(record, i, row, "feature")
			if err != nil {
				return err
			}
			outRow = append(outRow, strconv.FormatFloat(v, 'f', -1, 64))
		}
		classCols := record[13:]
		for _, classRaw := range classCols {
			cv, err := strconv.ParseFloat(strings.TrimSpace(classRaw), 64)
			if err != nil {
				return fmt.Errorf("parse vowel class row %d: %w", row, err)
			}
			outRow = append(outRow, strconv.FormatFloat(cv, 'f', -1, 64))
		}
		if err := writer.Write(outRow); err != nil {
			return fmt.Errorf("write vowel row %d: %w", row, err)
		}
	}
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush vowel csv: %w", err)
	}
	return nil
}

func writeVowelHeader(writer *csv.Writer, classCount int) error {
	header := []string{"type0", "type1", "type2"}
	for i := 0; i < 10; i++ {
		header = append(header, fmt.Sprintf("feature%d", i))
	}
	for i := 0; i < classCount; i++ {
		header = append(header, fmt.Sprintf("class%d", i))
	}
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("write vowel header: %w", err)
	}
	return nil
}

func ExtractABCPred1CSV(in io.Reader, out io.Writer, opts ABCPred1Options) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	if opts.HasHeader {
		if _, err := reader.Read(); err != nil && err != io.EOF {
			return fmt.Errorf("read abc_pred1 header: %w", err)
		}
	}
	if err := writer.Write([]string{"sequence", "class"}); err != nil {
		return fmt.Errorf("write abc_pred1 header: %w", err)
	}
	row := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read abc_pred1 row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if len(record) < 2 {
			return fmt.Errorf("abc_pred1 row %d requires 2 columns, got %d", row, len(record))
		}
		outRow := []string{
			strings.TrimSpace(record[0]),
			strings.TrimSpace(record[1]),
		}
		if err := writer.Write(outRow); err != nil {
			return fmt.Errorf("write abc_pred1 row %d: %w", row, err)
		}
	}
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush abc_pred1 csv: %w", err)
	}
	return nil
}

func ExtractHedgeFundCSV(in io.Reader, out io.Writer, opts HedgeFundOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	header, err := reader.Read()
	if err == io.EOF {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read hedge_fund header: %w", err)
	}
	row := 1
	if !opts.HasHeader {
		// The reference helper expects header names in first row.
		// Keep behavior by treating row as names regardless.
	}
	if len(header) < 3 {
		return fmt.Errorf("hedge_fund header requires at least 3 columns")
	}
	featureHeader := append([]string(nil), header[1:len(header)-1]...)
	featureHeader = append(featureHeader, "date")
	if err := writer.Write(featureHeader); err != nil {
		return fmt.Errorf("write hedge_fund header: %w", err)
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read hedge_fund row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if len(record) < 3 {
			return fmt.Errorf("hedge_fund row %d requires at least 3 columns, got %d", row, len(record))
		}
		date := strings.TrimSpace(record[0])
		vector := append([]string(nil), record[1:len(record)-1]...)
		outRow := append(vector, date)
		if err := writer.Write(outRow); err != nil {
			return fmt.Errorf("write hedge_fund row %d: %w", row, err)
		}
	}
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush hedge_fund csv: %w", err)
	}
	return nil
}

func ExtractMinesVsRocksCSV(in io.Reader, out io.Writer, opts MinesVsRocksOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	row := 0
	headerWritten := false
	featureCount := -1
	if opts.HasHeader {
		if _, err := reader.Read(); err != nil && err != io.EOF {
			return fmt.Errorf("read mines_vs_rocks header: %w", err)
		}
		row++
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read mines_vs_rocks row %d: %w", row+1, err)
		}
		row++
		if blankRecord(record) {
			continue
		}
		if len(record) < 2 {
			return fmt.Errorf("mines_vs_rocks row %d requires at least 2 columns", row)
		}
		split, classVec, ok := minesVsRocksLabel(strings.TrimSpace(record[0]))
		if !ok {
			continue
		}
		features := record[1:]
		if featureCount < 0 {
			featureCount = len(features)
		}
		if len(features) != featureCount {
			return fmt.Errorf(
				"mines_vs_rocks row %d feature width mismatch: got=%d want=%d",
				row,
				len(features),
				featureCount,
			)
		}
		if !headerWritten {
			header := []string{"split"}
			for i := 0; i < featureCount; i++ {
				header = append(header, fmt.Sprintf("feature%d", i))
			}
			header = append(header, "class0", "class1")
			if err := writer.Write(header); err != nil {
				return fmt.Errorf("write mines_vs_rocks header: %w", err)
			}
			headerWritten = true
		}
		outRow := make([]string, 0, featureCount+3)
		outRow = append(outRow, strconv.Itoa(split))
		for _, feature := range features {
			outRow = append(outRow, strings.TrimSpace(feature))
		}
		outRow = append(outRow, strconv.Itoa(classVec[0]), strconv.Itoa(classVec[1]))
		if err := writer.Write(outRow); err != nil {
			return fmt.Errorf("write mines_vs_rocks row %d: %w", row, err)
		}
	}
	if !headerWritten {
		if err := writer.Write([]string{"split", "class0", "class1"}); err != nil {
			return fmt.Errorf("write mines_vs_rocks empty header: %w", err)
		}
	}
	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush mines_vs_rocks csv: %w", err)
	}
	return nil
}

func minesVsRocksLabel(identifier string) (split int, class [2]int, ok bool) {
	switch {
	case strings.HasPrefix(identifier, "*CM"):
		return 0, [2]int{1, 0}, true
	case strings.HasPrefix(identifier, "*CR"):
		return 0, [2]int{0, 1}, true
	case strings.HasPrefix(identifier, "CM"):
		return 1, [2]int{1, 0}, true
	case strings.HasPrefix(identifier, "CR"):
		return 1, [2]int{0, 1}, true
	default:
		return 0, [2]int{}, false
	}
}

func ExtractChromHMMExpandedCSV(in io.Reader, out io.Writer, opts ChromHMMExpandedOptions) error {
	reader := csv.NewReader(in)
	reader.FieldsPerRecord = -1
	writer := csv.NewWriter(out)
	defer writer.Flush()

	chromIdx := opts.ChromIndex
	fromIdx := opts.FromIndex
	toIdx := opts.ToIndex
	tagIdx := opts.TagIndex
	if strings.TrimSpace(opts.FromColumn) == "" && fromIdx == 0 {
		fromIdx = -1
	}
	if strings.TrimSpace(opts.ToColumn) == "" && toIdx == 0 {
		toIdx = -1
	}
	if strings.TrimSpace(opts.TagColumn) == "" && tagIdx == 0 {
		tagIdx = -1
	}
	step := opts.Step
	if step <= 0 {
		step = 200
	}

	rows := make([][]string, 0, 1024)
	if opts.HasHeader {
		header, err := reader.Read()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read chrom_hmm header: %w", err)
		}
		if strings.TrimSpace(opts.ChromColumn) != "" {
			idx, err := columnIndexByName(header, opts.ChromColumn)
			if err != nil {
				return err
			}
			chromIdx = idx
		}
		if strings.TrimSpace(opts.FromColumn) != "" {
			idx, err := columnIndexByName(header, opts.FromColumn)
			if err != nil {
				return err
			}
			fromIdx = idx
		}
		if strings.TrimSpace(opts.ToColumn) != "" {
			idx, err := columnIndexByName(header, opts.ToColumn)
			if err != nil {
				return err
			}
			toIdx = idx
		}
		if strings.TrimSpace(opts.TagColumn) != "" {
			idx, err := columnIndexByName(header, opts.TagColumn)
			if err != nil {
				return err
			}
			tagIdx = idx
		}
	}
	if chromIdx < 0 {
		chromIdx = 0
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

	tagSet := make([]string, 0, 32)
	tagSeen := map[string]struct{}{}
	if opts.UseKnownTagSet {
		for _, tag := range ChromHMMKnownTags() {
			tagSet = append(tagSet, tag)
			tagSeen[tag] = struct{}{}
		}
	}

	rowIndex := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read chrom_hmm row %d: %w", rowIndex+1, err)
		}
		rowIndex++
		if blankRecord(record) {
			continue
		}
		if chromIdx >= len(record) || fromIdx >= len(record) || toIdx >= len(record) || tagIdx >= len(record) {
			return fmt.Errorf("chrom_hmm row %d missing required columns", rowIndex)
		}
		rowCopy := append([]string(nil), record...)
		rows = append(rows, rowCopy)
		tag := strings.TrimSpace(record[tagIdx])
		if _, ok := tagSeen[tag]; ok {
			continue
		}
		// match reference find_unique_tags/2 behavior (prepend new tags).
		tagSet = append([]string{tag}, tagSet...)
		tagSeen[tag] = struct{}{}
	}

	header := make([]string, 0, len(tagSet)+3)
	for _, tag := range tagSet {
		header = append(header, "tag_"+sanitizeHeaderToken(tag))
	}
	header = append(header, "bp_index", "tag", "chrom")
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("write chrom_hmm header: %w", err)
	}

	tagPos := map[string]int{}
	for i, tag := range tagSet {
		tagPos[tag] = i
	}
	for rowNum, record := range rows {
		chrom := strings.TrimSpace(record[chromIdx])
		fromVal, err := strconv.ParseFloat(strings.TrimSpace(record[fromIdx]), 64)
		if err != nil {
			return fmt.Errorf("parse chrom_hmm from row %d: %w", rowNum+1, err)
		}
		toVal, err := strconv.ParseFloat(strings.TrimSpace(record[toIdx]), 64)
		if err != nil {
			return fmt.Errorf("parse chrom_hmm to row %d: %w", rowNum+1, err)
		}
		tag := strings.TrimSpace(record[tagIdx])
		pos, ok := tagPos[tag]
		if !ok {
			return fmt.Errorf("chrom_hmm row %d tag not found in tag set: %s", rowNum+1, tag)
		}

		steps := int(math.Round((toVal - fromVal) / float64(step)))
		if steps < 0 {
			steps = 0
		}
		for i := 0; i <= steps; i++ {
			vector := make([]string, len(tagSet))
			for j := range vector {
				vector[j] = "0"
			}
			vector[pos] = "1"
			bp := int(math.Round(fromVal + float64(i*step)))
			outRow := append(vector,
				strconv.Itoa(bp),
				tag,
				chrom,
			)
			if err := writer.Write(outRow); err != nil {
				return fmt.Errorf("write chrom_hmm expanded row %d: %w", rowNum+1, err)
			}
		}
	}

	if err := writer.Error(); err != nil {
		return fmt.Errorf("flush chrom_hmm csv: %w", err)
	}
	return nil
}

func ChromHMMKnownTags() []string {
	return []string{
		"ReprD",
		"EnhF",
		"PromP",
		"H4K20",
		"Enh",
		"Art",
		"Gen5'",
		"Gen3'",
		"ElonW",
		"Tss",
		"EnhW",
		"EnhWF",
		"CtcfO",
		"Repr",
		"ReprW",
		"Ctcf",
		"DnaseD",
		"Elon",
		"Pol2",
		"DnaseU",
		"Low",
		"FaireW",
		"TssF",
		"PromF",
		"Quies",
	}
}

func sanitizeHeaderToken(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "unknown"
	}
	parts := strings.FieldsFunc(trimmed, func(r rune) bool {
		switch {
		case r >= 'a' && r <= 'z':
			return false
		case r >= 'A' && r <= 'Z':
			return false
		case r >= '0' && r <= '9':
			return false
		case r == '_':
			return false
		default:
			return true
		}
	})
	if len(parts) == 0 {
		return "unknown"
	}
	return strings.Join(parts, "_")
}
