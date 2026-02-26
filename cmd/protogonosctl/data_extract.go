package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"protogonos/internal/dataextract"
)

func runDataExtract(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("data-extract", flag.ContinueOnError)
	scapeName := fs.String("scape", "", "target dataset shape: gtsa|fx|epitopes")
	inputPath := fs.String("in", "", "input CSV path")
	outputPath := fs.String("out", "", "output CSV path")
	hasHeader := fs.Bool("has-header", true, "input CSV has header row")
	valueCol := fs.String("value-col", "", "series value column name (header mode)")
	valueIndex := fs.Int("value-index", -1, "series value column index (no-header mode)")
	normalize := fs.String("normalize", "none", "series normalization mode: none|minmax|zscore")
	signalCol := fs.String("signal-col", "signal", "epitopes signal column name")
	memoryCol := fs.String("memory-col", "memory", "epitopes memory column name")
	classCol := fs.String("class-col", "class", "epitopes class column name")
	signalIndex := fs.Int("signal-index", 0, "epitopes signal column index")
	memoryIndex := fs.Int("memory-index", 1, "epitopes memory column index")
	classIndex := fs.Int("class-index", 2, "epitopes class column index")
	sequenceCols := fs.String("sequence-cols", "", "comma-separated epitopes sequence column names")
	sequenceIndexes := fs.String("sequence-indexes", "", "comma-separated epitopes sequence column indexes")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if strings.TrimSpace(*scapeName) == "" {
		return errors.New("data-extract requires --scape")
	}
	if strings.TrimSpace(*inputPath) == "" {
		return errors.New("data-extract requires --in")
	}
	if strings.TrimSpace(*outputPath) == "" {
		return errors.New("data-extract requires --out")
	}

	in, err := os.Open(*inputPath)
	if err != nil {
		return err
	}
	defer func() {
		_ = in.Close()
	}()

	out, err := os.Create(*outputPath)
	if err != nil {
		return err
	}
	defer func() {
		_ = out.Close()
	}()

	switch strings.TrimSpace(strings.ToLower(*scapeName)) {
	case "gtsa":
		columnName := strings.TrimSpace(*valueCol)
		if columnName == "" && *hasHeader {
			columnName = "value"
		}
		err = dataextract.ExtractSeriesCSV(in, out, dataextract.SeriesOptions{
			HasHeader:         *hasHeader,
			ValueColumnName:   columnName,
			ValueColumnIndex:  *valueIndex,
			OutputValueHeader: "value",
			Normalize:         *normalize,
		})
	case "fx":
		columnName := strings.TrimSpace(*valueCol)
		if columnName == "" && *hasHeader {
			columnName = "close"
		}
		err = dataextract.ExtractSeriesCSV(in, out, dataextract.SeriesOptions{
			HasHeader:         *hasHeader,
			ValueColumnName:   columnName,
			ValueColumnIndex:  *valueIndex,
			OutputValueHeader: "close",
			Normalize:         *normalize,
		})
	case "epitopes":
		sequenceNames := parseCommaSeparated(*sequenceCols)
		seqIndexes, parseErr := parseIndexList(*sequenceIndexes)
		if parseErr != nil {
			return parseErr
		}
		sCol := strings.TrimSpace(*signalCol)
		mCol := strings.TrimSpace(*memoryCol)
		cCol := strings.TrimSpace(*classCol)
		if !*hasHeader {
			sCol, mCol, cCol = "", "", ""
		}
		err = dataextract.ExtractEpitopesCSV(in, out, dataextract.EpitopesOptions{
			HasHeader:             *hasHeader,
			SignalColumnName:      sCol,
			SignalColumnIndex:     *signalIndex,
			MemoryColumnName:      mCol,
			MemoryColumnIndex:     *memoryIndex,
			ClassColumnName:       cCol,
			ClassColumnIndex:      *classIndex,
			SequenceColumnNames:   sequenceNames,
			SequenceColumnIndexes: seqIndexes,
		})
	default:
		return fmt.Errorf("unsupported data-extract scape: %s", *scapeName)
	}
	if err != nil {
		return err
	}

	fmt.Printf("data_extract scape=%s in=%s out=%s normalize=%s\n", strings.ToLower(strings.TrimSpace(*scapeName)), *inputPath, *outputPath, strings.ToLower(strings.TrimSpace(*normalize)))
	return nil
}

func parseCommaSeparated(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func parseIndexList(raw string) ([]int, error) {
	parts := parseCommaSeparated(raw)
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		idx, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("parse sequence-indexes value %q: %w", part, err)
		}
		if idx < 0 {
			return nil, fmt.Errorf("sequence-indexes value must be >= 0: %d", idx)
		}
		out = append(out, idx)
	}
	return out, nil
}
