package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"protogonos/internal/dataextract"
)

func runDataExtract(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("data-extract", flag.ContinueOnError)
	scapeName := fs.String("scape", "", "target dataset shape: gtsa|fx|epitopes|mnist|wine|chr-hmm")
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
	labelCol := fs.String("label-col", "", "mnist/wine class column name")
	labelIndex := fs.Int("label-index", -1, "mnist/wine class column index")
	featureCols := fs.String("feature-cols", "", "comma-separated feature column names for mnist/wine")
	featureIndexes := fs.String("feature-indexes", "", "comma-separated feature column indexes for mnist/wine")
	oneHot := fs.Bool("one-hot", true, "emit one-hot classes for mnist/wine")
	fromCol := fs.String("from-col", "from", "chr-hmm from column name")
	toCol := fs.String("to-col", "to", "chr-hmm to column name")
	tagCol := fs.String("tag-col", "tag", "chr-hmm tag column name")
	fromIndex := fs.Int("from-index", 1, "chr-hmm from column index")
	toIndex := fs.Int("to-index", 2, "chr-hmm to column index")
	tagIndex := fs.Int("tag-index", 3, "chr-hmm tag column index")
	tableOut := fs.String("table-out", "", "optional ETS-like table file (.json) output path")
	tableName := fs.String("table-name", "", "optional table name for --table-out")
	tableCheck := fs.String("table-check", "", "check/dump existing table file and exit")
	dumpLimit := fs.Int("dump-limit", 10, "max table rows to print for --table-check")
	tableSave := fs.String("table-save", "", "write transformed --table-check result to table file path")
	tableScaleMax := fs.Bool("table-scale-max", false, "scale table input columns by per-column max (dg_scale1 analog)")
	tableScaleAsinh := fs.Bool("table-scale-asinh", false, "apply asinh scaling to table inputs (dg_scale2 analog)")
	tableBinarize := fs.Bool("table-binarize", false, "binarize table inputs (dg_bin analog)")
	tableCleanZeroInputs := fs.Bool("table-clean-zero-inputs", false, "remove rows whose input-vector sum is zero (dg_clean analog)")
	tableResolution := fs.Int("table-resolution", 0, "resolutionator window size for table inputs (deep_gene_full analog)")
	tableResolutionDropZeroRun := fs.Int("table-resolution-drop-zero-run", 200, "drop zero-input runs longer than this threshold during table resolution")
	tableResolutionAsinh := fs.Bool("table-resolution-asinh", true, "apply asinh transform to resolved input-window averages")
	tableStats := fs.Bool("table-stats", false, "print per-input-column min/avg/max stats")
	tableZeroCounts := fs.Bool("table-zero-counts", false, "print zero/non-zero input counts and ratio")
	if err := fs.Parse(args); err != nil {
		return err
	}

	transformOpts := tableTransformOptions{
		ScaleMax:           *tableScaleMax,
		ScaleAsinh:         *tableScaleAsinh,
		Binarize:           *tableBinarize,
		CleanZeroInputs:    *tableCleanZeroInputs,
		Resolution:         *tableResolution,
		ResolutionDropZero: *tableResolutionDropZeroRun,
		ResolutionUseAsinh: *tableResolutionAsinh,
	}

	if strings.TrimSpace(*tableCheck) != "" {
		table, err := dataextract.ReadTableFile(*tableCheck)
		if err != nil {
			return err
		}
		if err := applyTableTransforms(&table, transformOpts); err != nil {
			return err
		}
		if *tableStats {
			if err := printTableStats(table); err != nil {
				return err
			}
		}
		if *tableZeroCounts {
			zeroes, nonZeroes, ratio := dataextract.CountZeroInputs(table)
			fmt.Printf("table_zeroes zeroes=%d non_zeroes=%d ratio=%g\n", zeroes, nonZeroes, ratio)
		}
		fmt.Printf("table_check name=%s rows=%d ivl=%d ovl=%d trn_end=%d val_end=%d tst_end=%d\n",
			table.Info.Name,
			len(table.Rows),
			table.Info.IVL,
			table.Info.OVL,
			table.Info.TrnEnd,
			table.Info.ValEnd,
			table.Info.TstEnd,
		)
		for _, row := range dataextract.DumpTable(table, *dumpLimit) {
			fmt.Printf("row index=%d inputs=%v targets=%v fields=%v\n", row.Index, row.Inputs, row.Targets, row.Fields)
		}
		if strings.TrimSpace(*tableSave) != "" {
			if err := dataextract.WriteTableFile(*tableSave, table); err != nil {
				return err
			}
		}
		return nil
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
	case "mnist":
		featureNames := parseCommaSeparated(*featureCols)
		featureIdx, parseErr := parseIndexList(*featureIndexes)
		if parseErr != nil {
			return parseErr
		}
		lCol := strings.TrimSpace(*labelCol)
		if !*hasHeader {
			lCol = ""
		}
		err = dataextract.ExtractMNISTCSV(in, out, dataextract.MNISTOptions{
			HasHeader:            *hasHeader,
			LabelColumnName:      lCol,
			LabelColumnIndex:     *labelIndex,
			FeatureColumnNames:   featureNames,
			FeatureColumnIndexes: featureIdx,
			OneHotClassification: *oneHot,
		})
	case "wine":
		featureNames := parseCommaSeparated(*featureCols)
		featureIdx, parseErr := parseIndexList(*featureIndexes)
		if parseErr != nil {
			return parseErr
		}
		lCol := strings.TrimSpace(*labelCol)
		if !*hasHeader {
			lCol = ""
		}
		err = dataextract.ExtractWineCSV(in, out, dataextract.WineOptions{
			HasHeader:            *hasHeader,
			LabelColumnName:      lCol,
			LabelColumnIndex:     *labelIndex,
			FeatureColumnNames:   featureNames,
			FeatureColumnIndexes: featureIdx,
			OneHotClassification: *oneHot,
		})
	case "chr-hmm":
		fCol := strings.TrimSpace(*fromCol)
		tCol := strings.TrimSpace(*toCol)
		taCol := strings.TrimSpace(*tagCol)
		if !*hasHeader {
			fCol, tCol, taCol = "", "", ""
		}
		err = dataextract.ExtractChrHMMCSV(in, out, dataextract.ChrHMMOptions{
			HasHeader:       *hasHeader,
			FromColumnName:  fCol,
			FromColumnIndex: *fromIndex,
			ToColumnName:    tCol,
			ToColumnIndex:   *toIndex,
			TagColumnName:   taCol,
			TagColumnIndex:  *tagIndex,
		})
	default:
		return fmt.Errorf("unsupported data-extract scape: %s", *scapeName)
	}
	if err != nil {
		return err
	}

	if strings.TrimSpace(*tableOut) != "" {
		outputCSV, err := os.Open(*outputPath)
		if err != nil {
			return err
		}
		defer func() {
			_ = outputCSV.Close()
		}()

		name := strings.TrimSpace(*tableName)
		if name == "" {
			base := filepath.Base(*tableOut)
			ext := filepath.Ext(base)
			name = strings.TrimSuffix(base, ext)
			if name == "" {
				name = "table"
			}
		}
		table, err := dataextract.BuildTableFromExtractedCSV(outputCSV, dataextract.BuildTableOptions{
			Scape: strings.ToLower(strings.TrimSpace(*scapeName)),
			Name:  name,
		})
		if err != nil {
			return err
		}
		if err := applyTableTransforms(&table, transformOpts); err != nil {
			return err
		}
		if *tableStats {
			if err := printTableStats(table); err != nil {
				return err
			}
		}
		if *tableZeroCounts {
			zeroes, nonZeroes, ratio := dataextract.CountZeroInputs(table)
			fmt.Printf("table_zeroes zeroes=%d non_zeroes=%d ratio=%g\n", zeroes, nonZeroes, ratio)
		}
		if err := dataextract.WriteTableFile(*tableOut, table); err != nil {
			return err
		}
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

type tableTransformOptions struct {
	ScaleMax           bool
	ScaleAsinh         bool
	Binarize           bool
	CleanZeroInputs    bool
	Resolution         int
	ResolutionDropZero int
	ResolutionUseAsinh bool
}

func applyTableTransforms(table *dataextract.TableFile, opts tableTransformOptions) error {
	if table == nil {
		return errors.New("table is required")
	}
	if opts.Resolution > 1 {
		if err := dataextract.ResolutionateInputs(table, opts.Resolution, opts.ResolutionDropZero, opts.ResolutionUseAsinh); err != nil {
			return err
		}
	}
	if opts.ScaleMax {
		stats, err := dataextract.InputColumnStats(*table)
		if err != nil {
			return err
		}
		if err := dataextract.ScaleInputsByColumnMax(table, stats); err != nil {
			return err
		}
	}
	if opts.ScaleAsinh {
		dataextract.ScaleInputsAsinh(table)
	}
	if opts.Binarize {
		dataextract.BinarizeInputs(table)
	}
	if opts.CleanZeroInputs {
		dataextract.CleanZeroInputRows(table)
	}
	return nil
}

func printTableStats(table dataextract.TableFile) error {
	stats, err := dataextract.InputColumnStats(table)
	if err != nil {
		return err
	}
	for i, stat := range stats {
		fmt.Printf("table_stats col=%d min=%g avg=%g max=%g\n", i, stat.Min, stat.Avg, stat.Max)
	}
	return nil
}
