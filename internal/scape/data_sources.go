package scape

import (
	"context"
	"fmt"
	"strings"
)

// DataSources configures optional per-run dataset sources.
// When a CSV path is unset, the scape uses its default source.
type DataSources struct {
	GTSA     GTSADataSource
	FX       FXDataSource
	Epitopes EpitopesDataSource
	LLVM     LLVMDataSource
}

// GTSADataSource configures an optional GTSA CSV table and bounds.
type GTSADataSource struct {
	CSVPath string
	Bounds  GTSATableBounds
}

// FXDataSource configures an optional FX CSV price series.
type FXDataSource struct {
	CSVPath string
}

// EpitopesDataSource configures an optional epitopes CSV table and windows.
type EpitopesDataSource struct {
	CSVPath   string
	TableName string
	Bounds    EpitopesTableBounds
}

// LLVMDataSource configures an optional LLVM workflow JSON file.
type LLVMDataSource struct {
	WorkflowJSONPath string
}

// WithDataSources returns a context carrying optional per-run dataset overrides.
func WithDataSources(ctx context.Context, sources DataSources) (context.Context, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	if strings.TrimSpace(sources.GTSA.CSVPath) != "" {
		table, err := loadGTSATableCSV(sources.GTSA.CSVPath, sources.GTSA.Bounds)
		if err != nil {
			return nil, fmt.Errorf("configure gtsa data source: %w", err)
		}
		ctx = context.WithValue(ctx, gtsaDataSourceContextKey{}, table)
	}

	if strings.TrimSpace(sources.FX.CSVPath) != "" {
		series, err := loadFXSeriesCSV(sources.FX.CSVPath)
		if err != nil {
			return nil, fmt.Errorf("configure fx data source: %w", err)
		}
		ctx = context.WithValue(ctx, fxDataSourceContextKey{}, series)
	}

	epitopesCSVPath := strings.TrimSpace(sources.Epitopes.CSVPath)
	epitopesTableName := strings.TrimSpace(sources.Epitopes.TableName)
	switch {
	case epitopesCSVPath != "":
		source, err := loadEpitopesSourceCSVWithName(epitopesCSVPath, sources.Epitopes.Bounds, epitopesTableName)
		if err != nil {
			return nil, fmt.Errorf("configure epitopes data source: %w", err)
		}
		ctx = context.WithValue(ctx, epitopesDataSourceContextKey{}, source)
	case epitopesTableName != "" || hasAnyEpitopesBounds(sources.Epitopes.Bounds):
		source, err := loadDefaultEpitopesSource(epitopesTableName, sources.Epitopes.Bounds)
		if err != nil {
			return nil, fmt.Errorf("configure epitopes data source: %w", err)
		}
		ctx = context.WithValue(ctx, epitopesDataSourceContextKey{}, source)
	}
	if strings.TrimSpace(sources.LLVM.WorkflowJSONPath) != "" {
		workflow, err := loadLLVMWorkflowJSON(sources.LLVM.WorkflowJSONPath)
		if err != nil {
			return nil, fmt.Errorf("configure llvm workflow source: %w", err)
		}
		ctx = context.WithValue(ctx, llvmDataSourceContextKey{}, workflow)
	}

	return ctx, nil
}

type gtsaDataSourceContextKey struct{}

func gtsaTableFromContext(ctx context.Context) (gtsaTable, bool) {
	if ctx == nil {
		return gtsaTable{}, false
	}
	table, ok := ctx.Value(gtsaDataSourceContextKey{}).(gtsaTable)
	if !ok || len(table.values) <= 1 {
		return gtsaTable{}, false
	}
	return table, true
}

type fxDataSourceContextKey struct{}

func fxSeriesFromContext(ctx context.Context) (fxSeries, bool) {
	if ctx == nil {
		return fxSeries{}, false
	}
	series, ok := ctx.Value(fxDataSourceContextKey{}).(fxSeries)
	if !ok || len(series.values) == 0 {
		return fxSeries{}, false
	}
	return series, true
}

type epitopesDataSourceContextKey struct{}

func epitopesSourceFromContext(ctx context.Context) (epitopesSource, bool) {
	if ctx == nil {
		return epitopesSource{}, false
	}
	source, ok := ctx.Value(epitopesDataSourceContextKey{}).(epitopesSource)
	if !ok || len(source.table.rows) <= 1 {
		return epitopesSource{}, false
	}
	return source, true
}

type llvmDataSourceContextKey struct{}

func llvmWorkflowFromContext(ctx context.Context) (llvmWorkflow, bool) {
	if ctx == nil {
		return llvmWorkflow{}, false
	}
	workflow, ok := ctx.Value(llvmDataSourceContextKey{}).(llvmWorkflow)
	if !ok || len(workflow.optimizations) == 0 {
		return llvmWorkflow{}, false
	}
	return workflow, true
}
