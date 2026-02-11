package storage

import (
	"context"

	"protogonos/internal/model"
)

// Store defines transaction-like persistence operations for core DXNN entities.
type Store interface {
	Init(ctx context.Context) error
	SaveGenome(ctx context.Context, genome model.Genome) error
	GetGenome(ctx context.Context, id string) (model.Genome, bool, error)
	SavePopulation(ctx context.Context, population model.Population) error
	GetPopulation(ctx context.Context, id string) (model.Population, bool, error)
	SaveScapeSummary(ctx context.Context, summary model.ScapeSummary) error
	GetScapeSummary(ctx context.Context, name string) (model.ScapeSummary, bool, error)
	SaveFitnessHistory(ctx context.Context, runID string, history []float64) error
	GetFitnessHistory(ctx context.Context, runID string) ([]float64, bool, error)
	SaveGenerationDiagnostics(ctx context.Context, runID string, diagnostics []model.GenerationDiagnostics) error
	GetGenerationDiagnostics(ctx context.Context, runID string) ([]model.GenerationDiagnostics, bool, error)
	SaveTopGenomes(ctx context.Context, runID string, top []model.TopGenomeRecord) error
	GetTopGenomes(ctx context.Context, runID string) ([]model.TopGenomeRecord, bool, error)
	SaveLineage(ctx context.Context, runID string, lineage []model.LineageRecord) error
	GetLineage(ctx context.Context, runID string) ([]model.LineageRecord, bool, error)
}
