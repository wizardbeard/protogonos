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
	SaveFitnessHistory(ctx context.Context, runID string, history []float64) error
	GetFitnessHistory(ctx context.Context, runID string) ([]float64, bool, error)
	SaveLineage(ctx context.Context, runID string, lineage []model.LineageRecord) error
	GetLineage(ctx context.Context, runID string) ([]model.LineageRecord, bool, error)
}
