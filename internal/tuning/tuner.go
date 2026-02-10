package tuning

import (
	"context"

	"protogonos/internal/model"
)

type FitnessFn func(ctx context.Context, genome model.Genome) (float64, error)

type Tuner interface {
	Name() string
	Tune(ctx context.Context, genome model.Genome, attempts int, fitness FitnessFn) (model.Genome, error)
}
