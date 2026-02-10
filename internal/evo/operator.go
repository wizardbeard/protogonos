package evo

import (
	"context"

	"protogonos/internal/model"
)

type Operator interface {
	Name() string
	Apply(ctx context.Context, genome model.Genome) (model.Genome, error)
}
