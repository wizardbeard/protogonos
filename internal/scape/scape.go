package scape

import "context"

type Fitness float64

type Trace map[string]any

type Agent interface {
	ID() string
}

type TickAgent interface {
	Agent
	Tick(ctx context.Context) ([]float64, error)
}

type Scape interface {
	Name() string
	Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error)
}
