package tuning

import (
	"context"
	"errors"
	"math/rand"
	"sync"

	"protogonos/internal/model"
)

type Exoself struct {
	Rand     *rand.Rand
	Steps    int
	StepSize float64
	mu       sync.Mutex
}

func (e *Exoself) Name() string {
	return "exoself_hillclimb"
}

func (e *Exoself) Tune(ctx context.Context, genome model.Genome, attempts int, fitness FitnessFn) (model.Genome, error) {
	if err := ctx.Err(); err != nil {
		return model.Genome{}, err
	}
	if e == nil || e.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if attempts <= 0 {
		return cloneGenome(genome), nil
	}
	if e.Steps <= 0 {
		return model.Genome{}, errors.New("steps must be > 0")
	}
	if e.StepSize <= 0 {
		return model.Genome{}, errors.New("step size must be > 0")
	}
	if fitness == nil {
		return model.Genome{}, errors.New("fitness function is required")
	}
	if len(genome.Synapses) == 0 {
		return cloneGenome(genome), nil
	}

	best := cloneGenome(genome)
	bestFitness, err := fitness(ctx, best)
	if err != nil {
		return model.Genome{}, err
	}

	for a := 0; a < attempts; a++ {
		candidate := cloneGenome(best)
		for s := 0; s < e.Steps; s++ {
			if err := ctx.Err(); err != nil {
				return model.Genome{}, err
			}
			if len(candidate.Synapses) == 0 {
				break
			}
			idx := e.randIntn(len(candidate.Synapses))
			delta := (e.randFloat64()*2 - 1) * e.StepSize
			candidate.Synapses[idx].Weight += delta
		}

		candidateFitness, err := fitness(ctx, candidate)
		if err != nil {
			return model.Genome{}, err
		}
		if candidateFitness > bestFitness {
			best = candidate
			bestFitness = candidateFitness
		}
	}

	return best, nil
}

func (e *Exoself) randIntn(n int) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.Rand.Intn(n)
}

func (e *Exoself) randFloat64() float64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.Rand.Float64()
}

func cloneGenome(g model.Genome) model.Genome {
	out := g
	out.Neurons = append([]model.Neuron(nil), g.Neurons...)
	out.Synapses = append([]model.Synapse(nil), g.Synapses...)
	out.SensorIDs = append([]string(nil), g.SensorIDs...)
	out.ActuatorIDs = append([]string(nil), g.ActuatorIDs...)
	return out
}
