package tuning

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"sync"

	"protogonos/internal/model"
)

type Exoself struct {
	Rand               *rand.Rand
	Steps              int
	StepSize           float64
	PerturbationRange  float64
	AnnealingFactor    float64
	MinImprovement     float64
	GoalFitness        float64
	CandidateSelection string
	mu                 sync.Mutex
}

const (
	CandidateSelectBestSoFar = "best_so_far"
	CandidateSelectOriginal  = "original"
	CandidateSelectDynamicA  = "dynamic"
	CandidateSelectDynamic   = "dynamic_random"
	CandidateSelectAll       = "all"
	CandidateSelectAllRandom = "all_random"
	CandidateSelectActive    = "active"
	CandidateSelectActiveRnd = "active_random"
	CandidateSelectRecent    = "recent"
	CandidateSelectRecentRnd = "recent_random"
	CandidateSelectCurrent   = "current"
	CandidateSelectCurrentRd = "current_random"
	CandidateSelectLastGen   = "lastgen"
	CandidateSelectLastGenRd = "lastgen_random"
)

func (e *Exoself) Name() string {
	return "exoself_hillclimb"
}

func (e *Exoself) SetGoalFitness(goal float64) {
	e.GoalFitness = goal
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
	if e.PerturbationRange < 0 {
		return model.Genome{}, errors.New("perturbation range must be >= 0")
	}
	if e.AnnealingFactor < 0 {
		return model.Genome{}, errors.New("annealing factor must be >= 0")
	}
	if e.MinImprovement < 0 {
		return model.Genome{}, errors.New("min improvement must be >= 0")
	}
	if fitness == nil {
		return model.Genome{}, errors.New("fitness function is required")
	}
	if len(genome.Synapses) == 0 {
		return cloneGenome(genome), nil
	}
	perturbationRange := e.PerturbationRange
	if perturbationRange == 0 {
		perturbationRange = 1.0
	}
	annealingFactor := e.AnnealingFactor
	if annealingFactor == 0 {
		annealingFactor = 1.0
	}

	best := cloneGenome(genome)
	bestFitness, err := fitness(ctx, best)
	if err != nil {
		return model.Genome{}, err
	}
	if e.GoalFitness > 0 && bestFitness >= e.GoalFitness {
		return best, nil
	}
	recentBase := cloneGenome(best)

	for a := 0; a < attempts; a++ {
		bases, err := e.candidateBases(best, genome, recentBase)
		if err != nil {
			return model.Genome{}, err
		}
		localBest := cloneGenome(best)
		localBestFitness := bestFitness
		for _, base := range bases {
			candidate, err := e.perturbCandidate(ctx, base, perturbationRange, annealingFactor)
			if err != nil {
				return model.Genome{}, err
			}
			candidateFitness, err := fitness(ctx, candidate)
			if err != nil {
				return model.Genome{}, err
			}
			if candidateFitness > localBestFitness+e.MinImprovement {
				localBest = candidate
				localBestFitness = candidateFitness
			}
		}
		recentBase = cloneGenome(localBest)
		if localBestFitness > bestFitness+e.MinImprovement {
			best = localBest
			bestFitness = localBestFitness
		}
		if e.GoalFitness > 0 && bestFitness >= e.GoalFitness {
			break
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

func NormalizeCandidateSelectionName(name string) string {
	switch name {
	case "", CandidateSelectBestSoFar:
		return CandidateSelectBestSoFar
	case CandidateSelectOriginal:
		return CandidateSelectOriginal
	case CandidateSelectDynamicA:
		return CandidateSelectDynamicA
	case CandidateSelectDynamic:
		return CandidateSelectDynamic
	case CandidateSelectAll:
		return CandidateSelectAll
	case CandidateSelectAllRandom:
		return CandidateSelectAllRandom
	case CandidateSelectActive:
		return CandidateSelectActive
	case CandidateSelectActiveRnd:
		return CandidateSelectActiveRnd
	case CandidateSelectRecent:
		return CandidateSelectRecent
	case CandidateSelectRecentRnd:
		return CandidateSelectRecentRnd
	case CandidateSelectCurrent:
		return CandidateSelectCurrent
	case CandidateSelectCurrentRd:
		return CandidateSelectCurrentRd
	case CandidateSelectLastGen:
		return CandidateSelectLastGen
	case CandidateSelectLastGenRd:
		return CandidateSelectLastGenRd
	default:
		return name
	}
}

func (e *Exoself) candidateBases(best, original, recent model.Genome) ([]model.Genome, error) {
	mode := NormalizeCandidateSelectionName(e.CandidateSelection)
	if isRandomSelection(mode) {
		baseMode := nonRandomModeFor(mode)
		pool, err := e.candidateBasesForMode(baseMode, best, original, recent)
		if err != nil {
			return nil, err
		}
		return e.randomSubset(pool), nil
	}
	return e.candidateBasesForMode(mode, best, original, recent)
}

func (e *Exoself) candidateBasesForMode(mode string, best, original, recent model.Genome) ([]model.Genome, error) {
	switch mode {
	case CandidateSelectBestSoFar:
		return []model.Genome{cloneGenome(best)}, nil
	case CandidateSelectOriginal, CandidateSelectLastGen:
		return []model.Genome{cloneGenome(original)}, nil
	case CandidateSelectDynamicA:
		return []model.Genome{cloneGenome(best), cloneGenome(original)}, nil
	case CandidateSelectActive, CandidateSelectRecent:
		return []model.Genome{cloneGenome(recent)}, nil
	case CandidateSelectCurrent, CandidateSelectAll:
		return []model.Genome{cloneGenome(best), cloneGenome(original), cloneGenome(recent)}, nil
	default:
		return nil, errors.New("unsupported candidate selection")
	}
}

func isRandomSelection(mode string) bool {
	switch mode {
	case CandidateSelectDynamic, CandidateSelectAllRandom, CandidateSelectActiveRnd, CandidateSelectRecentRnd, CandidateSelectCurrentRd, CandidateSelectLastGenRd:
		return true
	default:
		return false
	}
}

func nonRandomModeFor(mode string) string {
	switch mode {
	case CandidateSelectDynamic:
		return CandidateSelectDynamicA
	case CandidateSelectAllRandom:
		return CandidateSelectAll
	case CandidateSelectActiveRnd:
		return CandidateSelectActive
	case CandidateSelectRecentRnd:
		return CandidateSelectRecent
	case CandidateSelectCurrentRd:
		return CandidateSelectCurrent
	case CandidateSelectLastGenRd:
		return CandidateSelectLastGen
	default:
		return mode
	}
}

func (e *Exoself) randomSubset(pool []model.Genome) []model.Genome {
	if len(pool) <= 1 {
		return pool
	}
	mutationP := 1 / math.Sqrt(float64(len(pool)))
	chosen := make([]model.Genome, 0, len(pool))
	for i := range pool {
		if e.randFloat64() < mutationP {
			chosen = append(chosen, cloneGenome(pool[i]))
		}
	}
	if len(chosen) > 0 {
		return chosen
	}
	return []model.Genome{cloneGenome(pool[e.randIntn(len(pool))])}
}

func (e *Exoself) perturbCandidate(ctx context.Context, base model.Genome, perturbationRange, annealingFactor float64) (model.Genome, error) {
	candidate := cloneGenome(base)
	for s := 0; s < e.Steps; s++ {
		if err := ctx.Err(); err != nil {
			return model.Genome{}, err
		}
		if len(candidate.Synapses) == 0 {
			break
		}
		idx := e.randIntn(len(candidate.Synapses))
		spread := e.StepSize * perturbationRange * math.Pow(annealingFactor, float64(s))
		delta := (e.randFloat64()*2 - 1) * spread
		candidate.Synapses[idx].Weight += delta
	}
	return candidate, nil
}
