package tuning

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"strconv"
	"strings"
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
	tuned, _, err := e.TuneWithReport(ctx, genome, attempts, fitness)
	return tuned, err
}

func (e *Exoself) TuneWithReport(ctx context.Context, genome model.Genome, attempts int, fitness FitnessFn) (model.Genome, TuneReport, error) {
	report := TuneReport{AttemptsPlanned: attempts}
	if err := ctx.Err(); err != nil {
		return model.Genome{}, report, err
	}
	if e == nil || e.Rand == nil {
		return model.Genome{}, report, errors.New("random source is required")
	}
	if attempts <= 0 {
		return cloneGenome(genome), report, nil
	}
	if e.Steps <= 0 {
		return model.Genome{}, report, errors.New("steps must be > 0")
	}
	if e.StepSize <= 0 {
		return model.Genome{}, report, errors.New("step size must be > 0")
	}
	if e.PerturbationRange < 0 {
		return model.Genome{}, report, errors.New("perturbation range must be >= 0")
	}
	if e.AnnealingFactor < 0 {
		return model.Genome{}, report, errors.New("annealing factor must be >= 0")
	}
	if e.MinImprovement < 0 {
		return model.Genome{}, report, errors.New("min improvement must be >= 0")
	}
	if fitness == nil {
		return model.Genome{}, report, errors.New("fitness function is required")
	}
	if len(genome.Synapses) == 0 {
		return cloneGenome(genome), report, nil
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
		return model.Genome{}, report, err
	}
	report.CandidateEvaluations++
	if e.GoalFitness > 0 && bestFitness >= e.GoalFitness {
		report.GoalReached = true
		return best, report, nil
	}
	recentBase := cloneGenome(best)

	for a := 0; a < attempts; a++ {
		report.AttemptsExecuted++
		bases, err := e.candidateBases(best, genome, recentBase)
		if err != nil {
			return model.Genome{}, report, err
		}
		localBest := cloneGenome(best)
		localBestFitness := bestFitness
		for _, base := range bases {
			candidate, err := e.perturbCandidate(ctx, base, perturbationRange, annealingFactor)
			if err != nil {
				return model.Genome{}, report, err
			}
			candidateFitness, err := fitness(ctx, candidate)
			if err != nil {
				return model.Genome{}, report, err
			}
			report.CandidateEvaluations++
			if candidateFitness > localBestFitness+e.MinImprovement {
				report.AcceptedCandidates++
				localBest = candidate
				localBestFitness = candidateFitness
			} else {
				report.RejectedCandidates++
			}
		}
		recentBase = cloneGenome(localBest)
		if localBestFitness > bestFitness+e.MinImprovement {
			best = localBest
			bestFitness = localBestFitness
		}
		if e.GoalFitness > 0 && bestFitness >= e.GoalFitness {
			report.GoalReached = true
			break
		}
	}

	return best, report, nil
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
	candidates := uniqueCandidatePool(best, original, recent)
	if len(candidates) == 0 {
		return nil, errors.New("empty candidate pool")
	}
	switch mode {
	case CandidateSelectBestSoFar:
		return []model.Genome{cloneGenome(best)}, nil
	case CandidateSelectOriginal:
		return []model.Genome{cloneGenome(original)}, nil
	case CandidateSelectLastGen:
		return filterCandidatesByAge(candidates, 0), nil
	case CandidateSelectDynamicA:
		limit := dynamicAgeLimit(e.randFloat64())
		return filterCandidatesByAge(candidates, limit), nil
	case CandidateSelectActive, CandidateSelectRecent:
		return filterCandidatesByAge(candidates, 3), nil
	case CandidateSelectCurrent, CandidateSelectAll:
		return filterCandidatesByAge(candidates, 0), nil
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
		return CandidateSelectCurrent
	default:
		return mode
	}
}

func dynamicAgeLimit(u float64) float64 {
	// Mirror tuning_selection.erl dynamic age-limit shape: sqrt(1/U).
	if u <= 0 {
		u = math.SmallestNonzeroFloat64
	}
	return math.Sqrt(1 / u)
}

func uniqueCandidatePool(best, original, recent model.Genome) []model.Genome {
	seen := map[string]struct{}{}
	out := make([]model.Genome, 0, 3)
	for _, g := range []model.Genome{best, original, recent} {
		key := g.ID
		if key == "" {
			key = strconv.Itoa(len(out))
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, cloneGenome(g))
	}
	return out
}

func filterCandidatesByAge(pool []model.Genome, maxAge float64) []model.Genome {
	if len(pool) == 0 {
		return nil
	}
	currentGen := 0
	knownCurrent := false
	for _, g := range pool {
		gen, ok := inferGenomeGeneration(g.ID)
		if !ok {
			continue
		}
		if !knownCurrent || gen > currentGen {
			currentGen = gen
			knownCurrent = true
		}
	}
	filtered := make([]model.Genome, 0, len(pool))
	for _, g := range pool {
		gen, ok := inferGenomeGeneration(g.ID)
		if !knownCurrent || !ok {
			filtered = append(filtered, cloneGenome(g))
			continue
		}
		age := currentGen - gen
		if float64(age) <= maxAge {
			filtered = append(filtered, cloneGenome(g))
		}
	}
	if len(filtered) > 0 {
		return filtered
	}
	return []model.Genome{cloneGenome(pool[0])}
}

func inferGenomeGeneration(id string) (int, bool) {
	if id == "" {
		return 0, false
	}
	parts := strings.Split(id, "-")
	for _, part := range parts {
		if len(part) > 1 && part[0] == 'g' {
			if gen, err := strconv.Atoi(part[1:]); err == nil {
				return gen, true
			}
		}
	}
	return 0, false
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
