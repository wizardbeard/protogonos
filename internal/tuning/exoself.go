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

	consecutiveNoImprovement := 0
	for consecutiveNoImprovement < attempts {
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
			if scalarFitnessDominates(candidateFitness, localBestFitness, e.MinImprovement) {
				report.AcceptedCandidates++
				localBest = candidate
				localBestFitness = candidateFitness
			} else {
				report.RejectedCandidates++
			}
		}
		recentBase = cloneGenome(localBest)
		improved := scalarFitnessDominates(localBestFitness, bestFitness, e.MinImprovement)
		if improved {
			best = localBest
			bestFitness = localBestFitness
		}
		if improved {
			consecutiveNoImprovement = 0
		} else {
			consecutiveNoImprovement++
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
	if g.ActuatorTunables != nil {
		out.ActuatorTunables = make(map[string]float64, len(g.ActuatorTunables))
		for k, v := range g.ActuatorTunables {
			out.ActuatorTunables[k] = v
		}
	}
	if g.ActuatorGenerations != nil {
		out.ActuatorGenerations = make(map[string]int, len(g.ActuatorGenerations))
		for k, v := range g.ActuatorGenerations {
			out.ActuatorGenerations[k] = v
		}
	}
	out.SensorNeuronLinks = append([]model.SensorNeuronLink(nil), g.SensorNeuronLinks...)
	out.NeuronActuatorLinks = append([]model.NeuronActuatorLink(nil), g.NeuronActuatorLinks...)
	if g.Substrate != nil {
		sub := *g.Substrate
		sub.Dimensions = append([]int(nil), g.Substrate.Dimensions...)
		if g.Substrate.Parameters != nil {
			sub.Parameters = make(map[string]float64, len(g.Substrate.Parameters))
			for k, v := range g.Substrate.Parameters {
				sub.Parameters[k] = v
			}
		}
		out.Substrate = &sub
	}
	if g.Plasticity != nil {
		p := *g.Plasticity
		out.Plasticity = &p
	}
	if g.Strategy != nil {
		s := *g.Strategy
		out.Strategy = &s
	}
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
	case CandidateSelectAll:
		return cloneCandidatePool(candidates), nil
	case CandidateSelectLastGen:
		return filterCandidatesByAge(candidates, 0), nil
	case CandidateSelectDynamicA:
		limit := dynamicAgeLimit(e.randFloat64())
		return filterCandidatesByAge(candidates, limit), nil
	case CandidateSelectActive, CandidateSelectRecent:
		return filterCandidatesByAge(candidates, 3), nil
	case CandidateSelectCurrent:
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

func cloneCandidatePool(pool []model.Genome) []model.Genome {
	out := make([]model.Genome, 0, len(pool))
	for i := range pool {
		out = append(out, cloneGenome(pool[i]))
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
	targets := e.selectedNeuronPerturbTargets(candidate, perturbationRange, annealingFactor)
	if len(targets) == 0 {
		return candidate, nil
	}
	currentGeneration := currentGenomeGeneration(candidate)
	for s := 0; s < e.Steps; s++ {
		if err := ctx.Err(); err != nil {
			return model.Genome{}, err
		}
		target := targets[e.randIntn(len(targets))]
		if target.sourceKind == tuningElementActuator {
			spread := e.StepSize * target.spread
			if spread <= 0 {
				continue
			}
			perturbActuatorTunable(&candidate, target.sourceID, spread, e.randFloat64)
			touchActuatorGeneration(&candidate, target.sourceID, currentGeneration)
			continue
		}
		if len(candidate.Synapses) == 0 || target.neuronID == "" {
			continue
		}
		incoming := incomingSynapseIndexes(candidate, target.neuronID)
		if len(incoming) == 0 {
			continue
		}
		idx := incoming[e.randIntn(len(incoming))]
		spread := e.StepSize * target.spread
		delta := (e.randFloat64()*2 - 1) * spread
		candidate.Synapses[idx].Weight += delta
		touchNeuronGeneration(candidate.Neurons, target.neuronID, currentGeneration)
	}
	return candidate, nil
}

type neuronPerturbTarget struct {
	neuronID   string
	spread     float64
	sourceKind string
	sourceID   string
	generation int
}

type tuningElementCandidate struct {
	kind       string
	id         string
	generation int
}

const (
	tuningElementNeuron   = "neuron"
	tuningElementActuator = "actuator"
)

func (e *Exoself) selectedNeuronPerturbTargets(
	genome model.Genome,
	perturbationRange float64,
	annealingFactor float64,
) []neuronPerturbTarget {
	if len(genome.Neurons) == 0 && len(genome.ActuatorIDs) == 0 {
		return nil
	}
	if perturbationRange <= 0 {
		perturbationRange = 1.0
	}
	if annealingFactor <= 0 {
		annealingFactor = 1.0
	}

	mode := NormalizeCandidateSelectionName(e.CandidateSelection)
	currentGeneration := currentGenomeGeneration(genome)
	candidates := tuningElementsForGenome(genome, currentGeneration)
	selected := filterTuningElementsByMode(candidates, nonRandomModeFor(mode), currentGeneration, e.randFloat64)
	targets := perturbTargetsFromElements(genome, selected, currentGeneration, perturbationRange, annealingFactor)
	if len(targets) == 0 && shouldFallbackToFirstTuningTarget(mode) {
		targets = fallbackNeuronTargetsFromCandidates(genome, candidates, currentGeneration, perturbationRange*math.Pi)
	}
	if len(targets) == 0 {
		return nil
	}
	if isRandomSelection(mode) {
		return e.randomNeuronTargetSubset(targets)
	}
	return targets
}

func shouldFallbackToFirstTuningTarget(mode string) bool {
	switch mode {
	case CandidateSelectDynamicA,
		CandidateSelectDynamic,
		CandidateSelectActiveRnd,
		CandidateSelectRecentRnd,
		CandidateSelectCurrent,
		CandidateSelectCurrentRd,
		CandidateSelectLastGen,
		CandidateSelectLastGenRd,
		CandidateSelectBestSoFar,
		CandidateSelectOriginal:
		return true
	default:
		return false
	}
}

func fallbackNeuronTargetsFromCandidates(
	genome model.Genome,
	candidates []tuningElementCandidate,
	currentGeneration int,
	spread float64,
) []neuronPerturbTarget {
	for _, candidate := range candidates {
		target := neuronPerturbTarget{
			spread:     spread,
			sourceKind: candidate.kind,
			sourceID:   candidate.id,
			generation: candidate.generation,
		}
		switch candidate.kind {
		case tuningElementNeuron:
			if candidate.id == "" || !hasNeuron(genome, candidate.id) {
				continue
			}
			target.neuronID = candidate.id
		case tuningElementActuator:
			if candidate.id == "" || !hasActuator(genome, candidate.id) {
				continue
			}
		default:
			continue
		}
		return []neuronPerturbTarget{target}
	}
	if len(genome.Neurons) > 0 {
		fallback := genome.Neurons[0]
		return []neuronPerturbTarget{{
			neuronID:   fallback.ID,
			spread:     spread,
			sourceKind: tuningElementNeuron,
			sourceID:   fallback.ID,
			generation: effectiveNeuronGeneration(fallback, currentGeneration),
		}}
	}
	if len(genome.ActuatorIDs) == 0 {
		return nil
	}
	fallback := genome.ActuatorIDs[0]
	return []neuronPerturbTarget{{
		neuronID:   "",
		spread:     spread,
		sourceKind: tuningElementActuator,
		sourceID:   fallback,
		generation: effectiveActuatorGeneration(genome, fallback, currentGeneration),
	}}
}

func tuningElementsForGenome(genome model.Genome, currentGeneration int) []tuningElementCandidate {
	out := make([]tuningElementCandidate, 0, len(genome.Neurons)+len(genome.ActuatorIDs))
	for _, neuron := range genome.Neurons {
		out = append(out, tuningElementCandidate{
			kind:       tuningElementNeuron,
			id:         neuron.ID,
			generation: effectiveNeuronGeneration(neuron, currentGeneration),
		})
	}
	for _, actuatorID := range uniqueStrings(genome.ActuatorIDs) {
		if actuatorID == "" {
			continue
		}
		out = append(out, tuningElementCandidate{
			kind:       tuningElementActuator,
			id:         actuatorID,
			generation: effectiveActuatorGeneration(genome, actuatorID, currentGeneration),
		})
	}
	return out
}

func filterTuningElementsByMode(
	candidates []tuningElementCandidate,
	mode string,
	currentGeneration int,
	randFloat64 func() float64,
) []tuningElementCandidate {
	if len(candidates) == 0 {
		return nil
	}
	switch mode {
	case CandidateSelectDynamicA:
		u := randFloat64()
		return filterTuningElementsByAge(candidates, currentGeneration, dynamicAgeLimit(u))
	case CandidateSelectActive, CandidateSelectRecent:
		return filterTuningElementsByAge(candidates, currentGeneration, 3)
	case CandidateSelectCurrent, CandidateSelectLastGen:
		return filterTuningElementsByAge(candidates, currentGeneration, 0)
	case CandidateSelectAll, CandidateSelectBestSoFar, CandidateSelectOriginal:
		return append([]tuningElementCandidate(nil), candidates...)
	default:
		return append([]tuningElementCandidate(nil), candidates...)
	}
}

func filterTuningElementsByAge(candidates []tuningElementCandidate, currentGeneration int, maxAge float64) []tuningElementCandidate {
	filtered := make([]tuningElementCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		age := currentGeneration - candidate.generation
		if age < 0 {
			age = 0
		}
		if float64(age) <= maxAge {
			filtered = append(filtered, candidate)
		}
	}
	return filtered
}

func perturbTargetsFromElements(
	genome model.Genome,
	selected []tuningElementCandidate,
	currentGeneration int,
	perturbationRange float64,
	annealingFactor float64,
) []neuronPerturbTarget {
	out := make([]neuronPerturbTarget, 0, len(selected))
	for _, candidate := range selected {
		age := currentGeneration - candidate.generation
		if age < 0 {
			age = 0
		}
		spread := perturbationRange * math.Pi * math.Pow(annealingFactor, float64(age))
		if spread <= 0 {
			spread = perturbationRange * math.Pi
		}
		target := neuronPerturbTarget{
			spread:     spread,
			sourceKind: candidate.kind,
			sourceID:   candidate.id,
			generation: candidate.generation,
		}
		switch candidate.kind {
		case tuningElementNeuron:
			if candidate.id == "" || !hasNeuron(genome, candidate.id) {
				continue
			}
			target.neuronID = candidate.id
		case tuningElementActuator:
			if candidate.id == "" || !hasActuator(genome, candidate.id) {
				continue
			}
		default:
			continue
		}
		out = append(out, target)
	}
	return out
}

func currentGenomeGeneration(genome model.Genome) int {
	if gen, ok := inferGenomeGeneration(genome.ID); ok {
		return gen
	}
	maxGen := 0
	for _, neuron := range genome.Neurons {
		if neuron.Generation > maxGen {
			maxGen = neuron.Generation
		}
	}
	for _, actuatorGen := range genome.ActuatorGenerations {
		if actuatorGen > maxGen {
			maxGen = actuatorGen
		}
	}
	for _, actuatorID := range genome.ActuatorIDs {
		if gen, ok := inferGenomeGeneration(actuatorID); ok && gen > maxGen {
			maxGen = gen
		}
	}
	return maxGen
}

func effectiveNeuronGeneration(neuron model.Neuron, fallback int) int {
	if neuron.Generation > 0 {
		return neuron.Generation
	}
	if gen, ok := inferGenomeGeneration(neuron.ID); ok {
		return gen
	}
	return fallback
}

func effectiveActuatorGeneration(genome model.Genome, actuatorID string, fallback int) int {
	if genome.ActuatorGenerations != nil {
		if generation, ok := genome.ActuatorGenerations[actuatorID]; ok && generation > 0 {
			return generation
		}
	}
	if gen, ok := inferGenomeGeneration(actuatorID); ok {
		return gen
	}
	return fallback
}

func (e *Exoself) randomNeuronTargetSubset(targets []neuronPerturbTarget) []neuronPerturbTarget {
	if len(targets) <= 1 {
		return append([]neuronPerturbTarget(nil), targets...)
	}
	mutationP := 1 / math.Sqrt(float64(len(targets)))
	chosen := make([]neuronPerturbTarget, 0, len(targets))
	for i := range targets {
		if e.randFloat64() < mutationP {
			chosen = append(chosen, targets[i])
		}
	}
	if len(chosen) > 0 {
		return chosen
	}
	return []neuronPerturbTarget{targets[e.randIntn(len(targets))]}
}

func hasNeuron(genome model.Genome, neuronID string) bool {
	for _, neuron := range genome.Neurons {
		if neuron.ID == neuronID {
			return true
		}
	}
	return false
}

func hasActuator(genome model.Genome, actuatorID string) bool {
	for _, id := range genome.ActuatorIDs {
		if id == actuatorID {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func incomingSynapseIndexes(genome model.Genome, neuronID string) []int {
	indexes := make([]int, 0, len(genome.Synapses))
	for i, syn := range genome.Synapses {
		if syn.To == neuronID {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

func perturbActuatorTunable(genome *model.Genome, actuatorID string, spread float64, randFloat64 func() float64) {
	if genome == nil || actuatorID == "" || spread <= 0 || randFloat64 == nil {
		return
	}
	if genome.ActuatorTunables == nil {
		genome.ActuatorTunables = map[string]float64{}
	}
	delta := (randFloat64()*2 - 1) * spread
	genome.ActuatorTunables[actuatorID] += delta
}

func touchNeuronGeneration(neurons []model.Neuron, neuronID string, generation int) {
	if generation < 0 {
		generation = 0
	}
	for i := range neurons {
		if neurons[i].ID != neuronID {
			continue
		}
		neurons[i].Generation = generation
		return
	}
}

func touchActuatorGeneration(genome *model.Genome, actuatorID string, generation int) {
	if genome == nil || actuatorID == "" {
		return
	}
	if generation < 0 {
		generation = 0
	}
	if genome.ActuatorGenerations == nil {
		genome.ActuatorGenerations = map[string]int{}
	}
	genome.ActuatorGenerations[actuatorID] = generation
}

func scalarFitnessDominates(candidate, incumbent, minImprovement float64) bool {
	if candidate <= incumbent {
		return false
	}
	threshold := incumbent + incumbent*minImprovement
	return candidate > threshold
}

func vectorFitnessDominates(candidate, incumbent []float64, minImprovement float64) bool {
	if len(candidate) == 0 || len(candidate) != len(incumbent) {
		return false
	}
	for i := range candidate {
		if candidate[i] <= incumbent[i] {
			return false
		}
		threshold := incumbent[i] + incumbent[i]*minImprovement
		if candidate[i] <= threshold {
			return false
		}
	}
	return true
}

func transposeVectors(vectors [][]float64) [][]float64 {
	if len(vectors) == 0 {
		return nil
	}
	minLen := -1
	for _, vector := range vectors {
		if minLen == -1 || len(vector) < minLen {
			minLen = len(vector)
		}
	}
	if minLen <= 0 {
		return nil
	}
	transposed := make([][]float64, minLen)
	for i := 0; i < minLen; i++ {
		column := make([]float64, 0, len(vectors))
		for _, vector := range vectors {
			column = append(column, vector[i])
		}
		transposed[i] = column
	}
	return transposed
}

func vectorAvg(vectors [][]float64) []float64 {
	transposed := transposeVectors(vectors)
	if len(transposed) == 0 {
		return nil
	}
	averages := make([]float64, len(transposed))
	for i, column := range transposed {
		if len(column) == 0 {
			continue
		}
		total := 0.0
		for _, value := range column {
			total += value
		}
		averages[i] = total / float64(len(column))
	}
	return averages
}

func vectorBasicStats(vectors [][]float64) (max []float64, min []float64, avg []float64, std []float64) {
	if len(vectors) == 0 {
		return nil, nil, nil, nil
	}

	avg = vectorAvg(vectors)
	transposed := transposeVectors(vectors)
	if len(transposed) == 0 {
		return cloneFloatSlice(vectors[0]), cloneFloatSlice(vectors[0]), nil, nil
	}
	std = make([]float64, len(transposed))
	for i, column := range transposed {
		if len(column) == 0 {
			continue
		}
		mean := avg[i]
		sumSq := 0.0
		for _, value := range column {
			diff := mean - value
			sumSq += diff * diff
		}
		std[i] = math.Sqrt(sumSq / float64(len(column)))
	}

	max = cloneFloatSlice(vectors[0])
	min = cloneFloatSlice(vectors[0])
	for _, vector := range vectors[1:] {
		if compareFloatSlicesLex(vector, max) > 0 {
			max = cloneFloatSlice(vector)
		}
		if compareFloatSlicesLex(vector, min) < 0 {
			min = cloneFloatSlice(vector)
		}
	}
	return max, min, avg, std
}

func compareFloatSlicesLex(left, right []float64) int {
	shared := len(left)
	if len(right) < shared {
		shared = len(right)
	}
	for i := 0; i < shared; i++ {
		if left[i] > right[i] {
			return 1
		}
		if left[i] < right[i] {
			return -1
		}
	}
	switch {
	case len(left) > len(right):
		return 1
	case len(left) < len(right):
		return -1
	default:
		return 0
	}
}

func cloneFloatSlice(values []float64) []float64 {
	if values == nil {
		return nil
	}
	return append([]float64(nil), values...)
}
