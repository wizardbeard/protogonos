package evo

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"

	"protogonos/internal/model"
)

// Selector chooses parents from ranked genomes for replication.
type Selector interface {
	Name() string
	PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error)
}

type GenerationAwareSelector interface {
	Selector
	PickParentForGeneration(rng *rand.Rand, ranked []ScoredGenome, eliteCount, generation int) (model.Genome, error)
}

// EliteSelector picks uniformly from the top elite set.
type EliteSelector struct{}

func (EliteSelector) Name() string {
	return "elite"
}

func (EliteSelector) PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	return ranked[rng.Intn(eliteCount)].Genome, nil
}

// TournamentSelector samples candidates and picks the best fitness among them.
type TournamentSelector struct {
	PoolSize       int
	TournamentSize int
}

func (TournamentSelector) Name() string {
	return "tournament"
}

func (s TournamentSelector) PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}

	poolSize := s.PoolSize
	if poolSize <= 0 {
		poolSize = eliteCount * 2
	}
	if poolSize < eliteCount {
		poolSize = eliteCount
	}
	if poolSize > len(ranked) {
		poolSize = len(ranked)
	}

	tournamentSize := s.TournamentSize
	if tournamentSize <= 0 {
		tournamentSize = 3
	}
	if tournamentSize > poolSize {
		tournamentSize = poolSize
	}

	best := ranked[rng.Intn(poolSize)]
	for i := 1; i < tournamentSize; i++ {
		candidate := ranked[rng.Intn(poolSize)]
		if candidate.Fitness > best.Fitness {
			best = candidate
		}
	}
	return best.Genome, nil
}

// SpeciesTournamentSelector first samples a species uniformly and then runs
// tournament selection inside that species.
type SpeciesTournamentSelector struct {
	Identifier     SpecieIdentifier
	PoolSize       int
	TournamentSize int
}

func (SpeciesTournamentSelector) Name() string {
	return "species_tournament"
}

func (s SpeciesTournamentSelector) PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	if s.Identifier == nil {
		return model.Genome{}, fmt.Errorf("species identifier is required")
	}

	poolSize := s.PoolSize
	if poolSize <= 0 {
		poolSize = eliteCount * 2
	}
	if poolSize < eliteCount {
		poolSize = eliteCount
	}
	if poolSize > len(ranked) {
		poolSize = len(ranked)
	}
	pool := ranked[:poolSize]

	bySpecies := make(map[string][]ScoredGenome, poolSize)
	for _, scored := range pool {
		key := s.Identifier.Identify(scored.Genome)
		bySpecies[key] = append(bySpecies[key], scored)
	}

	speciesKeys := make([]string, 0, len(bySpecies))
	for key := range bySpecies {
		speciesKeys = append(speciesKeys, key)
	}
	sort.Strings(speciesKeys)
	chosenSpecies := speciesKeys[rng.Intn(len(speciesKeys))]
	candidates := bySpecies[chosenSpecies]

	tournamentSize := s.TournamentSize
	if tournamentSize <= 0 {
		tournamentSize = 3
	}
	if tournamentSize > len(candidates) {
		tournamentSize = len(candidates)
	}

	best := candidates[rng.Intn(len(candidates))]
	for i := 1; i < tournamentSize; i++ {
		candidate := candidates[rng.Intn(len(candidates))]
		if candidate.Fitness > best.Fitness {
			best = candidate
		}
	}
	return best.Genome, nil
}

type speciesState struct {
	bestFitness    float64
	lastImprovedAt int
}

// SpeciesSharedTournamentSelector picks a species using shared-fitness weighting,
// optionally filters stagnant species, and then runs tournament inside it.
type SpeciesSharedTournamentSelector struct {
	Identifier            SpecieIdentifier
	PoolSize              int
	TournamentSize        int
	StagnationGenerations int

	mu    sync.Mutex
	state map[string]speciesState
}

func (SpeciesSharedTournamentSelector) Name() string {
	return "species_shared_tournament"
}

func (s *SpeciesSharedTournamentSelector) PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error) {
	return s.PickParentForGeneration(rng, ranked, eliteCount, 0)
}

func (s *SpeciesSharedTournamentSelector) PickParentForGeneration(rng *rand.Rand, ranked []ScoredGenome, eliteCount, generation int) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	if s.Identifier == nil {
		return model.Genome{}, fmt.Errorf("species identifier is required")
	}

	poolSize := s.PoolSize
	if poolSize <= 0 {
		poolSize = eliteCount * 2
	}
	if poolSize < eliteCount {
		poolSize = eliteCount
	}
	if poolSize > len(ranked) {
		poolSize = len(ranked)
	}
	pool := ranked[:poolSize]

	bySpecies := make(map[string][]ScoredGenome, poolSize)
	for _, scored := range pool {
		key := s.Identifier.Identify(scored.Genome)
		bySpecies[key] = append(bySpecies[key], scored)
	}
	speciesKeys := make([]string, 0, len(bySpecies))
	for key := range bySpecies {
		speciesKeys = append(speciesKeys, key)
	}
	sort.Strings(speciesKeys)
	if len(speciesKeys) == 0 {
		return model.Genome{}, fmt.Errorf("no species available")
	}

	filtered := speciesKeys
	if s.StagnationGenerations > 0 {
		filtered = make([]string, 0, len(speciesKeys))
		for _, key := range speciesKeys {
			best := bySpecies[key][0].Fitness
			for _, cand := range bySpecies[key][1:] {
				if cand.Fitness > best {
					best = cand.Fitness
				}
			}
			if s.shouldKeepSpecies(key, best, generation) {
				filtered = append(filtered, key)
			}
		}
		if len(filtered) == 0 {
			filtered = speciesKeys
		}
	}

	means := make([]float64, len(filtered))
	minMean := 0.0
	for i, key := range filtered {
		sum := 0.0
		for _, cand := range bySpecies[key] {
			sum += cand.Fitness
		}
		mean := sum / float64(len(bySpecies[key]))
		means[i] = mean
		if i == 0 || mean < minMean {
			minMean = mean
		}
	}
	shift := 0.0
	if minMean <= 0 {
		shift = -minMean + 1e-9
	}
	total := 0.0
	for i := range means {
		means[i] += shift
		total += means[i]
	}
	if total <= 0 {
		for i := range means {
			means[i] = 1
		}
		total = float64(len(means))
	}
	pick := rng.Float64() * total
	acc := 0.0
	chosenKey := filtered[len(filtered)-1]
	for i, key := range filtered {
		acc += means[i]
		if pick <= acc {
			chosenKey = key
			break
		}
	}
	candidates := bySpecies[chosenKey]

	tournamentSize := s.TournamentSize
	if tournamentSize <= 0 {
		tournamentSize = 3
	}
	if tournamentSize > len(candidates) {
		tournamentSize = len(candidates)
	}

	best := candidates[rng.Intn(len(candidates))]
	for i := 1; i < tournamentSize; i++ {
		candidate := candidates[rng.Intn(len(candidates))]
		if candidate.Fitness > best.Fitness {
			best = candidate
		}
	}
	return best.Genome, nil
}

func (s *SpeciesSharedTournamentSelector) shouldKeepSpecies(key string, bestFitness float64, generation int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == nil {
		s.state = make(map[string]speciesState)
	}
	prev, ok := s.state[key]
	if !ok || bestFitness > prev.bestFitness {
		s.state[key] = speciesState{bestFitness: bestFitness, lastImprovedAt: generation}
		return true
	}
	return generation-prev.lastImprovedAt <= s.StagnationGenerations
}
