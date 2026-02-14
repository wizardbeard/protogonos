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

type SpeciesAwareGenerationSelector interface {
	GenerationAwareSelector
	PickParentForGenerationWithSpecies(rng *rand.Rand, ranked []ScoredGenome, eliteCount, generation int, speciesByGenomeID map[string]string) (model.Genome, error)
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

// RankSelector picks from a pool weighted by descending rank.
type RankSelector struct {
	PoolSize int
}

func (RankSelector) Name() string {
	return "rank"
}

func (s RankSelector) PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	pool := boundedPool(ranked, eliteCount, s.PoolSize)
	total := 0.0
	weights := make([]float64, len(pool))
	for i := range pool {
		weights[i] = float64(len(pool) - i)
		total += weights[i]
	}
	choice := rng.Float64() * total
	acc := 0.0
	for i, weight := range weights {
		acc += weight
		if choice <= acc {
			return pool[i].Genome, nil
		}
	}
	return pool[len(pool)-1].Genome, nil
}

// EfficiencySelector picks from a pool weighted by fitness divided by network size.
type EfficiencySelector struct {
	PoolSize int
}

func (EfficiencySelector) Name() string {
	return "efficiency"
}

func (s EfficiencySelector) PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	pool := boundedPool(ranked, eliteCount, s.PoolSize)
	weights := make([]float64, len(pool))
	total := 0.0
	for i, candidate := range pool {
		size := float64(len(candidate.Genome.Neurons) + len(candidate.Genome.Synapses))
		if size <= 0 {
			size = 1
		}
		weight := candidate.Fitness / size
		if weight < 0 {
			weight = 0
		}
		weights[i] = weight
		total += weight
	}
	if total <= 0 {
		return pool[rng.Intn(len(pool))].Genome, nil
	}
	choice := rng.Float64() * total
	acc := 0.0
	for i, weight := range weights {
		acc += weight
		if choice <= acc {
			return pool[i].Genome, nil
		}
	}
	return pool[len(pool)-1].Genome, nil
}

// RandomSelector picks uniformly from a bounded pool.
type RandomSelector struct {
	PoolSize int
}

func (RandomSelector) Name() string {
	return "random"
}

func (s RandomSelector) PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	pool := boundedPool(ranked, eliteCount, s.PoolSize)
	return pool[rng.Intn(len(pool))].Genome, nil
}

// TopKFitnessSelector picks from the top-k ranked pool weighted by fitness.
type TopKFitnessSelector struct {
	K int
}

func (TopKFitnessSelector) Name() string {
	return "topk_fitness"
}

func (s TopKFitnessSelector) PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	k := s.K
	if k <= 0 {
		k = 3
	}
	if k > len(ranked) {
		k = len(ranked)
	}
	pool := ranked[:k]

	minFitness := pool[0].Fitness
	for _, candidate := range pool[1:] {
		if candidate.Fitness < minFitness {
			minFitness = candidate.Fitness
		}
	}
	shift := 0.0
	if minFitness <= 0 {
		shift = -minFitness + 1e-9
	}

	total := 0.0
	weights := make([]float64, len(pool))
	for i, candidate := range pool {
		weight := candidate.Fitness + shift
		if weight < 0 {
			weight = 0
		}
		weights[i] = weight
		total += weight
	}
	if total <= 0 {
		return pool[rng.Intn(len(pool))].Genome, nil
	}

	choice := rng.Float64() * total
	acc := 0.0
	for i, weight := range weights {
		acc += weight
		if choice <= acc {
			return pool[i].Genome, nil
		}
	}
	return pool[len(pool)-1].Genome, nil
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
	return s.pickParentInternal(rng, ranked, eliteCount, nil)
}

func (s SpeciesTournamentSelector) PickParentForGeneration(rng *rand.Rand, ranked []ScoredGenome, eliteCount, _ int) (model.Genome, error) {
	return s.pickParentInternal(rng, ranked, eliteCount, nil)
}

func (s SpeciesTournamentSelector) PickParentForGenerationWithSpecies(rng *rand.Rand, ranked []ScoredGenome, eliteCount, _ int, speciesByGenomeID map[string]string) (model.Genome, error) {
	return s.pickParentInternal(rng, ranked, eliteCount, speciesByGenomeID)
}

func (s SpeciesTournamentSelector) pickParentInternal(rng *rand.Rand, ranked []ScoredGenome, eliteCount int, speciesByGenomeID map[string]string) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	if s.Identifier == nil && speciesByGenomeID == nil {
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

	bySpecies := buildSpeciesBuckets(pool, s.Identifier, speciesByGenomeID)

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
	return s.pickParentInternal(rng, ranked, eliteCount, generation, nil)
}

func (s *SpeciesSharedTournamentSelector) PickParentForGenerationWithSpecies(rng *rand.Rand, ranked []ScoredGenome, eliteCount, generation int, speciesByGenomeID map[string]string) (model.Genome, error) {
	return s.pickParentInternal(rng, ranked, eliteCount, generation, speciesByGenomeID)
}

func (s *SpeciesSharedTournamentSelector) pickParentInternal(rng *rand.Rand, ranked []ScoredGenome, eliteCount, generation int, speciesByGenomeID map[string]string) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, fmt.Errorf("random source is required")
	}
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, fmt.Errorf("invalid elite count: %d", eliteCount)
	}
	if s.Identifier == nil && speciesByGenomeID == nil {
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

	bySpecies := buildSpeciesBuckets(pool, s.Identifier, speciesByGenomeID)
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

func buildSpeciesBuckets(pool []ScoredGenome, identifier SpecieIdentifier, speciesByGenomeID map[string]string) map[string][]ScoredGenome {
	bySpecies := make(map[string][]ScoredGenome, len(pool))
	for _, scored := range pool {
		key := ""
		if speciesByGenomeID != nil {
			key = speciesByGenomeID[scored.Genome.ID]
		}
		if key == "" && identifier != nil {
			key = identifier.Identify(scored.Genome)
		}
		if key == "" {
			key = "species:unknown"
		}
		bySpecies[key] = append(bySpecies[key], scored)
	}
	return bySpecies
}

func boundedPool(ranked []ScoredGenome, eliteCount, poolSize int) []ScoredGenome {
	if poolSize <= 0 {
		poolSize = eliteCount * 2
	}
	if poolSize < eliteCount {
		poolSize = eliteCount
	}
	if poolSize > len(ranked) {
		poolSize = len(ranked)
	}
	return ranked[:poolSize]
}
