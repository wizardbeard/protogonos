package evo

import (
	"fmt"
	"math/rand"
	"sort"

	"protogonos/internal/model"
)

// Selector chooses parents from ranked genomes for replication.
type Selector interface {
	Name() string
	PickParent(rng *rand.Rand, ranked []ScoredGenome, eliteCount int) (model.Genome, error)
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
