package evo

import (
	"fmt"
	"math"
	"math/rand"

	"protogonos/internal/model"
)

// TopologicalMutationPolicy determines how many mutation operations are applied
// to each replicated child genome.
type TopologicalMutationPolicy interface {
	Name() string
	MutationCount(genome model.Genome, generation int, rng *rand.Rand) (int, error)
}

type ConstTopologicalMutations struct {
	Count int
}

func (ConstTopologicalMutations) Name() string {
	return "const"
}

func (p ConstTopologicalMutations) MutationCount(_ model.Genome, _ int, _ *rand.Rand) (int, error) {
	if p.Count <= 0 {
		return 0, fmt.Errorf("const topological mutation count must be > 0")
	}
	return p.Count, nil
}

type NCountLinearTopologicalMutations struct {
	Multiplier float64
	MaxCount   int
}

func (NCountLinearTopologicalMutations) Name() string {
	return "ncount_linear"
}

func (p NCountLinearTopologicalMutations) MutationCount(genome model.Genome, _ int, _ *rand.Rand) (int, error) {
	if p.Multiplier <= 0 {
		return 0, fmt.Errorf("linear multiplier must be > 0")
	}
	count := int(math.Round(float64(len(genome.Neurons)) * p.Multiplier))
	if count < 1 {
		count = 1
	}
	if p.MaxCount > 0 && count > p.MaxCount {
		count = p.MaxCount
	}
	return count, nil
}

type NCountExponentialTopologicalMutations struct {
	Power    float64
	MaxCount int
}

func (NCountExponentialTopologicalMutations) Name() string {
	return "ncount_exponential"
}

func (p NCountExponentialTopologicalMutations) MutationCount(genome model.Genome, _ int, _ *rand.Rand) (int, error) {
	if p.Power <= 0 {
		return 0, fmt.Errorf("exponential power must be > 0")
	}
	count := int(math.Round(math.Pow(float64(max(1, len(genome.Neurons))), p.Power)))
	if count < 1 {
		count = 1
	}
	if p.MaxCount > 0 && count > p.MaxCount {
		count = p.MaxCount
	}
	return count, nil
}
