package evo

import (
	"math"
)

const sizeProportionalEfficiency = 0.05

// FitnessPostprocessor adjusts fitness values after scape evaluation and
// before ranking/selection.
type FitnessPostprocessor interface {
	Name() string
	Process(scored []ScoredGenome) []ScoredGenome
}

type NoopFitnessPostprocessor struct{}

func (NoopFitnessPostprocessor) Name() string {
	return "none"
}

func (NoopFitnessPostprocessor) Process(scored []ScoredGenome) []ScoredGenome {
	return cloneScored(scored)
}

// SizeProportionalPostprocessor penalizes larger genomes by complexity.
type SizeProportionalPostprocessor struct{}

func (SizeProportionalPostprocessor) Name() string {
	return "size_proportional"
}

func (SizeProportionalPostprocessor) Process(scored []ScoredGenome) []ScoredGenome {
	out := cloneScored(scored)
	for i := range out {
		complexity := float64(len(out[i].Genome.Neurons) + len(out[i].Genome.Synapses))
		if complexity < 1 {
			complexity = 1
		}
		out[i].Fitness = out[i].Fitness / math.Pow(complexity, sizeProportionalEfficiency)
	}
	return out
}

// NoveltyProportionalPostprocessor boosts genomes with greater population
// novelty based on topology-level differences.
//
// Reference DXNN2 leaves novelty_proportional as a placeholder (`void`).
// Keep this as a no-op for parity and stability until a reference-backed
// novelty definition is adopted.
type NoveltyProportionalPostprocessor struct{}

func (NoveltyProportionalPostprocessor) Name() string {
	return "novelty_proportional"
}

func (NoveltyProportionalPostprocessor) Process(scored []ScoredGenome) []ScoredGenome {
	return cloneScored(scored)
}

func cloneScored(scored []ScoredGenome) []ScoredGenome {
	out := make([]ScoredGenome, len(scored))
	copy(out, scored)
	return out
}
