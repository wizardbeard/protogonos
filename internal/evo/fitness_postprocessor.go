package evo

import (
	"math"

	"protogonos/internal/model"
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
type NoveltyProportionalPostprocessor struct{}

func (NoveltyProportionalPostprocessor) Name() string {
	return "novelty_proportional"
}

func (NoveltyProportionalPostprocessor) Process(scored []ScoredGenome) []ScoredGenome {
	out := cloneScored(scored)
	if len(out) <= 1 {
		return out
	}

	for i := range out {
		novelty := 0.0
		for j := range out {
			if i == j {
				continue
			}
			novelty += topologyDistance(out[i].Genome, out[j].Genome)
		}
		novelty /= float64(len(out) - 1)
		out[i].Fitness = out[i].Fitness * (1.0 + novelty)
	}
	return out
}

func topologyDistance(a, b model.Genome) float64 {
	an := float64(len(a.Neurons))
	bn := float64(len(b.Neurons))
	as := float64(len(a.Synapses))
	bs := float64(len(b.Synapses))

	nd := math.Abs(an - bn)
	sd := math.Abs(as - bs)
	return (nd + sd) / 2.0
}

func cloneScored(scored []ScoredGenome) []ScoredGenome {
	out := make([]ScoredGenome, len(scored))
	copy(out, scored)
	return out
}
