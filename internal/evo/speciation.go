package evo

import (
	"fmt"
	"math"
	"sort"

	"protogonos/internal/model"
)

// SpeciationStats captures per-generation species partitioning diagnostics.
type SpeciationStats struct {
	SpeciesCount       int
	TargetSpeciesCount int
	Threshold          float64
	MeanSpeciesSize    float64
	LargestSpeciesSize int
}

// AdaptiveSpeciation tracks a compatibility threshold and nudges it toward a
// target species count each generation.
type AdaptiveSpeciation struct {
	TargetSpeciesCount int
	Threshold          float64
	MinThreshold       float64
	MaxThreshold       float64
	AdjustStep         float64
}

func NewAdaptiveSpeciation(populationSize int) *AdaptiveSpeciation {
	target := int(math.Sqrt(float64(populationSize)))
	if target < 2 {
		target = 2
	}
	return &AdaptiveSpeciation{
		TargetSpeciesCount: target,
		Threshold:          1.0,
		MinThreshold:       0.05,
		MaxThreshold:       8.0,
		AdjustStep:         0.1,
	}
}

func (s *AdaptiveSpeciation) Assign(genomes []model.Genome) (map[string][]model.Genome, SpeciationStats) {
	if len(genomes) == 0 {
		return map[string][]model.Genome{}, SpeciationStats{
			TargetSpeciesCount: s.TargetSpeciesCount,
			Threshold:          s.Threshold,
		}
	}

	type speciesGroup struct {
		key            string
		representative model.Genome
		members        []model.Genome
	}
	groups := make([]*speciesGroup, 0, len(genomes))

	for _, genome := range genomes {
		bestIdx := -1
		bestDistance := math.MaxFloat64
		for i, grp := range groups {
			dist := GenomeCompatibilityDistance(genome, grp.representative)
			if dist < bestDistance {
				bestDistance = dist
				bestIdx = i
			}
		}
		if bestIdx == -1 || bestDistance > s.Threshold {
			key := fmt.Sprintf("sp-%03d", len(groups)+1)
			groups = append(groups, &speciesGroup{
				key:            key,
				representative: genome,
				members:        []model.Genome{genome},
			})
			continue
		}
		groups[bestIdx].members = append(groups[bestIdx].members, genome)
	}

	if len(groups) > s.TargetSpeciesCount {
		s.Threshold = math.Min(s.MaxThreshold, s.Threshold+s.AdjustStep)
	} else if len(groups) < s.TargetSpeciesCount {
		s.Threshold = math.Max(s.MinThreshold, s.Threshold-s.AdjustStep)
	}

	speciesByKey := make(map[string][]model.Genome, len(groups))
	totalMembers := 0
	largest := 0
	for _, group := range groups {
		members := append([]model.Genome(nil), group.members...)
		speciesByKey[group.key] = members
		totalMembers += len(members)
		if len(members) > largest {
			largest = len(members)
		}
	}

	stats := SpeciationStats{
		SpeciesCount:       len(groups),
		TargetSpeciesCount: s.TargetSpeciesCount,
		Threshold:          s.Threshold,
		MeanSpeciesSize:    float64(totalMembers) / float64(len(groups)),
		LargestSpeciesSize: largest,
	}
	return speciesByKey, stats
}

// GenomeCompatibilityDistance provides a coarse, deterministic compatibility
// score between two genomes based on topology summary and operator mix.
func GenomeCompatibilityDistance(a, b model.Genome) float64 {
	sa := ComputeGenomeSignature(a).Summary
	sb := ComputeGenomeSignature(b).Summary

	weightedCountDelta := func(x, y int, weight float64) float64 {
		maxv := x
		if y > maxv {
			maxv = y
		}
		if maxv == 0 {
			return 0
		}
		return weight * math.Abs(float64(x-y)) / float64(maxv)
	}

	dist := 0.0
	dist += weightedCountDelta(sa.TotalNeurons, sb.TotalNeurons, 1.0)
	dist += weightedCountDelta(sa.TotalSynapses, sb.TotalSynapses, 1.0)
	dist += weightedCountDelta(sa.TotalRecurrentSynapses, sb.TotalRecurrentSynapses, 0.5)
	dist += weightedCountDelta(sa.TotalSensors, sb.TotalSensors, 0.3)
	dist += weightedCountDelta(sa.TotalActuators, sb.TotalActuators, 0.3)
	dist += distributionDistance(sa.ActivationDistribution, sb.ActivationDistribution, sa.TotalNeurons, sb.TotalNeurons)
	dist += distributionDistance(sa.AggregatorDistribution, sb.AggregatorDistribution, sa.TotalNeurons, sb.TotalNeurons)
	return dist
}

func distributionDistance(a, b map[string]int, aTotal, bTotal int) float64 {
	keys := make(map[string]struct{}, len(a)+len(b))
	for k := range a {
		keys[k] = struct{}{}
	}
	for k := range b {
		keys[k] = struct{}{}
	}
	ordered := make([]string, 0, len(keys))
	for k := range keys {
		ordered = append(ordered, k)
	}
	sort.Strings(ordered)

	total := 0.0
	for _, k := range ordered {
		pa := proportion(a[k], aTotal)
		pb := proportion(b[k], bTotal)
		total += math.Abs(pa - pb)
	}
	return total
}

func proportion(v, total int) float64 {
	if total <= 0 {
		return 0
	}
	return float64(v) / float64(total)
}
