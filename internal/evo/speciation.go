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
	representatives    map[string]model.Genome
	nextSpeciesID      int
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
		representatives:    map[string]model.Genome{},
		nextSpeciesID:      1,
	}
}

func (s *AdaptiveSpeciation) Assign(genomes []model.Genome) (map[string][]model.Genome, SpeciationStats) {
	if len(genomes) == 0 {
		return map[string][]model.Genome{}, SpeciationStats{
			TargetSpeciesCount: s.TargetSpeciesCount,
			Threshold:          s.Threshold,
		}
	}

	ordered := append([]model.Genome(nil), genomes...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].ID < ordered[j].ID })

	if s.representatives == nil {
		s.representatives = map[string]model.Genome{}
	}
	speciesByKey := make(map[string][]model.Genome, len(ordered))

	repKeys := make([]string, 0, len(s.representatives))
	for key := range s.representatives {
		repKeys = append(repKeys, key)
	}
	sort.Strings(repKeys)

	for _, genome := range ordered {
		bestKey := ""
		bestDistance := math.MaxFloat64
		for _, key := range repKeys {
			rep := s.representatives[key]
			dist := GenomeCompatibilityDistance(genome, rep)
			if dist < bestDistance {
				bestDistance = dist
				bestKey = key
			}
		}
		if bestKey == "" || bestDistance > s.Threshold {
			bestKey = s.nextSpeciesKey()
			repKeys = append(repKeys, bestKey)
			sort.Strings(repKeys)
			s.representatives[bestKey] = genome
		}
		speciesByKey[bestKey] = append(speciesByKey[bestKey], genome)
	}

	if len(speciesByKey) > s.TargetSpeciesCount {
		s.Threshold = math.Min(s.MaxThreshold, s.Threshold+s.AdjustStep)
	} else if len(speciesByKey) < s.TargetSpeciesCount {
		s.Threshold = math.Max(s.MinThreshold, s.Threshold-s.AdjustStep)
	}

	updatedRepresentatives := map[string]model.Genome{}
	activeKeys := make([]string, 0, len(speciesByKey))
	for key := range speciesByKey {
		activeKeys = append(activeKeys, key)
	}
	sort.Strings(activeKeys)
	totalMembers := 0
	largest := 0
	for _, key := range activeKeys {
		members := speciesByKey[key]
		speciesByKey[key] = append([]model.Genome(nil), members...)
		updatedRepresentatives[key] = chooseRepresentative(members)
		totalMembers += len(members)
		if len(members) > largest {
			largest = len(members)
		}
	}
	s.representatives = updatedRepresentatives

	stats := SpeciationStats{
		SpeciesCount:       len(speciesByKey),
		TargetSpeciesCount: s.TargetSpeciesCount,
		Threshold:          s.Threshold,
		MeanSpeciesSize:    float64(totalMembers) / float64(len(speciesByKey)),
		LargestSpeciesSize: largest,
	}
	return speciesByKey, stats
}

func (s *AdaptiveSpeciation) nextSpeciesKey() string {
	key := fmt.Sprintf("sp-%03d", s.nextSpeciesID)
	s.nextSpeciesID++
	return key
}

func chooseRepresentative(members []model.Genome) model.Genome {
	if len(members) == 0 {
		return model.Genome{}
	}
	bestIdx := 0
	bestScore := math.MaxFloat64
	for i := range members {
		sum := 0.0
		for j := range members {
			if i == j {
				continue
			}
			sum += GenomeCompatibilityDistance(members[i], members[j])
		}
		if sum < bestScore {
			bestScore = sum
			bestIdx = i
		}
	}
	return members[bestIdx]
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
