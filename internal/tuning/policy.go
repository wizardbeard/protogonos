package tuning

import (
	"fmt"
	"math"

	"protogonos/internal/model"
)

type AttemptPolicy interface {
	Name() string
	Attempts(baseAttempts, generation, totalGenerations int, genome model.Genome) int
}

type FixedAttemptPolicy struct{}

func (FixedAttemptPolicy) Name() string { return "fixed" }

func (FixedAttemptPolicy) Attempts(baseAttempts, _generation, _totalGenerations int, _ model.Genome) int {
	if baseAttempts < 0 {
		return 0
	}
	return baseAttempts
}

type LinearDecayAttemptPolicy struct {
	MinAttempts int
}

func (LinearDecayAttemptPolicy) Name() string { return "linear_decay" }

func (p LinearDecayAttemptPolicy) Attempts(baseAttempts, generation, totalGenerations int, _ model.Genome) int {
	if baseAttempts <= 0 {
		return 0
	}
	if totalGenerations <= 0 {
		return baseAttempts
	}
	remaining := totalGenerations - generation
	if remaining < 1 {
		remaining = 1
	}
	attempts := (baseAttempts * remaining) / totalGenerations
	if attempts < p.MinAttempts {
		attempts = p.MinAttempts
	}
	if attempts < 0 {
		return 0
	}
	return attempts
}

type TopologyScaledAttemptPolicy struct {
	Scale       float64
	MinAttempts int
	MaxAttempts int
}

func (TopologyScaledAttemptPolicy) Name() string { return "topology_scaled" }

func (p TopologyScaledAttemptPolicy) Attempts(baseAttempts, _generation, _totalGenerations int, genome model.Genome) int {
	if baseAttempts <= 0 {
		return 0
	}
	scale := p.Scale
	if scale <= 0 {
		scale = 1.0
	}
	attempts := int(float64(baseAttempts) * scale * (1.0 + float64(len(genome.Synapses))/10.0))
	if attempts < p.MinAttempts {
		attempts = p.MinAttempts
	}
	if p.MaxAttempts > 0 && attempts > p.MaxAttempts {
		attempts = p.MaxAttempts
	}
	return attempts
}

type NSizeProportionalAttemptPolicy struct {
	Power float64
}

func (NSizeProportionalAttemptPolicy) Name() string { return "nsize_proportional" }

func (p NSizeProportionalAttemptPolicy) Attempts(baseAttempts, _generation, _totalGenerations int, genome model.Genome) int {
	if baseAttempts <= 0 {
		return 0
	}
	power := p.Power
	if power <= 0 {
		power = 1.0
	}
	recentNeuronCount := len(genome.Neurons)
	scaled := satInt(int(math.Round(math.Pow(float64(recentNeuronCount), power))), 0, 100)
	return 20 + scaled
}

type WSizeProportionalAttemptPolicy struct {
	Power float64
}

func (WSizeProportionalAttemptPolicy) Name() string { return "wsize_proportional" }

func (p WSizeProportionalAttemptPolicy) Attempts(baseAttempts, _generation, _totalGenerations int, genome model.Genome) int {
	if baseAttempts <= 0 {
		return 0
	}
	power := p.Power
	if power <= 0 {
		power = 1.0
	}
	recentWeightCount := len(genome.Synapses)
	scaled := satInt(int(math.Round(math.Pow(float64(recentWeightCount), power))), 0, 100)
	return 10 + scaled
}

func AttemptPolicyFromConfig(name string, param float64) (AttemptPolicy, error) {
	switch NormalizeAttemptPolicyName(name) {
	case "", "fixed":
		return FixedAttemptPolicy{}, nil
	case "linear_decay":
		min := int(param)
		if min < 1 {
			min = 1
		}
		return LinearDecayAttemptPolicy{MinAttempts: min}, nil
	case "topology_scaled":
		scale := param
		if scale <= 0 {
			scale = 1.0
		}
		return TopologyScaledAttemptPolicy{Scale: scale, MinAttempts: 1, MaxAttempts: 0}, nil
	case "nsize_proportional":
		power := param
		if power <= 0 {
			power = 1.0
		}
		return NSizeProportionalAttemptPolicy{Power: power}, nil
	case "wsize_proportional":
		power := param
		if power <= 0 {
			power = 1.0
		}
		return WSizeProportionalAttemptPolicy{Power: power}, nil
	default:
		return nil, fmt.Errorf("unsupported tune duration policy: %s", name)
	}
}

func NormalizeAttemptPolicyName(name string) string {
	switch name {
	case "", "fixed", "const":
		return "fixed"
	case "linear_decay":
		return "linear_decay"
	case "topology_scaled":
		return "topology_scaled"
	case "nsize_proportional":
		return "nsize_proportional"
	case "wsize_proportional":
		return "wsize_proportional"
	default:
		return name
	}
}

func satInt(v, minV, maxV int) int {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}
