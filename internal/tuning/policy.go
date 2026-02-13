package tuning

import (
	"fmt"

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
	case "topology_scaled", "nsize_proportional", "wsize_proportional":
		return "topology_scaled"
	default:
		return name
	}
}
