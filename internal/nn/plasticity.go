package nn

import (
	"fmt"
	"math"
	"strings"

	"protogonos/internal/model"
)

const (
	PlasticityNone    = "none"
	PlasticityHebbian = "hebbian"
	PlasticityOja     = "oja"
)

func NormalizePlasticityRuleName(rule string) string {
	switch strings.ToLower(strings.TrimSpace(rule)) {
	case "", PlasticityNone:
		return PlasticityNone
	case PlasticityHebbian, "hebbian_w":
		return PlasticityHebbian
	case PlasticityOja, "ojas", "ojas_w":
		return PlasticityOja
	default:
		return strings.ToLower(strings.TrimSpace(rule))
	}
}

func ApplyPlasticity(genome *model.Genome, neuronValues map[string]float64, cfg model.PlasticityConfig) error {
	if genome == nil {
		return fmt.Errorf("genome is required")
	}
	defaultRule := NormalizePlasticityRuleName(cfg.Rule)
	if err := validatePlasticityRule(defaultRule, cfg.Rule); err != nil {
		return err
	}

	limit := cfg.SaturationLimit
	if limit <= 0 {
		limit = math.Pi * 2
	}
	neuronByID := make(map[string]model.Neuron, len(genome.Neurons))
	for _, neuron := range genome.Neurons {
		neuronByID[neuron.ID] = neuron
	}

	for i := range genome.Synapses {
		s := &genome.Synapses[i]
		if !s.Enabled {
			continue
		}

		rule := defaultRule
		rate := cfg.Rate
		if neuron, ok := neuronByID[s.To]; ok {
			if neuronRule := NormalizePlasticityRuleName(neuron.PlasticityRule); neuronRule != PlasticityNone {
				rule = neuronRule
			}
			if neuron.PlasticityRate != 0 {
				rate = neuron.PlasticityRate
			}
		}
		if rule == PlasticityNone || rate == 0 {
			continue
		}
		if err := validatePlasticityRule(rule, rule); err != nil {
			return err
		}

		pre := neuronValues[s.From]
		post := neuronValues[s.To]

		var delta float64
		switch rule {
		case PlasticityHebbian:
			delta = rate * pre * post
		case PlasticityOja:
			delta = rate * post * (pre - (post * s.Weight))
		}

		next := s.Weight + delta
		if next > limit {
			next = limit
		} else if next < -limit {
			next = -limit
		}
		s.Weight = next
	}
	return nil
}

func validatePlasticityRule(rule, original string) error {
	switch rule {
	case PlasticityNone, PlasticityHebbian, PlasticityOja:
		return nil
	default:
		return fmt.Errorf("unsupported plasticity rule: %s", original)
	}
}
