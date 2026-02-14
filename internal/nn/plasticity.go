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
	rule := NormalizePlasticityRuleName(cfg.Rule)
	if rule == PlasticityNone {
		return nil
	}
	switch rule {
	case PlasticityHebbian, PlasticityOja:
	default:
		return fmt.Errorf("unsupported plasticity rule: %s", cfg.Rule)
	}
	if cfg.Rate == 0 {
		return nil
	}

	limit := cfg.SaturationLimit
	if limit <= 0 {
		limit = math.Pi * 2
	}

	for i := range genome.Synapses {
		s := &genome.Synapses[i]
		if !s.Enabled {
			continue
		}
		pre := neuronValues[s.From]
		post := neuronValues[s.To]

		var delta float64
		switch rule {
		case PlasticityHebbian:
			delta = cfg.Rate * pre * post
		case PlasticityOja:
			delta = cfg.Rate * post * (pre - (post * s.Weight))
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
