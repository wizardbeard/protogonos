package nn

import (
	"fmt"
	"math"
	"strings"

	"protogonos/internal/model"
)

const (
	PlasticityNone             = "none"
	PlasticityHebbian          = "hebbian"
	PlasticityHebbianW         = "hebbian_w"
	PlasticityOja              = "oja"
	PlasticityOjaW             = "ojas_w"
	PlasticityNeuromodulation  = "neuromodulation"
	PlasticitySelfModulationV1 = "self_modulationv1"
	PlasticitySelfModulationV2 = "self_modulationv2"
	PlasticitySelfModulationV3 = "self_modulationv3"
	PlasticitySelfModulationV4 = "self_modulationv4"
	PlasticitySelfModulationV5 = "self_modulationv5"
	PlasticitySelfModulationV6 = "self_modulationv6"
)

func NormalizePlasticityRuleName(rule string) string {
	switch strings.ToLower(strings.TrimSpace(rule)) {
	case "", PlasticityNone:
		return PlasticityNone
	case PlasticityHebbian:
		return PlasticityHebbian
	case PlasticityHebbianW:
		return PlasticityHebbianW
	case PlasticityOja, "ojas":
		return PlasticityOja
	case PlasticityOjaW:
		return PlasticityOjaW
	case PlasticityNeuromodulation:
		return PlasticityNeuromodulation
	case PlasticitySelfModulationV1, "self_modulation_v1":
		return PlasticitySelfModulationV1
	case PlasticitySelfModulationV2, "self_modulation_v2":
		return PlasticitySelfModulationV2
	case PlasticitySelfModulationV3, "self_modulation_v3":
		return PlasticitySelfModulationV3
	case PlasticitySelfModulationV4, "self_modulation_v4":
		return PlasticitySelfModulationV4
	case PlasticitySelfModulationV5, "self_modulation_v5":
		return PlasticitySelfModulationV5
	case PlasticitySelfModulationV6, "self_modulation_v6":
		return PlasticitySelfModulationV6
	default:
		return strings.ToLower(strings.TrimSpace(rule))
	}
}

type plasticityCoefficients struct {
	A float64
	B float64
	C float64
	D float64
}

type selfModulationDynamics struct {
	H            float64
	Coefficients plasticityCoefficients
}

func ApplyPlasticity(genome *model.Genome, neuronValues map[string]float64, cfg model.PlasticityConfig) error {
	if genome == nil {
		return fmt.Errorf("genome is required")
	}
	defaultRule := NormalizePlasticityRuleName(cfg.Rule)
	if err := validatePlasticityRule(defaultRule, cfg.Rule); err != nil {
		return err
	}
	defaultCoefficients := defaultPlasticityCoefficients(cfg)

	limit := cfg.SaturationLimit
	if limit <= 0 {
		limit = math.Pi * 2
	}
	neuronByID := make(map[string]model.Neuron, len(genome.Neurons))
	for _, neuron := range genome.Neurons {
		neuronByID[neuron.ID] = neuron
	}
	incomingByTarget := make(map[string][]model.Synapse, len(genome.Neurons))
	for _, synapse := range genome.Synapses {
		if !synapse.Enabled {
			continue
		}
		incomingByTarget[synapse.To] = append(incomingByTarget[synapse.To], synapse)
	}

	for i := range genome.Synapses {
		s := &genome.Synapses[i]
		if !s.Enabled {
			continue
		}

		rule := defaultRule
		rate := cfg.Rate
		coeffs := defaultCoefficients
		var biasParams []float64
		if neuron, ok := neuronByID[s.To]; ok {
			if neuronRule := NormalizePlasticityRuleName(neuron.PlasticityRule); neuronRule != PlasticityNone {
				rule = neuronRule
			}
			if neuron.PlasticityRate != 0 {
				rate = neuron.PlasticityRate
			}
			coeffs = withNeuronPlasticityCoefficients(coeffs, neuron)
			biasParams = neuron.PlasticityBiasParams
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
		case PlasticityHebbianW:
			h := synapsePlasticityParameter(*s, 0, rate)
			delta = h * pre * post
		case PlasticityOja:
			delta = rate * post * (pre - (post * s.Weight))
		case PlasticityOjaW:
			h := synapsePlasticityParameter(*s, 0, rate)
			delta = h * post * (pre - (post * s.Weight))
		case PlasticityNeuromodulation:
			modulator := scaleDeadzone(post, 0.33, limit)
			delta = modulator * generalizedHebbianDelta(rate, coeffs, pre, post)
		case PlasticitySelfModulationV1, PlasticitySelfModulationV2, PlasticitySelfModulationV3, PlasticitySelfModulationV4, PlasticitySelfModulationV5, PlasticitySelfModulationV6:
			dynamics := deriveSelfModulationDynamics(rule, coeffs, incomingByTarget[s.To], biasParams, neuronValues)
			delta = dynamics.H * generalizedHebbianDelta(rate, dynamics.Coefficients, pre, post)
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
	case PlasticityNone,
		PlasticityHebbian,
		PlasticityHebbianW,
		PlasticityOja,
		PlasticityOjaW,
		PlasticityNeuromodulation,
		PlasticitySelfModulationV1,
		PlasticitySelfModulationV2,
		PlasticitySelfModulationV3,
		PlasticitySelfModulationV4,
		PlasticitySelfModulationV5,
		PlasticitySelfModulationV6:
		return nil
	default:
		return fmt.Errorf("unsupported plasticity rule: %s", original)
	}
}

func synapsePlasticityParameter(synapse model.Synapse, index int, fallback float64) float64 {
	if index >= 0 && index < len(synapse.PlasticityParams) {
		return synapse.PlasticityParams[index]
	}
	return fallback
}

func defaultPlasticityCoefficients(cfg model.PlasticityConfig) plasticityCoefficients {
	return normalizePlasticityCoefficients(plasticityCoefficients{
		A: cfg.CoeffA,
		B: cfg.CoeffB,
		C: cfg.CoeffC,
		D: cfg.CoeffD,
	})
}

func withNeuronPlasticityCoefficients(base plasticityCoefficients, neuron model.Neuron) plasticityCoefficients {
	if neuron.PlasticityA != 0 {
		base.A = neuron.PlasticityA
	}
	if neuron.PlasticityB != 0 {
		base.B = neuron.PlasticityB
	}
	if neuron.PlasticityC != 0 {
		base.C = neuron.PlasticityC
	}
	if neuron.PlasticityD != 0 {
		base.D = neuron.PlasticityD
	}
	return normalizePlasticityCoefficients(base)
}

func normalizePlasticityCoefficients(coeffs plasticityCoefficients) plasticityCoefficients {
	if coeffs.A == 0 && coeffs.B == 0 && coeffs.C == 0 && coeffs.D == 0 {
		coeffs.A = 1
	}
	return coeffs
}

func generalizedHebbianDelta(rate float64, coeffs plasticityCoefficients, pre, post float64) float64 {
	return rate * (coeffs.A*pre*post + coeffs.B*pre + coeffs.C*post + coeffs.D)
}

func deriveSelfModulationDynamics(
	rule string,
	coeffs plasticityCoefficients,
	incoming []model.Synapse,
	biasParams []float64,
	neuronValues map[string]float64,
) selfModulationDynamics {
	dynamics := selfModulationDynamics{
		H:            1,
		Coefficients: coeffs,
	}

	switch rule {
	case PlasticitySelfModulationV1, PlasticitySelfModulationV2, PlasticitySelfModulationV3:
		if dot, ok := dotPlasticityParameter(incoming, biasParams, neuronValues, 0); ok {
			dynamics.H = math.Tanh(dot)
		}
	case PlasticitySelfModulationV4, PlasticitySelfModulationV5:
		if dot, ok := dotPlasticityParameter(incoming, biasParams, neuronValues, 0); ok {
			dynamics.H = math.Tanh(dot)
		}
		if dot, ok := dotPlasticityParameter(incoming, biasParams, neuronValues, 1); ok {
			dynamics.Coefficients.A = math.Tanh(dot)
		}
	case PlasticitySelfModulationV6:
		if dot, ok := dotPlasticityParameter(incoming, biasParams, neuronValues, 0); ok {
			dynamics.H = math.Tanh(dot)
		}
		if dot, ok := dotPlasticityParameter(incoming, biasParams, neuronValues, 1); ok {
			dynamics.Coefficients.A = math.Tanh(dot)
		}
		if dot, ok := dotPlasticityParameter(incoming, biasParams, neuronValues, 2); ok {
			dynamics.Coefficients.B = math.Tanh(dot)
		}
		if dot, ok := dotPlasticityParameter(incoming, biasParams, neuronValues, 3); ok {
			dynamics.Coefficients.C = math.Tanh(dot)
		}
		if dot, ok := dotPlasticityParameter(incoming, biasParams, neuronValues, 4); ok {
			dynamics.Coefficients.D = math.Tanh(dot)
		}
	}

	dynamics.Coefficients = normalizePlasticityCoefficients(dynamics.Coefficients)
	return dynamics
}

func dotPlasticityParameter(incoming []model.Synapse, biasParams []float64, neuronValues map[string]float64, index int) (float64, bool) {
	total := 0.0
	used := false
	if index >= 0 && index < len(biasParams) {
		total += biasParams[index]
		used = true
	}
	for _, synapse := range incoming {
		if len(synapse.PlasticityParams) <= index {
			continue
		}
		total += neuronValues[synapse.From] * synapse.PlasticityParams[index]
		used = true
	}
	return total, used
}

func scaleDeadzone(value, threshold, maxMagnitude float64) float64 {
	switch {
	case value > threshold:
		return (scaleLinear(value, maxMagnitude, threshold) + 1) * maxMagnitude / 2
	case value < -threshold:
		return (scaleLinear(value, -threshold, -maxMagnitude) - 1) * maxMagnitude / 2
	default:
		return 0
	}
}

func scaleLinear(value, max, min float64) float64 {
	if max == min {
		return 0
	}
	return (value*2 - (max + min)) / (max - min)
}
