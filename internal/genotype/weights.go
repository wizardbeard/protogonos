package genotype

import (
	"math/rand"
	"strings"
)

// NeuralWeightParam is a Go-native analog for reference {W,DW,LP,Ps} tuples.
type NeuralWeightParam struct {
	Weight           float64
	DeltaWeight      float64
	LearningProgress float64
	PlasticityParams []float64
}

// InputIDP is a Go-native analog for reference {Input_Id,WeightsP} tuples.
type InputIDP struct {
	FromID  string
	Weights []NeuralWeightParam
}

// CreateWeight mirrors genotype:create_weight/1.
func CreateWeight(rng *rand.Rand) float64 {
	rng = ensureRNG(rng)
	return randomCentered(rng)
}

// CreateNeuralWeightsP mirrors genotype:create_NeuralWeightsP/3.
func CreateNeuralWeightsP(pfRule string, count int, rng *rand.Rand) []NeuralWeightParam {
	if count <= 0 {
		return nil
	}
	rng = ensureRNG(rng)
	weights := make([]NeuralWeightParam, 0, count)
	for i := 0; i < count; i++ {
		weights = append(weights, NeuralWeightParam{
			Weight:           CreateWeight(rng),
			DeltaWeight:      0,
			LearningProgress: 0,
			PlasticityParams: defaultPFWeightParameters(pfRule, rng),
		})
	}
	return weights
}

// CreateInputIDPs mirrors genotype:create_InputIdPs/3.
func CreateInputIDPs(pfRule string, inputSpecs []InputSpec, rng *rand.Rand) []InputIDP {
	rng = ensureRNG(rng)
	out := make([]InputIDP, 0, len(inputSpecs))
	for _, spec := range inputSpecs {
		fromID := strings.TrimSpace(spec.FromID)
		if fromID == "" || spec.Width <= 0 {
			continue
		}
		out = append(out, InputIDP{
			FromID:  fromID,
			Weights: CreateNeuralWeightsP(pfRule, spec.Width, rng),
		})
	}
	return out
}
