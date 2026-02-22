package genotype

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"protogonos/internal/model"
	"protogonos/internal/nn"
)

// InputSpec mirrors genotype:construct_Neuron/6 input vector specs:
// one source ID and the expected input width from that source.
type InputSpec struct {
	FromID string
	Width  int
}

// ConstructNeuron is a Go analog of genotype:construct_Neuron/6.
// It returns the constructed neuron, generated inbound synapses, and
// reference-style recurrent-output IDs derived from output layer ordering.
func ConstructNeuron(
	generation int,
	neuronID string,
	inputSpecs []InputSpec,
	outputIDs []string,
	neuralAFs []string,
	neuralPFs []string,
	neuralAggrFs []string,
	rng *rand.Rand,
) (model.Neuron, []model.Synapse, []string, error) {
	if strings.TrimSpace(neuronID) == "" {
		return model.Neuron{}, nil, nil, fmt.Errorf("neuron id is required")
	}
	rng = ensureRNG(rng)

	pfRule, pfParams := GenerateNeuronPF(rng, neuralPFs)
	neuron := model.Neuron{
		ID:         neuronID,
		Generation: generation,
		Activation: GenerateNeuronAF(rng, neuralAFs),
		Aggregator: GenerateNeuronAggrF(rng, neuralAggrFs),
	}
	applyPFNeuralParams(&neuron, pfRule, pfParams)

	synapses := make([]model.Synapse, 0)
	for _, spec := range inputSpecs {
		fromID := strings.TrimSpace(spec.FromID)
		if fromID == "" || spec.Width <= 0 {
			continue
		}
		for i := 0; i < spec.Width; i++ {
			synapse := model.Synapse{
				ID:               fmt.Sprintf("%s:in:%s:%d", neuronID, sanitizeID(fromID), i),
				From:             fromID,
				To:               neuronID,
				Weight:           randomCentered(rng),
				Enabled:          true,
				PlasticityParams: defaultPFWeightParameters(pfRule, rng),
			}
			synapses = append(synapses, synapse)
		}
	}

	return neuron, synapses, CalculateROIDs(neuronID, outputIDs), nil
}

// GenerateNeuronAF mirrors genotype:generate_NeuronAF/1.
// Empty inputs default to tanh.
func GenerateNeuronAF(rng *rand.Rand, activationFunctions []string) string {
	rng = ensureRNG(rng)
	if len(activationFunctions) == 0 {
		return "tanh"
	}
	choice, err := RandomElement(rng, activationFunctions)
	if err != nil {
		return "tanh"
	}
	choice = strings.TrimSpace(choice)
	if choice == "" {
		return "tanh"
	}
	return choice
}

// GenerateNeuronPF mirrors genotype:generate_NeuronPF/1 and returns the
// normalized PF rule plus default neural-parameter vector.
func GenerateNeuronPF(rng *rand.Rand, pfNames []string) (string, []float64) {
	rng = ensureRNG(rng)
	if len(pfNames) == 0 {
		return nn.PlasticityNone, nil
	}
	choice, err := RandomElement(rng, pfNames)
	if err != nil {
		return nn.PlasticityNone, nil
	}
	rule := nn.NormalizePlasticityRuleName(choice)
	if rule == "" {
		rule = nn.PlasticityNone
	}
	return rule, defaultPFNeuralParameters(rule, rng)
}

// GenerateNeuronAggrF mirrors genotype:generate_NeuronAggrF/1.
// Empty inputs default to none.
func GenerateNeuronAggrF(rng *rand.Rand, aggregationFunctions []string) string {
	rng = ensureRNG(rng)
	if len(aggregationFunctions) == 0 {
		return "none"
	}
	choice, err := RandomElement(rng, aggregationFunctions)
	if err != nil {
		return "none"
	}
	choice = strings.TrimSpace(choice)
	if choice == "" {
		return "none"
	}
	return choice
}

// CalculateROIDs mirrors genotype:calculate_ROIds/3 behavior.
// Output IDs at the same or lower parsed layer index are considered recurrent.
func CalculateROIDs(selfID string, outputIDs []string) []string {
	selfLayer, ok := parseLayerIndex(selfID)
	if !ok {
		return nil
	}
	roIDs := make([]string, 0, len(outputIDs))
	for _, outputID := range outputIDs {
		layer, layerOK := parseLayerIndex(outputID)
		if !layerOK {
			continue
		}
		if layer <= selfLayer {
			roIDs = append(roIDs, outputID)
		}
	}
	return roIDs
}

func ensureRNG(rng *rand.Rand) *rand.Rand {
	if rng != nil {
		return rng
	}
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

func randomCentered(rng *rand.Rand) float64 {
	return rng.Float64() - 0.5
}

func defaultPFNeuralParameters(rule string, rng *rand.Rand) []float64 {
	switch nn.NormalizePlasticityRuleName(rule) {
	case nn.PlasticityNone, nn.PlasticityHebbianW, nn.PlasticityOjaW, nn.PlasticitySelfModulationV6:
		return nil
	case nn.PlasticityHebbian, nn.PlasticityOja:
		return []float64{randomCentered(rng)}
	case nn.PlasticitySelfModulationV1:
		return []float64{0.1, 0, 0, 0}
	case nn.PlasticitySelfModulationV2:
		return []float64{randomCentered(rng), 0, 0, 0}
	case nn.PlasticitySelfModulationV3:
		return []float64{randomCentered(rng), randomCentered(rng), randomCentered(rng), randomCentered(rng)}
	case nn.PlasticitySelfModulationV4:
		return []float64{0, 0, 0}
	case nn.PlasticitySelfModulationV5:
		return []float64{randomCentered(rng), randomCentered(rng), randomCentered(rng)}
	case nn.PlasticityNeuromodulation:
		return []float64{randomCentered(rng), randomCentered(rng), randomCentered(rng), randomCentered(rng), randomCentered(rng)}
	default:
		return nil
	}
}

func defaultPFWeightParameters(rule string, rng *rand.Rand) []float64 {
	width := defaultPFWeightParameterWidth(rule)
	if width <= 0 {
		return nil
	}
	params := make([]float64, width)
	for i := range params {
		params[i] = randomCentered(rng)
	}
	return params
}

func defaultPFWeightParameterWidth(rule string) int {
	switch nn.NormalizePlasticityRuleName(rule) {
	case nn.PlasticityHebbianW, nn.PlasticityOjaW:
		return 1
	case nn.PlasticitySelfModulationV1, nn.PlasticitySelfModulationV2, nn.PlasticitySelfModulationV3:
		return 1
	case nn.PlasticitySelfModulationV4, nn.PlasticitySelfModulationV5:
		return 2
	case nn.PlasticitySelfModulationV6:
		return 5
	default:
		return 0
	}
}

func applyPFNeuralParams(neuron *model.Neuron, rule string, params []float64) {
	if neuron == nil {
		return
	}
	rule = nn.NormalizePlasticityRuleName(rule)
	if rule == nn.PlasticityNone {
		neuron.PlasticityRule = ""
		return
	}
	neuron.PlasticityRule = rule

	switch rule {
	case nn.PlasticityHebbian, nn.PlasticityOja:
		if len(params) > 0 {
			neuron.PlasticityRate = params[0]
		}
	case nn.PlasticityHebbianW, nn.PlasticityOjaW:
		neuron.PlasticityRate = 0.1
	case nn.PlasticityNeuromodulation:
		neuron.PlasticityRate = pickParamOrDefault(params, 0, 0.1)
		neuron.PlasticityA = pickParamOrDefault(params, 1, 0)
		neuron.PlasticityB = pickParamOrDefault(params, 2, 0)
		neuron.PlasticityC = pickParamOrDefault(params, 3, 0)
		neuron.PlasticityD = pickParamOrDefault(params, 4, 0)
	case nn.PlasticitySelfModulationV1, nn.PlasticitySelfModulationV2, nn.PlasticitySelfModulationV3:
		neuron.PlasticityRate = 1
		neuron.PlasticityA = pickParamOrDefault(params, 0, 0.1)
		neuron.PlasticityB = pickParamOrDefault(params, 1, 0)
		neuron.PlasticityC = pickParamOrDefault(params, 2, 0)
		neuron.PlasticityD = pickParamOrDefault(params, 3, 0)
	case nn.PlasticitySelfModulationV4, nn.PlasticitySelfModulationV5:
		neuron.PlasticityRate = 1
		neuron.PlasticityA = 0
		neuron.PlasticityB = pickParamOrDefault(params, 0, 0)
		neuron.PlasticityC = pickParamOrDefault(params, 1, 0)
		neuron.PlasticityD = pickParamOrDefault(params, 2, 0)
	case nn.PlasticitySelfModulationV6:
		neuron.PlasticityRate = 1
	}
}

func pickParamOrDefault(values []float64, index int, fallback float64) float64 {
	if index >= 0 && index < len(values) {
		return values[index]
	}
	return fallback
}

func parseLayerIndex(id string) (float64, bool) {
	id = strings.TrimSpace(id)
	if id == "" {
		return 0, false
	}
	if layer, ok := parseFloatToken(id); ok {
		return layer, true
	}

	for _, sep := range []string{":", "|", "/", ","} {
		if token, _, ok := strings.Cut(id, sep); ok {
			if layer, layerOK := parseFloatToken(token); layerOK {
				return layer, true
			}
		}
	}
	return 0, false
}

func parseFloatToken(token string) (float64, bool) {
	token = strings.TrimSpace(token)
	token = strings.TrimPrefix(token, "layer")
	token = strings.TrimPrefix(token, "li")
	token = strings.TrimPrefix(token, "l")
	token = strings.TrimPrefix(token, "L")
	token = strings.TrimPrefix(token, "=")
	token = strings.TrimSpace(token)
	if token == "" {
		return 0, false
	}
	layer, err := strconv.ParseFloat(token, 64)
	if err != nil {
		return 0, false
	}
	return layer, true
}

func sanitizeID(id string) string {
	id = strings.TrimSpace(id)
	replacer := strings.NewReplacer(":", "_", "|", "_", "/", "_", " ", "_")
	return replacer.Replace(id)
}
