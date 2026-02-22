package genotype

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"sync/atomic"
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

// PatternLayer mirrors genotype:create_InitPattern/1 grouping by layer index.
type PatternLayer struct {
	Layer     float64
	NeuronIDs []string
}

// SeedNetwork captures the initial topology scaffold produced by ConstructSeedNN.
type SeedNetwork struct {
	Neurons             []model.Neuron
	Synapses            []model.Synapse
	SensorNeuronLinks   []model.SensorNeuronLink
	NeuronActuatorLinks []model.NeuronActuatorLink
	InputNeuronIDs      []string
	OutputNeuronIDs     []string
	Pattern             []PatternLayer
}

var uniqueIDSequence uint64

// ConstructSeedNN is a Go analog of genotype:construct_SeedNN/6.
// If neural activation choices include a "circuit" tag, this builds a
// relay+output two-stage scaffold per actuator; otherwise it builds the
// baseline one-output-neuron-per-actuator scaffold.
func ConstructSeedNN(
	generation int,
	sensors []string,
	actuators []string,
	neuralAFs []string,
	neuralPFs []string,
	neuralAggrFs []string,
	rng *rand.Rand,
) (SeedNetwork, error) {
	rng = ensureRNG(rng)
	uniqSensors := uniqueNonEmpty(sensors)
	uniqActuators := uniqueNonEmpty(actuators)
	if len(uniqSensors) == 0 {
		return SeedNetwork{}, fmt.Errorf("at least one sensor is required")
	}
	if len(uniqActuators) == 0 {
		return SeedNetwork{}, fmt.Errorf("at least one actuator is required")
	}

	inputNeuronIDs := make([]string, 0, len(uniqSensors))
	outputNeuronIDs := make([]string, 0, len(uniqActuators))
	neurons := make([]model.Neuron, 0, len(uniqSensors)+len(uniqActuators))
	synapses := make([]model.Synapse, 0, len(uniqSensors)*len(uniqActuators))
	sensorLinks := make([]model.SensorNeuronLink, 0, len(uniqSensors))
	actuatorLinks := make([]model.NeuronActuatorLink, 0, len(uniqActuators))

	for i, sensorID := range uniqSensors {
		neuronID := fmt.Sprintf("L0:in:%d", i)
		inputNeuronIDs = append(inputNeuronIDs, neuronID)
		neurons = append(neurons, model.Neuron{
			ID:         neuronID,
			Generation: generation,
			Activation: "identity",
			Aggregator: "none",
		})
		sensorLinks = append(sensorLinks, model.SensorNeuronLink{
			SensorID: sensorID,
			NeuronID: neuronID,
		})
	}

	inputSpecs := make([]InputSpec, 0, len(inputNeuronIDs))
	for _, inputID := range inputNeuronIDs {
		inputSpecs = append(inputSpecs, InputSpec{FromID: inputID, Width: 1})
	}

	circuitMode, circuitActivation := circuitActivationTag(neuralAFs)
	if circuitMode {
		relayNeuronIDs := make([]string, 0, len(uniqActuators))
		relayAFs := stripCircuitActivations(neuralAFs)
		for i, actuatorID := range uniqActuators {
			relayID := fmt.Sprintf("L0.5:relay:%d", i)
			circuitID := fmt.Sprintf("L0.99:circuit:%d", i)
			relayNeuronIDs = append(relayNeuronIDs, relayID)
			outputNeuronIDs = append(outputNeuronIDs, circuitID)

			relay, relayInboundSynapses, _, err := ConstructNeuron(
				generation,
				relayID,
				inputSpecs,
				[]string{circuitID},
				relayAFs,
				neuralPFs,
				neuralAggrFs,
				rng,
			)
			if err != nil {
				return SeedNetwork{}, err
			}
			circuit, circuitInboundSynapses, _, err := ConstructNeuron(
				generation,
				circuitID,
				[]InputSpec{{FromID: relayID, Width: 1}},
				nil,
				[]string{circuitActivation},
				neuralPFs,
				neuralAggrFs,
				rng,
			)
			if err != nil {
				return SeedNetwork{}, err
			}
			neurons = append(neurons, relay, circuit)
			synapses = append(synapses, relayInboundSynapses...)
			synapses = append(synapses, circuitInboundSynapses...)
			actuatorLinks = append(actuatorLinks, model.NeuronActuatorLink{
				NeuronID:   circuitID,
				ActuatorID: actuatorID,
			})
		}

		return SeedNetwork{
			Neurons:             neurons,
			Synapses:            synapses,
			SensorNeuronLinks:   sensorLinks,
			NeuronActuatorLinks: actuatorLinks,
			InputNeuronIDs:      inputNeuronIDs,
			OutputNeuronIDs:     outputNeuronIDs,
			Pattern: []PatternLayer{
				{Layer: 0, NeuronIDs: append([]string(nil), inputNeuronIDs...)},
				{Layer: 0.5, NeuronIDs: relayNeuronIDs},
				{Layer: 0.99, NeuronIDs: append([]string(nil), outputNeuronIDs...)},
			},
		}, nil
	}

	for i, actuatorID := range uniqActuators {
		neuronID := fmt.Sprintf("L1:out:%d", i)
		outputNeuronIDs = append(outputNeuronIDs, neuronID)
		neuron, inboundSynapses, _, err := ConstructNeuron(
			generation,
			neuronID,
			inputSpecs,
			nil,
			neuralAFs,
			neuralPFs,
			neuralAggrFs,
			rng,
		)
		if err != nil {
			return SeedNetwork{}, err
		}
		neurons = append(neurons, neuron)
		synapses = append(synapses, inboundSynapses...)
		actuatorLinks = append(actuatorLinks, model.NeuronActuatorLink{
			NeuronID:   neuronID,
			ActuatorID: actuatorID,
		})
	}

	return SeedNetwork{
		Neurons:             neurons,
		Synapses:            synapses,
		SensorNeuronLinks:   sensorLinks,
		NeuronActuatorLinks: actuatorLinks,
		InputNeuronIDs:      inputNeuronIDs,
		OutputNeuronIDs:     outputNeuronIDs,
		Pattern: []PatternLayer{
			{Layer: 0, NeuronIDs: append([]string(nil), inputNeuronIDs...)},
			{Layer: 1, NeuronIDs: append([]string(nil), outputNeuronIDs...)},
		},
	}, nil
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

	inputIDPs := CreateInputIDPs(pfRule, inputSpecs, rng)
	synapses := make([]model.Synapse, 0, len(inputIDPs))
	for _, inputIDP := range inputIDPs {
		for i, weight := range inputIDP.Weights {
			synapse := model.Synapse{
				ID:               fmt.Sprintf("%s:in:%s:%d", neuronID, sanitizeID(inputIDP.FromID), i),
				From:             inputIDP.FromID,
				To:               neuronID,
				Weight:           weight.Weight,
				Enabled:          true,
				PlasticityParams: append([]float64(nil), weight.PlasticityParams...),
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

// LinkNeuron mirrors genotype:link_Neuron/4 by scaffolding inbound and outbound
// synapse records around a neuron ID.
func LinkNeuron(fromIDs []string, neuronID string, toIDs []string, rng *rand.Rand) ([]model.Synapse, error) {
	if strings.TrimSpace(neuronID) == "" {
		return nil, fmt.Errorf("neuron id is required")
	}
	rng = ensureRNG(rng)
	uniqFrom := uniqueNonEmpty(fromIDs)
	uniqTo := uniqueNonEmpty(toIDs)
	synapses := make([]model.Synapse, 0, len(uniqFrom)+len(uniqTo))

	for i, fromID := range uniqFrom {
		synapses = append(synapses, model.Synapse{
			ID:      fmt.Sprintf("%s:link:in:%s:%d", neuronID, sanitizeID(fromID), i),
			From:    fromID,
			To:      neuronID,
			Weight:  randomCentered(rng),
			Enabled: true,
		})
	}

	roIDSet := make(map[string]struct{})
	for _, roID := range CalculateROIDs(neuronID, uniqTo) {
		roIDSet[roID] = struct{}{}
	}
	for i, toID := range uniqTo {
		_, recurrent := roIDSet[toID]
		synapses = append(synapses, model.Synapse{
			ID:        fmt.Sprintf("%s:link:out:%s:%d", neuronID, sanitizeID(toID), i),
			From:      neuronID,
			To:        toID,
			Weight:    randomCentered(rng),
			Enabled:   true,
			Recurrent: recurrent,
		})
	}
	return synapses, nil
}

// GenerateUniqueID mirrors genotype:generate_UniqueId/0 intent.
func GenerateUniqueID(rng *rand.Rand) float64 {
	seq := atomic.AddUint64(&uniqueIDSequence, 1)
	if rng != nil {
		return float64(seq) + rng.Float64()
	}
	seconds := float64(time.Now().UnixNano()) / float64(time.Second)
	if seconds <= 0 {
		seconds = float64(seq)
	}
	return 1 / (seconds + float64(seq)/1e9)
}

// GenerateIDs mirrors genotype:generate_ids/2.
func GenerateIDs(count int, rng *rand.Rand) []float64 {
	if count <= 0 {
		return nil
	}
	ids := make([]float64, 0, count)
	for i := 0; i < count; i++ {
		ids = append(ids, GenerateUniqueID(rng))
	}
	return ids
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

func uniqueNonEmpty(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func circuitActivationTag(values []string) (bool, string) {
	for _, value := range values {
		candidate := strings.TrimSpace(value)
		if candidate == "" {
			continue
		}
		lower := strings.ToLower(candidate)
		if lower == "circuit" {
			return true, "tanh"
		}
		if strings.HasPrefix(lower, "circuit:") {
			_, raw, _ := strings.Cut(candidate, ":")
			raw = strings.TrimSpace(raw)
			if raw == "" {
				return true, "tanh"
			}
			return true, raw
		}
	}
	return false, ""
}

func stripCircuitActivations(values []string) []string {
	filtered := make([]string, 0, len(values))
	for _, value := range values {
		candidate := strings.TrimSpace(value)
		if candidate == "" {
			continue
		}
		if ok, _ := circuitActivationTag([]string{candidate}); ok {
			continue
		}
		filtered = append(filtered, candidate)
	}
	return filtered
}
