package genotype

import (
	"fmt"
	"math/rand"
	"strings"

	"protogonos/internal/model"
	"protogonos/internal/morphology"
	"protogonos/internal/storage"
)

// TopologicalMutationOption captures one candidate topological-policy mode.
type TopologicalMutationOption struct {
	Name  string
	Param float64
}

// ConstructConstraint is a Go analog of the subset of #constraint{} fields
// used by genotype construction helpers.
type ConstructConstraint struct {
	Morphology                string
	NeuralAFs                 []string
	NeuralPFNs                []string
	NeuralAggrFs              []string
	TuningSelectionFs         []string
	AnnealingParameters       []float64
	PerturbationRanges        []float64
	AgentEncodingTypes        []string
	SubstratePlasticities     []string
	SubstrateLinkforms        []string
	HeredityTypes             []string
	TotTopologicalMutationsFs []TopologicalMutationOption
}

// ConstructedCortex is the Go analog return payload for construct_Cortex/6.
type ConstructedCortex struct {
	Genome          model.Genome
	InputNeuronIDs  []string
	OutputNeuronIDs []string
	Pattern         []PatternLayer
}

// ConstructedAgent is the Go analog return payload for construct_Agent/3.
type ConstructedAgent struct {
	Genome              model.Genome
	SpecieID            string
	EncodingType        string
	SubstratePlasticity string
	SubstrateLinkform   string
	TuningSelection     string
	AnnealingParameter  float64
	PerturbationRange   float64
	TopologicalMode     string
	TopologicalParam    float64
	HeredityType        string
	Pattern             []PatternLayer
	InputNeuronIDs      []string
	OutputNeuronIDs     []string
}

// DefaultConstructConstraint mirrors the reference default constraint intent in
// a Go-native form for construction helpers.
func DefaultConstructConstraint() ConstructConstraint {
	return ConstructConstraint{
		Morphology:          "xor_mimic",
		NeuralAFs:           []string{"tanh", "cos", "gaussian"},
		NeuralPFNs:          []string{"none"},
		NeuralAggrFs:        []string{"dot_product"},
		TuningSelectionFs:   []string{"dynamic_random"},
		AnnealingParameters: []float64{0.5},
		PerturbationRanges:  []float64{1},
		AgentEncodingTypes:  []string{"neural"},
		SubstratePlasticities: []string{
			"none",
		},
		SubstrateLinkforms: []string{
			"l2l_feedforward",
		},
		HeredityTypes: []string{"darwinian"},
		TotTopologicalMutationsFs: []TopologicalMutationOption{
			{Name: "ncount_exponential", Param: 0.5},
		},
	}
}

// ConstructAgent is a Go analog of genotype:construct_Agent/3.
func ConstructAgent(specieID, agentID string, constraint ConstructConstraint, rng *rand.Rand) (ConstructedAgent, error) {
	if strings.TrimSpace(agentID) == "" {
		return ConstructedAgent{}, fmt.Errorf("agent id is required")
	}
	rng = ensureRNG(rng)

	encodingType := pickString(rng, constraint.AgentEncodingTypes, "neural")
	substratePlasticity := pickString(rng, constraint.SubstratePlasticities, "none")
	substrateLinkform := pickString(rng, constraint.SubstrateLinkforms, "l2l_feedforward")

	cortex, err := ConstructCortex(
		agentID,
		0,
		constraint,
		encodingType,
		substratePlasticity,
		substrateLinkform,
		rng,
	)
	if err != nil {
		return ConstructedAgent{}, err
	}

	tuningSelection := pickString(rng, constraint.TuningSelectionFs, "dynamic_random")
	annealing := pickFloat(rng, constraint.AnnealingParameters, 0.5)
	perturbationRange := pickFloat(rng, constraint.PerturbationRanges, 1)
	heredityType := pickString(rng, constraint.HeredityTypes, "darwinian")
	topologicalMode, topologicalParam := pickTopologicalMutationMode(rng, constraint.TotTopologicalMutationsFs)

	genome := cortex.Genome
	genome.Strategy = &model.StrategyConfig{
		TuningSelection:  tuningSelection,
		AnnealingFactor:  annealing,
		TopologicalMode:  topologicalMode,
		TopologicalParam: topologicalParam,
		HeredityType:     heredityType,
	}

	return ConstructedAgent{
		Genome:              genome,
		SpecieID:            specieID,
		EncodingType:        encodingType,
		SubstratePlasticity: substratePlasticity,
		SubstrateLinkform:   substrateLinkform,
		TuningSelection:     tuningSelection,
		AnnealingParameter:  annealing,
		PerturbationRange:   perturbationRange,
		TopologicalMode:     topologicalMode,
		TopologicalParam:    topologicalParam,
		HeredityType:        heredityType,
		Pattern:             append([]PatternLayer(nil), cortex.Pattern...),
		InputNeuronIDs:      append([]string(nil), cortex.InputNeuronIDs...),
		OutputNeuronIDs:     append([]string(nil), cortex.OutputNeuronIDs...),
	}, nil
}

// ConstructCortex is a Go analog of genotype:construct_Cortex/6.
func ConstructCortex(
	agentID string,
	generation int,
	constraint ConstructConstraint,
	encodingType string,
	substratePlasticity string,
	substrateLinkform string,
	rng *rand.Rand,
) (ConstructedCortex, error) {
	if strings.TrimSpace(agentID) == "" {
		return ConstructedCortex{}, fmt.Errorf("agent id is required")
	}
	rng = ensureRNG(rng)

	morph, err := resolveConstructMorphology(constraint.Morphology)
	if err != nil {
		return ConstructedCortex{}, err
	}

	sensors := append([]string(nil), morph.Sensors()...)
	actuators := append([]string(nil), morph.Actuators()...)
	seed, err := ConstructSeedNN(
		generation,
		sensors,
		actuators,
		constraint.NeuralAFs,
		constraint.NeuralPFNs,
		constraint.NeuralAggrFs,
		rng,
	)
	if err != nil {
		return ConstructedCortex{}, err
	}

	genome := model.Genome{
		VersionedRecord: model.VersionedRecord{
			SchemaVersion: storage.CurrentSchemaVersion,
			CodecVersion:  storage.CurrentCodecVersion,
		},
		ID:                  agentID,
		Neurons:             append([]model.Neuron(nil), seed.Neurons...),
		Synapses:            append([]model.Synapse(nil), seed.Synapses...),
		SensorIDs:           sensors,
		ActuatorIDs:         actuators,
		SensorNeuronLinks:   append([]model.SensorNeuronLink(nil), seed.SensorNeuronLinks...),
		NeuronActuatorLinks: append([]model.NeuronActuatorLink(nil), seed.NeuronActuatorLinks...),
		SensorLinks:         len(seed.SensorNeuronLinks),
		ActuatorLinks:       len(seed.NeuronActuatorLinks),
	}

	if strings.EqualFold(strings.TrimSpace(encodingType), "substrate") {
		formatInputs := make([]any, 0, len(sensors))
		for _, sensorID := range sensors {
			formatInputs = append(formatInputs, sensorID)
		}
		formatOutputs := make([]any, 0, len(actuators))
		for _, actuatorID := range actuators {
			formatOutputs = append(formatOutputs, actuatorID)
		}
		optimalDimension := CalculateOptimalSubstrateDimension(formatInputs, formatOutputs)
		densities := defaultSubstrateDensities(optimalDimension)
		genome.Substrate = &model.SubstrateConfig{
			CPPName:    strings.TrimSpace(substratePlasticity),
			CEPName:    strings.TrimSpace(substrateLinkform),
			Dimensions: densities,
			Parameters: map[string]float64{
				"depth":      1,
				"density":    5,
				"dimensions": float64(optimalDimension),
			},
		}
	}

	return ConstructedCortex{
		Genome:          genome,
		InputNeuronIDs:  append([]string(nil), seed.InputNeuronIDs...),
		OutputNeuronIDs: append([]string(nil), seed.OutputNeuronIDs...),
		Pattern:         append([]PatternLayer(nil), seed.Pattern...),
	}, nil
}

func resolveConstructMorphology(raw string) (morphology.Morphology, error) {
	key := strings.ToLower(strings.TrimSpace(raw))
	key = strings.ReplaceAll(key, "_", "-")
	switch key {
	case "", "xor", "xor-v1", "xor-mimic":
		return morphology.XORMorphology{}, nil
	case "regression-mimic", "regression-mimic-v1":
		return morphology.RegressionMimicMorphology{}, nil
	case "cart-pole-lite", "cart-pole-lite-v1":
		return morphology.CartPoleLiteMorphology{}, nil
	case "flatland", "flatland-v1":
		return morphology.FlatlandMorphology{}, nil
	case "gtsa", "gtsa-v1":
		return morphology.GTSAMorphology{}, nil
	case "fx", "fx-v1":
		return morphology.FXMorphology{}, nil
	default:
		return nil, fmt.Errorf("unsupported morphology: %s", raw)
	}
}

func pickString(rng *rand.Rand, values []string, fallback string) string {
	value, err := RandomElement(rng, values)
	if err != nil {
		return fallback
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func pickFloat(rng *rand.Rand, values []float64, fallback float64) float64 {
	value, err := RandomElement(rng, values)
	if err != nil {
		return fallback
	}
	return value
}

func pickTopologicalMutationMode(rng *rand.Rand, values []TopologicalMutationOption) (string, float64) {
	value, err := RandomElement(rng, values)
	if err != nil || strings.TrimSpace(value.Name) == "" {
		return "ncount_exponential", 0.5
	}
	return strings.TrimSpace(value.Name), value.Param
}

func defaultSubstrateDensities(dimension int) []int {
	if dimension < 2 {
		dimension = 2
	}
	densities := make([]int, 0, dimension)
	densities = append(densities, 1, 1)
	for i := 2; i < dimension; i++ {
		densities = append(densities, 5)
	}
	return densities
}
