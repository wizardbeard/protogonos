package morphology

import (
	"fmt"
	"strings"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/scapeid"
)

type RegressionMimicMorphology struct{}

func (RegressionMimicMorphology) Name() string {
	return "regression-mimic-v1"
}

func (RegressionMimicMorphology) Sensors() []string {
	return []string{protoio.ScalarInputSensorName}
}

func (RegressionMimicMorphology) Actuators() []string {
	return []string{protoio.ScalarOutputActuatorName}
}

func (RegressionMimicMorphology) Compatible(scape string) bool {
	return scape == "regression-mimic"
}

func EnsureScapeCompatibility(scapeName string) error {
	scapeName = scapeid.Normalize(scapeName)
	m, ok := defaultMorphologyForScape(scapeName)
	if !ok {
		return nil
	}
	return ValidateRegisteredComponents(scapeName, m)
}

func EnsureGenomeIOCompatibility(scapeName string, genome model.Genome) error {
	scapeName = scapeid.Normalize(scapeName)
	knownNeuronIDs := make(map[string]struct{}, len(genome.Neurons))
	for _, neuron := range genome.Neurons {
		id := strings.TrimSpace(neuron.ID)
		if id == "" {
			continue
		}
		knownNeuronIDs[id] = struct{}{}
	}
	knownSensorIDs := make(map[string]struct{}, len(genome.SensorIDs))
	for _, sensorName := range genome.SensorIDs {
		if _, err := protoio.ResolveSensor(sensorName, scapeName); err != nil {
			return fmt.Errorf("genome %s sensor %s incompatible with scape %s: %w", genome.ID, sensorName, scapeName, err)
		}
		knownSensorIDs[sensorName] = struct{}{}
	}
	if genome.Substrate != nil {
		for _, sensorID := range genome.Substrate.CPPIDs {
			sensorID = strings.TrimSpace(sensorID)
			if sensorID == "" {
				continue
			}
			knownSensorIDs[sensorID] = struct{}{}
		}
	}
	knownActuatorIDs := make(map[string]struct{}, len(genome.ActuatorIDs))
	for _, actuatorName := range genome.ActuatorIDs {
		if _, err := protoio.ResolveActuator(actuatorName, scapeName); err != nil {
			return fmt.Errorf("genome %s actuator %s incompatible with scape %s: %w", genome.ID, actuatorName, scapeName, err)
		}
		knownActuatorIDs[actuatorName] = struct{}{}
	}
	if genome.Substrate != nil {
		for _, actuatorID := range genome.Substrate.CEPIDs {
			actuatorID = strings.TrimSpace(actuatorID)
			if actuatorID == "" {
				continue
			}
			knownActuatorIDs[actuatorID] = struct{}{}
		}
	}
	for _, link := range genome.SensorNeuronLinks {
		sensorID := strings.TrimSpace(link.SensorID)
		neuronID := strings.TrimSpace(link.NeuronID)
		if sensorID == "" || neuronID == "" {
			return fmt.Errorf("genome %s has malformed sensor link: sensor=%q neuron=%q", genome.ID, link.SensorID, link.NeuronID)
		}
		if _, ok := knownSensorIDs[sensorID]; !ok {
			return fmt.Errorf("genome %s sensor link references unknown sensor %s", genome.ID, sensorID)
		}
		if _, ok := knownNeuronIDs[neuronID]; !ok {
			return fmt.Errorf("genome %s sensor link references unknown neuron %s", genome.ID, neuronID)
		}
	}
	for _, link := range genome.NeuronActuatorLinks {
		neuronID := strings.TrimSpace(link.NeuronID)
		actuatorID := strings.TrimSpace(link.ActuatorID)
		if neuronID == "" || actuatorID == "" {
			return fmt.Errorf("genome %s has malformed actuator link: neuron=%q actuator=%q", genome.ID, link.NeuronID, link.ActuatorID)
		}
		if _, ok := knownNeuronIDs[neuronID]; !ok {
			return fmt.Errorf("genome %s actuator link references unknown neuron %s", genome.ID, neuronID)
		}
		if _, ok := knownActuatorIDs[actuatorID]; !ok {
			return fmt.Errorf("genome %s actuator link references unknown actuator %s", genome.ID, actuatorID)
		}
	}
	return nil
}

func EnsurePopulationIOCompatibility(scapeName string, genomes []model.Genome) error {
	for _, genome := range genomes {
		if err := EnsureGenomeIOCompatibility(scapeName, genome); err != nil {
			return err
		}
	}
	return nil
}

func ValidateRegisteredComponents(scapeName string, m Morphology) error {
	if !m.Compatible(scapeName) {
		return fmt.Errorf("morphology %s incompatible with scape %s", m.Name(), scapeName)
	}

	for _, sensorName := range m.Sensors() {
		if _, err := protoio.ResolveSensor(sensorName, scapeName); err != nil {
			return fmt.Errorf("resolve sensor %s: %w", sensorName, err)
		}
	}
	for _, actuatorName := range m.Actuators() {
		if _, err := protoio.ResolveActuator(actuatorName, scapeName); err != nil {
			return fmt.Errorf("resolve actuator %s: %w", actuatorName, err)
		}
	}
	return nil
}

func defaultMorphologyForScape(scapeName string) (Morphology, bool) {
	scapeName = scapeid.Normalize(scapeName)
	switch scapeName {
	case "xor":
		return XORMorphology{}, true
	case "regression-mimic":
		return RegressionMimicMorphology{}, true
	case "cart-pole-lite":
		return CartPoleLiteMorphology{}, true
	case "pole2-balancing":
		return Pole2BalancingMorphology{}, true
	case "flatland":
		return FlatlandMorphology{}, true
	case "dtm":
		return DTMMorphology{}, true
	case "gtsa":
		return GTSAMorphology{}, true
	case "fx":
		return FXMorphology{}, true
	case "epitopes":
		return EpitopesMorphology{}, true
	case "llvm-phase-ordering":
		return LLVMPhaseOrderingMorphology{}, true
	default:
		return nil, false
	}
}
