package morphology

import (
	"fmt"

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
	for _, sensorName := range genome.SensorIDs {
		if _, err := protoio.ResolveSensor(sensorName, scapeName); err != nil {
			return fmt.Errorf("genome %s sensor %s incompatible with scape %s: %w", genome.ID, sensorName, scapeName, err)
		}
	}
	for _, actuatorName := range genome.ActuatorIDs {
		if _, err := protoio.ResolveActuator(actuatorName, scapeName); err != nil {
			return fmt.Errorf("genome %s actuator %s incompatible with scape %s: %w", genome.ID, actuatorName, scapeName, err)
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
