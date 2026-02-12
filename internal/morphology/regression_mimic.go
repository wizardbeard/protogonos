package morphology

import (
	"fmt"

	protoio "protogonos/internal/io"
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
	m, ok := defaultMorphologyForScape(scapeName)
	if !ok {
		return nil
	}
	return ValidateRegisteredComponents(scapeName, m)
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
	switch scapeName {
	case "xor":
		return XORMorphology{}, true
	case "regression-mimic":
		return RegressionMimicMorphology{}, true
	case "cart-pole-lite":
		return CartPoleLiteMorphology{}, true
	case "flatland":
		return FlatlandMorphology{}, true
	case "gtsa":
		return GTSAMorphology{}, true
	default:
		return nil, false
	}
}
