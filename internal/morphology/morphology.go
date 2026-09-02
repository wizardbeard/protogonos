package morphology

import protoio "protogonos/internal/io"

// Morphology defines allowed sensor/actuator combinations for a scape.
type Morphology interface {
	Name() string
	Sensors() []string
	Actuators() []string
	Compatible(scape string) bool
}

func GetSensors(scapeName, profile string) ([]string, error) {
	m, err := ConstructMorphology(scapeName, profile)
	if err != nil {
		return nil, err
	}
	return Sensors(m), nil
}

func GetActuators(scapeName, profile string) ([]string, error) {
	m, err := ConstructMorphology(scapeName, profile)
	if err != nil {
		return nil, err
	}
	return Actuators(m), nil
}

func GetInitSensors(scapeName, profile string) ([]string, error) {
	m, err := ConstructMorphology(scapeName, profile)
	if err != nil {
		return nil, err
	}
	return InitSensors(m), nil
}

func GetInitActuators(scapeName, profile string) ([]string, error) {
	m, err := ConstructMorphology(scapeName, profile)
	if err != nil {
		return nil, err
	}
	return InitActuators(m), nil
}

func Sensors(m Morphology) []string {
	if m == nil {
		return nil
	}
	return canonicalSensorNames(m.Sensors())
}

func Actuators(m Morphology) []string {
	if m == nil {
		return nil
	}
	return canonicalActuatorNames(m.Actuators())
}

func InitSensors(m Morphology) []string {
	sensors := Sensors(m)
	if len(sensors) == 0 {
		return nil
	}
	return sensors[:1]
}

func InitActuators(m Morphology) []string {
	actuators := Actuators(m)
	if len(actuators) == 0 {
		return nil
	}
	return actuators[:1]
}

func canonicalSensorNames(names []string) []string {
	out := make([]string, 0, len(names))
	for _, name := range names {
		if canonical := protoio.CanonicalSensorName(name); canonical != "" {
			out = append(out, canonical)
		}
	}
	return out
}

func canonicalActuatorNames(names []string) []string {
	out := make([]string, 0, len(names))
	for _, name := range names {
		if canonical := protoio.CanonicalActuatorName(name); canonical != "" {
			out = append(out, canonical)
		}
	}
	return out
}
