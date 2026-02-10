package morphology

import protoio "protogonos/internal/io"

type XORMorphology struct{}

func (XORMorphology) Name() string {
	return "xor-v1"
}

func (XORMorphology) Sensors() []string {
	return []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName}
}

func (XORMorphology) Actuators() []string {
	return []string{protoio.XOROutputActuatorName}
}

func (XORMorphology) Compatible(scape string) bool {
	return scape == "xor"
}
