package morphology

import protoio "protogonos/internal/io"

type GTSAMorphology struct{}

func (GTSAMorphology) Name() string {
	return "gtsa-v1"
}

func (GTSAMorphology) Sensors() []string {
	return []string{
		protoio.GTSAInputSensorName,
		protoio.GTSADeltaSensorName,
		protoio.GTSAWindowMeanSensorName,
		protoio.GTSAProgressSensorName,
	}
}

func (GTSAMorphology) Actuators() []string {
	return []string{protoio.GTSAPredictActuatorName}
}

func (GTSAMorphology) Compatible(scape string) bool {
	return scape == "gtsa"
}
