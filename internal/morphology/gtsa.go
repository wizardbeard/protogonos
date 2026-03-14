package morphology

import protoio "protogonos/internal/io"

type GTSAMorphology struct{}
type GTSACoreMorphology struct{}

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

func (GTSACoreMorphology) Name() string {
	return "gtsa-core-v1"
}

func (GTSACoreMorphology) Sensors() []string {
	return []string{
		protoio.GTSAInputSensorName,
	}
}

func (GTSACoreMorphology) Actuators() []string {
	return []string{protoio.GTSAPredictActuatorName}
}

func (GTSACoreMorphology) Compatible(scape string) bool {
	return scape == "gtsa"
}
