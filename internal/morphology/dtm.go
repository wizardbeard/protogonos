package morphology

import protoio "protogonos/internal/io"

type DTMMorphology struct{}

func (DTMMorphology) Name() string {
	return "dtm-v1"
}

func (DTMMorphology) Sensors() []string {
	return []string{
		protoio.DTMRangeLeftSensorName,
		protoio.DTMRangeFrontSensorName,
		protoio.DTMRangeRightSensorName,
		protoio.DTMRewardSensorName,
	}
}

func (DTMMorphology) Actuators() []string {
	return []string{protoio.DTMMoveActuatorName}
}

func (DTMMorphology) Compatible(scape string) bool {
	return scape == "dtm"
}
