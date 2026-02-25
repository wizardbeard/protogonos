package morphology

import protoio "protogonos/internal/io"

type EpitopesMorphology struct{}

func (EpitopesMorphology) Name() string {
	return "epitopes-v1"
}

func (EpitopesMorphology) Sensors() []string {
	return []string{
		protoio.EpitopesSignalSensorName,
		protoio.EpitopesMemorySensorName,
		protoio.EpitopesTargetSensorName,
		protoio.EpitopesProgressSensorName,
		protoio.EpitopesMarginSensorName,
	}
}

func (EpitopesMorphology) Actuators() []string {
	return []string{protoio.EpitopesResponseActuatorName}
}

func (EpitopesMorphology) Compatible(scape string) bool {
	return scape == "epitopes"
}
