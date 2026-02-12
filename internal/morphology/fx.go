package morphology

import protoio "protogonos/internal/io"

type FXMorphology struct{}

func (FXMorphology) Name() string {
	return "fx-v1"
}

func (FXMorphology) Sensors() []string {
	return []string{
		protoio.FXPriceSensorName,
		protoio.FXSignalSensorName,
	}
}

func (FXMorphology) Actuators() []string {
	return []string{protoio.FXTradeActuatorName}
}

func (FXMorphology) Compatible(scape string) bool {
	return scape == "fx"
}
