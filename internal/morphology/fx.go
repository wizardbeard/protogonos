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
		protoio.FXMomentumSensorName,
		protoio.FXVolatilitySensorName,
		protoio.FXNAVSensorName,
		protoio.FXDrawdownSensorName,
		protoio.FXPositionSensorName,
	}
}

func (FXMorphology) Actuators() []string {
	return []string{protoio.FXTradeActuatorName}
}

func (FXMorphology) Compatible(scape string) bool {
	return scape == "fx"
}
