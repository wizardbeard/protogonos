package morphology

import protoio "protogonos/internal/io"

type FXMorphology struct{}
type FXMarketMorphology struct{}

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
		protoio.FXEntrySensorName,
		protoio.FXPercentChangeSensorName,
		protoio.FXPrevPercentChangeSensorName,
		protoio.FXProfitSensorName,
	}
}

func (FXMorphology) Actuators() []string {
	return []string{protoio.FXTradeActuatorName}
}

func (FXMorphology) Compatible(scape string) bool {
	return scape == "fx"
}

func (FXMarketMorphology) Name() string {
	return "fx-market-v1"
}

func (FXMarketMorphology) Sensors() []string {
	return []string{
		protoio.FXPriceSensorName,
		protoio.FXSignalSensorName,
	}
}

func (FXMarketMorphology) Actuators() []string {
	return []string{protoio.FXTradeActuatorName}
}

func (FXMarketMorphology) Compatible(scape string) bool {
	return scape == "fx"
}
