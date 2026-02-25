package morphology

import protoio "protogonos/internal/io"

type FlatlandMorphology struct{}

func (FlatlandMorphology) Name() string {
	return "flatland-v1"
}

func (FlatlandMorphology) Sensors() []string {
	return []string{
		protoio.FlatlandDistanceSensorName,
		protoio.FlatlandEnergySensorName,
		protoio.FlatlandPoisonSensorName,
		protoio.FlatlandWallSensorName,
		protoio.FlatlandFoodProximitySensorName,
		protoio.FlatlandPoisonProximitySensorName,
		protoio.FlatlandWallProximitySensorName,
		protoio.FlatlandResourceBalanceSensorName,
	}
}

func (FlatlandMorphology) Actuators() []string {
	return []string{protoio.FlatlandMoveActuatorName}
}

func (FlatlandMorphology) Compatible(scape string) bool {
	return scape == "flatland"
}
