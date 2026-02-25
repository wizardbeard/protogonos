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
		protoio.FlatlandDistanceScan0SensorName,
		protoio.FlatlandDistanceScan1SensorName,
		protoio.FlatlandDistanceScan2SensorName,
		protoio.FlatlandDistanceScan3SensorName,
		protoio.FlatlandDistanceScan4SensorName,
		protoio.FlatlandColorScan0SensorName,
		protoio.FlatlandColorScan1SensorName,
		protoio.FlatlandColorScan2SensorName,
		protoio.FlatlandColorScan3SensorName,
		protoio.FlatlandColorScan4SensorName,
		protoio.FlatlandEnergyScan0SensorName,
		protoio.FlatlandEnergyScan1SensorName,
		protoio.FlatlandEnergyScan2SensorName,
		protoio.FlatlandEnergyScan3SensorName,
		protoio.FlatlandEnergyScan4SensorName,
	}
}

func (FlatlandMorphology) Actuators() []string {
	return []string{protoio.FlatlandMoveActuatorName}
}

func (FlatlandMorphology) Compatible(scape string) bool {
	return scape == "flatland"
}
