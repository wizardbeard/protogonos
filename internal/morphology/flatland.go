package morphology

import protoio "protogonos/internal/io"

type FlatlandMorphology struct{}
type FlatlandScannerMorphology struct{}

func (FlatlandMorphology) Name() string {
	return "flatland-v1"
}

func (FlatlandMorphology) Sensors() []string {
	return flatlandExtendedSensors()
}

func (FlatlandMorphology) Actuators() []string {
	return []string{protoio.FlatlandMoveActuatorName}
}

func (FlatlandMorphology) Compatible(scape string) bool {
	return scape == "flatland"
}

func (FlatlandScannerMorphology) Name() string {
	return "flatland-scanner-v1"
}

func (FlatlandScannerMorphology) Sensors() []string {
	// Mirrors reference-style scanner-heavy prey profile:
	// distance/color/energy scanners + internal scalar channels.
	return append(flatlandScannerSensors(), flatlandScannerStateSensors()...)
}

func (FlatlandScannerMorphology) Actuators() []string {
	return []string{protoio.FlatlandTwoWheelsActuatorName}
}

func (FlatlandScannerMorphology) Compatible(scape string) bool {
	return scape == "flatland"
}

func flatlandExtendedSensors() []string {
	base := []string{
		protoio.FlatlandDistanceSensorName,
		protoio.FlatlandEnergySensorName,
		protoio.FlatlandPreySensorName,
		protoio.FlatlandPredatorSensorName,
		protoio.FlatlandPoisonSensorName,
		protoio.FlatlandWallSensorName,
		protoio.FlatlandFoodProximitySensorName,
		protoio.FlatlandPreyProximitySensorName,
		protoio.FlatlandPredatorProximitySensorName,
		protoio.FlatlandPoisonProximitySensorName,
		protoio.FlatlandWallProximitySensorName,
		protoio.FlatlandResourceBalanceSensorName,
	}
	return append(base, flatlandScannerSensors()...)
}

func flatlandScannerStateSensors() []string {
	return []string{
		protoio.FlatlandEnergySensorName,
		protoio.FlatlandPreySensorName,
		protoio.FlatlandPredatorSensorName,
		protoio.FlatlandPreyProximitySensorName,
		protoio.FlatlandPredatorProximitySensorName,
	}
}

func flatlandScannerSensors() []string {
	return []string{
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
