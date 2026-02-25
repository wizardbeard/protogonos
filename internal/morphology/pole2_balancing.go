package morphology

import protoio "protogonos/internal/io"

type Pole2BalancingMorphology struct{}

func (Pole2BalancingMorphology) Name() string {
	return "pole2-balancing-v1"
}

func (Pole2BalancingMorphology) Sensors() []string {
	return []string{
		protoio.Pole2CartPositionSensorName,
		protoio.Pole2CartVelocitySensorName,
		protoio.Pole2Angle1SensorName,
		protoio.Pole2Velocity1SensorName,
		protoio.Pole2Angle2SensorName,
		protoio.Pole2Velocity2SensorName,
		protoio.Pole2RunProgressSensorName,
		protoio.Pole2StepProgressSensorName,
		protoio.Pole2FitnessSignalSensorName,
	}
}

func (Pole2BalancingMorphology) Actuators() []string {
	return []string{protoio.Pole2PushActuatorName}
}

func (Pole2BalancingMorphology) Compatible(scape string) bool {
	return scape == "pole2-balancing"
}
