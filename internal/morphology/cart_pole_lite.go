package morphology

import protoio "protogonos/internal/io"

type CartPoleLiteMorphology struct{}

func (CartPoleLiteMorphology) Name() string {
	return "cart-pole-lite-v1"
}

func (CartPoleLiteMorphology) Sensors() []string {
	return []string{protoio.CartPolePositionSensorName, protoio.CartPoleVelocitySensorName}
}

func (CartPoleLiteMorphology) Actuators() []string {
	return []string{protoio.CartPoleForceActuatorName}
}

func (CartPoleLiteMorphology) Compatible(scape string) bool {
	return scape == "cart-pole-lite"
}
