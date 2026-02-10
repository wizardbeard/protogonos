package morphology

// Morphology defines allowed sensor/actuator combinations for a scape.
type Morphology interface {
	Name() string
	Sensors() []string
	Actuators() []string
	Compatible(scape string) bool
}
