package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

type aliasMorphology struct{}

func (aliasMorphology) Name() string { return "alias-v1" }
func (aliasMorphology) Sensors() []string {
	return []string{
		protoio.XORGetInputSensorAliasName,
		protoio.DistanceScannerSensorAliasName,
	}
}
func (aliasMorphology) Actuators() []string {
	return []string{protoio.XORSendOutputActuatorAliasName}
}
func (aliasMorphology) Compatible(string) bool { return true }

func TestGetInitSensorsAndActuators(t *testing.T) {
	sensors, err := GetInitSensors("pb_sim", "6")
	if err != nil {
		t.Fatalf("get init sensors: %v", err)
	}
	if len(sensors) != 1 || sensors[0] != protoio.Pole2CartPositionSensorName {
		t.Fatalf("unexpected initial pole2 sensors: %v", sensors)
	}

	actuators, err := GetInitActuators("pb_sim", "6")
	if err != nil {
		t.Fatalf("get init actuators: %v", err)
	}
	if len(actuators) != 1 || actuators[0] != protoio.Pole2PushActuatorName {
		t.Fatalf("unexpected initial pole2 actuators: %v", actuators)
	}
}

func TestGetSensorsAndActuatorsReturnCanonicalCopies(t *testing.T) {
	sensors, err := GetSensors("flatland", "scanner")
	if err != nil {
		t.Fatalf("get sensors: %v", err)
	}
	if len(sensors) == 0 || sensors[0] != protoio.FlatlandDistanceScan0SensorName {
		t.Fatalf("unexpected flatland scanner sensors: %v", sensors)
	}
	sensors[0] = "mutated"
	again, err := GetSensors("flatland", "scanner")
	if err != nil {
		t.Fatalf("get sensors again: %v", err)
	}
	if again[0] == "mutated" {
		t.Fatalf("expected copied sensor slice, got=%v", again)
	}

	actuators, err := GetActuators("fx_sim", "market")
	if err != nil {
		t.Fatalf("get actuators: %v", err)
	}
	if len(actuators) != 1 || actuators[0] != protoio.FXTradeActuatorName {
		t.Fatalf("unexpected fx actuators: %v", actuators)
	}
}

func TestDirectMorphologyHelpersCanonicalizeLegacyAliases(t *testing.T) {
	m := aliasMorphology{}

	sensors := Sensors(m)
	if len(sensors) != 2 ||
		sensors[0] != protoio.XORInputLeftSensorName ||
		sensors[1] != protoio.FlatlandDistanceScan0SensorName {
		t.Fatalf("unexpected canonical sensors: %v", sensors)
	}
	if initSensors := InitSensors(m); len(initSensors) != 1 || initSensors[0] != protoio.XORInputLeftSensorName {
		t.Fatalf("unexpected initial canonical sensor: %v", initSensors)
	}

	actuators := Actuators(m)
	if len(actuators) != 1 || actuators[0] != protoio.XOROutputActuatorName {
		t.Fatalf("unexpected canonical actuators: %v", actuators)
	}
	if initActuators := InitActuators(m); len(initActuators) != 1 || initActuators[0] != protoio.XOROutputActuatorName {
		t.Fatalf("unexpected initial canonical actuator: %v", initActuators)
	}
}

func TestDirectMorphologyHelpersHandleNil(t *testing.T) {
	if got := Sensors(nil); got != nil {
		t.Fatalf("expected nil sensors, got=%v", got)
	}
	if got := Actuators(nil); got != nil {
		t.Fatalf("expected nil actuators, got=%v", got)
	}
	if got := InitSensors(nil); got != nil {
		t.Fatalf("expected nil init sensors, got=%v", got)
	}
	if got := InitActuators(nil); got != nil {
		t.Fatalf("expected nil init actuators, got=%v", got)
	}
}
