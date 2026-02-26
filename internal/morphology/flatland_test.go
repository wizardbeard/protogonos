package morphology

import (
	"slices"
	"testing"

	protoio "protogonos/internal/io"
)

func TestFlatlandMorphologyCompatibility(t *testing.T) {
	m := FlatlandMorphology{}
	if !m.Compatible("flatland") {
		t.Fatal("expected flatland to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityFlatland(t *testing.T) {
	if err := EnsureScapeCompatibility("flatland"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}

func TestFlatlandMorphologyIncludesExtendedSensors(t *testing.T) {
	m := FlatlandMorphology{}
	sensors := m.Sensors()
	expected := []string{
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
	for _, sensor := range expected {
		if !slices.Contains(sensors, sensor) {
			t.Fatalf("expected flatland morphology to include sensor %s, got=%v", sensor, sensors)
		}
	}
}

func TestFlatlandScannerMorphologyCompatibility(t *testing.T) {
	m := FlatlandScannerMorphology{}
	if !m.Compatible("flatland") {
		t.Fatal("expected flatland scanner morphology to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestFlatlandScannerMorphologySurface(t *testing.T) {
	m := FlatlandScannerMorphology{}
	sensors := m.Sensors()
	if !slices.Contains(sensors, protoio.FlatlandEnergySensorName) {
		t.Fatalf("expected scanner profile to include energy reader channel, got=%v", sensors)
	}
	if !slices.Contains(sensors, protoio.FlatlandPreySensorName) ||
		!slices.Contains(sensors, protoio.FlatlandPredatorSensorName) ||
		!slices.Contains(sensors, protoio.FlatlandPreyProximitySensorName) ||
		!slices.Contains(sensors, protoio.FlatlandPredatorProximitySensorName) {
		t.Fatalf("expected scanner profile to include social channels, got=%v", sensors)
	}
	if !slices.Contains(sensors, protoio.FlatlandDistanceScan0SensorName) ||
		!slices.Contains(sensors, protoio.FlatlandDistanceScan4SensorName) ||
		!slices.Contains(sensors, protoio.FlatlandColorScan0SensorName) ||
		!slices.Contains(sensors, protoio.FlatlandEnergyScan4SensorName) {
		t.Fatalf("expected scanner profile to include all scanner families, got=%v", sensors)
	}
	actuators := m.Actuators()
	if len(actuators) != 1 || actuators[0] != protoio.FlatlandTwoWheelsActuatorName {
		t.Fatalf("expected two_wheels actuator profile, got=%v", actuators)
	}
	if err := ValidateRegisteredComponents("flatland", m); err != nil {
		t.Fatalf("validate scanner profile components: %v", err)
	}
}
