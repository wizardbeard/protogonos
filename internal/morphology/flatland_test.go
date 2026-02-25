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
	for _, sensor := range expected {
		if !slices.Contains(sensors, sensor) {
			t.Fatalf("expected flatland morphology to include sensor %s, got=%v", sensor, sensors)
		}
	}
}
