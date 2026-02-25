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
	}
	for _, sensor := range expected {
		if !slices.Contains(sensors, sensor) {
			t.Fatalf("expected flatland morphology to include sensor %s, got=%v", sensor, sensors)
		}
	}
}
