package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

func TestEpitopesMorphologyCompatibility(t *testing.T) {
	m := EpitopesMorphology{}
	if !m.Compatible("epitopes") {
		t.Fatal("expected epitopes to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityEpitopes(t *testing.T) {
	if err := EnsureScapeCompatibility("epitopes"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}

func TestEpitopesMorphologyDefaultSensors(t *testing.T) {
	m := EpitopesMorphology{}
	sensors := m.Sensors()
	if len(sensors) != 5 {
		t.Fatalf("expected 5 epitopes sensors, got %d: %#v", len(sensors), sensors)
	}
	if sensors[0] != protoio.EpitopesSignalSensorName ||
		sensors[1] != protoio.EpitopesMemorySensorName ||
		sensors[2] != protoio.EpitopesTargetSensorName ||
		sensors[3] != protoio.EpitopesProgressSensorName ||
		sensors[4] != protoio.EpitopesMarginSensorName {
		t.Fatalf("unexpected epitopes sensor order: %#v", sensors)
	}
}
