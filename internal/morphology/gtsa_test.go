package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

func TestGTSAMorphologyCompatibility(t *testing.T) {
	m := GTSAMorphology{}
	if !m.Compatible("gtsa") {
		t.Fatal("expected gtsa to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityGTSA(t *testing.T) {
	if err := EnsureScapeCompatibility("gtsa"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}

func TestGTSAMorphologyDefaultSensors(t *testing.T) {
	m := GTSAMorphology{}
	sensors := m.Sensors()
	if len(sensors) != 4 {
		t.Fatalf("expected 4 gtsa sensors, got %d: %#v", len(sensors), sensors)
	}
	if sensors[0] != protoio.GTSAInputSensorName ||
		sensors[1] != protoio.GTSADeltaSensorName ||
		sensors[2] != protoio.GTSAWindowMeanSensorName ||
		sensors[3] != protoio.GTSAProgressSensorName {
		t.Fatalf("unexpected gtsa sensor order: %#v", sensors)
	}
}
