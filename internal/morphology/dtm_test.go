package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

func TestDTMMorphologyCompatibility(t *testing.T) {
	m := DTMMorphology{}
	if !m.Compatible("dtm") {
		t.Fatal("expected dtm to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityDTM(t *testing.T) {
	if err := EnsureScapeCompatibility("dtm"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}

func TestDTMMorphologyDefaultSensors(t *testing.T) {
	m := DTMMorphology{}
	sensors := m.Sensors()
	if len(sensors) != 7 {
		t.Fatalf("expected 7 dtm sensors, got %d: %#v", len(sensors), sensors)
	}
	if sensors[0] != protoio.DTMRangeLeftSensorName ||
		sensors[1] != protoio.DTMRangeFrontSensorName ||
		sensors[2] != protoio.DTMRangeRightSensorName ||
		sensors[3] != protoio.DTMRewardSensorName ||
		sensors[4] != protoio.DTMRunProgressSensorName ||
		sensors[5] != protoio.DTMStepProgressSensorName ||
		sensors[6] != protoio.DTMSwitchedSensorName {
		t.Fatalf("unexpected dtm sensor order: %#v", sensors)
	}
}
