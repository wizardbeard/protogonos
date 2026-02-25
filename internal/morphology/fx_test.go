package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

func TestFXMorphologyCompatibility(t *testing.T) {
	m := FXMorphology{}
	if !m.Compatible("fx") {
		t.Fatal("expected fx to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
	sensors := m.Sensors()
	if len(sensors) != 11 {
		t.Fatalf("expected 11 fx sensors, got %d (%v)", len(sensors), sensors)
	}
	found := map[string]bool{}
	for _, id := range sensors {
		found[id] = true
	}
	if !found[protoio.FXPrevPercentChangeSensorName] {
		t.Fatalf("expected fx morphology sensor %s in %v", protoio.FXPrevPercentChangeSensorName, sensors)
	}
}

func TestEnsureScapeCompatibilityFX(t *testing.T) {
	if err := EnsureScapeCompatibility("fx"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
