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

func TestFXMarketMorphologyCompatibility(t *testing.T) {
	m := FXMarketMorphology{}
	if !m.Compatible("fx") {
		t.Fatal("expected fx market profile to be compatible")
	}
	sensors := m.Sensors()
	if len(sensors) != 2 {
		t.Fatalf("expected 2 fx market sensors, got %d (%v)", len(sensors), sensors)
	}
	if sensors[0] != protoio.FXPriceSensorName || sensors[1] != protoio.FXSignalSensorName {
		t.Fatalf("unexpected fx market sensor order: %v", sensors)
	}
	if err := ValidateRegisteredComponents("fx", m); err != nil {
		t.Fatalf("validate fx market profile components: %v", err)
	}
}
