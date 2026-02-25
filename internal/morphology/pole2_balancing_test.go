package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

func TestPole2BalancingMorphologyCompatibility(t *testing.T) {
	m := Pole2BalancingMorphology{}
	if !m.Compatible("pole2-balancing") {
		t.Fatal("expected pole2-balancing to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
	sensors := m.Sensors()
	if len(sensors) != 9 {
		t.Fatalf("expected 9 pole2 sensors, got %d (%v)", len(sensors), sensors)
	}
	found := map[string]bool{}
	for _, id := range sensors {
		found[id] = true
	}
	for _, want := range []string{
		protoio.Pole2RunProgressSensorName,
		protoio.Pole2StepProgressSensorName,
		protoio.Pole2FitnessSignalSensorName,
	} {
		if !found[want] {
			t.Fatalf("expected pole2 morphology sensor %s in %v", want, sensors)
		}
	}
}

func TestEnsureScapeCompatibilityPole2Balancing(t *testing.T) {
	if err := EnsureScapeCompatibility("pole2-balancing"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
