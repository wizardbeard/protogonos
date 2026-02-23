package morphology

import "testing"

func TestPole2BalancingMorphologyCompatibility(t *testing.T) {
	m := Pole2BalancingMorphology{}
	if !m.Compatible("pole2-balancing") {
		t.Fatal("expected pole2-balancing to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityPole2Balancing(t *testing.T) {
	if err := EnsureScapeCompatibility("pole2-balancing"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
