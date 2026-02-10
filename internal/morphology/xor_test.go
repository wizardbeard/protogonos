package morphology

import "testing"

func TestXORMorphologyCompatibility(t *testing.T) {
	m := XORMorphology{}
	if !m.Compatible("xor") {
		t.Fatal("expected xor to be compatible")
	}
	if m.Compatible("regression-mimic") {
		t.Fatal("expected regression-mimic to be incompatible")
	}
}

func TestEnsureScapeCompatibilityXOR(t *testing.T) {
	if err := EnsureScapeCompatibility("xor"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
