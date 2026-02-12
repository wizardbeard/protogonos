package morphology

import "testing"

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
