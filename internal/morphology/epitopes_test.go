package morphology

import "testing"

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
