package morphology

import "testing"

func TestRegressionMimicMorphologyCompatibility(t *testing.T) {
	m := RegressionMimicMorphology{}
	if !m.Compatible("regression-mimic") {
		t.Fatal("expected regression-mimic to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityRegressionMimic(t *testing.T) {
	if err := EnsureScapeCompatibility("regression-mimic"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
