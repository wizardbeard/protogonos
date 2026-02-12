package morphology

import "testing"

func TestFXMorphologyCompatibility(t *testing.T) {
	m := FXMorphology{}
	if !m.Compatible("fx") {
		t.Fatal("expected fx to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityFX(t *testing.T) {
	if err := EnsureScapeCompatibility("fx"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
