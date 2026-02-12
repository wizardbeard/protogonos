package morphology

import "testing"

func TestFlatlandMorphologyCompatibility(t *testing.T) {
	m := FlatlandMorphology{}
	if !m.Compatible("flatland") {
		t.Fatal("expected flatland to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityFlatland(t *testing.T) {
	if err := EnsureScapeCompatibility("flatland"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
