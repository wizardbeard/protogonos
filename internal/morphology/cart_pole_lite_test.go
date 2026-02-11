package morphology

import "testing"

func TestCartPoleLiteMorphologyCompatibility(t *testing.T) {
	m := CartPoleLiteMorphology{}
	if !m.Compatible("cart-pole-lite") {
		t.Fatal("expected cart-pole-lite to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityCartPoleLite(t *testing.T) {
	if err := EnsureScapeCompatibility("cart-pole-lite"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
