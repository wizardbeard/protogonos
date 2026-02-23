package morphology

import "testing"

func TestDTMMorphologyCompatibility(t *testing.T) {
	m := DTMMorphology{}
	if !m.Compatible("dtm") {
		t.Fatal("expected dtm to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityDTM(t *testing.T) {
	if err := EnsureScapeCompatibility("dtm"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}
