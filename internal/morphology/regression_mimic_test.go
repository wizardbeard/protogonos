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

func TestEnsureScapeCompatibilityCommGridMentor(t *testing.T) {
	m := CommGridMentorMorphology{}
	if !m.Compatible("comm-grid-mentor") {
		t.Fatal("expected comm-grid-mentor to be compatible")
	}
	if m.Compatible("comm-grid") {
		t.Fatal("expected comm-grid to be incompatible")
	}
	if err := EnsureScapeCompatibility("scape_comm_grid_mentor_sim"); err != nil {
		t.Fatalf("ensure comm-grid mentor compatibility: %v", err)
	}
}
