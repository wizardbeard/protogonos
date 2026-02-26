package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

func TestConstructMorphologyFlatlandProfiles(t *testing.T) {
	defaultMorph, err := ConstructMorphology("flatland", "default")
	if err != nil {
		t.Fatalf("construct default flatland morphology: %v", err)
	}
	if defaultMorph.Name() != "flatland-v1" {
		t.Fatalf("expected flatland-v1 default profile, got=%s", defaultMorph.Name())
	}

	scannerMorph, err := ConstructMorphology("flatland", "flatland-scanner")
	if err != nil {
		t.Fatalf("construct scanner flatland morphology: %v", err)
	}
	if scannerMorph.Name() != "flatland-scanner-v1" {
		t.Fatalf("expected flatland-scanner-v1, got=%s", scannerMorph.Name())
	}

	classicMorph, err := ConstructMorphology("flatland", "classic")
	if err != nil {
		t.Fatalf("construct classic flatland morphology: %v", err)
	}
	if classicMorph.Name() != "flatland-classic-v1" {
		t.Fatalf("expected flatland-classic-v1, got=%s", classicMorph.Name())
	}
	if got := classicMorph.Sensors(); len(got) != 2 || got[0] != protoio.FlatlandDistanceSensorName || got[1] != protoio.FlatlandEnergySensorName {
		t.Fatalf("unexpected classic flatland sensors: %v", got)
	}
}

func TestConstructMorphologyDTMProfiles(t *testing.T) {
	rangeMorph, err := ConstructMorphology("dtm", "range_sense")
	if err != nil {
		t.Fatalf("construct dtm range profile: %v", err)
	}
	if rangeMorph.Name() != "dtm-range-sense-v1" {
		t.Fatalf("expected dtm-range-sense-v1, got=%s", rangeMorph.Name())
	}
	if got := rangeMorph.Sensors(); len(got) != 3 {
		t.Fatalf("expected 3 range sensors, got=%v", got)
	}

	rewardMorph, err := ConstructMorphology("dtm", "reward")
	if err != nil {
		t.Fatalf("construct dtm reward profile: %v", err)
	}
	if got := rewardMorph.Sensors(); len(got) != 1 || got[0] != protoio.DTMRewardSensorName {
		t.Fatalf("expected reward-only sensor profile, got=%v", got)
	}
}

func TestConstructMorphologyPole2Profiles(t *testing.T) {
	m3, err := ConstructMorphology("pole2-balancing", "3")
	if err != nil {
		t.Fatalf("construct pole2 profile 3: %v", err)
	}
	if m3.Name() != "pole2-balancing-3-v1" {
		t.Fatalf("expected pole2-balancing-3-v1, got=%s", m3.Name())
	}
	if got := m3.Sensors(); len(got) != 3 {
		t.Fatalf("expected 3-sensor profile, got=%v", got)
	}

	m6, err := ConstructMorphology("pb_sim", "6")
	if err != nil {
		t.Fatalf("construct pole2 alias profile 6: %v", err)
	}
	if got := m6.Sensors(); len(got) != 6 {
		t.Fatalf("expected 6-sensor profile, got=%v", got)
	}
}

func TestConstructMorphologyRejectsUnsupportedProfile(t *testing.T) {
	if _, err := ConstructMorphology("flatland", "unsupported"); err == nil {
		t.Fatal("expected unsupported profile error")
	}
}

func TestEnsureScapeCompatibilityWithProfile(t *testing.T) {
	if err := EnsureScapeCompatibilityWithProfile("flatland", "classic"); err != nil {
		t.Fatalf("ensure compatibility with profile: %v", err)
	}
	if err := EnsureScapeCompatibilityWithProfile("dtm_sim", "range_sense"); err != nil {
		t.Fatalf("ensure dtm alias profile compatibility: %v", err)
	}
}

func TestAvailableMorphologyProfiles(t *testing.T) {
	p := AvailableMorphologyProfiles("flatland")
	if len(p) == 0 {
		t.Fatal("expected flatland profiles")
	}
	if p[0] != "classic" || p[len(p)-1] != "scanner" {
		t.Fatalf("expected sorted profile list, got=%v", p)
	}
	if got := AvailableMorphologyProfiles("xor"); len(got) != 1 || got[0] != "default" {
		t.Fatalf("expected default-only profile for xor, got=%v", got)
	}
}
