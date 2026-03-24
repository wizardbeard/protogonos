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

	defaultAliasMorph, err := ConstructMorphology("flatland", "flatland_v1")
	if err != nil {
		t.Fatalf("construct flatland_v1 alias morphology: %v", err)
	}
	if defaultAliasMorph.Name() != "flatland-v1" {
		t.Fatalf("expected flatland_v1 alias to resolve to flatland-v1, got=%s", defaultAliasMorph.Name())
	}

	classicAliasMorph, err := ConstructMorphology("flatland", "flatland_classic_v1")
	if err != nil {
		t.Fatalf("construct flatland_classic_v1 alias morphology: %v", err)
	}
	if classicAliasMorph.Name() != "flatland-classic-v1" {
		t.Fatalf("expected flatland_classic_v1 alias to resolve to flatland-classic-v1, got=%s", classicAliasMorph.Name())
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

func TestConstructMorphologyFXProfiles(t *testing.T) {
	marketMorph, err := ConstructMorphology("fx", "market")
	if err != nil {
		t.Fatalf("construct fx market profile: %v", err)
	}
	if marketMorph.Name() != "fx-market-v1" {
		t.Fatalf("expected fx-market-v1, got=%s", marketMorph.Name())
	}
	if got := marketMorph.Sensors(); len(got) != 2 || got[0] != protoio.FXPriceSensorName || got[1] != protoio.FXSignalSensorName {
		t.Fatalf("expected market-only fx sensors, got=%v", got)
	}
}

func TestConstructMorphologyGTSAProfiles(t *testing.T) {
	coreMorph, err := ConstructMorphology("gtsa", "core")
	if err != nil {
		t.Fatalf("construct gtsa core profile: %v", err)
	}
	if coreMorph.Name() != "gtsa-core-v1" {
		t.Fatalf("expected gtsa-core-v1, got=%s", coreMorph.Name())
	}
	if got := coreMorph.Sensors(); len(got) != 1 || got[0] != protoio.GTSAInputSensorName {
		t.Fatalf("expected core-only gtsa sensor profile, got=%v", got)
	}
}

func TestConstructMorphologyEpitopesProfiles(t *testing.T) {
	coreMorph, err := ConstructMorphology("epitopes", "core")
	if err != nil {
		t.Fatalf("construct epitopes core profile: %v", err)
	}
	if coreMorph.Name() != "epitopes-core-v1" {
		t.Fatalf("expected epitopes-core-v1, got=%s", coreMorph.Name())
	}
	if got := coreMorph.Sensors(); len(got) != 2 || got[0] != protoio.EpitopesSignalSensorName || got[1] != protoio.EpitopesMemorySensorName {
		t.Fatalf("expected core-only epitopes sensors, got=%v", got)
	}
}

func TestConstructMorphologyLLVMProfiles(t *testing.T) {
	coreMorph, err := ConstructMorphology("llvm-phase-ordering", "core")
	if err != nil {
		t.Fatalf("construct llvm core profile: %v", err)
	}
	if coreMorph.Name() != "llvm-phase-ordering-core-v1" {
		t.Fatalf("expected llvm-phase-ordering-core-v1, got=%s", coreMorph.Name())
	}
	if got := coreMorph.Sensors(); len(got) != 2 || got[0] != protoio.LLVMComplexitySensorName || got[1] != protoio.LLVMPassIndexSensorName {
		t.Fatalf("expected core-only llvm sensors, got=%v", got)
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
	if err := EnsureScapeCompatibilityWithProfile("fx_sim", "market"); err != nil {
		t.Fatalf("ensure fx alias market profile compatibility: %v", err)
	}
	if err := EnsureScapeCompatibilityWithProfile("scape_GTSA", "core"); err != nil {
		t.Fatalf("ensure gtsa alias core profile compatibility: %v", err)
	}
	if err := EnsureScapeCompatibilityWithProfile("epitopes_sim", "core"); err != nil {
		t.Fatalf("ensure epitopes alias core profile compatibility: %v", err)
	}
	if err := EnsureScapeCompatibilityWithProfile("scape_LLVMPhaseOrdering", "core"); err != nil {
		t.Fatalf("ensure llvm alias core profile compatibility: %v", err)
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
	if got := AvailableMorphologyProfiles("fx"); len(got) != 2 || got[0] != "default" || got[1] != "market" {
		t.Fatalf("expected fx market/default profiles, got=%v", got)
	}
	if got := AvailableMorphologyProfiles("gtsa"); len(got) != 2 || got[0] != "core" || got[1] != "default" {
		t.Fatalf("expected gtsa core/default profiles, got=%v", got)
	}
	if got := AvailableMorphologyProfiles("epitopes"); len(got) != 2 || got[0] != "core" || got[1] != "default" {
		t.Fatalf("expected epitopes core/default profiles, got=%v", got)
	}
	if got := AvailableMorphologyProfiles("llvm-phase-ordering"); len(got) != 2 || got[0] != "core" || got[1] != "default" {
		t.Fatalf("expected llvm core/default profiles, got=%v", got)
	}
	if got := AvailableMorphologyProfiles("xor"); len(got) != 1 || got[0] != "default" {
		t.Fatalf("expected default-only profile for xor, got=%v", got)
	}
}
