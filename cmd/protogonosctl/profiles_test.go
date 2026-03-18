package main

import (
	"strings"
	"testing"
)

func TestLoadParityPreset(t *testing.T) {
	preset, err := loadParityPreset("ref-default-xorandxor")
	if err != nil {
		t.Fatalf("load preset: %v", err)
	}
	if preset.Selection != "species_shared_tournament" {
		t.Fatalf("unexpected selection mapping: %s", preset.Selection)
	}
	if preset.TuneSelection == "" {
		t.Fatal("expected tune selection mapping")
	}
	if preset.WeightAddSyn <= 0 || preset.WeightAddNeuro <= 0 {
		t.Fatalf("expected mapped structural mutation weights, got syn=%f neuro=%f", preset.WeightAddSyn, preset.WeightAddNeuro)
	}
}

func TestLoadParityPresetCarriesSeedProfiles(t *testing.T) {
	preset, err := loadParityPreset("parity-fx-market")
	if err != nil {
		t.Fatalf("load preset: %v", err)
	}
	if preset.FXProfile != "market" {
		t.Fatalf("expected fx market preset profile, got %q", preset.FXProfile)
	}

	preset, err = loadParityPreset("parity-gtsa-core")
	if err != nil {
		t.Fatalf("load preset: %v", err)
	}
	if preset.GTSAProfile != "core" {
		t.Fatalf("expected gtsa core preset profile, got %q", preset.GTSAProfile)
	}
}

func TestLoadParityPresetMissing(t *testing.T) {
	_, err := loadParityPreset("missing-profile")
	if err == nil {
		t.Fatal("expected missing profile error")
	}
}

func TestResolveParityProfile(t *testing.T) {
	resolved, err := resolveParityProfile("ref-default-xorandxor")
	if err != nil {
		t.Fatalf("resolve profile: %v", err)
	}
	if resolved.ID != "ref-default-xorandxor" {
		t.Fatalf("unexpected id: %s", resolved.ID)
	}
	if resolved.PopulationSelection != resolved.ExpectedSelection {
		t.Fatalf("selection mismatch: got=%s expected=%s", resolved.PopulationSelection, resolved.ExpectedSelection)
	}
	if resolved.TuningSelection != resolved.ExpectedTuning {
		t.Fatalf("tuning mismatch: got=%s expected=%s", resolved.TuningSelection, resolved.ExpectedTuning)
	}
}

func TestResolveParityProfileIncludesSeedProfiles(t *testing.T) {
	resolved, err := resolveParityProfile("parity-fx-market")
	if err != nil {
		t.Fatalf("resolve profile: %v", err)
	}
	if resolved.FXProfile != "market" {
		t.Fatalf("expected fx market profile, got %q", resolved.FXProfile)
	}

	resolved, err = resolveParityProfile("parity-gtsa-core")
	if err != nil {
		t.Fatalf("resolve profile: %v", err)
	}
	if resolved.GTSAProfile != "core" {
		t.Fatalf("expected gtsa core profile, got %q", resolved.GTSAProfile)
	}
}

func TestListParityProfiles(t *testing.T) {
	profiles, err := listParityProfiles()
	if err != nil {
		t.Fatalf("list profiles: %v", err)
	}
	if len(profiles) == 0 {
		t.Fatal("expected at least one profile")
	}
	if profiles[0].ID == "" {
		t.Fatal("expected profile id")
	}
	for _, profile := range profiles {
		if profile.ExpectedSelection == "" || profile.ExpectedTuning == "" {
			t.Fatalf("expected mapping metadata for profile %s", profile.ID)
		}
		if profile.PopulationSelection != profile.ExpectedSelection {
			t.Fatalf("selection mapping mismatch for %s: got=%s expected=%s", profile.ID, profile.PopulationSelection, profile.ExpectedSelection)
		}
		if profile.TuningSelection != profile.ExpectedTuning {
			t.Fatalf("tuning mapping mismatch for %s: got=%s expected=%s", profile.ID, profile.TuningSelection, profile.ExpectedTuning)
		}
	}
}

func TestMapPopulationSelectionAliases(t *testing.T) {
	cases := map[string]string{
		"hof_competition": "species_shared_tournament",
		"hof_rank":        "hof_rank",
		"hof_top3":        "hof_top3",
		"hof_efficiency":  "hof_efficiency",
		"hof_random":      "hof_random",
		"competition":     "competition",
		"top3":            "top3",
		"":                "species_shared_tournament",
		"custom":          "custom",
	}
	for in, want := range cases {
		if got := mapPopulationSelection(in); got != want {
			t.Fatalf("mapPopulationSelection(%q)=%q want=%q", in, got, want)
		}
	}
}

func TestParseBenchmarkMorphologyTag(t *testing.T) {
	scape, gtsa, fx, epitopes, llvm, flatland, err := parseBenchmarkMorphologyTag("fx[market]")
	if err != nil {
		t.Fatalf("parse fx morphology tag: %v", err)
	}
	if scape != "fx" || fx != "market" || gtsa != "" || epitopes != "" || llvm != "" || flatland != "" {
		t.Fatalf("unexpected fx morphology parse: scape=%q gtsa=%q fx=%q epitopes=%q llvm=%q flatland=%q", scape, gtsa, fx, epitopes, llvm, flatland)
	}

	scape, gtsa, fx, epitopes, llvm, flatland, err = parseBenchmarkMorphologyTag("gtsa[core]")
	if err != nil {
		t.Fatalf("parse gtsa morphology tag: %v", err)
	}
	if scape != "gtsa" || gtsa != "core" || fx != "" || epitopes != "" || llvm != "" || flatland != "" {
		t.Fatalf("unexpected gtsa morphology parse: scape=%q gtsa=%q fx=%q epitopes=%q llvm=%q flatland=%q", scape, gtsa, fx, epitopes, llvm, flatland)
	}

	scape, gtsa, fx, epitopes, llvm, flatland, err = parseBenchmarkMorphologyTag("epitopes[core]")
	if err != nil {
		t.Fatalf("parse epitopes morphology tag: %v", err)
	}
	if scape != "epitopes" || epitopes != "core" || gtsa != "" || fx != "" || llvm != "" || flatland != "" {
		t.Fatalf("unexpected epitopes morphology parse: scape=%q gtsa=%q fx=%q epitopes=%q llvm=%q flatland=%q", scape, gtsa, fx, epitopes, llvm, flatland)
	}

	scape, gtsa, fx, epitopes, llvm, flatland, err = parseBenchmarkMorphologyTag("llvm-phase-ordering[core]")
	if err != nil {
		t.Fatalf("parse llvm morphology tag: %v", err)
	}
	if scape != "llvm-phase-ordering" || llvm != "core" || gtsa != "" || fx != "" || epitopes != "" || flatland != "" {
		t.Fatalf("unexpected llvm morphology parse: scape=%q gtsa=%q fx=%q epitopes=%q llvm=%q flatland=%q", scape, gtsa, fx, epitopes, llvm, flatland)
	}
}

func TestRewriteBenchmarkMorphologyArgs(t *testing.T) {
	args := rewriteBenchmarkMorphologyArgs([]string{"--scape", "gtsa", "--gtsa-profile", "core", "--fx-profile", "market", "--epitopes-profile", "core", "--llvm-profile", "core"}, "", "market", "", "", "")
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "gtsa-profile") {
		t.Fatalf("expected gtsa-profile to be removed, args=%v", args)
	}
	if !strings.Contains(joined, "--fx-profile market") {
		t.Fatalf("expected fx-profile to be added, args=%v", args)
	}
	if strings.Contains(joined, "epitopes-profile") {
		t.Fatalf("expected epitopes-profile to be removed, args=%v", args)
	}
	if strings.Contains(joined, "llvm-profile") {
		t.Fatalf("expected llvm-profile to be removed, args=%v", args)
	}
}
