package main

import "testing"

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

func TestLoadParityPresetMissing(t *testing.T) {
	_, err := loadParityPreset("missing-profile")
	if err == nil {
		t.Fatal("expected missing profile error")
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
}
