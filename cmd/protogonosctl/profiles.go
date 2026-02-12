package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"protogonos/internal/tuning"
)

const parityProfileFixturePath = "testdata/fixtures/parity/ref_benchmarker_profiles.json"

type parityPreset struct {
	Selection         string
	TuneSelection     string
	WeightPerturb     float64
	WeightAddSyn      float64
	WeightRemoveSyn   float64
	WeightAddNeuro    float64
	WeightRemoveNeuro float64
	WeightPlasticity  float64
	WeightSubstrate   float64
}

type parityProfileFixture struct {
	Profiles []struct {
		ID                  string `json:"id"`
		PopulationSelection string `json:"population_selection"`
		TuningSelection     string `json:"tuning_selection"`
		MutationOperators   []struct {
			Name   string `json:"name"`
			Weight int    `json:"weight"`
		} `json:"mutation_operators"`
	} `json:"profiles"`
}

func loadParityPreset(profileID string) (parityPreset, error) {
	if profileID == "" {
		return parityPreset{}, fmt.Errorf("profile id is required")
	}
	path, err := resolveParityFixturePath()
	if err != nil {
		return parityPreset{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return parityPreset{}, err
	}
	var fixture parityProfileFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		return parityPreset{}, err
	}
	for _, profile := range fixture.Profiles {
		if profile.ID != profileID {
			continue
		}
		preset := parityPreset{
			Selection:     mapPopulationSelection(profile.PopulationSelection),
			TuneSelection: mapTuningSelection(profile.TuningSelection),
		}
		for _, op := range profile.MutationOperators {
			switch op.Name {
			case "mutate_weights", "add_bias":
				preset.WeightPerturb += float64(op.Weight)
			case "add_outlink", "add_inlink":
				preset.WeightAddSyn += float64(op.Weight)
			case "add_neuron", "outsplice", "insplice":
				preset.WeightAddNeuro += float64(op.Weight)
			case "remove_outlink", "remove_inlink":
				preset.WeightRemoveSyn += float64(op.Weight)
			case "remove_neuron":
				preset.WeightRemoveNeuro += float64(op.Weight)
			case "mutate_plasticity_parameters":
				preset.WeightPlasticity += float64(op.Weight)
			case "add_sensor", "add_sensorlink", "add_actuator", "add_cpp", "add_cep":
				preset.WeightSubstrate += float64(op.Weight)
			}
		}
		if preset.WeightPerturb+preset.WeightAddSyn+preset.WeightRemoveSyn+preset.WeightAddNeuro+preset.WeightRemoveNeuro+preset.WeightPlasticity+preset.WeightSubstrate <= 0 {
			return parityPreset{}, fmt.Errorf("profile %s has no mapped mutation weights", profileID)
		}
		return preset, nil
	}
	return parityPreset{}, fmt.Errorf("profile not found: %s", profileID)
}

func resolveParityFixturePath() (string, error) {
	candidates := []string{
		parityProfileFixturePath,
		filepath.Join("..", "..", parityProfileFixturePath),
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("parity profile fixture not found: %s", parityProfileFixturePath)
}

func mapPopulationSelection(name string) string {
	switch name {
	case "hof_competition":
		return "species_shared_tournament"
	default:
		return "species_shared_tournament"
	}
}

func mapTuningSelection(name string) string {
	switch name {
	case "dynamic_random":
		return tuning.CandidateSelectBestSoFar
	case "":
		return tuning.CandidateSelectBestSoFar
	default:
		return name
	}
}
