package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"protogonos/internal/map2rec"
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
	Profiles []parityProfileFixtureProfile `json:"profiles"`
}

type parityProfileFixtureProfile struct {
	ID                  string                   `json:"id"`
	PopulationSelection string                   `json:"population_selection"`
	ExpectedSelection   string                   `json:"expected_selection"`
	TuningSelection     string                   `json:"tuning_selection"`
	ExpectedTuning      string                   `json:"expected_tuning_selection"`
	MutationOperators   []parityMutationOperator `json:"mutation_operators"`
}

type parityMutationOperator struct {
	Name   string `json:"name"`
	Weight int    `json:"weight"`
}

type parityProfileInfo struct {
	ID                  string
	PopulationSelection string
	TuningSelection     string
	ExpectedSelection   string
	ExpectedTuning      string
	MutationOperatorLen int
}

func loadParityFixture() (parityProfileFixture, error) {
	path, err := resolveParityFixturePath()
	if err != nil {
		return parityProfileFixture{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return parityProfileFixture{}, err
	}
	var fixture parityProfileFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		return parityProfileFixture{}, err
	}
	return fixture, nil
}

func loadParityPreset(profileID string) (parityPreset, error) {
	if profileID == "" {
		return parityPreset{}, fmt.Errorf("profile id is required")
	}
	fixture, err := loadParityFixture()
	if err != nil {
		return parityPreset{}, err
	}
	for _, profile := range fixture.Profiles {
		if profile.ID != profileID {
			continue
		}
		constraint := map2rec.ConvertConstraint(profileToConstraintMap(profile))
		preset := parityPreset{
			Selection:     mapPopulationSelection(constraint.PopulationSelectionF),
			TuneSelection: mapTuningSelection(firstOrEmpty(constraint.TuningSelectionFs)),
		}
		for _, op := range constraint.MutationOperators {
			switch op.Name {
			case "mutate_weights", "add_bias":
				preset.WeightPerturb += op.Weight
			case "add_outlink", "add_inlink":
				preset.WeightAddSyn += op.Weight
			case "add_neuron", "outsplice", "insplice":
				preset.WeightAddNeuro += op.Weight
			case "remove_outlink", "remove_inlink":
				preset.WeightRemoveSyn += op.Weight
			case "remove_neuron":
				preset.WeightRemoveNeuro += op.Weight
			case "mutate_plasticity_parameters":
				preset.WeightPlasticity += op.Weight
			case "add_sensor", "add_sensorlink", "add_actuator", "add_cpp", "add_cep":
				preset.WeightSubstrate += op.Weight
			}
		}
		if preset.WeightPerturb+preset.WeightAddSyn+preset.WeightRemoveSyn+preset.WeightAddNeuro+preset.WeightRemoveNeuro+preset.WeightPlasticity+preset.WeightSubstrate <= 0 {
			return parityPreset{}, fmt.Errorf("profile %s has no mapped mutation weights", profileID)
		}
		return preset, nil
	}
	return parityPreset{}, fmt.Errorf("profile not found: %s", profileID)
}

func profileToConstraintMap(profile parityProfileFixtureProfile) map[string]any {
	ops := make([]any, 0, len(profile.MutationOperators))
	for _, op := range profile.MutationOperators {
		ops = append(ops, map[string]any{
			"name":   op.Name,
			"weight": op.Weight,
		})
	}
	return map[string]any{
		"population_selection_f": profile.PopulationSelection,
		"tuning_selection_fs":    []any{profile.TuningSelection},
		"mutation_operators":     ops,
	}
}

func firstOrEmpty(xs []string) string {
	if len(xs) == 0 {
		return ""
	}
	return xs[0]
}

func listParityProfiles() ([]parityProfileInfo, error) {
	fixture, err := loadParityFixture()
	if err != nil {
		return nil, err
	}
	out := make([]parityProfileInfo, 0, len(fixture.Profiles))
	for _, profile := range fixture.Profiles {
		out = append(out, parityProfileInfo{
			ID:                  profile.ID,
			PopulationSelection: mapPopulationSelection(profile.PopulationSelection),
			TuningSelection:     mapTuningSelection(profile.TuningSelection),
			ExpectedSelection:   profile.ExpectedSelection,
			ExpectedTuning:      profile.ExpectedTuning,
			MutationOperatorLen: len(profile.MutationOperators),
		})
	}
	return out, nil
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
	case tuning.CandidateSelectDynamic:
		return tuning.CandidateSelectDynamic
	case "":
		return tuning.CandidateSelectBestSoFar
	default:
		return name
	}
}
