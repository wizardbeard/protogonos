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
	Selection            string
	TuneSelection        string
	WeightPerturb        float64
	WeightBias           float64
	WeightRemoveBias     float64
	WeightActivation     float64
	WeightAggregator     float64
	WeightAddSyn         float64
	WeightRemoveSyn      float64
	WeightAddNeuro       float64
	WeightRemoveNeuro    float64
	WeightPlasticityRule float64
	WeightPlasticity     float64
	WeightSubstrate      float64
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

type parityProfileResolved struct {
	ID                   string
	PopulationSelection  string
	TuningSelection      string
	ExpectedSelection    string
	ExpectedTuning       string
	MutationOperatorLen  int
	WeightPerturb        float64
	WeightBias           float64
	WeightRemoveBias     float64
	WeightActivation     float64
	WeightAggregator     float64
	WeightAddSyn         float64
	WeightRemoveSyn      float64
	WeightAddNeuro       float64
	WeightRemoveNeuro    float64
	WeightPlasticityRule float64
	WeightPlasticity     float64
	WeightSubstrate      float64
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
	resolved, err := resolveParityProfile(profileID)
	if err != nil {
		return parityPreset{}, err
	}
	return parityPreset{
		Selection:            resolved.PopulationSelection,
		TuneSelection:        resolved.TuningSelection,
		WeightPerturb:        resolved.WeightPerturb,
		WeightBias:           resolved.WeightBias,
		WeightRemoveBias:     resolved.WeightRemoveBias,
		WeightActivation:     resolved.WeightActivation,
		WeightAggregator:     resolved.WeightAggregator,
		WeightAddSyn:         resolved.WeightAddSyn,
		WeightRemoveSyn:      resolved.WeightRemoveSyn,
		WeightAddNeuro:       resolved.WeightAddNeuro,
		WeightRemoveNeuro:    resolved.WeightRemoveNeuro,
		WeightPlasticityRule: resolved.WeightPlasticityRule,
		WeightPlasticity:     resolved.WeightPlasticity,
		WeightSubstrate:      resolved.WeightSubstrate,
	}, nil
}

func resolveParityProfile(profileID string) (parityProfileResolved, error) {
	if profileID == "" {
		return parityProfileResolved{}, fmt.Errorf("profile id is required")
	}
	fixture, err := loadParityFixture()
	if err != nil {
		return parityProfileResolved{}, err
	}
	for _, profile := range fixture.Profiles {
		if profile.ID != profileID {
			continue
		}
		constraint := map2rec.ConvertConstraint(profileToConstraintMap(profile))
		resolved := parityProfileResolved{
			ID:                  profile.ID,
			PopulationSelection: mapPopulationSelection(constraint.PopulationSelectionF),
			TuningSelection:     mapTuningSelection(firstOrEmpty(constraint.TuningSelectionFs)),
			ExpectedSelection:   profile.ExpectedSelection,
			ExpectedTuning:      profile.ExpectedTuning,
			MutationOperatorLen: len(profile.MutationOperators),
		}
		for _, op := range constraint.MutationOperators {
			switch normalizeMutationOperatorName(op.Name) {
			case "mutate_weights":
				resolved.WeightPerturb += op.Weight
			case "add_bias":
				resolved.WeightBias += op.Weight
			case "remove_bias":
				resolved.WeightRemoveBias += op.Weight
			case "mutate_af":
				resolved.WeightActivation += op.Weight
			case "mutate_aggrf":
				resolved.WeightAggregator += op.Weight
			case "add_outlink", "add_inlink":
				resolved.WeightAddSyn += op.Weight
			case "add_neuron", "outsplice", "insplice":
				resolved.WeightAddNeuro += op.Weight
			case "remove_outlink", "remove_inlink", "cutlink_FromNeuronToNeuron", "cutlink_FromElementToElement":
				resolved.WeightRemoveSyn += op.Weight
			case "remove_neuron":
				resolved.WeightRemoveNeuro += op.Weight
			case "mutate_plasticity_parameters":
				resolved.WeightPlasticity += op.Weight
			case "mutate_pf":
				resolved.WeightPlasticityRule += op.Weight
			case "add_sensor", "add_sensorlink", "add_actuator", "add_cpp", "remove_cpp", "add_cep", "remove_cep", "add_circuit_node", "delete_circuit_node", "add_circuit_layer", "remove_sensor", "remove_actuator", "cutlink_FromSensorToNeuron", "cutlink_FromNeuronToActuator", "mutate_tuning_selection", "mutate_tuning_annealing", "mutate_tot_topological_mutations", "mutate_heredity_type":
				resolved.WeightSubstrate += op.Weight
			}
		}
		if resolved.WeightPerturb+resolved.WeightBias+resolved.WeightRemoveBias+resolved.WeightActivation+resolved.WeightAggregator+resolved.WeightAddSyn+resolved.WeightRemoveSyn+resolved.WeightAddNeuro+resolved.WeightRemoveNeuro+resolved.WeightPlasticityRule+resolved.WeightPlasticity+resolved.WeightSubstrate <= 0 {
			return parityProfileResolved{}, fmt.Errorf("profile %s has no mapped mutation weights", profileID)
		}
		return resolved, nil
	}
	return parityProfileResolved{}, fmt.Errorf("profile not found: %s", profileID)
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
	case "competition":
		return "competition"
	case "top3":
		return "top3"
	case "":
		return "species_shared_tournament"
	default:
		return name
	}
}

func mapTuningSelection(name string) string {
	return tuning.NormalizeCandidateSelectionName(name)
}
