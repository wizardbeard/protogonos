package stats

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type parityProfileFixture struct {
	Source   string `json:"source"`
	Profiles []struct {
		ID                    string `json:"id"`
		PopulationSelect      string `json:"population_selection"`
		ExpectedSelection     string `json:"expected_selection"`
		PopulationEvo         string `json:"population_evolution"`
		Morphology            string `json:"morphology"`
		ConnectionArch        string `json:"connection_architecture"`
		TuningSelection       string `json:"tuning_selection"`
		ExpectedTuneSelection string `json:"expected_tuning_selection"`
		MutationOperators     []struct {
			Name   string `json:"name"`
			Weight int    `json:"weight"`
		} `json:"mutation_operators"`
	} `json:"profiles"`
}

func TestReferenceBenchmarkerFixture(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "fixtures", "parity", "ref_benchmarker_profiles.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fixture parityProfileFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if fixture.Source == "" {
		t.Fatal("expected source in parity fixture")
	}
	if len(fixture.Profiles) < 3 {
		t.Fatalf("expected expanded benchmark profiles, got %d", len(fixture.Profiles))
	}
	for _, p := range fixture.Profiles {
		if p.ID == "" || p.Morphology == "" {
			t.Fatalf("invalid profile identity: %+v", p)
		}
		if p.ExpectedSelection == "" || p.ExpectedTuneSelection == "" {
			t.Fatalf("missing expected mapping in profile: %+v", p)
		}
		if len(p.MutationOperators) < 3 {
			t.Fatalf("expected representative mutation operators for profile %s, got %d", p.ID, len(p.MutationOperators))
		}
	}
}
