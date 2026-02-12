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
		ID                string `json:"id"`
		PopulationSelect  string `json:"population_selection"`
		PopulationEvo     string `json:"population_evolution"`
		Morphology        string `json:"morphology"`
		ConnectionArch    string `json:"connection_architecture"`
		TuningSelection   string `json:"tuning_selection"`
		MutationOperators []struct {
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
	if len(fixture.Profiles) == 0 {
		t.Fatal("expected at least one benchmark profile")
	}
	p := fixture.Profiles[0]
	if p.Morphology != "xorAndXor" || p.ConnectionArch != "recurrent" {
		t.Fatalf("unexpected profile defaults: %+v", p)
	}
	if len(p.MutationOperators) < 5 {
		t.Fatalf("expected representative mutation operators, got %d", len(p.MutationOperators))
	}
}
