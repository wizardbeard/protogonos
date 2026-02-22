package genotype

import "testing"

func TestMapIDsBuildsDeterministicDefaults(t *testing.T) {
	ids := []string{"n1", "n2", "n1", "", "n3"}
	got := MapIDs(ids, nil)
	if len(got) != 3 {
		t.Fatalf("expected 3 mapped ids, got=%d map=%v", len(got), got)
	}
	if got["n1"] == "" || got["n2"] == "" || got["n3"] == "" {
		t.Fatalf("expected non-empty mapped ids, got=%v", got)
	}
	if got["n1"] == got["n2"] || got["n1"] == got["n3"] || got["n2"] == got["n3"] {
		t.Fatalf("expected unique mapped ids, got=%v", got)
	}
}

func TestMapIDsUsesCallbackAndEnforcesUniqueness(t *testing.T) {
	ids := []string{"a", "b"}
	got := MapIDs(ids, func(_ string, _ int) string {
		return "shared"
	})
	if len(got) != 2 {
		t.Fatalf("expected 2 mapped ids, got=%d", len(got))
	}
	if got["a"] == got["b"] {
		t.Fatalf("expected uniqueness even with colliding callback ids, got=%v", got)
	}
}

func TestMapEvoHistoryRemapsKnownIDsAndPreservesOrder(t *testing.T) {
	history := []EvoHistoryEvent{
		{Mutation: "add_neuron", IDs: []string{"n1"}},
		{Mutation: "add_link", IDs: []string{"n1", "n2"}},
		{Mutation: "splice", IDs: []string{"n1", "n2", "n3"}},
	}
	idMap := map[string]string{
		"n1": "n1c",
		"n2": "n2c",
	}
	got := MapEvoHistory(history, idMap)
	if len(got) != 3 {
		t.Fatalf("expected 3 mapped events, got=%d", len(got))
	}
	if got[0].Mutation != "add_neuron" || got[1].Mutation != "add_link" || got[2].Mutation != "splice" {
		t.Fatalf("expected mutation ordering to be preserved, got=%v", got)
	}
	if len(got[0].IDs) != 1 || got[0].IDs[0] != "n1c" {
		t.Fatalf("unexpected first mapped ids: %v", got[0].IDs)
	}
	if len(got[1].IDs) != 2 || got[1].IDs[0] != "n1c" || got[1].IDs[1] != "n2c" {
		t.Fatalf("unexpected second mapped ids: %v", got[1].IDs)
	}
	if len(got[2].IDs) != 3 || got[2].IDs[2] != "n3" {
		t.Fatalf("expected unknown ids to remain unchanged, got=%v", got[2].IDs)
	}
}

func TestMapEvoHistoryHandlesEmptyInput(t *testing.T) {
	if got := MapEvoHistory(nil, nil); got != nil {
		t.Fatalf("expected nil result for empty history, got=%v", got)
	}
}
