package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestCommGridMentorCommandRunsFixturePlan(t *testing.T) {
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-mentor", "--run-id", "mentor-fixture", "--plan", "solve"})
	})
	if err != nil {
		t.Fatalf("comm-grid-mentor command: %v", err)
	}
	for _, want := range []string{
		"comm_grid_mentor run_id=mentor-fixture provider=fixture plan=solve grid=3x3 key=(1,0) goal=(2,0) agent=agent-1@(0,0) steps=4 completed=true",
		"step=1 hint=\"east\" hint_action=east action=east",
		"step=4 hint=\"drop\" hint_action=drop action=drop",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected output to contain %q, got %s", want, out)
		}
	}
}

func TestCommGridMentorCommandEmitsJSON(t *testing.T) {
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-mentor", "--run-id", "mentor-json", "--json"})
	})
	if err != nil {
		t.Fatalf("comm-grid-mentor json command: %v", err)
	}
	var summary commGridMentorCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode mentor json: %v\n%s", err, out)
	}
	if summary.RunID != "mentor-json" || summary.Provider != "fixture" || summary.Plan != "solve" || !summary.Completed {
		t.Fatalf("unexpected mentor summary: %+v", summary)
	}
	if len(summary.Steps) != 4 || summary.Steps[0].HintAction != "east" || summary.Steps[3].Action != "drop" {
		t.Fatalf("unexpected mentor steps: %+v", summary.Steps)
	}
	if summary.Trace["mentor_tokens"] != float64(4) {
		t.Fatalf("expected mentor token trace, got %+v", summary.Trace)
	}
}

func TestCommGridMentorCommandRejectsUnknownFixturePlan(t *testing.T) {
	err := run(context.Background(), []string{"comm-grid-mentor", "--plan", "missing"})
	if err == nil {
		t.Fatal("expected unsupported plan error")
	}
	if !strings.Contains(err.Error(), "unsupported comm-grid mentor fixture plan") {
		t.Fatalf("unexpected error: %v", err)
	}
}
