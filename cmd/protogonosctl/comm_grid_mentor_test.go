package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCommGridMentorCommandRunsFixturePlan(t *testing.T) {
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-mentor", "--run-id", "mentor-fixture", "--plan", "solve", "--artifacts=false"})
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
		return run(context.Background(), []string{"comm-grid-mentor", "--run-id", "mentor-json", "--json", "--artifacts=false"})
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

func TestCommGridMentorCommandComparesBaseline(t *testing.T) {
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-mentor",
			"--run-id", "mentor-compare",
			"--plan", "solve",
			"--compare-baseline",
			"--json",
			"--artifacts=false",
		})
	})
	if err != nil {
		t.Fatalf("comm-grid-mentor compare command: %v", err)
	}
	var summary commGridMentorCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode mentor compare json: %v\n%s", err, out)
	}
	if !summary.Completed || summary.Baseline == nil {
		t.Fatalf("expected completed run with baseline summary: %+v", summary)
	}
	if summary.Baseline.Completed || summary.Baseline.Fitness >= summary.Fitness || summary.Baseline.Improvement <= 0 {
		t.Fatalf("expected mentor to improve over no-hint baseline: %+v", summary.Baseline)
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

func TestCommGridMentorCommandWritesArtifactAndReplays(t *testing.T) {
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	workdir := t.TempDir()
	if err := os.Chdir(workdir); err != nil {
		t.Fatalf("chdir tempdir: %v", err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(origWD)
	})

	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-mentor", "--run-id", "mentor-artifact", "--plan", "solve", "--compare-baseline"})
	})
	if err != nil {
		t.Fatalf("comm-grid-mentor artifact command: %v", err)
	}
	if !strings.Contains(out, "artifacts_dir=benchmarks/mentor-artifact") {
		t.Fatalf("expected artifact directory in output, got %s", out)
	}
	path := filepath.Join(workdir, "benchmarks", "mentor-artifact", "comm_grid_mentor.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read mentor artifact: %v", err)
	}
	var artifact commGridMentorArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		t.Fatalf("decode mentor artifact: %v", err)
	}
	if artifact.RunID != "mentor-artifact" || artifact.Provider != "fixture" || artifact.Plan != "solve" || !artifact.Completed {
		t.Fatalf("unexpected mentor artifact: %+v", artifact)
	}
	if artifact.TotalTokens != 4 || len(artifact.Steps) != 4 || artifact.Steps[0].Hint != "east" {
		t.Fatalf("unexpected mentor artifact steps: %+v", artifact)
	}
	if artifact.Baseline == nil || artifact.Baseline.Completed || artifact.Baseline.Improvement <= 0 {
		t.Fatalf("expected artifact baseline comparison, got %+v", artifact.Baseline)
	}

	out, err = captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-mentor", "--replay-run-id", "mentor-artifact", "--json"})
	})
	if err != nil {
		t.Fatalf("comm-grid-mentor replay command: %v", err)
	}
	var summary commGridMentorCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode mentor replay json: %v\n%s", err, out)
	}
	if summary.Replay == nil || !summary.Replay.Matched || summary.Provider != "fixture-replay" || !summary.Completed {
		t.Fatalf("unexpected mentor replay summary: %+v", summary)
	}
}
