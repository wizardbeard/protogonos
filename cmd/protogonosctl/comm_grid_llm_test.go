package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"
)

func TestCommGridLLMCommandRunsFixturePlan(t *testing.T) {
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm", "--plan", "solve"})
	})
	if err != nil {
		t.Fatalf("comm-grid-llm command: %v", err)
	}
	for _, want := range []string{
		"comm_grid_llm_fixture plan=solve steps=4 completed=true",
		"step=1 action=east",
		"step=4 action=drop",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected output to contain %q, got %s", want, out)
		}
	}
}

func TestCommGridLLMCommandEmitsJSON(t *testing.T) {
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm", "--plan", "tool", "--json"})
	})
	if err != nil {
		t.Fatalf("comm-grid-llm command: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode json output: %v\n%s", err, out)
	}
	if summary.Plan != "tool" || !summary.Completed || len(summary.Steps) != 4 {
		t.Fatalf("unexpected summary: %+v", summary)
	}
	if summary.Steps[0].ProviderTrace["finish_reason"] != "tool_calls" {
		t.Fatalf("expected tool-call provider trace, got %+v", summary.Steps[0].ProviderTrace)
	}
}

func TestCommGridLLMCommandRejectsUnknownFixturePlan(t *testing.T) {
	err := run(context.Background(), []string{"comm-grid-llm", "--plan", "missing"})
	if err == nil {
		t.Fatal("expected unsupported plan error")
	}
	if !strings.Contains(err.Error(), "unsupported comm-grid llm fixture plan") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func captureStdoutForCommGridLLM(fn func() error) (string, error) {
	origStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		return "", err
	}

	os.Stdout = w
	runErr := fn()
	_ = w.Close()
	os.Stdout = origStdout

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		_ = r.Close()
		return "", err
	}
	_ = r.Close()
	return buf.String(), runErr
}
