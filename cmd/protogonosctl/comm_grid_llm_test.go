package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
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
		"comm_grid_llm provider=fixture plan=solve steps=4 completed=true",
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
	if summary.Provider != "fixture" {
		t.Fatalf("unexpected provider: %+v", summary)
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

func TestCommGridLLMCommandRunsOpenAICompatibleProvider(t *testing.T) {
	responses := []string{
		`{"action":"east","message":"move to key","to":"all","tokens":3}`,
		`{"action":"pick","message":"picked key","to":"all","tokens":2}`,
		`{"action":"east","message":"move to goal","to":"all","tokens":3}`,
		`{"action":"drop","message":"delivered key","to":"all","tokens":2}`,
	}
	var gotAuth string
	var gotReqs []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		var got map[string]any
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		gotReqs = append(gotReqs, got)
		idx := len(gotReqs) - 1
		if idx >= len(responses) {
			idx = len(responses) - 1
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "fake-local",
			"choices": [{
				"finish_reason": "stop",
				"message": {"content": ` + strconvQuoteForTest(responses[idx]) + `}
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
		}`))
	}))
	defer server.Close()

	t.Setenv("PROTOGONOS_TEST_LLM_KEY", "secret")
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--provider", "openai-compatible",
			"--base-url", server.URL + "/v1",
			"--api-key-env", "PROTOGONOS_TEST_LLM_KEY",
			"--model", "fake-local",
			"--seed", "9",
			"--max-tokens", "32",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("comm-grid-llm openai-compatible command: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode json output: %v\n%s", err, out)
	}
	if summary.Provider != "openai-compatible" || summary.Plan != "live" || !summary.Completed || len(summary.Steps) != 4 {
		t.Fatalf("unexpected summary: %+v", summary)
	}
	if gotAuth != "Bearer secret" {
		t.Fatalf("unexpected auth header: %q", gotAuth)
	}
	if len(gotReqs) != 4 {
		t.Fatalf("expected four provider requests, got %d", len(gotReqs))
	}
	if gotReqs[0]["model"] != "fake-local" || gotReqs[0]["max_tokens"].(float64) != 32 {
		t.Fatalf("unexpected first request: %+v", gotReqs[0])
	}
	if gotReqs[0]["response_format"] == nil {
		t.Fatalf("expected response_format in request: %+v", gotReqs[0])
	}
	if gotReqs[0]["seed"].(float64) != 9 {
		t.Fatalf("expected seed in request: %+v", gotReqs[0])
	}
}

func TestCommGridLLMCommandRequiresOpenAICompatibleModel(t *testing.T) {
	err := run(context.Background(), []string{
		"comm-grid-llm",
		"--provider", "openai-compatible",
		"--base-url", "http://127.0.0.1:1/v1",
	})
	if err == nil {
		t.Fatal("expected model error")
	}
	if !strings.Contains(err.Error(), "llm model required") {
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

func strconvQuoteForTest(s string) string {
	data, _ := json.Marshal(s)
	return string(data)
}
