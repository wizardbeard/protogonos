package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCommGridLLMCommandRunsFixturePlan(t *testing.T) {
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm", "--plan", "solve", "--artifacts=false"})
	})
	if err != nil {
		t.Fatalf("comm-grid-llm command: %v", err)
	}
	for _, want := range []string{
		"comm_grid_llm run_id=comm-grid-llm-fixture-",
		"provider=fixture plan=solve grid=3x3 key=(1,0) goal=(2,0) agents=agent-1@(0,0) turn_order=agent-1 steps=4 completed=true",
		"step=1 actor=agent-1 action=east",
		"step=4 actor=agent-1 action=drop",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("expected output to contain %q, got %s", want, out)
		}
	}
}

func TestCommGridLLMCommandEmitsJSON(t *testing.T) {
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm", "--plan", "tool", "--json", "--artifacts=false"})
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
			"--artifacts=false",
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

func TestCommGridLLMCommandRetriesOpenAICompatibleProvider(t *testing.T) {
	responses := []string{
		`{"action":"east","message":"move to key","to":"all","tokens":3}`,
		`{"action":"pick","message":"picked key","to":"all","tokens":2}`,
		`{"action":"east","message":"move to goal","to":"all","tokens":3}`,
		`{"action":"drop","message":"delivered key","to":"all","tokens":2}`,
	}
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			http.Error(w, "temporary failure", http.StatusBadGateway)
			return
		}
		idx := calls - 2
		if idx >= len(responses) {
			idx = len(responses) - 1
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "retry-local",
			"choices": [{
				"finish_reason": "stop",
				"message": {"content": ` + strconvQuoteForTest(responses[idx]) + `}
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
		}`))
	}))
	defer server.Close()

	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--provider", "openai-compatible",
			"--base-url", server.URL + "/v1",
			"--model", "retry-local",
			"--provider-retries", "1",
			"--retry-backoff-ms", "0",
			"--json",
			"--artifacts=false",
		})
	})
	if err != nil {
		t.Fatalf("retry command: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode retry json: %v\n%s", err, out)
	}
	if !summary.Completed || calls != 5 {
		t.Fatalf("expected completed retry run and five provider calls, calls=%d summary=%+v", calls, summary)
	}
	if len(summary.Steps[0].Attempts) != 2 {
		t.Fatalf("expected two attempts on first step, step=%+v", summary.Steps[0])
	}
	if summary.Steps[0].Attempts[0].Success || !strings.Contains(summary.Steps[0].Attempts[0].Error, "502") {
		t.Fatalf("expected failed first attempt, attempts=%+v", summary.Steps[0].Attempts)
	}
	if !summary.Steps[0].Attempts[1].Success || summary.Steps[0].Attempts[1].Tokens != 8 {
		t.Fatalf("expected successful second attempt, attempts=%+v", summary.Steps[0].Attempts)
	}
	if got, ok := summary.Steps[0].ProviderTrace["attempts"].(float64); !ok || got != 2 {
		t.Fatalf("expected provider trace attempt count, trace=%+v", summary.Steps[0].ProviderTrace)
	}
}

func TestCommGridLLMCommandConvertsProviderTimeoutToFailureStep(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(25 * time.Millisecond)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"action\":\"east\"}"}}]}`))
	}))
	defer server.Close()

	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--provider", "openai-compatible",
			"--base-url", server.URL + "/v1",
			"--model", "slow-model",
			"--timeout-ms", "1",
			"--steps", "1",
			"--json",
			"--artifacts=false",
		})
	})
	if err != nil {
		t.Fatalf("comm-grid-llm timeout command: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode timeout json: %v\n%s", err, out)
	}
	if len(summary.Steps) != 1 || summary.Steps[0].ErrorKind != "llm_failure" {
		t.Fatalf("expected timeout failure step, summary=%+v", summary)
	}
	if !strings.Contains(summary.Steps[0].Error, "llm failure") {
		t.Fatalf("expected failure message, step=%+v", summary.Steps[0])
	}
}

func TestCommGridLLMCommandWritesArtifacts(t *testing.T) {
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
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--run-id", "fixture-artifact-run",
			"--plan", "solve",
		})
	})
	if err != nil {
		t.Fatalf("comm-grid-llm command: %v", err)
	}
	if !strings.Contains(out, "artifacts_dir=benchmarks/fixture-artifact-run") {
		t.Fatalf("expected artifact directory in output, got %s", out)
	}

	path := filepath.Join(workdir, "benchmarks", "fixture-artifact-run", "comm_grid_llm.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	var artifact commGridLLMArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		t.Fatalf("decode artifact: %v", err)
	}
	if artifact.RunID != "fixture-artifact-run" || artifact.Provider != "fixture" || !artifact.Completed || len(artifact.Steps) != 4 {
		t.Fatalf("unexpected artifact: %+v", artifact)
	}
	first := artifact.Steps[0]
	if first.Request.SystemPrompt == "" || len(first.Request.Messages) != 1 {
		t.Fatalf("expected captured request prompt, step=%+v", first)
	}
	if first.Response.Message == "" || first.Payload == "" || first.Parsed.Action != "east" {
		t.Fatalf("expected captured response and parsed action, step=%+v", first)
	}
	if first.Result.Messages[0].Text != "move to key" {
		t.Fatalf("expected captured messages, step=%+v", first)
	}
	if tokens, ok := first.Result.ProviderTrace["tokens"].(float64); !ok || tokens != 8 {
		t.Fatalf("expected provider token trace, got %+v", first.Result.ProviderTrace)
	}

	transcriptPath := filepath.Join(workdir, "benchmarks", "fixture-artifact-run", "comm_grid_llm_transcript.md")
	transcript, err := os.ReadFile(transcriptPath)
	if err != nil {
		t.Fatalf("read transcript: %v", err)
	}
	transcriptText := string(transcript)
	for _, want := range []string{
		"# Comm Grid LLM Transcript",
		"- replay: `protogonosctl comm-grid-llm --replay-run-id fixture-artifact-run`",
		"## Step 1: agent-1",
		"### System Prompt",
		"### Response Payload",
		"agent=agent-1 action=east",
		"## Final Trace",
	} {
		if !strings.Contains(transcriptText, want) {
			t.Fatalf("expected transcript to contain %q, got %s", want, transcriptText)
		}
	}

	indexPath := filepath.Join(workdir, "benchmarks", "comm_grid_llm_runs.jsonl")
	indexData, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatalf("read run index: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(indexData)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected one index line, got %d: %s", len(lines), string(indexData))
	}
	var indexEntry commGridLLMRunIndexEntry
	if err := json.Unmarshal([]byte(lines[0]), &indexEntry); err != nil {
		t.Fatalf("decode run index line: %v\n%s", err, lines[0])
	}
	if indexEntry.RunID != "fixture-artifact-run" || indexEntry.Provider != "fixture" || !indexEntry.Completed || indexEntry.Steps != 4 {
		t.Fatalf("unexpected run index entry: %+v", indexEntry)
	}
	if indexEntry.FailureCount != 0 || indexEntry.RetryCount != 0 {
		t.Fatalf("unexpected run index counts: %+v", indexEntry)
	}
	if indexEntry.ArtifactPath != "benchmarks/fixture-artifact-run/comm_grid_llm.json" {
		t.Fatalf("unexpected artifact path: %+v", indexEntry)
	}
	if indexEntry.TranscriptPath != "benchmarks/fixture-artifact-run/comm_grid_llm_transcript.md" {
		t.Fatalf("unexpected transcript path: %+v", indexEntry)
	}
}

func TestCommGridLLMCommandWritesCustomTaskArtifactAndReplays(t *testing.T) {
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
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--run-id", "custom-task-run",
			"--plan", "solve",
			"--width", "4",
			"--height", "2",
			"--key", "1,0",
			"--goal", "2,0",
			"--agent", "worker-a",
			"--agent-pos", "0,0",
			"--message-limit", "12",
		})
	})
	if err != nil {
		t.Fatalf("custom task command: %v", err)
	}
	if !strings.Contains(out, "grid=4x2") || !strings.Contains(out, "agents=worker-a@(0,0)") {
		t.Fatalf("expected custom task output, got %s", out)
	}

	artifact := readCommGridLLMTestArtifact(t, workdir, "custom-task-run")
	if artifact.Task.Width != 4 || artifact.Task.Height != 2 || artifact.Task.AgentID != "worker-a" || artifact.Task.MessageLimit != 12 {
		t.Fatalf("unexpected stored task: %+v", artifact.Task)
	}
	if artifact.Steps[0].Request.Messages[0].Content == "" {
		t.Fatalf("expected stored LLM prompt, step=%+v", artifact.Steps[0])
	}
	if !strings.Contains(artifact.Steps[0].Request.Messages[0].Content, "agent_id=worker-a") {
		t.Fatalf("expected prompt to use custom agent, got %s", artifact.Steps[0].Request.Messages[0].Content)
	}
	if len(artifact.Steps[0].Result.Messages) != 1 || artifact.Steps[0].Result.Messages[0].Text != "move to key" {
		t.Fatalf("expected message limit to truncate stored text, step=%+v", artifact.Steps[0])
	}

	replayOut, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--replay-run-id", "custom-task-run",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("replay custom task command: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(replayOut), &summary); err != nil {
		t.Fatalf("decode replay json: %v\n%s", err, replayOut)
	}
	if summary.Replay == nil || !summary.Replay.Matched || summary.Task.AgentID != "worker-a" {
		t.Fatalf("expected matched custom replay, summary=%+v", summary)
	}
}

func TestCommGridLLMCommandWritesMultiAgentTurnsAndReplays(t *testing.T) {
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
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--run-id", "multi-agent-run",
			"--plan", "multi-solve",
			"--agents", "agent-a@0,0:agent-b@0,1",
			"--turn-order", "agent-a,agent-b",
		})
	})
	if err != nil {
		t.Fatalf("multi-agent command: %v", err)
	}
	if !strings.Contains(out, "agents=agent-a@(0,0),agent-b@(0,1)") || !strings.Contains(out, "turn_order=agent-a,agent-b") {
		t.Fatalf("expected multi-agent output, got %s", out)
	}
	if !strings.Contains(out, "step=2 actor=agent-b action=stay") || !strings.Contains(out, "step=7 actor=agent-a action=drop") {
		t.Fatalf("expected alternating actor output, got %s", out)
	}

	artifact := readCommGridLLMTestArtifact(t, workdir, "multi-agent-run")
	if !artifact.Completed || len(artifact.Steps) != 7 {
		t.Fatalf("unexpected multi-agent artifact summary: %+v", artifact)
	}
	if len(artifact.Task.Agents) != 2 || strings.Join(artifact.Task.TurnOrder, ",") != "agent-a,agent-b" {
		t.Fatalf("unexpected stored multi-agent task: %+v", artifact.Task)
	}
	if artifact.Steps[0].Result.ActorID != "agent-a" || artifact.Steps[1].Result.ActorID != "agent-b" || artifact.Steps[6].Result.ActorID != "agent-a" {
		t.Fatalf("unexpected stored actor order: %+v", artifact.Steps)
	}
	if artifact.Steps[1].Parsed.AgentID != "agent-b" {
		t.Fatalf("expected parsed action to use scheduled actor, step=%+v", artifact.Steps[1])
	}

	replayOut, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--replay-run-id", "multi-agent-run",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("replay multi-agent command: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(replayOut), &summary); err != nil {
		t.Fatalf("decode multi-agent replay json: %v\n%s", err, replayOut)
	}
	if summary.Replay == nil || !summary.Replay.Matched || len(summary.Task.Agents) != 2 {
		t.Fatalf("expected matched multi-agent replay, summary=%+v", summary)
	}
}

func TestCommGridLLMCommandWritesAgentRolesAndPrompts(t *testing.T) {
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

	if err := run(context.Background(), []string{
		"comm-grid-llm",
		"--run-id", "role-prompt-run",
		"--plan", "multi-solve",
		"--agents", "agent-a@0,0:agent-b@0,1",
		"--turn-order", "agent-a,agent-b",
		"--system-prompt", "Return JSON only.",
		"--agent-roles", "agent-a=carrier:agent-b=observer",
		"--agent-prompts", "agent-b=Return JSON only. Wait unless asked.",
	}); err != nil {
		t.Fatalf("role prompt command: %v", err)
	}

	artifact := readCommGridLLMTestArtifact(t, workdir, "role-prompt-run")
	if artifact.Task.SystemPrompt != "Return JSON only." {
		t.Fatalf("unexpected global prompt: %+v", artifact.Task)
	}
	if artifact.Task.Agents[0].Role != "carrier" || artifact.Task.Agents[1].Role != "observer" {
		t.Fatalf("unexpected stored roles: %+v", artifact.Task.Agents)
	}
	if artifact.Task.Agents[1].SystemPrompt != "Return JSON only. Wait unless asked." {
		t.Fatalf("unexpected stored prompt: %+v", artifact.Task.Agents[1])
	}
	if artifact.Steps[0].Request.SystemPrompt != "Return JSON only.\nRole: carrier" {
		t.Fatalf("expected role prompt for agent-a, got %q", artifact.Steps[0].Request.SystemPrompt)
	}
	if artifact.Steps[1].Request.SystemPrompt != "Return JSON only. Wait unless asked." {
		t.Fatalf("expected explicit prompt for agent-b, got %q", artifact.Steps[1].Request.SystemPrompt)
	}
	transcriptPath := filepath.Join(workdir, "benchmarks", "role-prompt-run", "comm_grid_llm_transcript.md")
	transcript, err := os.ReadFile(transcriptPath)
	if err != nil {
		t.Fatalf("read role transcript: %v", err)
	}
	transcriptText := string(transcript)
	if !strings.Contains(transcriptText, "Return JSON only.\nRole: carrier") || !strings.Contains(transcriptText, "Return JSON only. Wait unless asked.") {
		t.Fatalf("expected transcript prompts, got %s", transcriptText)
	}

	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--replay-run-id", "role-prompt-run",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("replay role prompt command: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode role prompt replay json: %v\n%s", err, out)
	}
	if summary.Replay == nil || !summary.Replay.Matched {
		t.Fatalf("expected matched role prompt replay, summary=%+v", summary)
	}
}

func TestCommGridLLMCommandRejectsUnknownAgentRole(t *testing.T) {
	err := run(context.Background(), []string{
		"comm-grid-llm",
		"--agents", "agent-a@0,0",
		"--agent-roles", "missing=observer",
		"--artifacts=false",
	})
	if err == nil {
		t.Fatal("expected unknown agent role error")
	}
	if !strings.Contains(err.Error(), "agent-roles references unknown agent") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCommGridLLMCommandRejectsOutOfBoundsCustomTask(t *testing.T) {
	err := run(context.Background(), []string{
		"comm-grid-llm",
		"--width", "2",
		"--height", "2",
		"--goal", "2,0",
		"--artifacts=false",
	})
	if err == nil {
		t.Fatal("expected out-of-bounds goal error")
	}
	if !strings.Contains(err.Error(), "goal out of bounds") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCommGridLLMCommandWritesMalformedFailureArtifact(t *testing.T) {
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
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--run-id", "malformed-artifact-run",
			"--plan", "malformed",
			"--steps", "2",
		})
	})
	if err != nil {
		t.Fatalf("comm-grid-llm malformed command: %v", err)
	}
	if !strings.Contains(out, "completed=false") || !strings.Contains(out, "action=llm_failure") {
		t.Fatalf("expected bounded failure output, got %s", out)
	}

	artifact := readCommGridLLMTestArtifact(t, workdir, "malformed-artifact-run")
	if artifact.Completed || len(artifact.Steps) != 2 {
		t.Fatalf("unexpected artifact summary: %+v", artifact)
	}
	first := artifact.Steps[0]
	if first.ErrorKind != "llm_failure" || !strings.Contains(first.Error, "unsupported comm-grid action") {
		t.Fatalf("expected malformed action failure, step=%+v", first)
	}
	if first.Response.Message == "" || first.Payload == "" {
		t.Fatalf("expected response payload to be stored, step=%+v", first)
	}
}

func TestCommGridLLMCommandWritesProviderFailureArtifactAndReplays(t *testing.T) {
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

	if err := run(context.Background(), []string{
		"comm-grid-llm",
		"--run-id", "provider-error-artifact-run",
		"--plan", "provider-error",
		"--steps", "2",
		"--provider-retries", "2",
		"--retry-backoff-ms", "0",
	}); err != nil {
		t.Fatalf("comm-grid-llm provider error command: %v", err)
	}
	artifact := readCommGridLLMTestArtifact(t, workdir, "provider-error-artifact-run")
	if artifact.Completed || len(artifact.Steps) != 2 {
		t.Fatalf("unexpected artifact summary: %+v", artifact)
	}
	first := artifact.Steps[0]
	if first.ErrorKind != "llm_failure" || !strings.Contains(first.Error, "fixture provider failure") {
		t.Fatalf("expected provider failure, step=%+v", first)
	}
	if first.Response.Message != "" || len(first.Response.ToolCalls) != 0 {
		t.Fatalf("expected empty provider response on provider failure, step=%+v", first)
	}
	if len(first.Attempts) != 3 || first.Attempts[0].Success || first.Attempts[2].Success {
		t.Fatalf("expected stored failed retry attempts, step=%+v", first)
	}
	transcriptPath := filepath.Join(workdir, "benchmarks", "provider-error-artifact-run", "comm_grid_llm_transcript.md")
	transcript, err := os.ReadFile(transcriptPath)
	if err != nil {
		t.Fatalf("read provider failure transcript: %v", err)
	}
	if !strings.Contains(string(transcript), "### Provider Attempts") || !strings.Contains(string(transcript), "fixture provider failure") {
		t.Fatalf("expected retry attempts in transcript, got %s", string(transcript))
	}
	indexData, err := os.ReadFile(filepath.Join(workdir, "benchmarks", "comm_grid_llm_runs.jsonl"))
	if err != nil {
		t.Fatalf("read provider failure run index: %v", err)
	}
	var indexEntry commGridLLMRunIndexEntry
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(indexData))), &indexEntry); err != nil {
		t.Fatalf("decode provider failure run index: %v\n%s", err, string(indexData))
	}
	if indexEntry.FailureCount != 2 || indexEntry.RetryCount != 4 || indexEntry.Completed {
		t.Fatalf("unexpected provider failure index entry: %+v", indexEntry)
	}

	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--replay-run-id", "provider-error-artifact-run",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("replay provider failure artifact: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode replay json: %v\n%s", err, out)
	}
	if summary.Replay == nil || !summary.Replay.Matched {
		t.Fatalf("expected matched replay, summary=%+v", summary)
	}
}

func TestCommGridLLMCommandAppendsRunIndex(t *testing.T) {
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

	for _, runID := range []string{"index-run-a", "index-run-b"} {
		if err := run(context.Background(), []string{
			"comm-grid-llm",
			"--run-id", runID,
			"--plan", "solve",
		}); err != nil {
			t.Fatalf("write indexed run %s: %v", runID, err)
		}
	}
	data, err := os.ReadFile(filepath.Join(workdir, "benchmarks", "comm_grid_llm_runs.jsonl"))
	if err != nil {
		t.Fatalf("read appended index: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected two index lines, got %d: %s", len(lines), string(data))
	}
	var first, second commGridLLMRunIndexEntry
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("decode first index line: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &second); err != nil {
		t.Fatalf("decode second index line: %v", err)
	}
	if first.RunID != "index-run-a" || second.RunID != "index-run-b" {
		t.Fatalf("unexpected index order: first=%+v second=%+v", first, second)
	}
}

func TestCommGridLLMRunsCommandPrintsTableAndFilters(t *testing.T) {
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

	if err := run(context.Background(), []string{"comm-grid-llm", "--run-id", "runs-good", "--plan", "solve"}); err != nil {
		t.Fatalf("write good run: %v", err)
	}
	if err := run(context.Background(), []string{"comm-grid-llm", "--run-id", "runs-bad", "--plan", "provider-error", "--steps", "1"}); err != nil {
		t.Fatalf("write failed run: %v", err)
	}

	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm-runs"})
	})
	if err != nil {
		t.Fatalf("list run index: %v", err)
	}
	if !strings.Contains(out, "RUN_ID\tPROVIDER\tPLAN\tSTEPS\tDONE\tFITNESS\tFAIL\tRETRY\tARTIFACT") {
		t.Fatalf("expected table header, got %s", out)
	}
	if !strings.Contains(out, "runs-good") || !strings.Contains(out, "runs-bad") {
		t.Fatalf("expected both runs, got %s", out)
	}

	filtered, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm-runs", "--completed", "false"})
	})
	if err != nil {
		t.Fatalf("filter run index: %v", err)
	}
	if strings.Contains(filtered, "runs-good") || !strings.Contains(filtered, "runs-bad") {
		t.Fatalf("expected only failed run, got %s", filtered)
	}
}

func TestCommGridLLMRunsCommandEmitsJSONWithLimit(t *testing.T) {
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

	for _, runID := range []string{"json-index-a", "json-index-b"} {
		if err := run(context.Background(), []string{"comm-grid-llm", "--run-id", runID, "--plan", "solve"}); err != nil {
			t.Fatalf("write indexed run %s: %v", runID, err)
		}
	}
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm-runs", "--json", "--limit", "1"})
	})
	if err != nil {
		t.Fatalf("json run index: %v", err)
	}
	var entries []commGridLLMRunIndexEntry
	if err := json.Unmarshal([]byte(out), &entries); err != nil {
		t.Fatalf("decode run index json: %v\n%s", err, out)
	}
	if len(entries) != 1 || entries[0].RunID != "json-index-b" {
		t.Fatalf("expected latest indexed run, got %+v", entries)
	}
}

func TestCommGridLLMRunsCommandPrintsLatestFilteredTranscript(t *testing.T) {
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

	if err := run(context.Background(), []string{"comm-grid-llm", "--run-id", "transcript-index-a", "--plan", "solve"}); err != nil {
		t.Fatalf("write first transcript run: %v", err)
	}
	if err := run(context.Background(), []string{"comm-grid-llm", "--run-id", "transcript-index-b", "--plan", "invalid"}); err != nil {
		t.Fatalf("write second transcript run: %v", err)
	}

	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm-runs", "--plan", "invalid", "--transcript"})
	})
	if err != nil {
		t.Fatalf("print latest transcript: %v", err)
	}
	if !strings.Contains(out, "- run_id: `transcript-index-b`") || !strings.Contains(out, "## Step 1: agent-1") {
		t.Fatalf("expected latest filtered transcript, got %s", out)
	}
	if strings.Contains(out, "- run_id: `transcript-index-a`") {
		t.Fatalf("expected filtered transcript only, got %s", out)
	}
}

func TestCommGridLLMRunsCommandComparesRuns(t *testing.T) {
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

	for _, args := range [][]string{
		{"comm-grid-llm", "--run-id", "compare-good-a", "--plan", "solve"},
		{"comm-grid-llm", "--run-id", "compare-good-b", "--plan", "solve"},
		{"comm-grid-llm", "--run-id", "compare-bad", "--plan", "provider-error", "--steps", "1", "--provider-retries", "1", "--retry-backoff-ms", "0"},
	} {
		if err := run(context.Background(), args); err != nil {
			t.Fatalf("write compare run %v: %v", args, err)
		}
	}

	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm-runs", "--compare"})
	})
	if err != nil {
		t.Fatalf("compare run index: %v", err)
	}
	if !strings.Contains(out, "GROUP\tRUNS\tDONE_RATE\tBEST\tAVG_FIT\tAVG_FAIL\tAVG_RETRY\tBEST_RUN") {
		t.Fatalf("expected compare header, got %s", out)
	}
	if !strings.Contains(out, "fixture/solve/3x3:key(1,0):goal(2,0):agents1:turns[agent-1]:limit80\t2\t1.000\t1.450000") {
		t.Fatalf("expected solve aggregate, got %s", out)
	}
	if !strings.Contains(out, "fixture/provider-error/3x3:key(1,0):goal(2,0):agents1:turns[agent-1]:limit80\t1\t0.000\t0.000000") {
		t.Fatalf("expected provider-error aggregate, got %s", out)
	}
}

func TestCommGridLLMRunsCommandComparesRunsAsJSON(t *testing.T) {
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

	if err := run(context.Background(), []string{"comm-grid-llm", "--run-id", "compare-json-good", "--plan", "solve"}); err != nil {
		t.Fatalf("write compare json run: %v", err)
	}
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{"comm-grid-llm-runs", "--compare", "--json"})
	})
	if err != nil {
		t.Fatalf("compare json run index: %v", err)
	}
	var comparisons []commGridLLMRunComparison
	if err := json.Unmarshal([]byte(out), &comparisons); err != nil {
		t.Fatalf("decode compare json: %v\n%s", err, out)
	}
	if len(comparisons) != 1 {
		t.Fatalf("expected one comparison, got %+v", comparisons)
	}
	got := comparisons[0]
	if got.Provider != "fixture" || got.Plan != "solve" || got.Runs != 1 || got.Completed != 1 || got.CompletionRate != 1 {
		t.Fatalf("unexpected comparison: %+v", got)
	}
	if got.BestRunID != "compare-json-good" || got.BestFitness != 1.45 || got.AverageFitness != 1.45 {
		t.Fatalf("unexpected fitness aggregate: %+v", got)
	}
}

func TestCommGridLLMRunsCommandRejectsUnsafeTranscriptPath(t *testing.T) {
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
	if err := os.MkdirAll("benchmarks", 0o755); err != nil {
		t.Fatalf("mkdir benchmarks: %v", err)
	}
	entry := commGridLLMRunIndexEntry{
		RunID:          "unsafe",
		Provider:       "fixture",
		Plan:           "solve",
		TranscriptPath: "../outside.md",
	}
	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("marshal unsafe entry: %v", err)
	}
	if err := os.WriteFile(filepath.Join("benchmarks", "comm_grid_llm_runs.jsonl"), append(data, '\n'), 0o644); err != nil {
		t.Fatalf("write unsafe index: %v", err)
	}

	err = run(context.Background(), []string{"comm-grid-llm-runs", "--transcript"})
	if err == nil {
		t.Fatal("expected unsafe transcript path error")
	}
	if !strings.Contains(err.Error(), "invalid comm-grid llm transcript path") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCommGridLLMRunsCommandHandlesMissingIndex(t *testing.T) {
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
		return run(context.Background(), []string{"comm-grid-llm-runs"})
	})
	if err != nil {
		t.Fatalf("missing run index: %v", err)
	}
	if strings.TrimSpace(out) != "RUN_ID\tPROVIDER\tPLAN\tSTEPS\tDONE\tFITNESS\tFAIL\tRETRY\tARTIFACT" {
		t.Fatalf("expected header only, got %q", out)
	}
}

func TestCommGridLLMRunsCommandRejectsMalformedIndex(t *testing.T) {
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
	if err := os.MkdirAll("benchmarks", 0o755); err != nil {
		t.Fatalf("mkdir benchmarks: %v", err)
	}
	if err := os.WriteFile(filepath.Join("benchmarks", "comm_grid_llm_runs.jsonl"), []byte("{bad json}\n"), 0o644); err != nil {
		t.Fatalf("write malformed index: %v", err)
	}
	err = run(context.Background(), []string{"comm-grid-llm-runs"})
	if err == nil {
		t.Fatal("expected malformed index error")
	}
	if !strings.Contains(err.Error(), "decode comm-grid llm run index line 1") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCommGridLLMCommandReplaysArtifacts(t *testing.T) {
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

	if err := run(context.Background(), []string{
		"comm-grid-llm",
		"--run-id", "fixture-replay-run",
		"--plan", "tool",
	}); err != nil {
		t.Fatalf("write artifact command: %v", err)
	}
	out, err := captureStdoutForCommGridLLM(func() error {
		return run(context.Background(), []string{
			"comm-grid-llm",
			"--replay-run-id", "fixture-replay-run",
			"--json",
		})
	})
	if err != nil {
		t.Fatalf("replay artifact command: %v", err)
	}
	var summary commGridLLMCommandSummary
	if err := json.Unmarshal([]byte(out), &summary); err != nil {
		t.Fatalf("decode replay json: %v\n%s", err, out)
	}
	if summary.Replay == nil || !summary.Replay.Matched {
		t.Fatalf("expected matched replay, summary=%+v", summary)
	}
	if summary.Provider != "fixture-replay" || !summary.Completed {
		t.Fatalf("unexpected replay summary: %+v", summary)
	}
}

func TestCommGridLLMCommandReplayDetectsTraceMismatch(t *testing.T) {
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

	if err := run(context.Background(), []string{
		"comm-grid-llm",
		"--run-id", "fixture-mismatch-run",
		"--plan", "solve",
	}); err != nil {
		t.Fatalf("write artifact command: %v", err)
	}
	path := filepath.Join(workdir, "benchmarks", "fixture-mismatch-run", "comm_grid_llm.json")
	artifact := readCommGridLLMTestArtifact(t, workdir, "fixture-mismatch-run")
	artifact.Trace["completed"] = false
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		t.Fatalf("marshal tampered artifact: %v", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("write tampered artifact: %v", err)
	}

	err = run(context.Background(), []string{
		"comm-grid-llm",
		"--replay-run-id", "fixture-mismatch-run",
	})
	if err == nil {
		t.Fatal("expected replay mismatch error")
	}
	if !strings.Contains(err.Error(), "replay trace mismatch") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCommGridLLMCommandRejectsPathLikeRunID(t *testing.T) {
	err := run(context.Background(), []string{
		"comm-grid-llm",
		"--run-id", "../escape",
		"--artifacts=false",
	})
	if err == nil {
		t.Fatal("expected path-like run id error")
	}
	if !strings.Contains(err.Error(), "single path segment") {
		t.Fatalf("unexpected error: %v", err)
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

func readCommGridLLMTestArtifact(t *testing.T, workdir, runID string) commGridLLMArtifact {
	t.Helper()
	path := filepath.Join(workdir, "benchmarks", runID, "comm_grid_llm.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	var artifact commGridLLMArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		t.Fatalf("decode artifact: %v", err)
	}
	return artifact
}
