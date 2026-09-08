package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func TestOpenAICompatibleProviderCompletesChat(t *testing.T) {
	var gotAuth string
	var gotReq chatCompletionRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "test-model",
			"choices": [{
				"finish_reason": "stop",
				"message": {"content": "move north"}
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
		}`))
	}))
	defer server.Close()

	t.Setenv("PROTOGONOS_TEST_LLM_KEY", "secret")
	provider, err := NewOpenAICompatibleProvider(ProviderConfig{
		BaseURL:   server.URL + "/v1",
		APIKeyEnv: "PROTOGONOS_TEST_LLM_KEY",
		Model:     "test-model",
		MaxTokens: 32,
		Seed:      9,
		Capabilities: Capabilities{
			Seed: true,
		},
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleProvider: %v", err)
	}

	res, err := provider.Complete(context.Background(), Request{
		SystemPrompt: "return a bounded action",
		Messages:     []Message{{Role: "user", Content: "state"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if res.Message != "move north" || res.FinishReason != "stop" || res.TokenCount() != 7 {
		t.Fatalf("unexpected response: %+v", res)
	}
	if gotReq.Model != "test-model" || gotReq.MaxTokens != 32 {
		t.Fatalf("unexpected request config: %+v", gotReq)
	}
	if len(gotReq.Messages) != 2 || gotReq.Messages[0].Role != "system" || gotReq.Messages[1].Content != "state" {
		t.Fatalf("unexpected messages: %+v", gotReq.Messages)
	}
	if gotReq.Seed == nil || *gotReq.Seed != 9 {
		t.Fatalf("expected seed in request, got %+v", gotReq.Seed)
	}
	if gotAuth != "Bearer secret" {
		t.Fatalf("unexpected auth header: %q", gotAuth)
	}
}

func TestOpenAICompatibleProviderListsModels(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{
			"object": "list",
			"data": [
				{"id": "local-a", "owned_by": "lmstudio"},
				{"id": "local-b", "owned_by": "local"}
			]
		}`))
	}))
	defer server.Close()

	t.Setenv("PROTOGONOS_TEST_LLM_KEY", "secret")
	provider, err := NewOpenAICompatibleProvider(ProviderConfig{
		BaseURL:   server.URL + "/v1",
		APIKeyEnv: "PROTOGONOS_TEST_LLM_KEY",
		Model:     "local-a",
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleProvider: %v", err)
	}

	models, err := provider.Models(context.Background())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models) != 2 || models[0].ID != "local-a" || models[1].OwnedBy != "local" {
		t.Fatalf("unexpected models: %+v", models)
	}
	if len(models[0].Raw) == 0 {
		t.Fatalf("expected raw model payload")
	}
	if gotAuth != "Bearer secret" {
		t.Fatalf("unexpected auth header: %q", gotAuth)
	}
}

func TestOpenAICompatibleProviderParsesToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "tool-model",
			"choices": [{
				"finish_reason": "tool_calls",
				"message": {
					"content": "",
					"tool_calls": [{
						"id": "call_1",
						"type": "function",
						"function": {
							"name": "choose_action",
							"arguments": "{\"action\":\"move_east\"}"
						}
					}]
				}
			}],
			"usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17}
		}`))
	}))
	defer server.Close()

	provider, err := NewOpenAICompatibleProvider(ProviderConfig{
		BaseURL: server.URL + "/v1",
		Model:   "tool-model",
	})
	if err != nil {
		t.Fatalf("NewOpenAICompatibleProvider: %v", err)
	}

	res, err := provider.Complete(context.Background(), Request{})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(res.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %+v", res.ToolCalls)
	}
	call := res.ToolCalls[0]
	if call.ID != "call_1" || call.Type != "function" || call.Name != "choose_action" || call.ArgumentsJSON != `{"action":"move_east"}` {
		t.Fatalf("unexpected tool call: %+v", call)
	}
}

func TestOpenAICompatibleProviderSendsToolsOnlyWhenEnabled(t *testing.T) {
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		var got chatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if requestCount == 1 && len(got.Tools) != 0 {
			t.Fatalf("tools should be omitted when capability is disabled: %+v", got.Tools)
		}
		if requestCount == 2 && len(got.Tools) != 1 {
			t.Fatalf("tools should be sent when capability is enabled: %+v", got.Tools)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer server.Close()

	tool := ToolSpec{Type: "function", Function: ToolFunction{Name: "choose_action"}}
	withoutTools, err := NewOpenAICompatibleProvider(ProviderConfig{BaseURL: server.URL + "/v1", Model: "m"})
	if err != nil {
		t.Fatalf("withoutTools provider: %v", err)
	}
	if _, err := withoutTools.Complete(context.Background(), Request{Tools: []ToolSpec{tool}}); err != nil {
		t.Fatalf("withoutTools complete: %v", err)
	}

	withTools, err := NewOpenAICompatibleProvider(ProviderConfig{
		BaseURL:      server.URL + "/v1",
		Model:        "m",
		Capabilities: Capabilities{Tools: true},
	})
	if err != nil {
		t.Fatalf("withTools provider: %v", err)
	}
	if _, err := withTools.Complete(context.Background(), Request{Tools: []ToolSpec{tool}}); err != nil {
		t.Fatalf("withTools complete: %v", err)
	}
}

func TestOpenAICompatibleProviderErrors(t *testing.T) {
	if _, err := NewOpenAICompatibleProvider(ProviderConfig{}); !errors.Is(err, ErrBaseURLRequired) {
		t.Fatalf("err=%v, want ErrBaseURLRequired", err)
	}

	provider, err := NewOpenAICompatibleProvider(ProviderConfig{BaseURL: "http://127.0.0.1:1/v1"})
	if err != nil {
		t.Fatalf("provider: %v", err)
	}
	if _, err := provider.Complete(context.Background(), Request{}); !errors.Is(err, ErrModelRequired) {
		t.Fatalf("err=%v, want ErrModelRequired", err)
	}
}

func TestOpenAICompatibleProviderReturnsHTTPStatusError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "model loading", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider, err := NewOpenAICompatibleProvider(ProviderConfig{BaseURL: server.URL + "/v1", Model: "m"})
	if err != nil {
		t.Fatalf("provider: %v", err)
	}
	_, err = provider.Complete(context.Background(), Request{})
	if err == nil {
		t.Fatal("expected status error")
	}
}

func TestOpenAICompatibleProviderOmitsEmptyAuthorization(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer server.Close()

	const envName = "PROTOGONOS_EMPTY_LLM_KEY"
	if err := os.Unsetenv(envName); err != nil {
		t.Fatalf("unset env: %v", err)
	}
	provider, err := NewOpenAICompatibleProvider(ProviderConfig{
		BaseURL:   server.URL + "/v1",
		APIKeyEnv: envName,
		Model:     "m",
	})
	if err != nil {
		t.Fatalf("provider: %v", err)
	}
	if _, err := provider.Complete(context.Background(), Request{}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if gotAuth != "" {
		t.Fatalf("expected empty auth header, got %q", gotAuth)
	}
}
