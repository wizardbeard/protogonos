package scape

import (
	"context"
	"errors"
	"strings"
	"testing"

	"protogonos/internal/llm"
)

func TestCommGridLLMActorAppliesStructuredMessage(t *testing.T) {
	provider := llm.NewFixtureProvider([]llm.Response{{
		Message:      `{"action":"move_east","message":"moving to key","to":"all","tokens":5}`,
		Usage:        llm.Usage{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18},
		FinishReason: "stop",
		Model:        "fixture-model",
	}})
	sim := NewCommGridSimulator(CommGridConfig{
		Width:    3,
		Height:   3,
		MaxSteps: 8,
		Key:      CommGridPoint{X: 1, Y: 0},
		Goal:     CommGridPoint{X: 2, Y: 0},
		Agents: []CommGridAgentState{{
			ID:       "agent-1",
			Position: CommGridPoint{},
		}},
	})
	actor := CommGridLLMActor{
		AgentID:     " agent-1 ",
		Provider:    provider,
		Model:       "fixture-model",
		MaxTokens:   64,
		Temperature: 0.1,
		Seed:        7,
	}

	result, trace, err := actor.Step(context.Background(), sim)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if result.InvalidAction {
		t.Fatalf("expected valid LLM action, result=%+v", result)
	}
	state, ok := sim.AgentState("agent-1")
	if !ok {
		t.Fatal("expected agent state")
	}
	if state.Position != (CommGridPoint{X: 1, Y: 0}) {
		t.Fatalf("unexpected position: %+v", state.Position)
	}
	if trace.ParsedAction != CommGridEast || trace.StepInput.Message != "moving to key" {
		t.Fatalf("unexpected trace: %+v", trace)
	}
	if result.Trace["llm_total_tokens"] != 18 || result.Trace["llm_action"] != "east" {
		t.Fatalf("unexpected result trace: %+v", result.Trace)
	}
	requests := provider.Requests()
	if len(requests) != 1 {
		t.Fatalf("expected one provider request, got %d", len(requests))
	}
	req := requests[0]
	if req.Model != "fixture-model" || req.MaxTokens != 64 || req.Temperature != 0.1 || req.Seed != 7 {
		t.Fatalf("unexpected provider request: %+v", req)
	}
	if req.ResponseFormat != "json_object" {
		t.Fatalf("expected JSON response format, got %q", req.ResponseFormat)
	}
	if len(req.Messages) != 1 || !strings.Contains(req.Messages[0].Content, "position=(0,0)") {
		t.Fatalf("expected state prompt, request=%+v", req)
	}
}

func TestCommGridLLMActorPrefersToolCallArguments(t *testing.T) {
	provider := llm.NewFixtureProvider([]llm.Response{{
		Message: `{"action":"stay"}`,
		ToolCalls: []llm.ToolCall{{
			Name:          "comm_grid_action",
			ArgumentsJSON: `{"action":"pick","message":"picked key","to":"all"}`,
		}},
		FinishReason: "tool_calls",
	}})
	sim := NewCommGridSimulator(CommGridConfig{
		Width:    3,
		Height:   3,
		MaxSteps: 8,
		Key:      CommGridPoint{X: 0, Y: 0},
		Goal:     CommGridPoint{X: 2, Y: 0},
		Agents: []CommGridAgentState{{
			ID:       "agent-1",
			Position: CommGridPoint{},
		}},
	})
	actor := CommGridLLMActor{
		AgentID:       "agent-1",
		Provider:      provider,
		ResponseTools: true,
	}

	result, trace, err := actor.Step(context.Background(), sim)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if result.InvalidAction || trace.ParsedAction != CommGridPick {
		t.Fatalf("expected tool-call pick action, result=%+v trace=%+v", result, trace)
	}
	state, ok := sim.AgentState("agent-1")
	if !ok || !state.Carrying {
		t.Fatalf("expected agent to carry key, state=%+v ok=%t", state, ok)
	}
	requests := provider.Requests()
	if len(requests) != 1 || len(requests[0].Tools) != 1 {
		t.Fatalf("expected tool-enabled request, requests=%+v", requests)
	}
}

func TestCommGridLLMActorPropagatesProviderError(t *testing.T) {
	wantErr := errors.New("fixture failure")
	provider := llm.NewFixtureProviderWithError(wantErr)
	sim := NewCommGridSimulator(CommGridConfig{
		Agents: []CommGridAgentState{{ID: "agent-1"}},
	})
	actor := CommGridLLMActor{
		AgentID:  "agent-1",
		Provider: provider,
	}

	if _, trace, err := actor.Step(context.Background(), sim); !errors.Is(err, wantErr) {
		t.Fatalf("expected provider error, got err=%v trace=%+v", err, trace)
	}
	if got := sim.Trace()["step"]; got != 0 {
		t.Fatalf("expected simulator to remain unstepped, trace=%+v", sim.Trace())
	}
}

func TestCommGridLLMActorRejectsMalformedActionWithoutStepping(t *testing.T) {
	provider := llm.NewFixtureProvider([]llm.Response{{
		Message: `{"action":"teleport"}`,
	}})
	sim := NewCommGridSimulator(CommGridConfig{
		Agents: []CommGridAgentState{{ID: "agent-1"}},
	})
	actor := CommGridLLMActor{
		AgentID:  "agent-1",
		Provider: provider,
	}

	_, trace, err := actor.Step(context.Background(), sim)
	if err == nil {
		t.Fatalf("expected malformed action error, trace=%+v", trace)
	}
	if trace.Payload != `{"action":"teleport"}` {
		t.Fatalf("expected payload in trace, got %+v", trace)
	}
	if got := sim.Trace()["step"]; got != 0 {
		t.Fatalf("expected simulator to remain unstepped, trace=%+v", sim.Trace())
	}
}
