package scape

import (
	"context"
	"errors"
	"testing"

	"protogonos/internal/llm"
)

func TestCommGridMentorScapeFixtureMentorCompletesAndTraces(t *testing.T) {
	provider := llm.NewFixtureProvider([]llm.Response{
		{Message: "east", Usage: llm.Usage{TotalTokens: 1}, Model: "fixture-mentor", FinishReason: "stop"},
		{Message: "pick", Usage: llm.Usage{TotalTokens: 1}, Model: "fixture-mentor", FinishReason: "stop"},
		{Message: "east", Usage: llm.Usage{TotalTokens: 1}, Model: "fixture-mentor", FinishReason: "stop"},
		{Message: "drop", Usage: llm.Usage{TotalTokens: 1}, Model: "fixture-mentor", FinishReason: "stop"},
	})
	sc := CommGridMentorScape{Config: CommGridMentorConfig{
		CommGridConfig: CommGridConfig{
			Width:    3,
			Height:   3,
			MaxSteps: 8,
			Key:      CommGridPoint{X: 1, Y: 0},
			Goal:     CommGridPoint{X: 2, Y: 0},
			Agents: []CommGridAgentState{{
				ID:       "agent-1",
				Position: CommGridPoint{},
			}},
		},
		Provider:  provider,
		Model:     "fixture-mentor",
		MaxTokens: 12,
		Seed:      42,
	}}
	agent := scriptedStepAgent{id: "agent-1", fn: commGridMentorHintPolicy}

	fitness, trace, err := sc.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if fitness <= 1.0 {
		t.Fatalf("expected completed run with useful fitness, got %f trace=%+v", fitness, trace)
	}
	if trace["completed"] != true || trace["mentor_tokens"] != 4 || trace["mentor_failures"] != 0 {
		t.Fatalf("unexpected mentor trace: %+v", trace)
	}
	steps, ok := trace["mentor_steps"].([]CommGridMentorStepTrace)
	if !ok || len(steps) != 4 {
		t.Fatalf("expected four mentor step traces, got %#v", trace["mentor_steps"])
	}
	if steps[0].Hint != "east" || steps[0].HintAction != "east" || steps[3].Action != "drop" {
		t.Fatalf("unexpected mentor steps: %+v", steps)
	}
	requests := provider.Requests()
	if len(requests) != 4 {
		t.Fatalf("expected four provider requests, got %d", len(requests))
	}
	if requests[0].Model != "fixture-mentor" || requests[0].MaxTokens != 12 || requests[0].Seed != 42 {
		t.Fatalf("unexpected provider request: %+v", requests[0])
	}
}

func TestCommGridMentorScapeProviderFailureUsesEmptyHint(t *testing.T) {
	provider := llm.NewFixtureProviderWithError(errors.New("mentor unavailable"))
	sc := CommGridMentorScape{Config: CommGridMentorConfig{
		CommGridConfig: CommGridConfig{
			Width:    3,
			Height:   3,
			MaxSteps: 1,
			Agents: []CommGridAgentState{{
				ID:       "agent-1",
				Position: CommGridPoint{},
			}},
		},
		Provider:       provider,
		FailurePenalty: 0.1,
	}}
	agent := scriptedStepAgent{id: "agent-1", fn: func([]float64) []float64 {
		return []float64{0, 0, 0}
	}}

	fitness, trace, err := sc.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if trace["mentor_failures"] != 1 {
		t.Fatalf("expected one mentor failure, trace=%+v", trace)
	}
	steps, ok := trace["mentor_steps"].([]CommGridMentorStepTrace)
	if !ok || len(steps) != 1 || steps[0].ProviderError == "" || steps[0].Hint != "" {
		t.Fatalf("expected bounded failed mentor step, steps=%#v", trace["mentor_steps"])
	}
	if fitness != Fitness(0) {
		t.Fatalf("expected clamped low fitness after failure and no progress, got %f", fitness)
	}
}

func TestCommGridMentorHintVectorEncodesActionWords(t *testing.T) {
	vector, action := commGridMentorHintVector("Please move east now.")
	if action != CommGridEast || len(vector) != 4 || vector[0] != 1 || vector[3] != 1 {
		t.Fatalf("unexpected east hint vector: action=%s vector=%+v", action, vector)
	}
	vector, action = commGridMentorHintVector("pick up the key")
	if action != CommGridPick || vector[2] != 1 || vector[3] != 1 {
		t.Fatalf("unexpected pick hint vector: action=%s vector=%+v", action, vector)
	}
	vector, action = commGridMentorHintVector("no useful hint")
	if action != "" || vector[3] != 0 {
		t.Fatalf("unexpected empty hint vector: action=%s vector=%+v", action, vector)
	}
}

func commGridMentorHintPolicy(input []float64) []float64 {
	if len(input) < 8 {
		return []float64{0, 0, 0}
	}
	hintX := input[4]
	hintY := input[5]
	hintTool := input[6]
	if hintTool > 0.5 {
		return []float64{0, 0, 1}
	}
	if hintTool < -0.5 {
		return []float64{0, 0, -1}
	}
	return []float64{hintX, hintY, 0}
}
