package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"protogonos/internal/llm"
	"protogonos/internal/scape"
)

type commGridLLMCommandStep struct {
	Step          int                     `json:"step"`
	Action        string                  `json:"action"`
	Message       string                  `json:"message,omitempty"`
	To            string                  `json:"to,omitempty"`
	InvalidAction bool                    `json:"invalid_action"`
	Done          bool                    `json:"done"`
	Completed     bool                    `json:"completed"`
	Fitness       float64                 `json:"fitness"`
	Trace         map[string]any          `json:"trace"`
	ProviderTrace map[string]any          `json:"provider_trace"`
	Agent         map[string]any          `json:"agent"`
	Messages      []scape.CommGridMessage `json:"messages,omitempty"`
}

type commGridLLMCommandSummary struct {
	Plan      string                   `json:"plan"`
	Steps     []commGridLLMCommandStep `json:"steps"`
	Completed bool                     `json:"completed"`
	Fitness   float64                  `json:"fitness"`
	Trace     map[string]any           `json:"trace"`
}

func runCommGridLLM(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("comm-grid-llm", flag.ContinueOnError)
	plan := fs.String("plan", "solve", "fixture plan: solve|tool|invalid")
	steps := fs.Int("steps", 8, "maximum fixture LLM decisions")
	jsonOut := fs.Bool("json", false, "emit summary as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *steps <= 0 {
		return errors.New("steps must be > 0")
	}

	responses, useTools, err := commGridLLMFixturePlan(*plan)
	if err != nil {
		return err
	}
	sim := scape.NewCommGridSimulator(scape.CommGridConfig{
		Width:    3,
		Height:   3,
		MaxSteps: *steps,
		Key:      scape.CommGridPoint{X: 1, Y: 0},
		Goal:     scape.CommGridPoint{X: 2, Y: 0},
		Agents: []scape.CommGridAgentState{{
			ID:       "agent-1",
			Position: scape.CommGridPoint{},
		}},
	})
	actor := scape.CommGridLLMActor{
		AgentID:       "agent-1",
		Provider:      llm.NewFixtureProvider(responses),
		Model:         "fixture-comm-grid",
		MaxTokens:     64,
		Temperature:   0,
		Seed:          1,
		ResponseTools: useTools,
	}

	summary := commGridLLMCommandSummary{Plan: strings.TrimSpace(strings.ToLower(*plan))}
	for !sim.Done() && len(summary.Steps) < *steps {
		result, trace, err := actor.Step(ctx, sim)
		if err != nil {
			return err
		}
		state, _ := sim.AgentState("agent-1")
		summary.Steps = append(summary.Steps, commGridLLMCommandStep{
			Step:          len(summary.Steps) + 1,
			Action:        string(trace.ParsedAction),
			Message:       trace.StepInput.Message,
			To:            trace.StepInput.To,
			InvalidAction: result.InvalidAction,
			Done:          result.Done,
			Completed:     result.Completed,
			Fitness:       float64(result.Fitness),
			Trace:         map[string]any(result.Trace),
			ProviderTrace: map[string]any{
				"finish_reason": trace.Response.FinishReason,
				"model":         trace.Response.Model,
				"tokens":        trace.Response.TokenCount(),
				"payload":       trace.Payload,
			},
			Agent: map[string]any{
				"id":       state.ID,
				"x":        state.Position.X,
				"y":        state.Position.Y,
				"carrying": state.Carrying,
			},
			Messages: sim.Messages(),
		})
	}
	summary.Completed = sim.Done() && sim.Trace()["completed"] == true
	summary.Fitness = float64(sim.Fitness())
	summary.Trace = map[string]any(sim.Trace())

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(summary)
	}
	fmt.Printf("comm_grid_llm_fixture plan=%s steps=%d completed=%t fitness=%.6f\n", summary.Plan, len(summary.Steps), summary.Completed, summary.Fitness)
	for _, step := range summary.Steps {
		fmt.Printf("step=%d action=%s invalid=%t done=%t fitness=%.6f message=%q to=%q\n",
			step.Step,
			step.Action,
			step.InvalidAction,
			step.Done,
			step.Fitness,
			step.Message,
			step.To,
		)
	}
	fmt.Printf("trace=%v\n", summary.Trace)
	return nil
}

func commGridLLMFixturePlan(plan string) ([]llm.Response, bool, error) {
	normalized := strings.TrimSpace(strings.ToLower(plan))
	switch normalized {
	case "", "solve":
		return []llm.Response{
			commGridLLMFixtureContent(`{"action":"east","message":"move to key","to":"all","tokens":3}`, 8),
			commGridLLMFixtureContent(`{"action":"pick","message":"picked key","to":"all","tokens":2}`, 7),
			commGridLLMFixtureContent(`{"action":"east","message":"move to goal","to":"all","tokens":3}`, 8),
			commGridLLMFixtureContent(`{"action":"drop","message":"delivered key","to":"all","tokens":2}`, 7),
		}, false, nil
	case "tool":
		return []llm.Response{
			commGridLLMFixtureTool(`{"action":"east","message":"move to key","to":"all","tokens":3}`, 8),
			commGridLLMFixtureTool(`{"action":"pick","message":"picked key","to":"all","tokens":2}`, 7),
			commGridLLMFixtureTool(`{"action":"east","message":"move to goal","to":"all","tokens":3}`, 8),
			commGridLLMFixtureTool(`{"action":"drop","message":"delivered key","to":"all","tokens":2}`, 7),
		}, true, nil
	case "invalid":
		return []llm.Response{
			commGridLLMFixtureContent(`{"action":"west","message":"bad wall move","to":"all","tokens":3}`, 8),
			commGridLLMFixtureContent(`{"action":"east","message":"recover","to":"all","tokens":1}`, 6),
		}, false, nil
	default:
		return nil, false, fmt.Errorf("unsupported comm-grid llm fixture plan: %s", plan)
	}
}

func commGridLLMFixtureContent(payload string, tokens int) llm.Response {
	return llm.Response{
		Message:      payload,
		Usage:        llm.Usage{TotalTokens: tokens},
		FinishReason: "stop",
		Model:        "fixture-comm-grid",
	}
}

func commGridLLMFixtureTool(payload string, tokens int) llm.Response {
	return llm.Response{
		ToolCalls: []llm.ToolCall{{
			Name:          "comm_grid_action",
			ArgumentsJSON: payload,
		}},
		Usage:        llm.Usage{TotalTokens: tokens},
		FinishReason: "tool_calls",
		Model:        "fixture-comm-grid",
	}
}
