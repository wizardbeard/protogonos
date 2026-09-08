package scape

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"protogonos/internal/llm"
)

const defaultCommGridLLMSystemPrompt = "Return one compact JSON object with fields action, message, to, and tokens. Use only these actions: stay, north, south, east, west, pick, drop."

var commGridActionToolParameters = json.RawMessage(`{"type":"object","properties":{"action":{"type":"string","enum":["stay","north","south","east","west","pick","drop"]},"message":{"type":"string"},"to":{"type":"string"},"tokens":{"type":"integer","minimum":0}},"required":["action"],"additionalProperties":false}`)

type CommGridLLMActor struct {
	AgentID       string
	Provider      llm.Provider
	Model         string
	SystemPrompt  string
	MaxTokens     int
	Temperature   float64
	Seed          int64
	ResponseTools bool
}

type CommGridLLMStepTrace struct {
	AgentID      string
	Request      llm.Request
	Response     llm.Response
	Payload      string
	StepInput    CommGridStepInput
	ParsedAction CommGridAction
}

func (a CommGridLLMActor) Step(ctx context.Context, sim *CommGridSimulator) (CommGridStepResult, CommGridLLMStepTrace, error) {
	if err := ctx.Err(); err != nil {
		return CommGridStepResult{}, CommGridLLMStepTrace{}, err
	}
	if sim == nil {
		return CommGridStepResult{}, CommGridLLMStepTrace{}, fmt.Errorf("comm-grid simulator is nil")
	}
	if a.Provider == nil {
		return CommGridStepResult{}, CommGridLLMStepTrace{}, fmt.Errorf("comm-grid llm provider is nil")
	}

	agentID := strings.TrimSpace(a.AgentID)
	state, ok := sim.AgentState(agentID)
	if !ok {
		return CommGridStepResult{}, CommGridLLMStepTrace{}, fmt.Errorf("comm-grid agent not found: %s", agentID)
	}

	req := a.request(sim, state)
	res, err := a.Provider.Complete(ctx, req)
	if err != nil {
		return CommGridStepResult{}, CommGridLLMStepTrace{
			AgentID:  agentID,
			Request:  req,
			Response: res,
		}, err
	}

	payload := commGridLLMPayload(res)
	input, err := DecodeCommGridLanguageAction(agentID, []byte(payload))
	if err != nil {
		return CommGridStepResult{}, CommGridLLMStepTrace{
			AgentID:  agentID,
			Request:  req,
			Response: res,
			Payload:  payload,
		}, err
	}
	result, err := sim.Step(ctx, input)
	trace := CommGridLLMStepTrace{
		AgentID:      agentID,
		Request:      req,
		Response:     res,
		Payload:      payload,
		StepInput:    input,
		ParsedAction: input.Action,
	}
	if err != nil {
		return CommGridStepResult{}, trace, err
	}
	result.Trace["llm_finish_reason"] = res.FinishReason
	result.Trace["llm_model"] = res.Model
	result.Trace["llm_total_tokens"] = res.TokenCount()
	result.Trace["llm_action"] = string(input.Action)
	return result, trace, nil
}

func (a CommGridLLMActor) request(sim *CommGridSimulator, state CommGridAgentState) llm.Request {
	prompt := strings.TrimSpace(a.SystemPrompt)
	if prompt == "" {
		prompt = defaultCommGridLLMSystemPrompt
	}
	req := llm.Request{
		Model:          strings.TrimSpace(a.Model),
		SystemPrompt:   prompt,
		Messages:       []llm.Message{{Role: "user", Content: commGridLLMUserPrompt(sim, state)}},
		MaxTokens:      a.MaxTokens,
		Temperature:    a.Temperature,
		Seed:           a.Seed,
		ResponseFormat: "json_object",
	}
	if a.ResponseTools {
		req.Tools = []llm.ToolSpec{{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "comm_grid_action",
				Description: "Choose one bounded comm-grid action for this turn.",
				Parameters:  commGridActionToolParameters,
			},
		}}
	}
	return req
}

func commGridLLMPayload(res llm.Response) string {
	if len(res.ToolCalls) > 0 {
		args := strings.TrimSpace(res.ToolCalls[0].ArgumentsJSON)
		if args != "" {
			return args
		}
	}
	return strings.TrimSpace(res.Message)
}

func commGridLLMUserPrompt(sim *CommGridSimulator, state CommGridAgentState) string {
	target := sim.cfg.Key
	if state.Carrying {
		target = sim.cfg.Goal
	}
	messages := sim.Messages()
	var b strings.Builder
	fmt.Fprintf(&b, "agent_id=%s\n", state.ID)
	fmt.Fprintf(&b, "grid_width=%d\n", sim.cfg.Width)
	fmt.Fprintf(&b, "grid_height=%d\n", sim.cfg.Height)
	fmt.Fprintf(&b, "step=%d\n", sim.step)
	fmt.Fprintf(&b, "max_steps=%d\n", sim.cfg.MaxSteps)
	fmt.Fprintf(&b, "position=(%d,%d)\n", state.Position.X, state.Position.Y)
	fmt.Fprintf(&b, "key=(%d,%d)\n", sim.cfg.Key.X, sim.cfg.Key.Y)
	fmt.Fprintf(&b, "goal=(%d,%d)\n", sim.cfg.Goal.X, sim.cfg.Goal.Y)
	fmt.Fprintf(&b, "target=(%d,%d)\n", target.X, target.Y)
	fmt.Fprintf(&b, "carrying_key=%t\n", state.Carrying)
	fmt.Fprintf(&b, "recent_messages=%s", commGridLLMMessageSummary(messages, 4))
	return b.String()
}

func commGridLLMMessageSummary(messages []CommGridMessage, limit int) string {
	if len(messages) == 0 {
		return "none"
	}
	start := len(messages) - limit
	if start < 0 {
		start = 0
	}
	parts := make([]string, 0, len(messages)-start)
	for _, msg := range messages[start:] {
		parts = append(parts, fmt.Sprintf("%d:%s->%s:%s", msg.Step, msg.From, msg.To, msg.Text))
	}
	return strings.Join(parts, " | ")
}
