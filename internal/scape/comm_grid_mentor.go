package scape

import (
	"context"
	"fmt"
	"strings"

	"protogonos/internal/llm"
)

const defaultCommGridMentorSystemPrompt = "Send one short hint for the learner. Use plain text. Prefer one action word: north, south, east, west, pick, drop, or wait."

type CommGridMentorConfig struct {
	CommGridConfig
	Provider       llm.Provider
	Model          string
	SystemPrompt   string
	MaxTokens      int
	Temperature    float64
	Seed           int64
	TokenCost      float64
	FailurePenalty float64
}

type CommGridMentorScape struct {
	Config CommGridMentorConfig
}

type CommGridMentorStepTrace struct {
	Step          int           `json:"step"`
	AgentID       string        `json:"agent_id"`
	Request       llm.Request   `json:"request"`
	Response      llm.Response  `json:"response"`
	Hint          string        `json:"hint,omitempty"`
	HintAction    string        `json:"hint_action,omitempty"`
	HintVector    []float64     `json:"hint_vector,omitempty"`
	Action        string        `json:"action"`
	InvalidAction bool          `json:"invalid_action"`
	ProviderError string        `json:"provider_error,omitempty"`
	Trace         CommGridTrace `json:"trace"`
}

type CommGridTrace map[string]any

func (CommGridMentorScape) Name() string {
	return "comm-grid-mentor"
}

func (s CommGridMentorScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	if s.Config.Provider == nil {
		return 0, nil, fmt.Errorf("comm-grid mentor provider is nil")
	}

	cfg := s.Config
	sim := NewCommGridSimulator(cfg.CommGridConfig)
	var steps []CommGridMentorStepTrace
	mentorFailures := 0
	mentorTokens := 0
	for !sim.Done() {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}
		state, ok := sim.AgentState(agent.ID())
		if !ok {
			return 0, nil, fmt.Errorf("comm-grid agent not found: %s", agent.ID())
		}

		req := commGridMentorRequest(cfg, sim, state)
		res, err := cfg.Provider.Complete(ctx, req)
		hint := ""
		providerErr := ""
		if err != nil {
			mentorFailures++
			providerErr = err.Error()
		} else {
			hint = strings.TrimSpace(res.Message)
			mentorTokens += res.TokenCount()
		}

		input, hintVector, hintAction := commGridMentorStepVector(sim, state, hint)
		out, err := runner.RunStep(ctx, input)
		if err != nil {
			return 0, nil, err
		}
		action, err := DecodeCommGridAction(out)
		if err != nil {
			return 0, nil, err
		}
		result, err := sim.Step(ctx, CommGridStepInput{AgentID: agent.ID(), Action: action})
		if err != nil {
			return 0, nil, err
		}
		steps = append(steps, CommGridMentorStepTrace{
			Step:          len(steps),
			AgentID:       agent.ID(),
			Request:       req,
			Response:      res,
			Hint:          hint,
			HintAction:    string(hintAction),
			HintVector:    hintVector,
			Action:        string(action),
			InvalidAction: result.InvalidAction,
			ProviderError: providerErr,
			Trace:         CommGridTrace(result.Trace),
		})
	}

	fitness := float64(sim.Fitness())
	fitness -= float64(mentorTokens) * commGridMentorTokenCost(cfg)
	fitness -= float64(mentorFailures) * commGridMentorFailurePenalty(cfg)
	fitness = commGridClamp(fitness, 0, 1.5)
	trace := sim.Trace()
	trace["scape"] = "comm-grid-mentor"
	trace["mentor_steps"] = steps
	trace["mentor_tokens"] = mentorTokens
	trace["mentor_failures"] = mentorFailures
	trace["fitness"] = fitness
	return Fitness(fitness), trace, nil
}

func commGridMentorRequest(cfg CommGridMentorConfig, sim *CommGridSimulator, state CommGridAgentState) llm.Request {
	prompt := strings.TrimSpace(cfg.SystemPrompt)
	if prompt == "" {
		prompt = defaultCommGridMentorSystemPrompt
	}
	return llm.Request{
		Model:        strings.TrimSpace(cfg.Model),
		SystemPrompt: prompt,
		Messages: []llm.Message{{
			Role:    "user",
			Content: commGridLLMUserPrompt(sim, state),
		}},
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		Seed:        cfg.Seed,
	}
}

func commGridMentorStepVector(sim *CommGridSimulator, state CommGridAgentState, hint string) ([]float64, []float64, CommGridAction) {
	base := commGridStepVector(sim, state)
	hintVector, action := commGridMentorHintVector(hint)
	out := make([]float64, 0, len(base)+len(hintVector))
	out = append(out, base...)
	out = append(out, hintVector...)
	return out, hintVector, action
}

func commGridMentorHintVector(hint string) ([]float64, CommGridAction) {
	action, ok := commGridMentorHintAction(hint)
	vector := []float64{0, 0, 0, 0}
	if !ok {
		return vector, ""
	}
	vector[3] = 1
	switch action {
	case CommGridNorth:
		vector[1] = -1
	case CommGridSouth:
		vector[1] = 1
	case CommGridEast:
		vector[0] = 1
	case CommGridWest:
		vector[0] = -1
	case CommGridPick:
		vector[2] = 1
	case CommGridDrop:
		vector[2] = -1
	}
	return vector, action
}

func commGridMentorHintAction(hint string) (CommGridAction, bool) {
	normalized := strings.NewReplacer(
		".", " ", ",", " ", ";", " ", ":", " ", "!", " ", "?", " ",
		"(", " ", ")", " ", "[", " ", "]", " ", "{", " ", "}", " ",
	).Replace(strings.ToLower(hint))
	for _, word := range strings.Fields(normalized) {
		action, err := ParseCommGridAction(word)
		if err == nil {
			return action, true
		}
	}
	return "", false
}

func commGridMentorTokenCost(cfg CommGridMentorConfig) float64 {
	if cfg.TokenCost == 0 {
		return 0.01
	}
	return cfg.TokenCost
}

func commGridMentorFailurePenalty(cfg CommGridMentorConfig) float64 {
	if cfg.FailurePenalty == 0 {
		return 0.2
	}
	return cfg.FailurePenalty
}
