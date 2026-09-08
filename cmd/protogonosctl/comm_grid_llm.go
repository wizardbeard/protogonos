package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

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
	RunID        string                   `json:"run_id"`
	ArtifactsDir string                   `json:"artifacts_dir,omitempty"`
	Replay       *commGridLLMReplayResult `json:"replay,omitempty"`
	Provider     string                   `json:"provider"`
	Plan         string                   `json:"plan"`
	Steps        []commGridLLMCommandStep `json:"steps"`
	Completed    bool                     `json:"completed"`
	Fitness      float64                  `json:"fitness"`
	Trace        map[string]any           `json:"trace"`
}

type commGridLLMArtifact struct {
	RunID     string                    `json:"run_id"`
	Provider  string                    `json:"provider"`
	Plan      string                    `json:"plan"`
	CreatedAt string                    `json:"created_at_utc"`
	Steps     []commGridLLMArtifactStep `json:"steps"`
	Completed bool                      `json:"completed"`
	Fitness   float64                   `json:"fitness"`
	Trace     map[string]any            `json:"trace"`
}

type commGridLLMReplayResult struct {
	SourceRunID string         `json:"source_run_id"`
	Matched     bool           `json:"matched"`
	Expected    map[string]any `json:"expected_trace"`
	Actual      map[string]any `json:"actual_trace"`
}

type commGridLLMArtifactStep struct {
	Step     int                     `json:"step"`
	Request  llm.Request             `json:"request"`
	Response llm.Response            `json:"response"`
	Payload  string                  `json:"payload"`
	Parsed   scape.CommGridStepInput `json:"parsed_action"`
	Result   commGridLLMCommandStep  `json:"result"`
}

func runCommGridLLM(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("comm-grid-llm", flag.ContinueOnError)
	runID := fs.String("run-id", "", "artifact run id")
	replayRunID := fs.String("replay-run-id", "", "replay stored artifacts from benchmarks/<run-id>/comm_grid_llm.json")
	providerName := fs.String("provider", "fixture", "provider: fixture|openai-compatible")
	plan := fs.String("plan", "solve", "fixture plan: solve|tool|invalid")
	steps := fs.Int("steps", 8, "maximum fixture LLM decisions")
	baseURL := fs.String("base-url", "", "OpenAI-compatible base URL ending in /v1")
	apiKeyEnv := fs.String("api-key-env", "PROTOGONOS_LLM_API_KEY", "environment variable containing provider API key")
	model := fs.String("model", "", "provider model id")
	timeoutMS := fs.Int("timeout-ms", 30000, "provider request timeout in milliseconds")
	maxTokens := fs.Int("max-tokens", 64, "maximum completion tokens per decision")
	temperature := fs.Float64("temperature", 0, "provider temperature")
	seed := fs.Int64("seed", 1, "provider seed when supported")
	jsonMode := fs.Bool("json-mode", true, "request OpenAI-compatible JSON mode")
	tools := fs.Bool("tools", false, "request tool-call action output when supported")
	writeArtifacts := fs.Bool("artifacts", true, "write comm-grid LLM artifacts under benchmarks/<run-id>")
	jsonOut := fs.Bool("json", false, "emit summary as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *steps <= 0 {
		return errors.New("steps must be > 0")
	}
	if strings.TrimSpace(*replayRunID) != "" {
		return runCommGridLLMReplay(ctx, strings.TrimSpace(*replayRunID), *jsonOut)
	}

	provider, actorModel, useTools, summaryProvider, summaryPlan, err := commGridLLMProviderFromFlags(commGridLLMProviderFlags{
		Provider:    *providerName,
		Plan:        *plan,
		BaseURL:     *baseURL,
		APIKeyEnv:   *apiKeyEnv,
		Model:       *model,
		TimeoutMS:   *timeoutMS,
		MaxTokens:   *maxTokens,
		Temperature: *temperature,
		Seed:        *seed,
		JSONMode:    *jsonMode,
		Tools:       *tools,
	})
	if err != nil {
		return err
	}

	now := time.Now().UTC()
	id := strings.TrimSpace(*runID)
	if id == "" {
		id = fmt.Sprintf("comm-grid-llm-%s-%d", summaryProvider, now.UnixNano())
	}
	if err := validateCommGridLLMRunID(id); err != nil {
		return err
	}
	summary, artifactSteps, err := executeCommGridLLM(ctx, commGridLLMExecutionConfig{
		RunID:           id,
		Provider:        provider,
		ActorModel:      actorModel,
		SummaryProvider: summaryProvider,
		SummaryPlan:     summaryPlan,
		MaxSteps:        *steps,
		MaxTokens:       *maxTokens,
		Temperature:     *temperature,
		Seed:            *seed,
		UseTools:        useTools,
	})
	if err != nil {
		return err
	}
	artifact := commGridLLMArtifact{
		RunID:     id,
		Provider:  summaryProvider,
		Plan:      summaryPlan,
		CreatedAt: now.Format(time.RFC3339Nano),
		Steps:     artifactSteps,
		Completed: summary.Completed,
		Fitness:   summary.Fitness,
		Trace:     summary.Trace,
	}

	if *writeArtifacts {
		artifactDir, err := writeCommGridLLMArtifact(benchmarksDir, artifact)
		if err != nil {
			return err
		}
		summary.ArtifactsDir = artifactDir
	}

	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(summary)
	}
	fmt.Printf("comm_grid_llm run_id=%s provider=%s plan=%s steps=%d completed=%t fitness=%.6f\n", summary.RunID, summary.Provider, summary.Plan, len(summary.Steps), summary.Completed, summary.Fitness)
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
	if summary.ArtifactsDir != "" {
		fmt.Printf("artifacts_dir=%s\n", filepath.Clean(summary.ArtifactsDir))
	}
	return nil
}

func runCommGridLLMReplay(ctx context.Context, runID string, jsonOut bool) error {
	if err := validateCommGridLLMRunID(runID); err != nil {
		return err
	}
	artifact, err := readCommGridLLMArtifact(benchmarksDir, runID)
	if err != nil {
		return err
	}
	if len(artifact.Steps) == 0 {
		return fmt.Errorf("comm-grid llm artifact has no steps: %s", runID)
	}
	responses := make([]llm.Response, 0, len(artifact.Steps))
	useTools := false
	model := "fixture-comm-grid"
	maxTokens := 64
	temperature := 0.0
	seed := int64(1)
	for _, step := range artifact.Steps {
		responses = append(responses, step.Response)
		if len(step.Response.ToolCalls) > 0 {
			useTools = true
		}
		if strings.TrimSpace(step.Request.Model) != "" {
			model = strings.TrimSpace(step.Request.Model)
		}
		if step.Request.MaxTokens > 0 {
			maxTokens = step.Request.MaxTokens
		}
		if step.Request.Temperature != 0 {
			temperature = step.Request.Temperature
		}
		if step.Request.Seed != 0 {
			seed = step.Request.Seed
		}
	}

	maxSteps := commGridTraceInt(artifact.Trace, "max_steps", len(artifact.Steps))
	summary, _, err := executeCommGridLLM(ctx, commGridLLMExecutionConfig{
		RunID:           runID,
		Provider:        llm.NewFixtureProvider(responses),
		ActorModel:      model,
		SummaryProvider: "fixture-replay",
		SummaryPlan:     strings.TrimSpace(artifact.Plan),
		MaxSteps:        maxSteps,
		MaxTokens:       maxTokens,
		Temperature:     temperature,
		Seed:            seed,
		UseTools:        useTools,
	})
	if err != nil {
		return err
	}
	replay := commGridLLMReplayResult{
		SourceRunID: runID,
		Matched:     commGridTraceEqual(artifact.Trace, summary.Trace),
		Expected:    artifact.Trace,
		Actual:      summary.Trace,
	}
	summary.Replay = &replay
	if jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(summary)
	}
	fmt.Printf("comm_grid_llm_replay run_id=%s matched=%t completed=%t fitness=%.6f\n", runID, replay.Matched, summary.Completed, summary.Fitness)
	fmt.Printf("expected_trace=%v\n", replay.Expected)
	fmt.Printf("actual_trace=%v\n", replay.Actual)
	if !replay.Matched {
		return fmt.Errorf("comm-grid llm replay trace mismatch: %s", runID)
	}
	return nil
}

type commGridLLMExecutionConfig struct {
	RunID           string
	Provider        llm.Provider
	ActorModel      string
	SummaryProvider string
	SummaryPlan     string
	MaxSteps        int
	MaxTokens       int
	Temperature     float64
	Seed            int64
	UseTools        bool
}

func executeCommGridLLM(ctx context.Context, cfg commGridLLMExecutionConfig) (commGridLLMCommandSummary, []commGridLLMArtifactStep, error) {
	sim := scape.NewCommGridSimulator(scape.CommGridConfig{
		Width:    3,
		Height:   3,
		MaxSteps: cfg.MaxSteps,
		Key:      scape.CommGridPoint{X: 1, Y: 0},
		Goal:     scape.CommGridPoint{X: 2, Y: 0},
		Agents: []scape.CommGridAgentState{{
			ID:       "agent-1",
			Position: scape.CommGridPoint{},
		}},
	})
	actor := scape.CommGridLLMActor{
		AgentID:       "agent-1",
		Provider:      cfg.Provider,
		Model:         cfg.ActorModel,
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		Seed:          cfg.Seed,
		ResponseTools: cfg.UseTools,
	}

	summary := commGridLLMCommandSummary{
		RunID:    cfg.RunID,
		Provider: cfg.SummaryProvider,
		Plan:     cfg.SummaryPlan,
	}
	artifactSteps := make([]commGridLLMArtifactStep, 0, cfg.MaxSteps)
	for !sim.Done() && len(summary.Steps) < cfg.MaxSteps {
		result, trace, err := actor.Step(ctx, sim)
		if err != nil {
			return commGridLLMCommandSummary{}, nil, err
		}
		state, _ := sim.AgentState("agent-1")
		step := commGridLLMCommandStep{
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
		}
		summary.Steps = append(summary.Steps, step)
		artifactSteps = append(artifactSteps, commGridLLMArtifactStep{
			Step:     step.Step,
			Request:  trace.Request,
			Response: trace.Response,
			Payload:  trace.Payload,
			Parsed:   trace.StepInput,
			Result:   step,
		})
	}
	summary.Completed = sim.Done() && sim.Trace()["completed"] == true
	summary.Fitness = float64(sim.Fitness())
	summary.Trace = map[string]any(sim.Trace())
	return summary, artifactSteps, nil
}

func writeCommGridLLMArtifact(baseDir string, artifact commGridLLMArtifact) (string, error) {
	runID := strings.TrimSpace(artifact.RunID)
	if err := validateCommGridLLMRunID(runID); err != nil {
		return "", err
	}
	runDir := filepath.Join(baseDir, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(runDir, "comm_grid_llm.json")
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return "", err
	}
	return filepath.Clean(runDir), nil
}

func readCommGridLLMArtifact(baseDir, runID string) (commGridLLMArtifact, error) {
	if err := validateCommGridLLMRunID(runID); err != nil {
		return commGridLLMArtifact{}, err
	}
	path := filepath.Join(baseDir, runID, "comm_grid_llm.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return commGridLLMArtifact{}, err
	}
	var artifact commGridLLMArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return commGridLLMArtifact{}, err
	}
	return artifact, nil
}

func validateCommGridLLMRunID(runID string) error {
	if strings.TrimSpace(runID) == "" {
		return errors.New("comm-grid llm run id required")
	}
	if runID == "." || runID == ".." || strings.ContainsAny(runID, `/\`) {
		return fmt.Errorf("comm-grid llm run id must be a single path segment: %s", runID)
	}
	return nil
}

func commGridTraceInt(trace map[string]any, key string, fallback int) int {
	if trace == nil {
		return fallback
	}
	switch v := trace[key].(type) {
	case int:
		if v > 0 {
			return v
		}
	case int64:
		if v > 0 {
			return int(v)
		}
	case float64:
		if v > 0 {
			return int(v)
		}
	case json.Number:
		n, err := v.Int64()
		if err == nil && n > 0 {
			return int(n)
		}
	}
	return fallback
}

func commGridTraceEqual(a, b map[string]any) bool {
	normalizedA, err := commGridNormalizeTraceJSON(a)
	if err != nil {
		return false
	}
	normalizedB, err := commGridNormalizeTraceJSON(b)
	if err != nil {
		return false
	}
	return string(normalizedA) == string(normalizedB)
}

func commGridNormalizeTraceJSON(trace map[string]any) ([]byte, error) {
	data, err := json.Marshal(trace)
	if err != nil {
		return nil, err
	}
	var normalized map[string]any
	if err := json.Unmarshal(data, &normalized); err != nil {
		return nil, err
	}
	return json.Marshal(normalized)
}

type commGridLLMProviderFlags struct {
	Provider    string
	Plan        string
	BaseURL     string
	APIKeyEnv   string
	Model       string
	TimeoutMS   int
	MaxTokens   int
	Temperature float64
	Seed        int64
	JSONMode    bool
	Tools       bool
}

func commGridLLMProviderFromFlags(flags commGridLLMProviderFlags) (llm.Provider, string, bool, string, string, error) {
	providerName := strings.TrimSpace(strings.ToLower(flags.Provider))
	switch providerName {
	case "", "fixture":
		responses, useTools, err := commGridLLMFixturePlan(flags.Plan)
		if err != nil {
			return nil, "", false, "", "", err
		}
		return llm.NewFixtureProvider(responses), "fixture-comm-grid", useTools, "fixture", strings.TrimSpace(strings.ToLower(flags.Plan)), nil
	case "openai-compatible":
		model := strings.TrimSpace(flags.Model)
		if model == "" {
			return nil, "", false, "", "", llm.ErrModelRequired
		}
		provider, err := llm.NewOpenAICompatibleProvider(llm.ProviderConfig{
			BaseURL:     flags.BaseURL,
			APIKeyEnv:   flags.APIKeyEnv,
			Model:       model,
			TimeoutMS:   flags.TimeoutMS,
			MaxTokens:   flags.MaxTokens,
			Temperature: flags.Temperature,
			Seed:        flags.Seed,
			Capabilities: llm.Capabilities{
				ChatCompletions: true,
				JSONMode:        flags.JSONMode,
				Tools:           flags.Tools,
				Seed:            flags.Seed != 0,
				UsageTokens:     true,
			},
		})
		if err != nil {
			return nil, "", false, "", "", err
		}
		return provider, model, flags.Tools, "openai-compatible", "live", nil
	default:
		return nil, "", false, "", "", fmt.Errorf("unsupported comm-grid llm provider: %s", flags.Provider)
	}
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
