package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"protogonos/internal/llm"
	"protogonos/internal/scape"
)

type commGridLLMCommandStep struct {
	Step          int                     `json:"step"`
	ActorID       string                  `json:"actor_id"`
	Action        string                  `json:"action"`
	Message       string                  `json:"message,omitempty"`
	To            string                  `json:"to,omitempty"`
	Error         string                  `json:"error,omitempty"`
	ErrorKind     string                  `json:"error_kind,omitempty"`
	InvalidAction bool                    `json:"invalid_action"`
	Done          bool                    `json:"done"`
	Completed     bool                    `json:"completed"`
	Fitness       float64                 `json:"fitness"`
	Trace         map[string]any          `json:"trace"`
	ProviderTrace map[string]any          `json:"provider_trace"`
	Attempts      []commGridLLMAttempt    `json:"attempts,omitempty"`
	Agent         map[string]any          `json:"agent"`
	Messages      []scape.CommGridMessage `json:"messages,omitempty"`
}

type commGridLLMCommandSummary struct {
	RunID        string                   `json:"run_id"`
	ArtifactsDir string                   `json:"artifacts_dir,omitempty"`
	Replay       *commGridLLMReplayResult `json:"replay,omitempty"`
	Provider     string                   `json:"provider"`
	Plan         string                   `json:"plan"`
	Task         commGridLLMTaskConfig    `json:"task"`
	Steps        []commGridLLMCommandStep `json:"steps"`
	Completed    bool                     `json:"completed"`
	Fitness      float64                  `json:"fitness"`
	Trace        map[string]any           `json:"trace"`
}

type commGridLLMArtifact struct {
	RunID                string                    `json:"run_id"`
	Provider             string                    `json:"provider"`
	Plan                 string                    `json:"plan"`
	CreatedAt            string                    `json:"created_at_utc"`
	DurationMS           int64                     `json:"duration_ms"`
	Task                 commGridLLMTaskConfig     `json:"task"`
	Steps                []commGridLLMArtifactStep `json:"steps"`
	Completed            bool                      `json:"completed"`
	Fitness              float64                   `json:"fitness"`
	TotalTokens          int                       `json:"total_tokens"`
	AverageTokensPerStep float64                   `json:"average_tokens_per_step"`
	Trace                map[string]any            `json:"trace"`
}

type commGridLLMTaskConfig struct {
	Width        int                 `json:"width"`
	Height       int                 `json:"height"`
	Key          scape.CommGridPoint `json:"key"`
	Goal         scape.CommGridPoint `json:"goal"`
	SystemPrompt string              `json:"system_prompt,omitempty"`
	AgentID      string              `json:"agent_id"`
	Agent        scape.CommGridPoint `json:"agent"`
	Agents       []commGridLLMAgent  `json:"agents,omitempty"`
	TurnOrder    []string            `json:"turn_order,omitempty"`
	MessageLimit int                 `json:"message_limit"`
}

type commGridLLMAgent struct {
	ID           string              `json:"id"`
	Position     scape.CommGridPoint `json:"position"`
	Role         string              `json:"role,omitempty"`
	SystemPrompt string              `json:"system_prompt,omitempty"`
}

type commGridLLMReplayResult struct {
	SourceRunID string         `json:"source_run_id"`
	Matched     bool           `json:"matched"`
	Expected    map[string]any `json:"expected_trace"`
	Actual      map[string]any `json:"actual_trace"`
}

type commGridLLMArtifactStep struct {
	Step      int                     `json:"step"`
	Request   llm.Request             `json:"request"`
	Response  llm.Response            `json:"response"`
	Attempts  []commGridLLMAttempt    `json:"attempts,omitempty"`
	Payload   string                  `json:"payload"`
	Parsed    scape.CommGridStepInput `json:"parsed_action"`
	Error     string                  `json:"error,omitempty"`
	ErrorKind string                  `json:"error_kind,omitempty"`
	Result    commGridLLMCommandStep  `json:"result"`
}

type commGridLLMRunIndexEntry struct {
	RunID                string                `json:"run_id"`
	Provider             string                `json:"provider"`
	Plan                 string                `json:"plan"`
	CreatedAt            string                `json:"created_at_utc"`
	DurationMS           int64                 `json:"duration_ms"`
	Task                 commGridLLMTaskConfig `json:"task"`
	Steps                int                   `json:"steps"`
	Completed            bool                  `json:"completed"`
	Fitness              float64               `json:"fitness"`
	TotalTokens          int                   `json:"total_tokens"`
	AverageTokensPerStep float64               `json:"average_tokens_per_step"`
	FailureCount         int                   `json:"failure_count"`
	RetryCount           int                   `json:"retry_count"`
	ArtifactPath         string                `json:"artifact_path"`
	TranscriptPath       string                `json:"transcript_path"`
}

type commGridLLMRunIndexFilter struct {
	Limit           int
	Provider        string
	Plan            string
	FilterCompleted bool
	Completed       bool
}

type commGridLLMRunComparison struct {
	Group                string  `json:"group"`
	Provider             string  `json:"provider"`
	Plan                 string  `json:"plan"`
	TaskShape            string  `json:"task_shape"`
	Runs                 int     `json:"runs"`
	Completed            int     `json:"completed"`
	CompletionRate       float64 `json:"completion_rate"`
	BestFitness          float64 `json:"best_fitness"`
	AverageFitness       float64 `json:"average_fitness"`
	AverageTokensPerRun  float64 `json:"average_tokens_per_run"`
	AverageTokensPerStep float64 `json:"average_tokens_per_step"`
	AverageDurationMS    float64 `json:"average_duration_ms"`
	AverageFailures      float64 `json:"average_failures"`
	AverageRetries       float64 `json:"average_retries"`
	BestRunID            string  `json:"best_run_id"`
}

type commGridLLMAttempt struct {
	Attempt      int    `json:"attempt"`
	Success      bool   `json:"success"`
	Error        string `json:"error,omitempty"`
	BackoffMS    int    `json:"backoff_ms,omitempty"`
	Model        string `json:"model,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
	Tokens       int    `json:"tokens,omitempty"`
}

type commGridLLMRetryProvider struct {
	inner     llm.Provider
	retries   int
	backoffMS int
	attempts  [][]commGridLLMAttempt
}

func (p *commGridLLMRetryProvider) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	if p == nil || p.inner == nil {
		return llm.Response{}, errors.New("comm-grid llm provider is nil")
	}
	maxAttempts := p.retries + 1
	if maxAttempts <= 0 {
		maxAttempts = 1
	}
	attempts := make([]commGridLLMAttempt, 0, maxAttempts)
	var last llm.Response
	var lastErr error
	for i := 1; i <= maxAttempts; i++ {
		res, err := p.inner.Complete(ctx, req)
		attempt := commGridLLMAttempt{
			Attempt:      i,
			Success:      err == nil,
			Model:        res.Model,
			FinishReason: res.FinishReason,
			Tokens:       res.TokenCount(),
		}
		if err != nil {
			attempt.Error = strings.TrimSpace(err.Error())
		}
		if err != nil && i < maxAttempts {
			attempt.BackoffMS = p.backoffMS
		}
		attempts = append(attempts, attempt)
		last = res
		lastErr = err
		if err == nil {
			break
		}
		if i < maxAttempts && p.backoffMS > 0 {
			timer := time.NewTimer(time.Duration(p.backoffMS) * time.Millisecond)
			select {
			case <-ctx.Done():
				timer.Stop()
				attempts = append(attempts, commGridLLMAttempt{
					Attempt: i + 1,
					Success: false,
					Error:   ctx.Err().Error(),
				})
				p.attempts = append(p.attempts, attempts)
				return llm.Response{}, ctx.Err()
			case <-timer.C:
			}
		}
	}
	p.attempts = append(p.attempts, attempts)
	return last, lastErr
}

func (p *commGridLLMRetryProvider) PopAttempts() []commGridLLMAttempt {
	if p == nil || len(p.attempts) == 0 {
		return nil
	}
	attempts := p.attempts[0]
	p.attempts = p.attempts[1:]
	return attempts
}

type commGridLLMReplayProvider struct {
	steps []commGridLLMArtifactStep
	index int
}

func (p *commGridLLMReplayProvider) Complete(_ context.Context, _ llm.Request) (llm.Response, error) {
	if p == nil || len(p.steps) == 0 {
		return llm.Response{}, llm.ErrNoFixtureResponses
	}
	if p.index >= len(p.steps) {
		last := p.steps[len(p.steps)-1]
		return last.Response, commGridLLMReplayStepError(last)
	}
	step := p.steps[p.index]
	p.index++
	return step.Response, commGridLLMReplayStepError(step)
}

func runCommGridLLM(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("comm-grid-llm", flag.ContinueOnError)
	runID := fs.String("run-id", "", "artifact run id")
	replayRunID := fs.String("replay-run-id", "", "replay stored artifacts from benchmarks/<run-id>/comm_grid_llm.json")
	providerName := fs.String("provider", "fixture", "provider: fixture|openai-compatible")
	plan := fs.String("plan", "solve", "fixture plan: solve|tool|invalid")
	steps := fs.Int("steps", 8, "maximum fixture LLM decisions")
	width := fs.Int("width", 3, "comm-grid width")
	height := fs.Int("height", 3, "comm-grid height")
	key := fs.String("key", "1,0", "comm-grid key position as x,y")
	goal := fs.String("goal", "2,0", "comm-grid goal position as x,y")
	agentID := fs.String("agent", "agent-1", "comm-grid agent id")
	agentPos := fs.String("agent-pos", "0,0", "comm-grid agent start position as x,y")
	agents := fs.String("agents", "", "comm-grid agents as id@x,y:id@x,y")
	turnOrder := fs.String("turn-order", "", "comm-grid turn order as comma-separated agent ids")
	systemPrompt := fs.String("system-prompt", "", "default comm-grid LLM system prompt")
	agentRoles := fs.String("agent-roles", "", "comm-grid agent roles as id=role:id=role")
	agentPrompts := fs.String("agent-prompts", "", "comm-grid agent system prompts as id=prompt:id=prompt")
	messageLimit := fs.Int("message-limit", 80, "maximum stored message characters")
	baseURL := fs.String("base-url", "", "OpenAI-compatible base URL ending in /v1")
	apiKeyEnv := fs.String("api-key-env", "PROTOGONOS_LLM_API_KEY", "environment variable containing provider API key")
	model := fs.String("model", "", "provider model id")
	timeoutMS := fs.Int("timeout-ms", 30000, "provider request timeout in milliseconds")
	providerRetries := fs.Int("provider-retries", 0, "provider retries after a failed request")
	retryBackoffMS := fs.Int("retry-backoff-ms", 250, "provider retry backoff in milliseconds")
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
	if *providerRetries < 0 {
		return errors.New("provider-retries must be >= 0")
	}
	if *retryBackoffMS < 0 {
		return errors.New("retry-backoff-ms must be >= 0")
	}
	task, err := commGridLLMTaskFromFlags(commGridLLMTaskFlagValues{
		Width:        *width,
		Height:       *height,
		Key:          *key,
		Goal:         *goal,
		AgentID:      *agentID,
		Agent:        *agentPos,
		Agents:       *agents,
		TurnOrder:    *turnOrder,
		SystemPrompt: strings.TrimSpace(*systemPrompt),
		AgentRoles:   *agentRoles,
		AgentPrompts: *agentPrompts,
		MessageLimit: *messageLimit,
	})
	if err != nil {
		return err
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

	startedAt := time.Now().UTC()
	id := strings.TrimSpace(*runID)
	if id == "" {
		id = fmt.Sprintf("comm-grid-llm-%s-%d", summaryProvider, startedAt.UnixNano())
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
		Task:            task,
		ProviderRetries: *providerRetries,
		RetryBackoffMS:  *retryBackoffMS,
		MaxTokens:       *maxTokens,
		Temperature:     *temperature,
		Seed:            *seed,
		UseTools:        useTools,
	})
	if err != nil {
		return err
	}
	durationMS := commGridLLMDurationMillis(time.Since(startedAt))
	totalTokens := commGridLLMTotalTokens(artifactSteps)
	artifact := commGridLLMArtifact{
		RunID:                id,
		Provider:             summaryProvider,
		Plan:                 summaryPlan,
		CreatedAt:            startedAt.Format(time.RFC3339Nano),
		DurationMS:           durationMS,
		Task:                 task,
		Steps:                artifactSteps,
		Completed:            summary.Completed,
		Fitness:              summary.Fitness,
		TotalTokens:          totalTokens,
		AverageTokensPerStep: commGridLLMAverageTokensPerStep(totalTokens, len(artifactSteps)),
		Trace:                summary.Trace,
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
	fmt.Printf("comm_grid_llm run_id=%s provider=%s plan=%s grid=%dx%d key=(%d,%d) goal=(%d,%d) agents=%s turn_order=%s steps=%d completed=%t fitness=%.6f\n",
		summary.RunID,
		summary.Provider,
		summary.Plan,
		summary.Task.Width,
		summary.Task.Height,
		summary.Task.Key.X,
		summary.Task.Key.Y,
		summary.Task.Goal.X,
		summary.Task.Goal.Y,
		commGridLLMAgentSummary(summary.Task.Agents),
		strings.Join(summary.Task.TurnOrder, ","),
		len(summary.Steps),
		summary.Completed,
		summary.Fitness,
	)
	for _, step := range summary.Steps {
		fmt.Printf("step=%d actor=%s action=%s invalid=%t done=%t fitness=%.6f message=%q to=%q\n",
			step.Step,
			step.ActorID,
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

func runCommGridLLMRuns(ctx context.Context, args []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	fs := flag.NewFlagSet("comm-grid-llm-runs", flag.ContinueOnError)
	jsonOut := fs.Bool("json", false, "emit run index as JSON")
	csvOut := fs.Bool("csv", false, "emit run index as CSV")
	transcript := fs.Bool("transcript", false, "print latest matching transcript")
	compare := fs.Bool("compare", false, "compare indexed runs by task, provider, and plan")
	limit := fs.Int("limit", 0, "maximum rows to print, 0 means all")
	provider := fs.String("provider", "", "filter by provider")
	plan := fs.String("plan", "", "filter by plan")
	completed := fs.String("completed", "", "filter by completion status: true|false")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *limit < 0 {
		return errors.New("limit must be >= 0")
	}
	if *jsonOut && *csvOut {
		return errors.New("use only one output format: --json or --csv")
	}
	if *transcript && *csvOut {
		return errors.New("--csv cannot be used with --transcript")
	}
	completedFilter, filterCompleted, err := parseOptionalBoolFlag("completed", *completed)
	if err != nil {
		return err
	}
	entries, err := readCommGridLLMRunIndex(benchmarksDir)
	if err != nil {
		return err
	}
	entries = filterCommGridLLMRunIndex(entries, commGridLLMRunIndexFilter{
		Limit:           *limit,
		Provider:        *provider,
		Plan:            *plan,
		FilterCompleted: filterCompleted,
		Completed:       completedFilter,
	})
	if *transcript {
		return printLatestCommGridLLMTranscript(entries)
	}
	if *compare {
		comparisons := compareCommGridLLMRuns(entries)
		if *jsonOut {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(comparisons)
		}
		if *csvOut {
			return writeCommGridLLMRunComparisonCSV(os.Stdout, comparisons)
		}
		printCommGridLLMRunComparisonTable(comparisons)
		return nil
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(entries)
	}
	if *csvOut {
		return writeCommGridLLMRunIndexCSV(os.Stdout, entries)
	}
	printCommGridLLMRunIndexTable(entries)
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
	useTools := false
	model := "fixture-comm-grid"
	maxTokens := 64
	temperature := 0.0
	seed := int64(1)
	for _, step := range artifact.Steps {
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
	task := artifact.Task
	if task.Width == 0 || task.Height == 0 || strings.TrimSpace(task.AgentID) == "" {
		task = defaultCommGridLLMTaskConfig()
	}
	task = normalizeCommGridLLMTask(task)
	summary, _, err := executeCommGridLLM(ctx, commGridLLMExecutionConfig{
		RunID:           runID,
		Provider:        &commGridLLMReplayProvider{steps: artifact.Steps},
		ActorModel:      model,
		SummaryProvider: "fixture-replay",
		SummaryPlan:     strings.TrimSpace(artifact.Plan),
		MaxSteps:        maxSteps,
		Task:            task,
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
	Task            commGridLLMTaskConfig
	ProviderRetries int
	RetryBackoffMS  int
	MaxTokens       int
	Temperature     float64
	Seed            int64
	UseTools        bool
}

func executeCommGridLLM(ctx context.Context, cfg commGridLLMExecutionConfig) (commGridLLMCommandSummary, []commGridLLMArtifactStep, error) {
	task := cfg.Task
	if task.Width == 0 || task.Height == 0 || strings.TrimSpace(task.AgentID) == "" {
		task = defaultCommGridLLMTaskConfig()
	}
	task = normalizeCommGridLLMTask(task)
	sim := scape.NewCommGridSimulator(scape.CommGridConfig{
		Width:        task.Width,
		Height:       task.Height,
		MaxSteps:     cfg.MaxSteps,
		MessageLimit: task.MessageLimit,
		Key:          task.Key,
		Goal:         task.Goal,
		Agents:       commGridLLMScapeAgents(task),
	})
	provider := &commGridLLMRetryProvider{
		inner:     cfg.Provider,
		retries:   cfg.ProviderRetries,
		backoffMS: cfg.RetryBackoffMS,
	}

	summary := commGridLLMCommandSummary{
		RunID:    cfg.RunID,
		Provider: cfg.SummaryProvider,
		Plan:     cfg.SummaryPlan,
		Task:     task,
	}
	artifactSteps := make([]commGridLLMArtifactStep, 0, cfg.MaxSteps)
	for !sim.Done() && len(summary.Steps) < cfg.MaxSteps {
		actorID := task.TurnOrder[len(summary.Steps)%len(task.TurnOrder)]
		actor := scape.CommGridLLMActor{
			AgentID:       actorID,
			Provider:      provider,
			Model:         cfg.ActorModel,
			SystemPrompt:  commGridLLMSystemPromptForActor(task, actorID),
			MaxTokens:     cfg.MaxTokens,
			Temperature:   cfg.Temperature,
			Seed:          cfg.Seed,
			ResponseTools: cfg.UseTools,
		}
		result, trace, err := actor.Step(ctx, sim)
		if err != nil {
			result, trace = commGridLLMFailedStep(ctx, sim, actorID, len(summary.Steps)+1, trace, err)
		}
		attempts := provider.PopAttempts()
		state, _ := sim.AgentState(actorID)
		errText := ""
		errKind := ""
		if result.InvalidAction && trace.StepInput.Action == scape.CommGridAction("llm_failure") {
			errText = trace.StepInput.Message
			errKind = "llm_failure"
		}
		step := commGridLLMCommandStep{
			Step:          len(summary.Steps) + 1,
			ActorID:       actorID,
			Action:        string(trace.ParsedAction),
			Message:       trace.StepInput.Message,
			To:            trace.StepInput.To,
			Error:         errText,
			ErrorKind:     errKind,
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
				"attempts":      len(attempts),
			},
			Attempts: attempts,
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
			Step:      step.Step,
			Request:   trace.Request,
			Response:  trace.Response,
			Attempts:  attempts,
			Payload:   trace.Payload,
			Parsed:    trace.StepInput,
			Error:     errText,
			ErrorKind: errKind,
			Result:    step,
		})
	}
	summary.Completed = sim.Done() && sim.Trace()["completed"] == true
	summary.Fitness = float64(sim.Fitness())
	summary.Trace = map[string]any(sim.Trace())
	return summary, artifactSteps, nil
}

func commGridLLMReplayStepError(step commGridLLMArtifactStep) error {
	if step.ErrorKind == "" || commGridLLMResponseHasPayload(step.Response) {
		return nil
	}
	message := strings.TrimSpace(step.Error)
	message = strings.TrimPrefix(message, "llm failure: ")
	if message == "" {
		message = "stored llm failure"
	}
	return errors.New(message)
}

func commGridLLMResponseHasPayload(res llm.Response) bool {
	return strings.TrimSpace(res.Message) != "" || len(res.ToolCalls) > 0
}

func commGridLLMFailedStep(ctx context.Context, sim *scape.CommGridSimulator, agentID string, step int, trace scape.CommGridLLMStepTrace, cause error) (scape.CommGridStepResult, scape.CommGridLLMStepTrace) {
	message := commGridLLMFailureMessage(cause)
	input := scape.CommGridStepInput{
		AgentID: agentID,
		Action:  scape.CommGridAction("llm_failure"),
		Message: message,
		To:      "system",
	}
	result, err := sim.Step(ctx, input)
	if err != nil {
		result = scape.CommGridStepResult{
			Done:          sim.Done(),
			Completed:     false,
			InvalidAction: true,
			Fitness:       sim.Fitness(),
			Trace:         sim.Trace(),
		}
		result.Trace["llm_failure_step_error"] = err.Error()
	}
	trace.StepInput = input
	trace.ParsedAction = input.Action
	if trace.AgentID == "" {
		trace.AgentID = agentID
	}
	if trace.Payload == "" {
		trace.Payload = message
	}
	if result.Trace == nil {
		result.Trace = map[string]any{}
	}
	result.Trace["llm_failure"] = true
	result.Trace["llm_failure_step"] = step
	result.Trace["llm_failure_error"] = message
	result.Trace["llm_action"] = string(input.Action)
	return result, trace
}

type commGridLLMTaskFlagValues struct {
	Width        int
	Height       int
	Key          string
	Goal         string
	SystemPrompt string
	AgentID      string
	Agent        string
	Agents       string
	TurnOrder    string
	AgentRoles   string
	AgentPrompts string
	MessageLimit int
}

func commGridLLMTaskFromFlags(flags commGridLLMTaskFlagValues) (commGridLLMTaskConfig, error) {
	if flags.Width <= 0 {
		return commGridLLMTaskConfig{}, errors.New("width must be > 0")
	}
	if flags.Height <= 0 {
		return commGridLLMTaskConfig{}, errors.New("height must be > 0")
	}
	if flags.MessageLimit <= 0 {
		return commGridLLMTaskConfig{}, errors.New("message-limit must be > 0")
	}
	agentID := strings.TrimSpace(flags.AgentID)
	if agentID == "" {
		return commGridLLMTaskConfig{}, errors.New("agent must not be empty")
	}
	key, err := parseCommGridLLMPoint("key", flags.Key)
	if err != nil {
		return commGridLLMTaskConfig{}, err
	}
	goal, err := parseCommGridLLMPoint("goal", flags.Goal)
	if err != nil {
		return commGridLLMTaskConfig{}, err
	}
	agent, err := parseCommGridLLMPoint("agent-pos", flags.Agent)
	if err != nil {
		return commGridLLMTaskConfig{}, err
	}
	agents := []commGridLLMAgent{{
		ID:       agentID,
		Position: agent,
	}}
	if strings.TrimSpace(flags.Agents) != "" {
		agents, err = parseCommGridLLMAgents(flags.Agents)
		if err != nil {
			return commGridLLMTaskConfig{}, err
		}
		agentID = agents[0].ID
		agent = agents[0].Position
	}
	if err := applyCommGridLLMAgentSettings(agents, flags.AgentRoles, flags.AgentPrompts); err != nil {
		return commGridLLMTaskConfig{}, err
	}
	if !commGridLLMPointInBounds(key, flags.Width, flags.Height) {
		return commGridLLMTaskConfig{}, fmt.Errorf("key out of bounds: %d,%d", key.X, key.Y)
	}
	if !commGridLLMPointInBounds(goal, flags.Width, flags.Height) {
		return commGridLLMTaskConfig{}, fmt.Errorf("goal out of bounds: %d,%d", goal.X, goal.Y)
	}
	if !commGridLLMPointInBounds(agent, flags.Width, flags.Height) {
		return commGridLLMTaskConfig{}, fmt.Errorf("agent-pos out of bounds: %d,%d", agent.X, agent.Y)
	}
	for _, configured := range agents {
		if !commGridLLMPointInBounds(configured.Position, flags.Width, flags.Height) {
			return commGridLLMTaskConfig{}, fmt.Errorf("agent out of bounds: %s@%d,%d", configured.ID, configured.Position.X, configured.Position.Y)
		}
	}
	order, err := commGridLLMTurnOrder(flags.TurnOrder, agents)
	if err != nil {
		return commGridLLMTaskConfig{}, err
	}
	return commGridLLMTaskConfig{
		Width:        flags.Width,
		Height:       flags.Height,
		Key:          key,
		Goal:         goal,
		SystemPrompt: strings.TrimSpace(flags.SystemPrompt),
		AgentID:      agentID,
		Agent:        agent,
		Agents:       agents,
		TurnOrder:    order,
		MessageLimit: flags.MessageLimit,
	}, nil
}

func defaultCommGridLLMTaskConfig() commGridLLMTaskConfig {
	return commGridLLMTaskConfig{
		Width:   3,
		Height:  3,
		Key:     scape.CommGridPoint{X: 1, Y: 0},
		Goal:    scape.CommGridPoint{X: 2, Y: 0},
		AgentID: "agent-1",
		Agent:   scape.CommGridPoint{},
		Agents: []commGridLLMAgent{{
			ID:       "agent-1",
			Position: scape.CommGridPoint{},
		}},
		TurnOrder:    []string{"agent-1"},
		MessageLimit: 80,
	}
}

func normalizeCommGridLLMTask(task commGridLLMTaskConfig) commGridLLMTaskConfig {
	task.SystemPrompt = strings.TrimSpace(task.SystemPrompt)
	task.AgentID = strings.TrimSpace(task.AgentID)
	if len(task.Agents) == 0 {
		task.Agents = []commGridLLMAgent{{
			ID:       task.AgentID,
			Position: task.Agent,
		}}
	}
	for i := range task.Agents {
		task.Agents[i].ID = strings.TrimSpace(task.Agents[i].ID)
		task.Agents[i].Role = strings.TrimSpace(task.Agents[i].Role)
		task.Agents[i].SystemPrompt = strings.TrimSpace(task.Agents[i].SystemPrompt)
	}
	if task.AgentID == "" && len(task.Agents) > 0 {
		task.AgentID = task.Agents[0].ID
		task.Agent = task.Agents[0].Position
	}
	if len(task.TurnOrder) == 0 {
		task.TurnOrder = make([]string, 0, len(task.Agents))
		for _, agent := range task.Agents {
			if agent.ID != "" {
				task.TurnOrder = append(task.TurnOrder, agent.ID)
			}
		}
	}
	if len(task.TurnOrder) == 0 && task.AgentID != "" {
		task.TurnOrder = []string{task.AgentID}
	}
	return task
}

func applyCommGridLLMAgentSettings(agents []commGridLLMAgent, rawRoles, rawPrompts string) error {
	roleByAgent, err := parseCommGridLLMAgentTextMap("agent-roles", rawRoles)
	if err != nil {
		return err
	}
	promptByAgent, err := parseCommGridLLMAgentTextMap("agent-prompts", rawPrompts)
	if err != nil {
		return err
	}
	known := map[string]int{}
	for i, agent := range agents {
		known[agent.ID] = i
	}
	for id, role := range roleByAgent {
		index, ok := known[id]
		if !ok {
			return fmt.Errorf("agent-roles references unknown agent: %s", id)
		}
		agents[index].Role = role
	}
	for id, prompt := range promptByAgent {
		index, ok := known[id]
		if !ok {
			return fmt.Errorf("agent-prompts references unknown agent: %s", id)
		}
		agents[index].SystemPrompt = prompt
	}
	return nil
}

func parseCommGridLLMAgentTextMap(name, raw string) (map[string]string, error) {
	out := map[string]string{}
	if strings.TrimSpace(raw) == "" {
		return out, nil
	}
	for _, part := range strings.Split(raw, ":") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, value, ok := strings.Cut(part, "=")
		if !ok {
			return nil, fmt.Errorf("%s entries must use id=value: %s", name, part)
		}
		id = strings.TrimSpace(id)
		value = strings.TrimSpace(value)
		if id == "" || value == "" {
			return nil, fmt.Errorf("%s entries must use non-empty id and value: %s", name, part)
		}
		if _, exists := out[id]; exists {
			return nil, fmt.Errorf("%s has duplicate agent id: %s", name, id)
		}
		out[id] = value
	}
	return out, nil
}

func commGridLLMSystemPromptForActor(task commGridLLMTaskConfig, actorID string) string {
	base := strings.TrimSpace(task.SystemPrompt)
	var role string
	for _, agent := range task.Agents {
		if agent.ID != actorID {
			continue
		}
		if agent.SystemPrompt != "" {
			return agent.SystemPrompt
		}
		role = agent.Role
		break
	}
	if role == "" {
		return base
	}
	if base == "" {
		base = "Return one compact JSON object with fields action, message, to, and tokens. Use only these actions: stay, north, south, east, west, pick, drop."
	}
	return base + "\nRole: " + role
}

func parseCommGridLLMAgents(raw string) ([]commGridLLMAgent, error) {
	parts := strings.Split(strings.TrimSpace(raw), ":")
	agents := make([]commGridLLMAgent, 0, len(parts))
	seen := map[string]bool{}
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, pointText, ok := strings.Cut(part, "@")
		if !ok {
			return nil, fmt.Errorf("agent entry must use id@x,y: %s", part)
		}
		id = strings.TrimSpace(id)
		if id == "" {
			return nil, fmt.Errorf("agent entry must use non-empty id: %s", part)
		}
		if seen[id] {
			return nil, fmt.Errorf("duplicate comm-grid agent id: %s", id)
		}
		point, err := parseCommGridLLMPoint("agent", pointText)
		if err != nil {
			return nil, err
		}
		seen[id] = true
		agents = append(agents, commGridLLMAgent{ID: id, Position: point})
	}
	if len(agents) == 0 {
		return nil, errors.New("agents must include at least one id@x,y entry")
	}
	return agents, nil
}

func commGridLLMTurnOrder(raw string, agents []commGridLLMAgent) ([]string, error) {
	known := map[string]bool{}
	for _, agent := range agents {
		known[agent.ID] = true
	}
	if strings.TrimSpace(raw) == "" {
		order := make([]string, 0, len(agents))
		for _, agent := range agents {
			order = append(order, agent.ID)
		}
		return order, nil
	}
	parts := strings.Split(raw, ",")
	order := make([]string, 0, len(parts))
	for _, part := range parts {
		id := strings.TrimSpace(part)
		if id == "" {
			continue
		}
		if !known[id] {
			return nil, fmt.Errorf("turn-order references unknown agent: %s", id)
		}
		order = append(order, id)
	}
	if len(order) == 0 {
		return nil, errors.New("turn-order must include at least one agent id")
	}
	return order, nil
}

func commGridLLMScapeAgents(task commGridLLMTaskConfig) []scape.CommGridAgentState {
	if len(task.Agents) == 0 {
		return []scape.CommGridAgentState{{
			ID:       task.AgentID,
			Position: task.Agent,
		}}
	}
	agents := make([]scape.CommGridAgentState, 0, len(task.Agents))
	for _, agent := range task.Agents {
		agents = append(agents, scape.CommGridAgentState{
			ID:       agent.ID,
			Position: agent.Position,
		})
	}
	return agents
}

func commGridLLMAgentSummary(agents []commGridLLMAgent) string {
	if len(agents) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(agents))
	for _, agent := range agents {
		parts = append(parts, fmt.Sprintf("%s@(%d,%d)", agent.ID, agent.Position.X, agent.Position.Y))
	}
	return strings.Join(parts, ",")
}

func parseCommGridLLMPoint(name, raw string) (scape.CommGridPoint, error) {
	parts := strings.Split(strings.TrimSpace(raw), ",")
	if len(parts) != 2 {
		return scape.CommGridPoint{}, fmt.Errorf("%s must use x,y", name)
	}
	x, err := parseCommGridLLMNonNegativeInt(name+".x", parts[0])
	if err != nil {
		return scape.CommGridPoint{}, err
	}
	y, err := parseCommGridLLMNonNegativeInt(name+".y", parts[1])
	if err != nil {
		return scape.CommGridPoint{}, err
	}
	return scape.CommGridPoint{X: x, Y: y}, nil
}

func parseCommGridLLMNonNegativeInt(name, raw string) (int, error) {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return 0, fmt.Errorf("%s must be a non-negative integer: %w", name, err)
	}
	if value < 0 {
		return 0, fmt.Errorf("%s must be a non-negative integer", name)
	}
	return value, nil
}

func commGridLLMPointInBounds(point scape.CommGridPoint, width, height int) bool {
	return point.X >= 0 && point.Y >= 0 && point.X < width && point.Y < height
}

func commGridLLMFailureMessage(err error) string {
	if err == nil {
		return "llm failure"
	}
	text := strings.TrimSpace(err.Error())
	if text == "" {
		return "llm failure"
	}
	if len(text) > 120 {
		text = text[:120]
	}
	return "llm failure: " + text
}

func commGridLLMDurationMillis(duration time.Duration) int64 {
	if duration <= 0 {
		return 0
	}
	ms := duration.Milliseconds()
	if ms == 0 {
		return 1
	}
	return ms
}

func commGridLLMTotalTokens(steps []commGridLLMArtifactStep) int {
	var total int
	for _, step := range steps {
		total += step.Response.TokenCount()
	}
	return total
}

func commGridLLMAverageTokensPerStep(totalTokens, steps int) float64 {
	if steps <= 0 {
		return 0
	}
	return float64(totalTokens) / float64(steps)
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
	transcriptPath := filepath.Join(runDir, "comm_grid_llm_transcript.md")
	transcript := renderCommGridLLMTranscript(artifact)
	if err := os.WriteFile(transcriptPath, []byte(transcript), 0o644); err != nil {
		return "", err
	}
	if err := appendCommGridLLMRunIndex(baseDir, artifact, path, transcriptPath); err != nil {
		return "", err
	}
	return filepath.Clean(runDir), nil
}

func appendCommGridLLMRunIndex(baseDir string, artifact commGridLLMArtifact, artifactPath, transcriptPath string) error {
	entry := commGridLLMRunIndexEntry{
		RunID:                artifact.RunID,
		Provider:             artifact.Provider,
		Plan:                 artifact.Plan,
		CreatedAt:            artifact.CreatedAt,
		DurationMS:           artifact.DurationMS,
		Task:                 normalizeCommGridLLMTask(artifact.Task),
		Steps:                len(artifact.Steps),
		Completed:            artifact.Completed,
		Fitness:              artifact.Fitness,
		TotalTokens:          artifact.TotalTokens,
		AverageTokensPerStep: artifact.AverageTokensPerStep,
		FailureCount:         commGridLLMFailureCount(artifact.Steps),
		RetryCount:           commGridLLMRetryCount(artifact.Steps),
		ArtifactPath:         filepath.ToSlash(filepath.Clean(artifactPath)),
		TranscriptPath:       filepath.ToSlash(filepath.Clean(transcriptPath)),
	}
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	indexPath := filepath.Join(baseDir, "comm_grid_llm_runs.jsonl")
	file, err := os.OpenFile(indexPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	if _, err := file.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}

func readCommGridLLMRunIndex(baseDir string) ([]commGridLLMRunIndexEntry, error) {
	path := filepath.Join(baseDir, "comm_grid_llm_runs.jsonl")
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer file.Close()

	var entries []commGridLLMRunIndexEntry
	scanner := bufio.NewScanner(file)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var entry commGridLLMRunIndexEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			return nil, fmt.Errorf("decode comm-grid llm run index line %d: %w", lineNumber, err)
		}
		entry.Task = normalizeCommGridLLMTask(entry.Task)
		entries = append(entries, entry)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return entries, nil
}

func filterCommGridLLMRunIndex(entries []commGridLLMRunIndexEntry, filter commGridLLMRunIndexFilter) []commGridLLMRunIndexEntry {
	provider := strings.TrimSpace(strings.ToLower(filter.Provider))
	plan := strings.TrimSpace(strings.ToLower(filter.Plan))
	out := make([]commGridLLMRunIndexEntry, 0, len(entries))
	for _, entry := range entries {
		if provider != "" && strings.ToLower(entry.Provider) != provider {
			continue
		}
		if plan != "" && strings.ToLower(entry.Plan) != plan {
			continue
		}
		if filter.FilterCompleted && entry.Completed != filter.Completed {
			continue
		}
		out = append(out, entry)
	}
	if filter.Limit > 0 && len(out) > filter.Limit {
		out = out[len(out)-filter.Limit:]
	}
	return out
}

func printCommGridLLMRunIndexTable(entries []commGridLLMRunIndexEntry) {
	fmt.Printf("RUN_ID\tPROVIDER\tPLAN\tSTEPS\tDONE\tFITNESS\tTOKENS\tAVG_TOK\tMS\tFAIL\tRETRY\tARTIFACT\n")
	for _, entry := range entries {
		fmt.Printf("%s\t%s\t%s\t%d\t%t\t%.6f\t%d\t%.3f\t%d\t%d\t%d\t%s\n",
			entry.RunID,
			entry.Provider,
			entry.Plan,
			entry.Steps,
			entry.Completed,
			entry.Fitness,
			entry.TotalTokens,
			entry.AverageTokensPerStep,
			entry.DurationMS,
			entry.FailureCount,
			entry.RetryCount,
			entry.ArtifactPath,
		)
	}
}

func writeCommGridLLMRunIndexCSV(file *os.File, entries []commGridLLMRunIndexEntry) error {
	writer := csv.NewWriter(file)
	if err := writer.Write([]string{
		"run_id",
		"provider",
		"plan",
		"created_at_utc",
		"steps",
		"completed",
		"fitness",
		"total_tokens",
		"average_tokens_per_step",
		"duration_ms",
		"failure_count",
		"retry_count",
		"artifact_path",
		"transcript_path",
	}); err != nil {
		return err
	}
	for _, entry := range entries {
		if err := writer.Write([]string{
			entry.RunID,
			entry.Provider,
			entry.Plan,
			entry.CreatedAt,
			strconv.Itoa(entry.Steps),
			strconv.FormatBool(entry.Completed),
			strconv.FormatFloat(entry.Fitness, 'f', 6, 64),
			strconv.Itoa(entry.TotalTokens),
			strconv.FormatFloat(entry.AverageTokensPerStep, 'f', 3, 64),
			strconv.FormatInt(entry.DurationMS, 10),
			strconv.Itoa(entry.FailureCount),
			strconv.Itoa(entry.RetryCount),
			entry.ArtifactPath,
			entry.TranscriptPath,
		}); err != nil {
			return err
		}
	}
	writer.Flush()
	return writer.Error()
}

func compareCommGridLLMRuns(entries []commGridLLMRunIndexEntry) []commGridLLMRunComparison {
	type accumulator struct {
		comparison      commGridLLMRunComparison
		totalFitness    float64
		totalTokens     int
		totalTokenSteps float64
		totalDurationMS int64
		totalFailures   int
		totalRetries    int
	}
	groups := map[string]*accumulator{}
	for _, entry := range entries {
		taskShape := commGridLLMTaskShape(entry.Task)
		key := entry.Provider + "\x00" + entry.Plan + "\x00" + taskShape
		acc, ok := groups[key]
		if !ok {
			acc = &accumulator{
				comparison: commGridLLMRunComparison{
					Group:       entry.Provider + "/" + entry.Plan + "/" + taskShape,
					Provider:    entry.Provider,
					Plan:        entry.Plan,
					TaskShape:   taskShape,
					BestFitness: entry.Fitness,
					BestRunID:   entry.RunID,
				},
			}
			groups[key] = acc
		}
		acc.comparison.Runs++
		if entry.Completed {
			acc.comparison.Completed++
		}
		if entry.Fitness > acc.comparison.BestFitness || acc.comparison.Runs == 1 {
			acc.comparison.BestFitness = entry.Fitness
			acc.comparison.BestRunID = entry.RunID
		}
		acc.totalFitness += entry.Fitness
		acc.totalTokens += entry.TotalTokens
		acc.totalTokenSteps += entry.AverageTokensPerStep
		acc.totalDurationMS += entry.DurationMS
		acc.totalFailures += entry.FailureCount
		acc.totalRetries += entry.RetryCount
	}
	out := make([]commGridLLMRunComparison, 0, len(groups))
	for _, acc := range groups {
		runs := float64(acc.comparison.Runs)
		if runs > 0 {
			acc.comparison.CompletionRate = float64(acc.comparison.Completed) / runs
			acc.comparison.AverageFitness = acc.totalFitness / runs
			acc.comparison.AverageTokensPerRun = float64(acc.totalTokens) / runs
			acc.comparison.AverageTokensPerStep = acc.totalTokenSteps / runs
			acc.comparison.AverageDurationMS = float64(acc.totalDurationMS) / runs
			acc.comparison.AverageFailures = float64(acc.totalFailures) / runs
			acc.comparison.AverageRetries = float64(acc.totalRetries) / runs
		}
		out = append(out, acc.comparison)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].TaskShape != out[j].TaskShape {
			return out[i].TaskShape < out[j].TaskShape
		}
		if out[i].Provider != out[j].Provider {
			return out[i].Provider < out[j].Provider
		}
		return out[i].Plan < out[j].Plan
	})
	return out
}

func printCommGridLLMRunComparisonTable(comparisons []commGridLLMRunComparison) {
	fmt.Printf("GROUP\tRUNS\tDONE_RATE\tBEST\tAVG_FIT\tAVG_TOK_RUN\tAVG_TOK_STEP\tAVG_MS\tAVG_FAIL\tAVG_RETRY\tBEST_RUN\n")
	for _, comparison := range comparisons {
		fmt.Printf("%s\t%d\t%.3f\t%.6f\t%.6f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\n",
			comparison.Group,
			comparison.Runs,
			comparison.CompletionRate,
			comparison.BestFitness,
			comparison.AverageFitness,
			comparison.AverageTokensPerRun,
			comparison.AverageTokensPerStep,
			comparison.AverageDurationMS,
			comparison.AverageFailures,
			comparison.AverageRetries,
			comparison.BestRunID,
		)
	}
}

func writeCommGridLLMRunComparisonCSV(file *os.File, comparisons []commGridLLMRunComparison) error {
	writer := csv.NewWriter(file)
	if err := writer.Write([]string{
		"group",
		"provider",
		"plan",
		"task_shape",
		"runs",
		"completed",
		"completion_rate",
		"best_fitness",
		"average_fitness",
		"average_tokens_per_run",
		"average_tokens_per_step",
		"average_duration_ms",
		"average_failures",
		"average_retries",
		"best_run_id",
	}); err != nil {
		return err
	}
	for _, comparison := range comparisons {
		if err := writer.Write([]string{
			comparison.Group,
			comparison.Provider,
			comparison.Plan,
			comparison.TaskShape,
			strconv.Itoa(comparison.Runs),
			strconv.Itoa(comparison.Completed),
			strconv.FormatFloat(comparison.CompletionRate, 'f', 3, 64),
			strconv.FormatFloat(comparison.BestFitness, 'f', 6, 64),
			strconv.FormatFloat(comparison.AverageFitness, 'f', 6, 64),
			strconv.FormatFloat(comparison.AverageTokensPerRun, 'f', 3, 64),
			strconv.FormatFloat(comparison.AverageTokensPerStep, 'f', 3, 64),
			strconv.FormatFloat(comparison.AverageDurationMS, 'f', 3, 64),
			strconv.FormatFloat(comparison.AverageFailures, 'f', 3, 64),
			strconv.FormatFloat(comparison.AverageRetries, 'f', 3, 64),
			comparison.BestRunID,
		}); err != nil {
			return err
		}
	}
	writer.Flush()
	return writer.Error()
}

func commGridLLMTaskShape(task commGridLLMTaskConfig) string {
	task = normalizeCommGridLLMTask(task)
	return fmt.Sprintf("%dx%d:key(%d,%d):goal(%d,%d):agents%d:turns[%s]:limit%d",
		task.Width,
		task.Height,
		task.Key.X,
		task.Key.Y,
		task.Goal.X,
		task.Goal.Y,
		len(task.Agents),
		strings.Join(task.TurnOrder, ","),
		task.MessageLimit,
	)
}

func printLatestCommGridLLMTranscript(entries []commGridLLMRunIndexEntry) error {
	if len(entries) == 0 {
		return errors.New("no comm-grid llm runs match filters")
	}
	entry := entries[len(entries)-1]
	path, err := validateCommGridLLMTranscriptPath(entry.TranscriptPath)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	_, err = os.Stdout.Write(data)
	return err
}

func validateCommGridLLMTranscriptPath(raw string) (string, error) {
	path := filepath.Clean(strings.TrimSpace(raw))
	if path == "" || path == "." {
		return "", errors.New("comm-grid llm transcript path missing")
	}
	parts := strings.Split(filepath.ToSlash(path), "/")
	if len(parts) != 3 || parts[0] != benchmarksDir || parts[2] != "comm_grid_llm_transcript.md" {
		return "", fmt.Errorf("invalid comm-grid llm transcript path: %s", raw)
	}
	if err := validateCommGridLLMRunID(parts[1]); err != nil {
		return "", err
	}
	return filepath.Join(parts[0], parts[1], parts[2]), nil
}

func parseOptionalBoolFlag(name, raw string) (bool, bool, error) {
	value := strings.TrimSpace(strings.ToLower(raw))
	if value == "" {
		return false, false, nil
	}
	switch value {
	case "true", "t", "1", "yes", "y":
		return true, true, nil
	case "false", "f", "0", "no", "n":
		return false, true, nil
	default:
		return false, false, fmt.Errorf("%s must be true or false", name)
	}
}

func commGridLLMFailureCount(steps []commGridLLMArtifactStep) int {
	count := 0
	for _, step := range steps {
		if step.ErrorKind != "" || step.Result.ErrorKind != "" {
			count++
		}
	}
	return count
}

func commGridLLMRetryCount(steps []commGridLLMArtifactStep) int {
	count := 0
	for _, step := range steps {
		if len(step.Attempts) > 1 {
			count += len(step.Attempts) - 1
		}
	}
	return count
}

func renderCommGridLLMTranscript(artifact commGridLLMArtifact) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Comm Grid LLM Transcript\n\n")
	fmt.Fprintf(&b, "- run_id: `%s`\n", artifact.RunID)
	fmt.Fprintf(&b, "- provider: `%s`\n", artifact.Provider)
	fmt.Fprintf(&b, "- plan: `%s`\n", artifact.Plan)
	fmt.Fprintf(&b, "- completed: `%t`\n", artifact.Completed)
	fmt.Fprintf(&b, "- fitness: `%.6f`\n", artifact.Fitness)
	fmt.Fprintf(&b, "- replay: `protogonosctl comm-grid-llm --replay-run-id %s`\n\n", artifact.RunID)

	task := normalizeCommGridLLMTask(artifact.Task)
	fmt.Fprintf(&b, "## Task\n\n")
	fmt.Fprintf(&b, "- grid: `%dx%d`\n", task.Width, task.Height)
	fmt.Fprintf(&b, "- key: `(%d,%d)`\n", task.Key.X, task.Key.Y)
	fmt.Fprintf(&b, "- goal: `(%d,%d)`\n", task.Goal.X, task.Goal.Y)
	fmt.Fprintf(&b, "- agents: `%s`\n", commGridLLMAgentSummary(task.Agents))
	fmt.Fprintf(&b, "- turn_order: `%s`\n", strings.Join(task.TurnOrder, ","))
	fmt.Fprintf(&b, "- message_limit: `%d`\n\n", task.MessageLimit)

	for _, step := range artifact.Steps {
		renderCommGridLLMTranscriptStep(&b, step)
	}

	fmt.Fprintf(&b, "## Final Trace\n\n")
	fmt.Fprintf(&b, "```text\n%v\n```\n", artifact.Trace)
	return b.String()
}

func renderCommGridLLMTranscriptStep(b *strings.Builder, step commGridLLMArtifactStep) {
	fmt.Fprintf(b, "## Step %d: %s\n\n", step.Step, step.Result.ActorID)
	fmt.Fprintf(b, "- action: `%s`\n", step.Result.Action)
	fmt.Fprintf(b, "- invalid: `%t`\n", step.Result.InvalidAction)
	fmt.Fprintf(b, "- done: `%t`\n", step.Result.Done)
	fmt.Fprintf(b, "- fitness: `%.6f`\n", step.Result.Fitness)
	if step.ErrorKind != "" {
		fmt.Fprintf(b, "- error_kind: `%s`\n", step.ErrorKind)
		fmt.Fprintf(b, "- error: `%s`\n", step.Error)
	}
	if len(step.Attempts) > 0 {
		fmt.Fprintf(b, "- attempts: `%d`\n", len(step.Attempts))
	}
	fmt.Fprintf(b, "- message: `%s`\n", step.Result.Message)
	fmt.Fprintf(b, "- to: `%s`\n\n", step.Result.To)

	fmt.Fprintf(b, "### System Prompt\n\n")
	fmt.Fprintf(b, "```text\n%s\n```\n\n", step.Request.SystemPrompt)
	fmt.Fprintf(b, "### User Prompt\n\n")
	userPrompt := ""
	if len(step.Request.Messages) > 0 {
		userPrompt = step.Request.Messages[len(step.Request.Messages)-1].Content
	}
	fmt.Fprintf(b, "```text\n%s\n```\n\n", userPrompt)
	fmt.Fprintf(b, "### Response Payload\n\n")
	fmt.Fprintf(b, "```json\n%s\n```\n\n", step.Payload)
	fmt.Fprintf(b, "### Parsed Action\n\n")
	fmt.Fprintf(b, "```text\nagent=%s action=%s message=%q to=%q tokens=%d\n```\n\n",
		step.Parsed.AgentID,
		step.Parsed.Action,
		step.Parsed.Message,
		step.Parsed.To,
		step.Parsed.Tokens,
	)
	if len(step.Attempts) > 0 {
		fmt.Fprintf(b, "### Provider Attempts\n\n")
		for _, attempt := range step.Attempts {
			fmt.Fprintf(b, "- attempt `%d`: success=`%t` tokens=`%d` finish_reason=`%s` error=`%s` backoff_ms=`%d`\n",
				attempt.Attempt,
				attempt.Success,
				attempt.Tokens,
				attempt.FinishReason,
				attempt.Error,
				attempt.BackoffMS,
			)
		}
		fmt.Fprintf(b, "\n")
	}
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
		if strings.TrimSpace(strings.ToLower(flags.Plan)) == "provider-error" {
			return llm.NewFixtureProviderWithError(errors.New("fixture provider failure")), "fixture-comm-grid", false, "fixture", "provider-error", nil
		}
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
	case "malformed":
		return []llm.Response{
			commGridLLMFixtureContent(`{"action":"teleport","message":"bad action","to":"all","tokens":2}`, 6),
		}, false, nil
	case "multi-solve":
		return []llm.Response{
			commGridLLMFixtureContent(`{"action":"east","message":"a moves to key","to":"all","tokens":4}`, 8),
			commGridLLMFixtureContent(`{"action":"stay","message":"b waits","to":"all","tokens":2}`, 6),
			commGridLLMFixtureContent(`{"action":"pick","message":"a picked key","to":"all","tokens":3}`, 7),
			commGridLLMFixtureContent(`{"action":"stay","message":"b still waits","to":"all","tokens":3}`, 6),
			commGridLLMFixtureContent(`{"action":"east","message":"a moves to goal","to":"all","tokens":4}`, 8),
			commGridLLMFixtureContent(`{"action":"stay","message":"b guards","to":"all","tokens":2}`, 6),
			commGridLLMFixtureContent(`{"action":"drop","message":"a delivered key","to":"all","tokens":3}`, 7),
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
