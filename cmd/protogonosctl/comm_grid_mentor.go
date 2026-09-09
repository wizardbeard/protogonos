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

type commGridMentorCommandSummary struct {
	RunID        string                          `json:"run_id"`
	ArtifactsDir string                          `json:"artifacts_dir,omitempty"`
	Replay       *commGridLLMReplayResult        `json:"replay,omitempty"`
	Provider     string                          `json:"provider"`
	Plan         string                          `json:"plan"`
	Task         commGridLLMTaskConfig           `json:"task"`
	Steps        []scape.CommGridMentorStepTrace `json:"steps"`
	Completed    bool                            `json:"completed"`
	Fitness      float64                         `json:"fitness"`
	Trace        map[string]any                  `json:"trace"`
}

type commGridMentorArtifact struct {
	RunID       string                          `json:"run_id"`
	Provider    string                          `json:"provider"`
	Plan        string                          `json:"plan"`
	CreatedAt   string                          `json:"created_at_utc"`
	DurationMS  int64                           `json:"duration_ms"`
	Task        commGridLLMTaskConfig           `json:"task"`
	Steps       []scape.CommGridMentorStepTrace `json:"steps"`
	Completed   bool                            `json:"completed"`
	Fitness     float64                         `json:"fitness"`
	TotalTokens int                             `json:"total_tokens"`
	Trace       map[string]any                  `json:"trace"`
}

type commGridMentorReplayProvider struct {
	steps []scape.CommGridMentorStepTrace
	index int
}

type commGridMentorPolicyAgent struct {
	id string
}

func (a commGridMentorPolicyAgent) ID() string { return a.id }

func (a commGridMentorPolicyAgent) RunStep(_ context.Context, input []float64) ([]float64, error) {
	if len(input) >= 8 && input[7] > 0 {
		return []float64{input[4], input[5], input[6]}, nil
	}
	if len(input) < 3 {
		return []float64{0, 0, 0}, nil
	}
	if input[2] > 0.5 {
		return []float64{0, 0, -1}, nil
	}
	if input[0] == 0 && input[1] == 0 {
		return []float64{0, 0, 1}, nil
	}
	return []float64{input[0], input[1], 0}, nil
}

func (p *commGridMentorReplayProvider) Complete(_ context.Context, _ llm.Request) (llm.Response, error) {
	if p == nil || len(p.steps) == 0 {
		return llm.Response{}, llm.ErrNoFixtureResponses
	}
	if p.index >= len(p.steps) {
		last := p.steps[len(p.steps)-1]
		return last.Response, commGridMentorReplayStepError(last)
	}
	step := p.steps[p.index]
	p.index++
	return step.Response, commGridMentorReplayStepError(step)
}

func runCommGridMentor(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("comm-grid-mentor", flag.ContinueOnError)
	runID := fs.String("run-id", "", "run id; generated when empty")
	replayRunID := fs.String("replay-run-id", "", "replay stored artifact from benchmarks/<run-id>/comm_grid_mentor.json")
	plan := fs.String("plan", "solve", "fixture mentor plan: solve|silent|provider-error")
	jsonOut := fs.Bool("json", false, "print JSON summary")
	writeArtifacts := fs.Bool("artifacts", true, "write comm-grid mentor artifact under benchmarks/<run-id>")
	steps := fs.Int("steps", 8, "maximum learner turns")
	width := fs.Int("width", 3, "grid width")
	height := fs.Int("height", 3, "grid height")
	keyRaw := fs.String("key", "1,0", "key point as x,y")
	goalRaw := fs.String("goal", "2,0", "goal point as x,y")
	agentID := fs.String("agent", "agent-1", "learner agent id")
	agentPosRaw := fs.String("agent-pos", "0,0", "learner start point as x,y")
	model := fs.String("model", "fixture-mentor", "fixture model name")
	maxTokens := fs.Int("max-tokens", 16, "mentor max tokens")
	seed := fs.Int64("seed", 1, "mentor request seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *steps <= 0 {
		return errors.New("steps must be > 0")
	}
	if *width <= 0 {
		return errors.New("width must be > 0")
	}
	if *height <= 0 {
		return errors.New("height must be > 0")
	}
	if *maxTokens <= 0 {
		return errors.New("max-tokens must be > 0")
	}
	if strings.TrimSpace(*replayRunID) != "" {
		return runCommGridMentorReplay(ctx, strings.TrimSpace(*replayRunID), *jsonOut)
	}
	id := strings.TrimSpace(*runID)
	if id == "" {
		id = fmt.Sprintf("comm-grid-mentor-fixture-%d", time.Now().UTC().UnixNano())
	}
	if err := validateCommGridLLMRunID(id); err != nil {
		return err
	}
	key, err := parseCommGridLLMPoint("key", *keyRaw)
	if err != nil {
		return err
	}
	goal, err := parseCommGridLLMPoint("goal", *goalRaw)
	if err != nil {
		return err
	}
	agentPos, err := parseCommGridLLMPoint("agent-pos", *agentPosRaw)
	if err != nil {
		return err
	}
	agentName := strings.TrimSpace(*agentID)
	if agentName == "" {
		return errors.New("agent must not be empty")
	}

	provider, normalizedPlan, err := commGridMentorFixtureProvider(*plan)
	if err != nil {
		return err
	}
	cfg := scape.CommGridMentorConfig{
		CommGridConfig: scape.CommGridConfig{
			Width:        *width,
			Height:       *height,
			MaxSteps:     *steps,
			MessageLimit: 80,
			Key:          key,
			Goal:         goal,
			Agents: []scape.CommGridAgentState{{
				ID:       agentName,
				Position: agentPos,
			}},
		},
		Provider:  provider,
		Model:     strings.TrimSpace(*model),
		MaxTokens: *maxTokens,
		Seed:      *seed,
	}
	startedAt := time.Now().UTC()
	summary, artifact, err := runCommGridMentorSingle(ctx, commGridMentorRunConfig{
		RunID:    id,
		Provider: provider,
		Plan:     normalizedPlan,
		Task: commGridLLMTaskConfig{
			Width:        *width,
			Height:       *height,
			Key:          key,
			Goal:         goal,
			AgentID:      agentName,
			Agent:        agentPos,
			MessageLimit: 80,
		},
		Config:    cfg,
		StartedAt: startedAt,
	})
	if err != nil {
		return err
	}
	if *writeArtifacts {
		artifactDir, err := writeCommGridMentorArtifact(benchmarksDir, artifact)
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
	printCommGridMentorSummary(summary)
	return nil
}

type commGridMentorRunConfig struct {
	RunID     string
	Provider  llm.Provider
	Plan      string
	Task      commGridLLMTaskConfig
	Config    scape.CommGridMentorConfig
	StartedAt time.Time
}

func runCommGridMentorSingle(ctx context.Context, cfg commGridMentorRunConfig) (commGridMentorCommandSummary, commGridMentorArtifact, error) {
	if err := validateCommGridLLMRunID(cfg.RunID); err != nil {
		return commGridMentorCommandSummary{}, commGridMentorArtifact{}, err
	}
	runCfg := cfg.Config
	runCfg.Provider = cfg.Provider
	sc := scape.CommGridMentorScape{Config: runCfg}
	fitness, trace, err := sc.Evaluate(ctx, commGridMentorPolicyAgent{id: cfg.Task.AgentID})
	if err != nil {
		return commGridMentorCommandSummary{}, commGridMentorArtifact{}, err
	}
	steps := commGridMentorSteps(trace)
	totalTokens := commGridMentorTotalTokens(steps)
	startedAt := cfg.StartedAt
	if startedAt.IsZero() {
		startedAt = time.Now().UTC()
	}
	summary := commGridMentorCommandSummary{
		RunID:     cfg.RunID,
		Provider:  "fixture",
		Plan:      cfg.Plan,
		Task:      normalizeCommGridLLMTask(cfg.Task),
		Steps:     steps,
		Completed: trace["completed"] == true,
		Fitness:   float64(fitness),
		Trace:     map[string]any(trace),
	}
	artifact := commGridMentorArtifact{
		RunID:       summary.RunID,
		Provider:    summary.Provider,
		Plan:        summary.Plan,
		CreatedAt:   startedAt.Format(time.RFC3339Nano),
		DurationMS:  commGridLLMDurationMillis(time.Since(startedAt)),
		Task:        summary.Task,
		Steps:       steps,
		Completed:   summary.Completed,
		Fitness:     summary.Fitness,
		TotalTokens: totalTokens,
		Trace:       summary.Trace,
	}
	return summary, artifact, nil
}

func runCommGridMentorReplay(ctx context.Context, runID string, jsonOut bool) error {
	if err := validateCommGridLLMRunID(runID); err != nil {
		return err
	}
	artifact, err := readCommGridMentorArtifact(benchmarksDir, runID)
	if err != nil {
		return err
	}
	if len(artifact.Steps) == 0 {
		return fmt.Errorf("comm-grid mentor artifact has no steps: %s", runID)
	}
	task := normalizeCommGridLLMTask(artifact.Task)
	model := "fixture-mentor"
	maxTokens := 16
	seed := int64(1)
	for _, step := range artifact.Steps {
		if strings.TrimSpace(step.Request.Model) != "" {
			model = strings.TrimSpace(step.Request.Model)
		}
		if step.Request.MaxTokens > 0 {
			maxTokens = step.Request.MaxTokens
		}
		if step.Request.Seed != 0 {
			seed = step.Request.Seed
		}
	}
	cfg := scape.CommGridMentorConfig{
		CommGridConfig: scape.CommGridConfig{
			Width:        task.Width,
			Height:       task.Height,
			MaxSteps:     commGridTraceInt(artifact.Trace, "max_steps", len(artifact.Steps)),
			MessageLimit: task.MessageLimit,
			Key:          task.Key,
			Goal:         task.Goal,
			Agents: []scape.CommGridAgentState{{
				ID:       task.AgentID,
				Position: task.Agent,
			}},
		},
		Provider:  &commGridMentorReplayProvider{steps: artifact.Steps},
		Model:     model,
		MaxTokens: maxTokens,
		Seed:      seed,
	}
	summary, _, err := runCommGridMentorSingle(ctx, commGridMentorRunConfig{
		RunID:    runID,
		Provider: cfg.Provider,
		Plan:     strings.TrimSpace(artifact.Plan),
		Task:     task,
		Config:   cfg,
	})
	if err != nil {
		return err
	}
	summary.Provider = "fixture-replay"
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
	fmt.Printf("comm_grid_mentor_replay run_id=%s matched=%t completed=%t fitness=%.6f\n", runID, replay.Matched, summary.Completed, summary.Fitness)
	fmt.Printf("expected_trace=%v\n", replay.Expected)
	fmt.Printf("actual_trace=%v\n", replay.Actual)
	if !replay.Matched {
		return fmt.Errorf("comm-grid mentor replay trace mismatch: %s", runID)
	}
	return nil
}

func commGridMentorFixtureProvider(plan string) (llm.Provider, string, error) {
	normalized := strings.TrimSpace(strings.ToLower(plan))
	switch normalized {
	case "", "solve":
		return llm.NewFixtureProvider([]llm.Response{
			commGridMentorFixtureHint("east", 1),
			commGridMentorFixtureHint("pick", 1),
			commGridMentorFixtureHint("east", 1),
			commGridMentorFixtureHint("drop", 1),
		}), "solve", nil
	case "silent":
		return llm.NewFixtureProvider([]llm.Response{
			commGridMentorFixtureHint("", 0),
			commGridMentorFixtureHint("", 0),
			commGridMentorFixtureHint("", 0),
			commGridMentorFixtureHint("", 0),
		}), "silent", nil
	case "provider-error":
		return llm.NewFixtureProviderWithError(errors.New("fixture mentor provider failure")), "provider-error", nil
	default:
		return nil, "", fmt.Errorf("unsupported comm-grid mentor fixture plan: %s", plan)
	}
}

func commGridMentorFixtureHint(hint string, tokens int) llm.Response {
	return llm.Response{
		Message:      hint,
		Usage:        llm.Usage{TotalTokens: tokens},
		FinishReason: "stop",
		Model:        "fixture-mentor",
	}
}

func commGridMentorSteps(trace map[string]any) []scape.CommGridMentorStepTrace {
	steps, ok := trace["mentor_steps"].([]scape.CommGridMentorStepTrace)
	if !ok {
		return nil
	}
	return steps
}

func printCommGridMentorSummary(summary commGridMentorCommandSummary) {
	fmt.Printf("comm_grid_mentor run_id=%s provider=%s plan=%s grid=%dx%d key=(%d,%d) goal=(%d,%d) agent=%s@(%d,%d) steps=%d completed=%t fitness=%.6f\n",
		summary.RunID,
		summary.Provider,
		summary.Plan,
		summary.Task.Width,
		summary.Task.Height,
		summary.Task.Key.X,
		summary.Task.Key.Y,
		summary.Task.Goal.X,
		summary.Task.Goal.Y,
		summary.Task.AgentID,
		summary.Task.Agent.X,
		summary.Task.Agent.Y,
		len(summary.Steps),
		summary.Completed,
		summary.Fitness,
	)
	for _, step := range summary.Steps {
		fmt.Printf("step=%d hint=%q hint_action=%s action=%s invalid=%t provider_error=%q\n",
			step.Step+1,
			step.Hint,
			step.HintAction,
			step.Action,
			step.InvalidAction,
			step.ProviderError,
		)
	}
	if summary.ArtifactsDir != "" {
		fmt.Printf("artifacts_dir=%s\n", filepath.Clean(summary.ArtifactsDir))
	}
}

func writeCommGridMentorArtifact(baseDir string, artifact commGridMentorArtifact) (string, error) {
	runID := strings.TrimSpace(artifact.RunID)
	if err := validateCommGridLLMRunID(runID); err != nil {
		return "", err
	}
	runDir := filepath.Join(baseDir, runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(runDir, "comm_grid_mentor.json")
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return "", err
	}
	return filepath.Clean(runDir), nil
}

func readCommGridMentorArtifact(baseDir, runID string) (commGridMentorArtifact, error) {
	if err := validateCommGridLLMRunID(runID); err != nil {
		return commGridMentorArtifact{}, err
	}
	path := filepath.Join(baseDir, runID, "comm_grid_mentor.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return commGridMentorArtifact{}, err
	}
	var artifact commGridMentorArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return commGridMentorArtifact{}, err
	}
	return artifact, nil
}

func commGridMentorTotalTokens(steps []scape.CommGridMentorStepTrace) int {
	total := 0
	for _, step := range steps {
		total += step.Response.TokenCount()
	}
	return total
}

func commGridMentorReplayStepError(step scape.CommGridMentorStepTrace) error {
	if strings.TrimSpace(step.ProviderError) == "" {
		return nil
	}
	return errors.New(step.ProviderError)
}
