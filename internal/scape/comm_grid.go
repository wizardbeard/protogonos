package scape

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

type CommGridAction string

const (
	CommGridStay  CommGridAction = "stay"
	CommGridNorth CommGridAction = "north"
	CommGridSouth CommGridAction = "south"
	CommGridEast  CommGridAction = "east"
	CommGridWest  CommGridAction = "west"
	CommGridPick  CommGridAction = "pick"
	CommGridDrop  CommGridAction = "drop"
)

type CommGridPoint struct {
	X int
	Y int
}

type CommGridMessage struct {
	From   string
	To     string
	Text   string
	Tokens int
	Step   int
}

type CommGridAgentState struct {
	ID       string
	Position CommGridPoint
	Carrying bool
}

type CommGridConfig struct {
	Width                int
	Height               int
	MaxSteps             int
	MessageLimit         int
	MessageCost          float64
	TokenCost            float64
	InvalidActionPenalty float64
	GoalReward           float64
	Key                  CommGridPoint
	Goal                 CommGridPoint
	Agents               []CommGridAgentState
}

type CommGridStepInput struct {
	AgentID string
	Action  CommGridAction
	Message string
	To      string
	Tokens  int
}

type CommGridLanguageAction struct {
	Action  string `json:"action"`
	Message string `json:"message,omitempty"`
	To      string `json:"to,omitempty"`
	Tokens  int    `json:"tokens,omitempty"`
}

type CommGridStepResult struct {
	Done          bool
	Completed     bool
	InvalidAction bool
	Fitness       Fitness
	Trace         Trace
}

type CommGridSimulator struct {
	cfg       CommGridConfig
	agents    map[string]CommGridAgentState
	messages  []CommGridMessage
	step      int
	completed bool
	invalids  int
	tokens    int
}

type CommGridScape struct {
	Config CommGridConfig
}

func (CommGridScape) Name() string {
	return "comm-grid"
}

func (s CommGridScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	sim := NewCommGridSimulator(s.Config)
	for !sim.Done() {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}
		state, ok := sim.AgentState(agent.ID())
		if !ok {
			return 0, nil, fmt.Errorf("comm-grid agent not found: %s", agent.ID())
		}
		input := commGridStepVector(sim, state)
		output, err := runner.RunStep(ctx, input)
		if err != nil {
			return 0, nil, err
		}
		action, err := DecodeCommGridAction(output)
		if err != nil {
			return 0, nil, err
		}
		if _, err := sim.Step(ctx, CommGridStepInput{AgentID: agent.ID(), Action: action}); err != nil {
			return 0, nil, err
		}
	}
	return sim.Fitness(), sim.Trace(), nil
}

func NewCommGridSimulator(cfg CommGridConfig) *CommGridSimulator {
	cfg = normalizeCommGridConfig(cfg)
	agents := make(map[string]CommGridAgentState, len(cfg.Agents))
	for _, agent := range cfg.Agents {
		agent.ID = strings.TrimSpace(agent.ID)
		if agent.ID == "" {
			continue
		}
		agent.Position = clampCommGridPoint(agent.Position, cfg.Width, cfg.Height)
		agents[agent.ID] = agent
	}
	return &CommGridSimulator{
		cfg:    cfg,
		agents: agents,
	}
}

func (s *CommGridSimulator) Step(ctx context.Context, input CommGridStepInput) (CommGridStepResult, error) {
	if err := ctx.Err(); err != nil {
		return CommGridStepResult{}, err
	}
	if s == nil {
		return CommGridStepResult{}, fmt.Errorf("comm-grid simulator is nil")
	}
	if s.Done() {
		return CommGridStepResult{Done: true, Completed: s.completed, Fitness: s.Fitness(), Trace: s.Trace()}, nil
	}

	agentID := strings.TrimSpace(input.AgentID)
	agent, ok := s.agents[agentID]
	if !ok {
		return CommGridStepResult{}, fmt.Errorf("comm-grid agent not found: %s", agentID)
	}

	invalid := false
	next := agent.Position
	switch input.Action {
	case CommGridStay:
	case CommGridNorth:
		next.Y--
	case CommGridSouth:
		next.Y++
	case CommGridEast:
		next.X++
	case CommGridWest:
		next.X--
	case CommGridPick:
		if agent.Position == s.cfg.Key && !agent.Carrying {
			agent.Carrying = true
		} else {
			invalid = true
		}
	case CommGridDrop:
		if agent.Position == s.cfg.Goal && agent.Carrying {
			agent.Carrying = false
			s.completed = true
		} else {
			invalid = true
		}
	default:
		invalid = true
	}
	if isCommGridMove(input.Action) {
		if next.X < 0 || next.Y < 0 || next.X >= s.cfg.Width || next.Y >= s.cfg.Height {
			invalid = true
		} else {
			agent.Position = next
		}
	}

	if invalid {
		s.invalids++
	}
	s.recordMessage(input)
	s.agents[agentID] = agent
	s.step++

	return CommGridStepResult{
		Done:          s.Done(),
		Completed:     s.completed,
		InvalidAction: invalid,
		Fitness:       s.Fitness(),
		Trace:         s.Trace(),
	}, nil
}

func (s *CommGridSimulator) Done() bool {
	if s == nil {
		return true
	}
	return s.completed || s.step >= s.cfg.MaxSteps
}

func (s *CommGridSimulator) AgentState(agentID string) (CommGridAgentState, bool) {
	if s == nil {
		return CommGridAgentState{}, false
	}
	agent, ok := s.agents[strings.TrimSpace(agentID)]
	return agent, ok
}

func (s *CommGridSimulator) Messages() []CommGridMessage {
	if s == nil || len(s.messages) == 0 {
		return nil
	}
	out := make([]CommGridMessage, len(s.messages))
	copy(out, s.messages)
	return out
}

func (s *CommGridSimulator) Fitness() Fitness {
	if s == nil {
		return 0
	}
	bestProgress := 0.0
	for _, agent := range s.agents {
		progress := s.agentProgress(agent)
		if progress > bestProgress {
			bestProgress = progress
		}
	}
	score := 0.55 * bestProgress
	if s.completed {
		score += s.cfg.GoalReward
	}
	score -= float64(s.invalids) * s.cfg.InvalidActionPenalty
	score -= float64(len(s.messages)) * s.cfg.MessageCost
	score -= float64(s.tokens) * s.cfg.TokenCost
	if !s.completed && s.step >= s.cfg.MaxSteps {
		score -= 0.05
	}
	return Fitness(commGridClamp(score, 0, 1.5))
}

func (s *CommGridSimulator) Trace() Trace {
	if s == nil {
		return Trace{}
	}
	return Trace{
		"step":            s.step,
		"max_steps":       s.cfg.MaxSteps,
		"completed":       s.completed,
		"invalid_actions": s.invalids,
		"messages":        len(s.messages),
		"tokens":          s.tokens,
		"fitness":         float64(s.Fitness()),
	}
}

func (s *CommGridSimulator) recordMessage(input CommGridStepInput) {
	text := strings.TrimSpace(input.Message)
	if text == "" {
		return
	}
	if len(text) > s.cfg.MessageLimit {
		text = text[:s.cfg.MessageLimit]
	}
	tokens := input.Tokens
	if tokens < 0 {
		tokens = 0
	}
	s.tokens += tokens
	s.messages = append(s.messages, CommGridMessage{
		From:   strings.TrimSpace(input.AgentID),
		To:     strings.TrimSpace(input.To),
		Text:   text,
		Tokens: tokens,
		Step:   s.step,
	})
}

func (s *CommGridSimulator) agentProgress(agent CommGridAgentState) float64 {
	start := s.cfg.Agents[0].Position
	if len(s.cfg.Agents) > 0 {
		for _, configured := range s.cfg.Agents {
			if configured.ID == agent.ID {
				start = configured.Position
				break
			}
		}
	}
	var target CommGridPoint
	if agent.Carrying || s.completed {
		target = s.cfg.Goal
	} else {
		target = s.cfg.Key
	}
	initial := math.Max(1, float64(commGridDistance(start, s.cfg.Key)+commGridDistance(s.cfg.Key, s.cfg.Goal)))
	current := float64(commGridDistance(agent.Position, target))
	if !agent.Carrying && !s.completed {
		current += float64(commGridDistance(s.cfg.Key, s.cfg.Goal))
	}
	return commGridClamp(1-current/initial, 0, 1)
}

func DecodeCommGridAction(output []float64) (CommGridAction, error) {
	if len(output) < 3 {
		return "", fmt.Errorf("comm-grid requires at least 3 outputs, got %d", len(output))
	}
	x := output[0]
	y := output[1]
	tool := output[2]
	if tool > 0.6 {
		return CommGridPick, nil
	}
	if tool < -0.6 {
		return CommGridDrop, nil
	}
	if math.Abs(x) < 0.2 && math.Abs(y) < 0.2 {
		return CommGridStay, nil
	}
	if math.Abs(x) >= math.Abs(y) {
		if x > 0 {
			return CommGridEast, nil
		}
		return CommGridWest, nil
	}
	if y > 0 {
		return CommGridSouth, nil
	}
	return CommGridNorth, nil
}

func DecodeCommGridLanguageAction(agentID string, payload []byte) (CommGridStepInput, error) {
	var decoded CommGridLanguageAction
	if err := json.Unmarshal(payload, &decoded); err != nil {
		return CommGridStepInput{}, err
	}
	action, err := ParseCommGridAction(decoded.Action)
	if err != nil {
		return CommGridStepInput{}, err
	}
	tokens := decoded.Tokens
	if tokens < 0 {
		tokens = 0
	}
	if tokens == 0 && strings.TrimSpace(decoded.Message) != "" {
		tokens = len(strings.Fields(decoded.Message))
	}
	return CommGridStepInput{
		AgentID: strings.TrimSpace(agentID),
		Action:  action,
		Message: strings.TrimSpace(decoded.Message),
		To:      strings.TrimSpace(decoded.To),
		Tokens:  tokens,
	}, nil
}

func ParseCommGridAction(action string) (CommGridAction, error) {
	normalized := strings.TrimSpace(strings.ToLower(action))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	normalized = strings.TrimPrefix(normalized, "move_")
	switch normalized {
	case "", "stay", "wait", "none":
		return CommGridStay, nil
	case "north", "up", "n":
		return CommGridNorth, nil
	case "south", "down", "s":
		return CommGridSouth, nil
	case "east", "right", "e":
		return CommGridEast, nil
	case "west", "left", "w":
		return CommGridWest, nil
	case "pick", "pickup", "pick_up":
		return CommGridPick, nil
	case "drop", "deliver":
		return CommGridDrop, nil
	default:
		return "", fmt.Errorf("unsupported comm-grid action: %s", action)
	}
}

func commGridStepVector(sim *CommGridSimulator, agent CommGridAgentState) []float64 {
	target := sim.cfg.Key
	if agent.Carrying {
		target = sim.cfg.Goal
	}
	width := math.Max(1, float64(sim.cfg.Width-1))
	height := math.Max(1, float64(sim.cfg.Height-1))
	return []float64{
		float64(target.X-agent.Position.X) / width,
		float64(target.Y-agent.Position.Y) / height,
		boolToFloat(agent.Carrying),
		float64(sim.step) / float64(sim.cfg.MaxSteps),
	}
}

func normalizeCommGridConfig(cfg CommGridConfig) CommGridConfig {
	if cfg.Width <= 0 {
		cfg.Width = 4
	}
	if cfg.Height <= 0 {
		cfg.Height = 4
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 16
	}
	if cfg.MessageLimit <= 0 {
		cfg.MessageLimit = 80
	}
	if cfg.MessageCost == 0 {
		cfg.MessageCost = 0.01
	}
	if cfg.TokenCost == 0 {
		cfg.TokenCost = 0.001
	}
	if cfg.InvalidActionPenalty == 0 {
		cfg.InvalidActionPenalty = 0.05
	}
	if cfg.GoalReward == 0 {
		cfg.GoalReward = 0.95
	}
	cfg.Key = clampCommGridPoint(cfg.Key, cfg.Width, cfg.Height)
	if cfg.Goal == (CommGridPoint{}) {
		cfg.Goal = CommGridPoint{X: cfg.Width - 1, Y: cfg.Height - 1}
	}
	cfg.Goal = clampCommGridPoint(cfg.Goal, cfg.Width, cfg.Height)
	if len(cfg.Agents) == 0 {
		cfg.Agents = []CommGridAgentState{{
			ID:       "agent-1",
			Position: CommGridPoint{},
		}}
	}
	return cfg
}

func clampCommGridPoint(point CommGridPoint, width, height int) CommGridPoint {
	if point.X < 0 {
		point.X = 0
	}
	if point.Y < 0 {
		point.Y = 0
	}
	if point.X >= width {
		point.X = width - 1
	}
	if point.Y >= height {
		point.Y = height - 1
	}
	return point
}

func isCommGridMove(action CommGridAction) bool {
	switch action {
	case CommGridNorth, CommGridSouth, CommGridEast, CommGridWest:
		return true
	default:
		return false
	}
}

func commGridDistance(a, b CommGridPoint) int {
	return commGridAbsInt(a.X-b.X) + commGridAbsInt(a.Y-b.Y)
}

func commGridAbsInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func boolToFloat(v bool) float64 {
	if v {
		return 1
	}
	return 0
}

func commGridClamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
