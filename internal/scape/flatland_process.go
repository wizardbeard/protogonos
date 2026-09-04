package scape

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

type FlatlandPublicMessage interface {
	isFlatlandPublicMessage()
}

type FlatlandPublicStartMessage struct{}

func (FlatlandPublicStartMessage) isFlatlandPublicMessage() {}

type FlatlandPublicStopMessage struct {
	Reason string
}

func (FlatlandPublicStopMessage) isFlatlandPublicMessage() {}

type FlatlandPublicSyncMessage struct{}

func (FlatlandPublicSyncMessage) isFlatlandPublicMessage() {}

type FlatlandPublicEnterMessage struct {
	Agent FlatlandPublicAgent
}

func (FlatlandPublicEnterMessage) isFlatlandPublicMessage() {}

type FlatlandPublicLeaveMessage struct {
	AgentID string
}

func (FlatlandPublicLeaveMessage) isFlatlandPublicMessage() {}

type FlatlandPublicUpdateAgentsMessage struct {
	Agents []FlatlandPublicAgent
}

func (FlatlandPublicUpdateAgentsMessage) isFlatlandPublicMessage() {}

type FlatlandPublicGetAllMessage struct{}

func (FlatlandPublicGetAllMessage) isFlatlandPublicMessage() {}

type FlatlandPublicTickMessage struct{}

func (FlatlandPublicTickMessage) isFlatlandPublicMessage() {}

type FlatlandPublicSenseMessage struct {
	AgentID string
}

func (FlatlandPublicSenseMessage) isFlatlandPublicMessage() {}

type FlatlandPublicActMessage struct {
	AgentID string
	Output  []float64
}

func (FlatlandPublicActMessage) isFlatlandPublicMessage() {}

type FlatlandPublicResponse struct {
	OK         bool
	Percept    []float64
	Fitness    Fitness
	End        bool
	Trace      Trace
	Agents     []Trace
	StopReason string
	Err        error
}

type FlatlandPublicProcess struct {
	runtime *flatlandPublicRuntime
}

func NewFlatlandPublicProcess() *FlatlandPublicProcess {
	return &FlatlandPublicProcess{runtime: newFlatlandPublicRuntime()}
}

func (p *FlatlandPublicProcess) Call(ctx context.Context, message FlatlandPublicMessage) FlatlandPublicResponse {
	if p == nil || message == nil {
		return FlatlandPublicResponse{Err: fmt.Errorf("invalid flatland public process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return FlatlandPublicResponse{Err: err}
	}
	if p.runtime == nil {
		p.runtime = newFlatlandPublicRuntime()
	}

	switch msg := message.(type) {
	case FlatlandPublicStartMessage:
		err := p.start(ctx)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicStopMessage:
		reason, err := p.stop(ctx, msg.Reason)
		return FlatlandPublicResponse{OK: err == nil, StopReason: reason, Err: err}
	case FlatlandPublicSyncMessage:
		err := p.sync(ctx)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicEnterMessage:
		err := p.enter(msg.Agent)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicLeaveMessage:
		err := p.leave(msg.AgentID)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicUpdateAgentsMessage:
		err := p.updateAgents(msg.Agents)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicGetAllMessage:
		agents, err := p.agents()
		return FlatlandPublicResponse{OK: err == nil, Agents: agents, Err: err}
	case FlatlandPublicTickMessage:
		trace, err := p.tick(ctx)
		return FlatlandPublicResponse{OK: err == nil, Trace: trace, Err: err}
	case FlatlandPublicSenseMessage:
		percept, trace, err := p.sense(ctx, msg.AgentID)
		return FlatlandPublicResponse{OK: err == nil, Percept: percept, Trace: trace, Err: err}
	case FlatlandPublicActMessage:
		fitness, end, trace, err := p.act(ctx, msg.AgentID, msg.Output)
		return FlatlandPublicResponse{OK: err == nil, Fitness: fitness, End: end, Trace: trace, Err: err}
	default:
		return FlatlandPublicResponse{Err: fmt.Errorf("unsupported flatland public process message %T", message)}
	}
}

func (p *FlatlandPublicProcess) start(_ context.Context) error {
	cfg, err := flatlandConfigForMode("gt")
	if err != nil {
		return err
	}

	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	p.runtime.started = true
	p.runtime.config = cfg
	p.runtime.tick = 0
	p.runtime.agents = make(map[string]*flatlandPublicAgentState)
	p.runtime.lastStopReason = ""
	return nil
}

func (p *FlatlandPublicProcess) stop(_ context.Context, reason string) (string, error) {
	rawReason := reason
	reason = normalizeFlatlandPublicStopReason(reason)
	if reason == "" {
		return "", fmt.Errorf("unsupported flatland public stop reason: %s", rawReason)
	}

	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	p.runtime.started = false
	p.runtime.tick = 0
	p.runtime.agents = make(map[string]*flatlandPublicAgentState)
	p.runtime.lastStopReason = reason
	return reason, nil
}

func (p *FlatlandPublicProcess) sync(_ context.Context) error {
	return nil
}

func (p *FlatlandPublicProcess) enter(agent FlatlandPublicAgent) error {
	agentID := strings.TrimSpace(agent.ID)
	if agentID == "" {
		return fmt.Errorf("flatland public agent id is required")
	}

	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	if !p.runtime.started {
		return fmt.Errorf("flatland public world is not started")
	}
	if _, exists := p.runtime.agents[agentID]; exists {
		return fmt.Errorf("flatland public agent already entered: %s", agentID)
	}

	cfg := p.runtime.config
	mode := cfg.mode
	if strings.TrimSpace(agent.Mode) != "" {
		modeCfg, err := flatlandConfigForMode(agent.Mode)
		if err != nil {
			return err
		}
		cfg = modeCfg
		mode = modeCfg.mode
	}
	p.runtime.agents[agentID] = &flatlandPublicAgentState{
		id:      agentID,
		mode:    mode,
		episode: newFlatlandEpisodeForAgent(cfg, agentID),
		decide:  agent.Decide,
	}
	return nil
}

func (p *FlatlandPublicProcess) updateAgents(agents []FlatlandPublicAgent) error {
	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	if !p.runtime.started {
		return fmt.Errorf("flatland public world is not started")
	}

	next := make(map[string]*flatlandPublicAgentState, len(agents))
	for _, agent := range agents {
		agentID := strings.TrimSpace(agent.ID)
		if agentID == "" {
			return fmt.Errorf("flatland public agent id is required")
		}
		if _, exists := next[agentID]; exists {
			return fmt.Errorf("duplicate public agent update for id: %s", agentID)
		}

		existing, hasExisting := p.runtime.agents[agentID]
		if hasExisting {
			mode := existing.mode
			if mode == "" {
				mode = p.runtime.config.mode
			}
			modeCfg, err := flatlandConfigForMode(mode)
			if err != nil {
				return err
			}
			if strings.TrimSpace(agent.Mode) != "" {
				modeCfg, err = flatlandConfigForMode(agent.Mode)
				if err != nil {
					return err
				}
				mode = modeCfg.mode
			}
			if mode != existing.mode || existing.terminated {
				existing.mode = mode
				existing.episode = newFlatlandEpisodeForAgent(modeCfg, agentID)
				existing.terminated = false
			}
			existing.decide = agent.Decide
			next[agentID] = existing
			continue
		}

		cfg := p.runtime.config
		mode := cfg.mode
		if strings.TrimSpace(agent.Mode) != "" {
			modeCfg, err := flatlandConfigForMode(agent.Mode)
			if err != nil {
				return err
			}
			cfg = modeCfg
			mode = modeCfg.mode
		}
		next[agentID] = &flatlandPublicAgentState{
			id:      agentID,
			mode:    mode,
			episode: newFlatlandEpisodeForAgent(cfg, agentID),
			decide:  agent.Decide,
		}
	}

	p.runtime.agents = next
	return nil
}

func (p *FlatlandPublicProcess) leave(agentID string) error {
	agentID = strings.TrimSpace(agentID)
	if agentID == "" {
		return fmt.Errorf("flatland public agent id is required")
	}

	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	if !p.runtime.started {
		return fmt.Errorf("flatland public world is not started")
	}
	if _, exists := p.runtime.agents[agentID]; !exists {
		return fmt.Errorf("flatland public agent not found: %s", agentID)
	}
	delete(p.runtime.agents, agentID)
	return nil
}

func (p *FlatlandPublicProcess) agents() ([]Trace, error) {
	p.runtime.mu.RLock()
	defer p.runtime.mu.RUnlock()
	if !p.runtime.started {
		return nil, fmt.Errorf("flatland public world is not started")
	}

	ids := make([]string, 0, len(p.runtime.agents))
	for id := range p.runtime.agents {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	out := make([]Trace, 0, len(ids))
	for _, id := range ids {
		out = append(out, flatlandPublicAgentTrace(p.runtime.agents[id]))
	}
	return out, nil
}

func (p *FlatlandPublicProcess) sense(ctx context.Context, agentID string) ([]float64, Trace, error) {
	agentID = strings.TrimSpace(agentID)
	if agentID == "" {
		return nil, nil, fmt.Errorf("flatland public agent id is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}

	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	if !p.runtime.started {
		return nil, nil, fmt.Errorf("flatland public world is not started")
	}
	state, exists := p.runtime.agents[agentID]
	if !exists {
		return nil, nil, fmt.Errorf("flatland public agent not found: %s", agentID)
	}
	if state.terminated {
		return nil, flatlandPublicAgentTrace(state), fmt.Errorf("flatland public agent terminated: %s", agentID)
	}

	state.episode.advanceRespawns()
	sense := state.episode.sense()
	trace := flatlandPublicAgentTrace(state)
	trace["sensor_surface"] = "step_input"
	trace["sensor_width"] = flatlandBaseFeatureWidth + flatlandScannerWidth
	trace["scanner_density"] = flatlandScannerDensity
	trace["scanner_density_effective"] = countActiveFlatlandScannerWeights(state.episode.scannerWeights)
	trace["scanner_profile_active_bins"] = flatlandActiveScannerBins(state.episode.scannerWeights)
	trace["last_food_distance"] = sense.distance
	trace["last_prey_signal"] = sense.prey
	trace["last_predator_signal"] = sense.predator
	trace["last_poison_signal"] = sense.poison
	trace["last_wall_signal"] = sense.wall
	trace["last_food_proximity"] = sense.foodProximity
	trace["last_prey_proximity"] = sense.preyProximity
	trace["last_predator_proximity"] = sense.predatorProximity
	trace["last_poison_proximity"] = sense.poisonProximity
	trace["last_wall_proximity"] = sense.wallProximity
	trace["last_resource_balance"] = sense.resourceBalance
	trace["last_distance_scan_bins"] = flatlandScanSlice(sense.distanceScan)
	trace["last_color_scan_bins"] = flatlandScanSlice(sense.colorScan)
	trace["last_energy_scan_bins"] = flatlandScanSlice(sense.energyScan)
	return flatlandStepInputVector(sense), trace, nil
}

func (p *FlatlandPublicProcess) act(ctx context.Context, agentID string, output []float64) (Fitness, bool, Trace, error) {
	agentID = strings.TrimSpace(agentID)
	if agentID == "" {
		return 0, false, nil, fmt.Errorf("flatland public agent id is required")
	}
	if err := ctx.Err(); err != nil {
		return 0, false, nil, err
	}
	control, err := flatlandControlFromOutput(output)
	if err != nil {
		return 0, false, nil, err
	}

	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	if !p.runtime.started {
		return 0, false, nil, fmt.Errorf("flatland public world is not started")
	}
	state, exists := p.runtime.agents[agentID]
	if !exists {
		return 0, false, nil, fmt.Errorf("flatland public agent not found: %s", agentID)
	}
	if state.terminated {
		trace := flatlandPublicAgentTrace(state)
		trace["end"] = true
		return 0, true, trace, nil
	}

	moveStep, hitFood, hitPoison, wallCollision, reason := state.episode.step(control.move)
	if reason != "" {
		state.terminated = true
	}
	trace := flatlandPublicAgentTrace(state)
	trace["move_step"] = moveStep
	trace["hit_food"] = hitFood
	trace["hit_poison"] = hitPoison
	trace["wall_collision"] = wallCollision
	trace["terminal_reason"] = reason
	trace["control_surface"] = "step_output"
	trace["last_control_width"] = control.width
	trace["end"] = state.terminated

	fitness := clamp(
		float64(state.episode.age)/float64(state.episode.maxAge)+
			0.25*state.episode.normalizedEnergy()+
			0.1*float64(state.episode.foodCollected-state.episode.poisonHits),
		0,
		1.4,
	)
	return Fitness(fitness), state.terminated, trace, nil
}

func (p *FlatlandPublicProcess) tick(ctx context.Context) (Trace, error) {
	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	if !p.runtime.started {
		return nil, fmt.Errorf("flatland public world is not started")
	}

	ids := make([]string, 0, len(p.runtime.agents))
	for id := range p.runtime.agents {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	terminated := 0
	totalEnergy := 0.0
	totalFood := 0
	totalPrey := 0
	totalPredatorHits := 0
	agentStates := make([]Trace, 0, len(ids))

	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		state := p.runtime.agents[id]
		if state.terminated {
			totalEnergy += state.episode.energy
			totalFood += state.episode.foodCollected
			totalPrey += state.episode.preyCollected
			totalPredatorHits += state.episode.predatorHits
			terminated++
			agentStates = append(agentStates, flatlandPublicAgentTrace(state))
			continue
		}

		state.episode.advanceRespawns()
		sense := state.episode.sense()
		out := defaultFlatlandPublicPolicy(sense)
		if state.decide != nil {
			candidate := state.decide(flatlandStepInputVector(sense))
			if len(candidate) > 0 {
				out = candidate
			}
		}
		control, err := flatlandControlFromOutput(out)
		if err != nil {
			return nil, err
		}
		_, _, _, _, reason := state.episode.step(control.move)
		if reason != "" {
			state.terminated = true
			terminated++
		}
		totalEnergy += state.episode.energy
		totalFood += state.episode.foodCollected
		totalPrey += state.episode.preyCollected
		totalPredatorHits += state.episode.predatorHits
		agentStates = append(agentStates, flatlandPublicAgentTrace(state))
	}

	p.runtime.tick++
	avgEnergy := 0.0
	if len(ids) > 0 {
		avgEnergy = totalEnergy / float64(len(ids))
	}
	return Trace{
		"tick":                p.runtime.tick,
		"active_agents":       len(ids) - terminated,
		"terminated_agents":   terminated,
		"total_agents":        len(ids),
		"avg_energy":          avgEnergy,
		"total_food":          totalFood,
		"total_prey":          totalPrey,
		"total_predator_hits": totalPredatorHits,
		"agents":              agentStates,
	}, nil
}
