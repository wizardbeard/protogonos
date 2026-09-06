package scape

import (
	"context"
	"fmt"
	"sort"
	"strings"

	protoio "protogonos/internal/io"
)

const (
	flatlandNeuralCost = 100.0
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
	AgentID      string
	Output       []float64
	ActuatorName string
}

func (FlatlandPublicActMessage) isFlatlandPublicMessage() {}

type FlatlandPublicResponse struct {
	OK         bool
	Percept    []float64
	Fitness    Fitness
	End        bool
	Trace      Trace
	Agents     []Trace
	Avatars    []FlatlandPublicAvatarSnapshot
	Update     FlatlandPublicUpdateSummary
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
		summary, avatars, err := p.updateAgents(msg.Agents)
		return FlatlandPublicResponse{OK: err == nil, Update: summary, Avatars: avatars, Trace: summary.trace(), Err: err}
	case FlatlandPublicGetAllMessage:
		agents, avatars, err := p.agents()
		return FlatlandPublicResponse{OK: err == nil, Agents: agents, Avatars: avatars, Err: err}
	case FlatlandPublicTickMessage:
		trace, avatars, err := p.tick(ctx)
		return FlatlandPublicResponse{OK: err == nil, Trace: trace, Avatars: avatars, Err: err}
	case FlatlandPublicSenseMessage:
		percept, trace, err := p.sense(ctx, msg.AgentID)
		return FlatlandPublicResponse{OK: err == nil, Percept: percept, Trace: trace, Err: err}
	case FlatlandPublicActMessage:
		fitness, end, trace, err := p.act(ctx, msg.AgentID, msg.ActuatorName, msg.Output)
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
	p.runtime.lastUpdate = FlatlandPublicUpdateSummary{}
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
	p.runtime.lastUpdate = FlatlandPublicUpdateSummary{}
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
		id:          agentID,
		mode:        mode,
		neuronCount: max(agent.NeuronCount, 0),
		episode:     newFlatlandEpisodeForAgent(cfg, agentID),
		decide:      agent.Decide,
	}
	return nil
}

func (p *FlatlandPublicProcess) updateAgents(agents []FlatlandPublicAgent) (FlatlandPublicUpdateSummary, []FlatlandPublicAvatarSnapshot, error) {
	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	if !p.runtime.started {
		return FlatlandPublicUpdateSummary{}, nil, fmt.Errorf("flatland public world is not started")
	}
	summary, err := p.runtime.updateAgentsLocked(agents)
	if err != nil {
		return FlatlandPublicUpdateSummary{}, nil, err
	}
	return summary, p.runtime.avatarSnapshots(), nil
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

func (p *FlatlandPublicProcess) agents() ([]Trace, []FlatlandPublicAvatarSnapshot, error) {
	p.runtime.mu.RLock()
	defer p.runtime.mu.RUnlock()
	if !p.runtime.started {
		return nil, nil, fmt.Errorf("flatland public world is not started")
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
	return out, p.runtime.avatarSnapshots(), nil
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

func (p *FlatlandPublicProcess) act(ctx context.Context, agentID string, actuatorName string, output []float64) (Fitness, bool, Trace, error) {
	agentID = strings.TrimSpace(agentID)
	if agentID == "" {
		return 0, false, nil, fmt.Errorf("flatland public agent id is required")
	}
	if err := ctx.Err(); err != nil {
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
		trace["reference_fitness"] = 0.0
		trace["shaped_fitness"] = float64(flatlandPublicShapedFitness(state.episode))
		trace["end"] = true
		return 0, true, trace, nil
	}

	previousKills := flatlandReferenceKills(state.episode)
	if action := flatlandApplyStateActuator(state, actuatorName, output); action.handled {
		terminalReason := action.terminalReason
		if terminalReason != "" {
			state.terminated = true
		}
		publicContact := flatlandPublicContactResult{}
		if action.attackReach > 0 {
			publicContact = p.runtime.resolvePublicAgentContact(agentID, action.attackReach, action.attackEnergyCredit)
		}
		referenceFitness := flatlandActuatorFeedback(!state.terminated, previousKills)
		trace := flatlandPublicAgentTrace(state)
		trace["move_step"] = 0
		trace["hit_food"] = false
		trace["hit_poison"] = false
		trace["spear_prey_hit"] = action.spearPreyHit
		trace["spear_predator_hit"] = action.spearPredatorHit
		trace["shot_fired"] = action.shotFired
		trace["shoot_prey_hit"] = action.shootPreyHit
		trace["shoot_predator_hit"] = action.shootPredatorHit
		trace["public_agent_collision"] = publicContact.collision
		trace["public_agent_target_id"] = publicContact.targetID
		trace["public_agent_target_terminated"] = publicContact.targetTerminated
		trace["wall_collision"] = false
		trace["terminal_reason"] = terminalReason
		trace["control_surface"] = strings.TrimSpace(actuatorName)
		trace["last_control_width"] = len(output)
		trace["reference_fitness"] = float64(referenceFitness)
		trace["shaped_fitness"] = float64(flatlandPublicShapedFitness(state.episode))
		trace["end"] = state.terminated
		return referenceFitness, state.terminated, trace, nil
	}
	control, err := flatlandControlFromActuatorOutput(actuatorName, output)
	if err != nil {
		return 0, false, nil, err
	}
	moveStep, hitFood, hitPoison, wallCollision, reason := state.episode.stepControl(control)
	publicContact := p.runtime.resolvePublicAgentContact(agentID, 0, 0)
	if reason != "" {
		state.terminated = true
	}
	referenceFitness := flatlandActuatorFeedback(!state.terminated, previousKills)
	shapedFitness := flatlandPublicShapedFitness(state.episode)
	trace := flatlandPublicAgentTrace(state)
	trace["move_step"] = moveStep
	trace["hit_food"] = hitFood
	trace["hit_poison"] = hitPoison
	trace["wall_collision"] = wallCollision
	trace["public_agent_collision"] = publicContact.collision
	trace["public_agent_target_id"] = publicContact.targetID
	trace["public_agent_target_terminated"] = publicContact.targetTerminated
	trace["terminal_reason"] = reason
	trace["control_surface"] = "step_output"
	trace["last_control_width"] = control.width
	trace["reference_fitness"] = float64(referenceFitness)
	trace["shaped_fitness"] = float64(shapedFitness)
	trace["end"] = state.terminated

	return referenceFitness, state.terminated, trace, nil
}

type flatlandStateActuatorResult struct {
	handled            bool
	terminalReason     string
	spearPreyHit       bool
	spearPredatorHit   bool
	shotFired          bool
	shootPreyHit       bool
	shootPredatorHit   bool
	attackReach        int
	attackEnergyCredit float64
}

func flatlandApplyStateActuator(state *flatlandPublicAgentState, actuatorName string, output []float64) flatlandStateActuatorResult {
	if state == nil {
		return flatlandStateActuatorResult{}
	}
	switch protoio.CanonicalActuatorName(actuatorName) {
	case protoio.FlatlandSpeakActuatorName:
		if len(output) == 0 {
			state.sound = 0
			return flatlandStateActuatorResult{handled: true}
		}
		state.sound = output[0]
		return flatlandStateActuatorResult{handled: true}
	case protoio.FlatlandGestaltActuatorName:
		state.gestalt = append([]float64(nil), output...)
		return flatlandStateActuatorResult{handled: true}
	case protoio.FlatlandSpearActuatorName:
		if len(output) == 0 || output[0] <= 0 {
			state.spear = false
			return flatlandStateActuatorResult{handled: true}
		}
		spearPreyHit := false
		spearPredatorHit := false
		if state.episode.energy > 100 {
			state.episode.energy -= 10
			state.spear = true
			spearPreyHit, spearPredatorHit = state.episode.spearForwardContact()
		} else {
			state.episode.energy -= 1
			state.spear = false
		}
		if state.episode.energy <= 0 {
			state.episode.energy = 0
			return flatlandStateActuatorResult{handled: true, terminalReason: "depleted", spearPreyHit: spearPreyHit, spearPredatorHit: spearPredatorHit}
		}
		return flatlandStateActuatorResult{
			handled:            true,
			spearPreyHit:       spearPreyHit,
			spearPredatorHit:   spearPredatorHit,
			attackReach:        flatlandSpearReach,
			attackEnergyCredit: flatlandSpearPreyEnergyCredit,
		}
	case protoio.FlatlandShootActuatorName:
		if len(output) == 0 || output[0] <= 0 {
			return flatlandStateActuatorResult{handled: true}
		}
		result := flatlandStateActuatorResult{handled: true}
		if state.episode.energy > 100 {
			state.episode.energy -= 20
			result.shotFired = true
			result.shootPreyHit, result.shootPredatorHit = state.episode.shootForwardContact()
			result.attackReach = flatlandShootReach
			result.attackEnergyCredit = flatlandShootPreyEnergyCredit
		} else {
			state.episode.energy -= 1
		}
		if state.episode.energy <= 0 {
			state.episode.energy = 0
			result.terminalReason = "depleted"
		}
		return result
	case protoio.FlatlandCreateOffspringActuatorName:
		state.offspringRequested = false
		state.offspringGranted = false
		state.offspringCost = 0
		state.offspringParentID = ""
		if len(output) == 0 || output[0] <= 0 {
			return flatlandStateActuatorResult{handled: true}
		}
		offspringCost := float64(max(state.neuronCount, 0)) * flatlandNeuralCost
		totalGrantCost := offspringCost + 1000
		state.offspringRequested = true
		state.offspringParentID = state.id
		if state.episode.energy > totalGrantCost {
			state.episode.energy -= totalGrantCost
			state.offspringGranted = true
			state.offspringCost = totalGrantCost
		} else {
			state.episode.energy -= 50
			state.offspringCost = 50
		}
		if state.episode.energy <= 0 {
			state.episode.energy = 0
			return flatlandStateActuatorResult{handled: true, terminalReason: "depleted"}
		}
		return flatlandStateActuatorResult{handled: true}
	default:
		return flatlandStateActuatorResult{}
	}
}

func (p *FlatlandPublicProcess) tick(ctx context.Context) (Trace, []FlatlandPublicAvatarSnapshot, error) {
	p.runtime.mu.Lock()
	defer p.runtime.mu.Unlock()
	if !p.runtime.started {
		return nil, nil, fmt.Errorf("flatland public world is not started")
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
	totalPublicCollisions := 0
	totalPublicKills := 0
	totalPublicDeaths := 0
	totalSpearKills := 0
	totalShootKills := 0
	agentStates := make([]Trace, 0, len(ids))

	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		state := p.runtime.agents[id]
		if state.terminated {
			totalEnergy += state.episode.energy
			totalFood += state.episode.foodCollected
			totalPrey += state.episode.preyCollected
			totalPredatorHits += state.episode.predatorHits
			totalPublicCollisions += state.episode.publicAgentCollisions
			totalPublicKills += state.episode.publicAgentKills
			totalPublicDeaths += state.episode.publicAgentDeaths
			totalSpearKills += state.episode.spearKills
			totalShootKills += state.episode.shootKills
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
			return nil, nil, err
		}
		_, _, _, _, reason := state.episode.step(control.move)
		p.runtime.resolvePublicAgentContact(id, 0, 0)
		if reason != "" {
			state.terminated = true
			terminated++
		}
		totalEnergy += state.episode.energy
		totalFood += state.episode.foodCollected
		totalPrey += state.episode.preyCollected
		totalPredatorHits += state.episode.predatorHits
		totalPublicCollisions += state.episode.publicAgentCollisions
		totalPublicKills += state.episode.publicAgentKills
		totalPublicDeaths += state.episode.publicAgentDeaths
		totalSpearKills += state.episode.spearKills
		totalShootKills += state.episode.shootKills
		agentStates = append(agentStates, flatlandPublicAgentTrace(state))
	}

	p.runtime.tick++
	avgEnergy := 0.0
	if len(ids) > 0 {
		avgEnergy = totalEnergy / float64(len(ids))
	}
	avatars := p.runtime.avatarSnapshots()
	return Trace{
		"tick":                          p.runtime.tick,
		"active_agents":                 len(ids) - terminated,
		"terminated_agents":             terminated,
		"total_agents":                  len(ids),
		"avg_energy":                    avgEnergy,
		"total_food":                    totalFood,
		"total_prey":                    totalPrey,
		"total_predator_hits":           totalPredatorHits,
		"total_public_agent_collisions": totalPublicCollisions,
		"total_public_agent_kills":      totalPublicKills,
		"total_public_agent_deaths":     totalPublicDeaths,
		"total_spear_kills":             totalSpearKills,
		"total_shoot_kills":             totalShootKills,
		"agents":                        agentStates,
		"avatars":                       avatars,
	}, avatars, nil
}
