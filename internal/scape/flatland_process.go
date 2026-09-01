package scape

import (
	"context"
	"fmt"
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
	scape FlatlandScape
}

func NewFlatlandPublicProcess() *FlatlandPublicProcess {
	return &FlatlandPublicProcess{scape: FlatlandScape{}}
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

	switch msg := message.(type) {
	case FlatlandPublicStartMessage:
		err := p.scape.Start(ctx)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicStopMessage:
		err := p.scape.StopWithReason(ctx, msg.Reason)
		return FlatlandPublicResponse{OK: err == nil, StopReason: p.scape.LastPublicStopReason(), Err: err}
	case FlatlandPublicSyncMessage:
		err := p.scape.Sync(ctx)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicEnterMessage:
		err := p.scape.EnterPublicAgent(msg.Agent)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicLeaveMessage:
		err := p.scape.LeavePublicAgent(msg.AgentID)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicUpdateAgentsMessage:
		err := p.scape.UpdatePublicAgents(msg.Agents)
		return FlatlandPublicResponse{OK: err == nil, Err: err}
	case FlatlandPublicGetAllMessage:
		agents, err := p.scape.PublicAgents()
		return FlatlandPublicResponse{OK: err == nil, Agents: agents, Err: err}
	case FlatlandPublicTickMessage:
		trace, err := p.scape.TickPublic(ctx)
		return FlatlandPublicResponse{OK: err == nil, Trace: trace, Err: err}
	case FlatlandPublicSenseMessage:
		percept, trace, err := p.scape.SensePublicAgent(ctx, msg.AgentID)
		return FlatlandPublicResponse{OK: err == nil, Percept: percept, Trace: trace, Err: err}
	case FlatlandPublicActMessage:
		fitness, end, trace, err := p.scape.ActPublicAgent(ctx, msg.AgentID, msg.Output)
		return FlatlandPublicResponse{OK: err == nil, Fitness: fitness, End: end, Trace: trace, Err: err}
	default:
		return FlatlandPublicResponse{Err: fmt.Errorf("unsupported flatland public process message %T", message)}
	}
}
