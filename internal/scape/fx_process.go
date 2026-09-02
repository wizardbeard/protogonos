package scape

import (
	"context"
	"fmt"
	"strings"
)

type FXMessage interface {
	isFXMessage()
}

type FXStartMessage struct {
	Mode string
}

func (FXStartMessage) isFXMessage() {}

type FXStopMessage struct {
	Reason string
}

func (FXStopMessage) isFXMessage() {}

type FXRestartMessage struct{}

func (FXRestartMessage) isFXMessage() {}

type FXSenseMessage struct{}

func (FXSenseMessage) isFXMessage() {}

type FXInternalsMessage struct{}

func (FXInternalsMessage) isFXMessage() {}

type FXTradeMessage struct {
	Action float64
}

func (FXTradeMessage) isFXMessage() {}

type FXStateMessage struct{}

func (FXStateMessage) isFXMessage() {}

type FXResponse struct {
	OK         bool
	Percept    []float64
	Internals  []float64
	Fitness    Fitness
	End        bool
	State      FXSimulatorState
	StopReason string
	Err        error
}

type FXProcess struct {
	sim        *FXSimulator
	mode       string
	stopReason string
}

func NewFXProcess() *FXProcess {
	return &FXProcess{}
}

func (p *FXProcess) Call(ctx context.Context, message FXMessage) FXResponse {
	if p == nil || message == nil {
		return FXResponse{Err: fmt.Errorf("invalid fx process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return FXResponse{Err: err}
	}

	switch msg := message.(type) {
	case FXStartMessage:
		return p.start(ctx, msg.Mode)
	case FXStopMessage:
		return p.stop(msg.Reason)
	case FXRestartMessage:
		return p.restart(ctx)
	case FXSenseMessage:
		return p.sense(ctx)
	case FXInternalsMessage:
		return p.internals(ctx)
	case FXTradeMessage:
		return p.trade(ctx, msg.Action)
	case FXStateMessage:
		return p.state()
	default:
		return FXResponse{Err: fmt.Errorf("unsupported fx process message %T", message)}
	}
}

func (p *FXProcess) start(ctx context.Context, mode string) FXResponse {
	sim, err := NewFXSimulator(ctx, mode)
	if err != nil {
		return FXResponse{Err: err}
	}
	p.sim = sim
	p.mode = sim.State().Mode
	p.stopReason = ""
	return FXResponse{OK: true, State: sim.State()}
}

func (p *FXProcess) stop(reason string) FXResponse {
	if p.sim == nil {
		return FXResponse{Err: fmt.Errorf("fx process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopReason = reason
	p.sim.halted = true
	p.sim.terminationReason = reason
	return FXResponse{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *FXProcess) restart(ctx context.Context) FXResponse {
	if p.sim == nil {
		return p.start(ctx, p.mode)
	}
	if err := ctx.Err(); err != nil {
		return FXResponse{Err: err}
	}
	p.sim.Restart()
	p.stopReason = ""
	return FXResponse{OK: true, State: p.sim.State()}
}

func (p *FXProcess) sense(ctx context.Context) FXResponse {
	if p.sim == nil {
		return FXResponse{Err: fmt.Errorf("fx process is not started")}
	}
	percept, err := p.sim.Sense(ctx)
	if err != nil {
		return FXResponse{Err: err}
	}
	return FXResponse{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *FXProcess) internals(ctx context.Context) FXResponse {
	if p.sim == nil {
		return FXResponse{Err: fmt.Errorf("fx process is not started")}
	}
	internals, err := p.sim.Internals(ctx)
	if err != nil {
		return FXResponse{Err: err}
	}
	return FXResponse{OK: true, Internals: internals, State: p.sim.State()}
}

func (p *FXProcess) trade(ctx context.Context, action float64) FXResponse {
	if p.sim == nil {
		return FXResponse{Err: fmt.Errorf("fx process is not started")}
	}
	fitness, end, err := p.sim.Trade(ctx, action)
	if err != nil {
		return FXResponse{Err: err}
	}
	return FXResponse{OK: true, Fitness: fitness, End: end, State: p.sim.State()}
}

func (p *FXProcess) state() FXResponse {
	if p.sim == nil {
		return FXResponse{Err: fmt.Errorf("fx process is not started")}
	}
	return FXResponse{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
