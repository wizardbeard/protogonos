package scape

import (
	"context"
	"fmt"
	"strings"
)

type DTMMessage interface {
	isDTMMessage()
}

type DTMStartMessage struct {
	Mode string
}

func (DTMStartMessage) isDTMMessage() {}

type DTMStopMessage struct {
	Reason string
}

func (DTMStopMessage) isDTMMessage() {}

type DTMRestartMessage struct{}

func (DTMRestartMessage) isDTMMessage() {}

type DTMSenseMessage struct {
	Parameter string
}

func (DTMSenseMessage) isDTMMessage() {}

type DTMMoveMessage struct {
	Output []float64
}

func (DTMMoveMessage) isDTMMessage() {}

type DTMStateMessage struct{}

func (DTMStateMessage) isDTMMessage() {}

type DTMResponse struct {
	OK         bool
	Percept    []float64
	Fitness    Fitness
	End        bool
	State      DTMSimulatorState
	StopReason string
	Err        error
}

type DTMProcess struct {
	sim        *DTMSimulator
	mode       string
	stopReason string
}

func NewDTMProcess() *DTMProcess {
	return &DTMProcess{}
}

func (p *DTMProcess) Call(ctx context.Context, message DTMMessage) DTMResponse {
	if p == nil || message == nil {
		return DTMResponse{Err: fmt.Errorf("invalid dtm process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return DTMResponse{Err: err}
	}

	switch msg := message.(type) {
	case DTMStartMessage:
		return p.start(msg.Mode)
	case DTMStopMessage:
		return p.stop(msg.Reason)
	case DTMRestartMessage:
		return p.restart()
	case DTMSenseMessage:
		return p.sense(ctx, msg.Parameter)
	case DTMMoveMessage:
		return p.move(ctx, msg.Output)
	case DTMStateMessage:
		return p.state()
	default:
		return DTMResponse{Err: fmt.Errorf("unsupported dtm process message %T", message)}
	}
}

func (p *DTMProcess) start(mode string) DTMResponse {
	sim, err := NewDTMSimulator(mode)
	if err != nil {
		return DTMResponse{Err: err}
	}
	p.sim = sim
	p.mode = sim.State().Mode
	p.stopReason = ""
	return DTMResponse{OK: true, State: sim.State()}
}

func (p *DTMProcess) stop(reason string) DTMResponse {
	if p.sim == nil {
		return DTMResponse{Err: fmt.Errorf("dtm process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopReason = reason
	p.sim.halted = true
	p.sim.terminationReason = reason
	return DTMResponse{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *DTMProcess) restart() DTMResponse {
	if p.sim == nil {
		return p.start(p.mode)
	}
	p.sim.Reset()
	p.stopReason = ""
	return DTMResponse{OK: true, State: p.sim.State()}
}

func (p *DTMProcess) sense(ctx context.Context, parameter string) DTMResponse {
	if p.sim == nil {
		return DTMResponse{Err: fmt.Errorf("dtm process is not started")}
	}
	percept, err := p.sim.Sense(ctx, parameter)
	if err != nil {
		return DTMResponse{Err: err}
	}
	return DTMResponse{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *DTMProcess) move(ctx context.Context, output []float64) DTMResponse {
	if p.sim == nil {
		return DTMResponse{Err: fmt.Errorf("dtm process is not started")}
	}
	fitness, end, err := p.sim.Move(ctx, output)
	if err != nil {
		return DTMResponse{Err: err}
	}
	return DTMResponse{OK: true, Fitness: fitness, End: end, State: p.sim.State()}
}

func (p *DTMProcess) state() DTMResponse {
	if p.sim == nil {
		return DTMResponse{Err: fmt.Errorf("dtm process is not started")}
	}
	return DTMResponse{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
