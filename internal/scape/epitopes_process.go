package scape

import (
	"context"
	"fmt"
	"strings"
)

type EpitopesMessage interface {
	isEpitopesMessage()
}

type EpitopesStartMessage struct {
	OpMode string
	Params EpitopesSimParameters
}

func (EpitopesStartMessage) isEpitopesMessage() {}

type EpitopesStopMessage struct {
	Reason string
}

func (EpitopesStopMessage) isEpitopesMessage() {}

type EpitopesRestartMessage struct{}

func (EpitopesRestartMessage) isEpitopesMessage() {}

type EpitopesSenseMessage struct{}

func (EpitopesSenseMessage) isEpitopesMessage() {}

type EpitopesClassifyMessage struct {
	Output []float64
}

func (EpitopesClassifyMessage) isEpitopesMessage() {}

type EpitopesStateMessage struct{}

func (EpitopesStateMessage) isEpitopesMessage() {}

type EpitopesResponse struct {
	OK         bool
	Percept    []float64
	Reward     int
	End        bool
	State      EpitopesSimState
	StopReason string
	Err        error
}

type EpitopesProcess struct {
	sim        *EpitopesSimulator
	opMode     string
	params     EpitopesSimParameters
	stopped    bool
	stopReason string
}

func NewEpitopesProcess() *EpitopesProcess {
	return &EpitopesProcess{}
}

func (p *EpitopesProcess) Call(ctx context.Context, message EpitopesMessage) EpitopesResponse {
	if p == nil || message == nil {
		return EpitopesResponse{Err: fmt.Errorf("invalid epitopes process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return EpitopesResponse{Err: err}
	}

	switch msg := message.(type) {
	case EpitopesStartMessage:
		return p.start(ctx, msg.OpMode, msg.Params)
	case EpitopesStopMessage:
		return p.stop(msg.Reason)
	case EpitopesRestartMessage:
		return p.restart(ctx)
	case EpitopesSenseMessage:
		return p.sense()
	case EpitopesClassifyMessage:
		return p.classify(msg.Output)
	case EpitopesStateMessage:
		return p.state()
	default:
		return EpitopesResponse{Err: fmt.Errorf("unsupported epitopes process message %T", message)}
	}
}

func (p *EpitopesProcess) start(ctx context.Context, opMode string, params EpitopesSimParameters) EpitopesResponse {
	sim, err := NewEpitopesSimulator(ctx, opMode, params)
	if err != nil {
		return EpitopesResponse{Err: err}
	}
	p.sim = sim
	p.opMode = sim.State().OpMode
	p.params = params
	p.stopped = false
	p.stopReason = ""
	return EpitopesResponse{OK: true, State: sim.State()}
}

func (p *EpitopesProcess) stop(reason string) EpitopesResponse {
	if p.sim == nil || p.sim.session == nil {
		return EpitopesResponse{Err: fmt.Errorf("epitopes process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopReason = reason
	p.stopped = true
	p.sim.session.halted = true
	p.sim.terminationReason = reason
	return EpitopesResponse{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *EpitopesProcess) restart(ctx context.Context) EpitopesResponse {
	if p.sim == nil {
		return p.start(ctx, p.opMode, p.params)
	}
	return p.start(ctx, p.opMode, p.params)
}

func (p *EpitopesProcess) sense() EpitopesResponse {
	if p.sim == nil {
		return EpitopesResponse{Err: fmt.Errorf("epitopes process is not started")}
	}
	if p.stopped {
		return EpitopesResponse{Err: fmt.Errorf("epitopes process is stopped")}
	}
	percept, err := p.sim.Sense()
	if err != nil {
		return EpitopesResponse{Err: err}
	}
	return EpitopesResponse{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *EpitopesProcess) classify(output []float64) EpitopesResponse {
	if p.sim == nil {
		return EpitopesResponse{Err: fmt.Errorf("epitopes process is not started")}
	}
	if p.stopped {
		return EpitopesResponse{Err: fmt.Errorf("epitopes process is stopped")}
	}
	reward, halt, err := p.sim.Classify(output)
	if err != nil {
		return EpitopesResponse{Err: err}
	}
	return EpitopesResponse{OK: true, Reward: reward, End: halt, State: p.sim.State()}
}

func (p *EpitopesProcess) state() EpitopesResponse {
	if p.sim == nil {
		return EpitopesResponse{Err: fmt.Errorf("epitopes process is not started")}
	}
	return EpitopesResponse{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
