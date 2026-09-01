package scape

import (
	"context"
	"fmt"
	"strings"
)

type GTSAMessage interface {
	isGTSAMessage()
}

type GTSAStartMessage struct {
	Mode string
}

func (GTSAStartMessage) isGTSAMessage() {}

type GTSAStopMessage struct {
	Reason string
}

func (GTSAStopMessage) isGTSAMessage() {}

type GTSARestartMessage struct{}

func (GTSARestartMessage) isGTSAMessage() {}

type GTSASenseMessage struct{}

func (GTSASenseMessage) isGTSAMessage() {}

type GTSASensePerceptMessage struct{}

func (GTSASensePerceptMessage) isGTSAMessage() {}

type GTSAPredictValueMessage struct {
	Prediction float64
}

func (GTSAPredictValueMessage) isGTSAMessage() {}

type GTSAStateMessage struct{}

func (GTSAStateMessage) isGTSAMessage() {}

type GTSAResponse struct {
	OK         bool
	Percept    []float64
	Current    float64
	Delta      float64
	WindowMean float64
	Progress   float64
	Fitness    Fitness
	End        bool
	State      GTSASimulatorState
	StopReason string
	Err        error
}

type GTSAProcess struct {
	sim        *GTSASimulator
	mode       string
	stopReason string
}

func NewGTSAProcess() *GTSAProcess {
	return &GTSAProcess{}
}

func (p *GTSAProcess) Call(ctx context.Context, message GTSAMessage) GTSAResponse {
	if p == nil || message == nil {
		return GTSAResponse{Err: fmt.Errorf("invalid gtsa process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return GTSAResponse{Err: err}
	}

	switch msg := message.(type) {
	case GTSAStartMessage:
		return p.start(ctx, msg.Mode)
	case GTSAStopMessage:
		return p.stop(msg.Reason)
	case GTSARestartMessage:
		return p.restart(ctx)
	case GTSASenseMessage:
		return p.sense(ctx)
	case GTSASensePerceptMessage:
		return p.sensePercept(ctx)
	case GTSAPredictValueMessage:
		return p.predictValue(ctx, msg.Prediction)
	case GTSAStateMessage:
		return p.state()
	default:
		return GTSAResponse{Err: fmt.Errorf("unsupported gtsa process message %T", message)}
	}
}

func (p *GTSAProcess) start(ctx context.Context, mode string) GTSAResponse {
	sim, err := NewGTSASimulator(ctx, mode)
	if err != nil {
		return GTSAResponse{Err: err}
	}
	p.sim = sim
	p.mode = sim.State().Mode
	p.stopReason = ""
	return GTSAResponse{OK: true, State: sim.State()}
}

func (p *GTSAProcess) stop(reason string) GTSAResponse {
	if p.sim == nil {
		return GTSAResponse{Err: fmt.Errorf("gtsa process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopReason = reason
	p.sim.halted = true
	return GTSAResponse{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *GTSAProcess) restart(ctx context.Context) GTSAResponse {
	if p.sim == nil {
		return p.start(ctx, p.mode)
	}
	if err := p.sim.Restart(ctx); err != nil {
		return GTSAResponse{Err: err}
	}
	p.stopReason = ""
	return GTSAResponse{OK: true, State: p.sim.State()}
}

func (p *GTSAProcess) sense(ctx context.Context) GTSAResponse {
	if p.sim == nil {
		return GTSAResponse{Err: fmt.Errorf("gtsa process is not started")}
	}
	percept, err := p.sim.Sense(ctx)
	if err != nil {
		return GTSAResponse{Err: err}
	}
	return GTSAResponse{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *GTSAProcess) sensePercept(ctx context.Context) GTSAResponse {
	if p.sim == nil {
		return GTSAResponse{Err: fmt.Errorf("gtsa process is not started")}
	}
	current, delta, windowMean, progress, percept, err := p.sim.SensePercept(ctx)
	if err != nil {
		return GTSAResponse{Err: err}
	}
	return GTSAResponse{
		OK:         true,
		Percept:    percept,
		Current:    current,
		Delta:      delta,
		WindowMean: windowMean,
		Progress:   progress,
		State:      p.sim.State(),
	}
}

func (p *GTSAProcess) predictValue(ctx context.Context, prediction float64) GTSAResponse {
	if p.sim == nil {
		return GTSAResponse{Err: fmt.Errorf("gtsa process is not started")}
	}
	fitness, end, err := p.sim.PredictValue(ctx, prediction)
	if err != nil {
		return GTSAResponse{Err: err}
	}
	return GTSAResponse{OK: true, Fitness: fitness, End: end, State: p.sim.State()}
}

func (p *GTSAProcess) state() GTSAResponse {
	if p.sim == nil {
		return GTSAResponse{Err: fmt.Errorf("gtsa process is not started")}
	}
	return GTSAResponse{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
