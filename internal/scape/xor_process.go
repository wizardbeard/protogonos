package scape

import (
	"context"
	"fmt"
	"strings"
)

type XORMessage interface {
	isXORMessage()
}

type XORStartMessage struct {
	Mode string
}

func (XORStartMessage) isXORMessage() {}

type XORStopMessage struct {
	Reason string
}

func (XORStopMessage) isXORMessage() {}

type XORRestartMessage struct{}

func (XORRestartMessage) isXORMessage() {}

type XORSenseMessage struct{}

func (XORSenseMessage) isXORMessage() {}

type XORPredictMessage struct {
	Output []float64
}

func (XORPredictMessage) isXORMessage() {}

type XORStateMessage struct{}

func (XORStateMessage) isXORMessage() {}

type XORResponse struct {
	OK         bool
	Percept    []float64
	Fitness    Fitness
	End        bool
	State      XORSimulatorState
	StopReason string
	Err        error
}

type XORProcess struct {
	sim        *XORSimulator
	mode       string
	stopped    bool
	stopReason string
}

func NewXORProcess() *XORProcess {
	return &XORProcess{}
}

func (p *XORProcess) Call(ctx context.Context, message XORMessage) XORResponse {
	if p == nil || message == nil {
		return XORResponse{Err: fmt.Errorf("invalid xor process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return XORResponse{Err: err}
	}

	switch msg := message.(type) {
	case XORStartMessage:
		return p.start(msg.Mode)
	case XORStopMessage:
		return p.stop(msg.Reason)
	case XORRestartMessage:
		return p.restart()
	case XORSenseMessage:
		return p.sense(ctx)
	case XORPredictMessage:
		return p.predict(ctx, msg.Output)
	case XORStateMessage:
		return p.state()
	default:
		return XORResponse{Err: fmt.Errorf("unsupported xor process message %T", message)}
	}
}

func (p *XORProcess) start(mode string) XORResponse {
	sim, err := NewXORSimulator(mode)
	if err != nil {
		return XORResponse{Err: err}
	}
	p.sim = sim
	p.mode = sim.State().Mode
	p.stopped = false
	p.stopReason = ""
	return XORResponse{OK: true, State: sim.State()}
}

func (p *XORProcess) stop(reason string) XORResponse {
	if p.sim == nil {
		return XORResponse{Err: fmt.Errorf("xor process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopped = true
	p.stopReason = reason
	return XORResponse{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *XORProcess) restart() XORResponse {
	if p.sim == nil {
		return p.start(p.mode)
	}
	p.sim.Reset()
	p.stopped = false
	p.stopReason = ""
	return XORResponse{OK: true, State: p.sim.State()}
}

func (p *XORProcess) sense(ctx context.Context) XORResponse {
	if p.sim == nil {
		return XORResponse{Err: fmt.Errorf("xor process is not started")}
	}
	if p.stopped {
		return XORResponse{Err: fmt.Errorf("xor process is stopped")}
	}
	percept, err := p.sim.Sense(ctx)
	if err != nil {
		return XORResponse{Err: err}
	}
	return XORResponse{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *XORProcess) predict(ctx context.Context, output []float64) XORResponse {
	if p.sim == nil {
		return XORResponse{Err: fmt.Errorf("xor process is not started")}
	}
	if p.stopped {
		return XORResponse{Err: fmt.Errorf("xor process is stopped")}
	}
	fitness, end, err := p.sim.Predict(ctx, output)
	if err != nil {
		return XORResponse{Err: err}
	}
	return XORResponse{OK: true, Fitness: fitness, End: end, State: p.sim.State()}
}

func (p *XORProcess) state() XORResponse {
	if p.sim == nil {
		return XORResponse{Err: fmt.Errorf("xor process is not started")}
	}
	return XORResponse{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
