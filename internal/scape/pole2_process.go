package scape

import (
	"context"
	"fmt"
	"strings"
)

type Pole2Message interface {
	isPole2Message()
}

type Pole2StartMessage struct {
	Mode string
}

func (Pole2StartMessage) isPole2Message() {}

type Pole2StopMessage struct {
	Reason string
}

func (Pole2StopMessage) isPole2Message() {}

type Pole2RestartMessage struct{}

func (Pole2RestartMessage) isPole2Message() {}

type Pole2SenseMessage struct {
	Parameter string
}

func (Pole2SenseMessage) isPole2Message() {}

type Pole2PushMessage struct {
	Output []float64
}

func (Pole2PushMessage) isPole2Message() {}

type Pole2StateMessage struct{}

func (Pole2StateMessage) isPole2Message() {}

type Pole2Response struct {
	OK         bool
	Percept    []float64
	Fitness    Fitness
	End        bool
	State      Pole2SimulatorState
	StopReason string
	Err        error
}

type Pole2Process struct {
	sim        *Pole2Simulator
	mode       string
	stopReason string
}

func NewPole2Process() *Pole2Process {
	return &Pole2Process{}
}

func (p *Pole2Process) Call(ctx context.Context, message Pole2Message) Pole2Response {
	if p == nil || message == nil {
		return Pole2Response{Err: fmt.Errorf("invalid pole2 process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return Pole2Response{Err: err}
	}

	switch msg := message.(type) {
	case Pole2StartMessage:
		return p.start(msg.Mode)
	case Pole2StopMessage:
		return p.stop(msg.Reason)
	case Pole2RestartMessage:
		return p.restart()
	case Pole2SenseMessage:
		return p.sense(ctx, msg.Parameter)
	case Pole2PushMessage:
		return p.push(ctx, msg.Output)
	case Pole2StateMessage:
		return p.state()
	default:
		return Pole2Response{Err: fmt.Errorf("unsupported pole2 process message %T", message)}
	}
}

func (p *Pole2Process) start(mode string) Pole2Response {
	sim, err := NewPole2Simulator(mode)
	if err != nil {
		return Pole2Response{Err: err}
	}
	p.sim = sim
	p.mode = sim.State().Mode
	p.stopReason = ""
	return Pole2Response{OK: true, State: sim.State()}
}

func (p *Pole2Process) stop(reason string) Pole2Response {
	if p.sim == nil {
		return Pole2Response{Err: fmt.Errorf("pole2 process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopReason = reason
	p.sim.halted = true
	p.sim.terminationReason = reason
	return Pole2Response{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *Pole2Process) restart() Pole2Response {
	if p.sim == nil {
		return p.start(p.mode)
	}
	p.sim.Reset()
	p.stopReason = ""
	return Pole2Response{OK: true, State: p.sim.State()}
}

func (p *Pole2Process) sense(ctx context.Context, parameter string) Pole2Response {
	if p.sim == nil {
		return Pole2Response{Err: fmt.Errorf("pole2 process is not started")}
	}
	percept, err := p.sim.Sense(ctx, parameter)
	if err != nil {
		return Pole2Response{Err: err}
	}
	return Pole2Response{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *Pole2Process) push(ctx context.Context, output []float64) Pole2Response {
	if p.sim == nil {
		return Pole2Response{Err: fmt.Errorf("pole2 process is not started")}
	}
	fitness, end, err := p.sim.Push(ctx, output)
	if err != nil {
		return Pole2Response{Err: err}
	}
	return Pole2Response{OK: true, Fitness: fitness, End: end, State: p.sim.State()}
}

func (p *Pole2Process) state() Pole2Response {
	if p.sim == nil {
		return Pole2Response{Err: fmt.Errorf("pole2 process is not started")}
	}
	return Pole2Response{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
