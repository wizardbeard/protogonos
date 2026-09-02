package scape

import (
	"context"
	"fmt"
	"strings"
)

type CartPoleLiteMessage interface {
	isCartPoleLiteMessage()
}

type CartPoleLiteStartMessage struct {
	Mode string
}

func (CartPoleLiteStartMessage) isCartPoleLiteMessage() {}

type CartPoleLiteStopMessage struct {
	Reason string
}

func (CartPoleLiteStopMessage) isCartPoleLiteMessage() {}

type CartPoleLiteRestartMessage struct{}

func (CartPoleLiteRestartMessage) isCartPoleLiteMessage() {}

type CartPoleLiteSenseMessage struct{}

func (CartPoleLiteSenseMessage) isCartPoleLiteMessage() {}

type CartPoleLitePushMessage struct {
	Output []float64
}

func (CartPoleLitePushMessage) isCartPoleLiteMessage() {}

type CartPoleLiteStateMessage struct{}

func (CartPoleLiteStateMessage) isCartPoleLiteMessage() {}

type CartPoleLiteResponse struct {
	OK         bool
	Percept    []float64
	Fitness    Fitness
	End        bool
	State      CartPoleLiteSimulatorState
	StopReason string
	Err        error
}

type CartPoleLiteProcess struct {
	sim        *CartPoleLiteSimulator
	mode       string
	stopReason string
}

func NewCartPoleLiteProcess() *CartPoleLiteProcess {
	return &CartPoleLiteProcess{}
}

func (p *CartPoleLiteProcess) Call(ctx context.Context, message CartPoleLiteMessage) CartPoleLiteResponse {
	if p == nil || message == nil {
		return CartPoleLiteResponse{Err: fmt.Errorf("invalid cart-pole-lite process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return CartPoleLiteResponse{Err: err}
	}

	switch msg := message.(type) {
	case CartPoleLiteStartMessage:
		return p.start(msg.Mode)
	case CartPoleLiteStopMessage:
		return p.stop(msg.Reason)
	case CartPoleLiteRestartMessage:
		return p.restart()
	case CartPoleLiteSenseMessage:
		return p.sense(ctx)
	case CartPoleLitePushMessage:
		return p.push(ctx, msg.Output)
	case CartPoleLiteStateMessage:
		return p.state()
	default:
		return CartPoleLiteResponse{Err: fmt.Errorf("unsupported cart-pole-lite process message %T", message)}
	}
}

func (p *CartPoleLiteProcess) start(mode string) CartPoleLiteResponse {
	sim, err := NewCartPoleLiteSimulator(mode)
	if err != nil {
		return CartPoleLiteResponse{Err: err}
	}
	p.sim = sim
	p.mode = sim.State().Mode
	p.stopReason = ""
	return CartPoleLiteResponse{OK: true, State: sim.State()}
}

func (p *CartPoleLiteProcess) stop(reason string) CartPoleLiteResponse {
	if p.sim == nil {
		return CartPoleLiteResponse{Err: fmt.Errorf("cart-pole-lite process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopReason = reason
	p.sim.halted = true
	p.sim.terminationReason = reason
	return CartPoleLiteResponse{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *CartPoleLiteProcess) restart() CartPoleLiteResponse {
	if p.sim == nil {
		return p.start(p.mode)
	}
	p.sim.Reset()
	p.stopReason = ""
	return CartPoleLiteResponse{OK: true, State: p.sim.State()}
}

func (p *CartPoleLiteProcess) sense(ctx context.Context) CartPoleLiteResponse {
	if p.sim == nil {
		return CartPoleLiteResponse{Err: fmt.Errorf("cart-pole-lite process is not started")}
	}
	percept, err := p.sim.Sense(ctx)
	if err != nil {
		return CartPoleLiteResponse{Err: err}
	}
	return CartPoleLiteResponse{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *CartPoleLiteProcess) push(ctx context.Context, output []float64) CartPoleLiteResponse {
	if p.sim == nil {
		return CartPoleLiteResponse{Err: fmt.Errorf("cart-pole-lite process is not started")}
	}
	fitness, end, err := p.sim.Push(ctx, output)
	if err != nil {
		return CartPoleLiteResponse{Err: err}
	}
	return CartPoleLiteResponse{OK: true, Fitness: fitness, End: end, State: p.sim.State()}
}

func (p *CartPoleLiteProcess) state() CartPoleLiteResponse {
	if p.sim == nil {
		return CartPoleLiteResponse{Err: fmt.Errorf("cart-pole-lite process is not started")}
	}
	return CartPoleLiteResponse{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
