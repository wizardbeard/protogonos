package scape

import (
	"context"
	"fmt"
	"strings"
)

type LLVMPhaseOrderingMessage interface {
	isLLVMPhaseOrderingMessage()
}

type LLVMPhaseOrderingStartMessage struct {
	Mode string
}

func (LLVMPhaseOrderingStartMessage) isLLVMPhaseOrderingMessage() {}

type LLVMPhaseOrderingStopMessage struct {
	Reason string
}

func (LLVMPhaseOrderingStopMessage) isLLVMPhaseOrderingMessage() {}

type LLVMPhaseOrderingRestartMessage struct{}

func (LLVMPhaseOrderingRestartMessage) isLLVMPhaseOrderingMessage() {}

type LLVMPhaseOrderingSenseMessage struct {
	Parameter string
}

func (LLVMPhaseOrderingSenseMessage) isLLVMPhaseOrderingMessage() {}

type LLVMPhaseOrderingOptimizeMessage struct {
	Output []float64
}

func (LLVMPhaseOrderingOptimizeMessage) isLLVMPhaseOrderingMessage() {}

type LLVMPhaseOrderingStateMessage struct{}

func (LLVMPhaseOrderingStateMessage) isLLVMPhaseOrderingMessage() {}

type LLVMPhaseOrderingResponse struct {
	OK         bool
	Percept    []float64
	Fitness    Fitness
	End        bool
	State      LLVMPhaseOrderingSimulatorState
	StopReason string
	Err        error
}

type LLVMPhaseOrderingProcess struct {
	sim        *LLVMPhaseOrderingSimulator
	mode       string
	stopReason string
}

func NewLLVMPhaseOrderingProcess() *LLVMPhaseOrderingProcess {
	return &LLVMPhaseOrderingProcess{}
}

func (p *LLVMPhaseOrderingProcess) Call(ctx context.Context, message LLVMPhaseOrderingMessage) LLVMPhaseOrderingResponse {
	if p == nil || message == nil {
		return LLVMPhaseOrderingResponse{Err: fmt.Errorf("invalid llvm-phase-ordering process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return LLVMPhaseOrderingResponse{Err: err}
	}

	switch msg := message.(type) {
	case LLVMPhaseOrderingStartMessage:
		return p.start(ctx, msg.Mode)
	case LLVMPhaseOrderingStopMessage:
		return p.stop(msg.Reason)
	case LLVMPhaseOrderingRestartMessage:
		return p.restart(ctx)
	case LLVMPhaseOrderingSenseMessage:
		return p.sense(ctx, msg.Parameter)
	case LLVMPhaseOrderingOptimizeMessage:
		return p.optimize(ctx, msg.Output)
	case LLVMPhaseOrderingStateMessage:
		return p.state()
	default:
		return LLVMPhaseOrderingResponse{Err: fmt.Errorf("unsupported llvm-phase-ordering process message %T", message)}
	}
}

func (p *LLVMPhaseOrderingProcess) start(ctx context.Context, mode string) LLVMPhaseOrderingResponse {
	sim, err := NewLLVMPhaseOrderingSimulator(ctx, mode)
	if err != nil {
		return LLVMPhaseOrderingResponse{Err: err}
	}
	p.sim = sim
	p.mode = sim.State().Mode
	p.stopReason = ""
	return LLVMPhaseOrderingResponse{OK: true, State: sim.State()}
}

func (p *LLVMPhaseOrderingProcess) stop(reason string) LLVMPhaseOrderingResponse {
	if p.sim == nil {
		return LLVMPhaseOrderingResponse{Err: fmt.Errorf("llvm-phase-ordering process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopReason = reason
	p.sim.halted = true
	p.sim.terminationReason = reason
	return LLVMPhaseOrderingResponse{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *LLVMPhaseOrderingProcess) restart(ctx context.Context) LLVMPhaseOrderingResponse {
	if p.sim == nil {
		return p.start(ctx, p.mode)
	}
	p.sim.Reset()
	p.stopReason = ""
	return LLVMPhaseOrderingResponse{OK: true, State: p.sim.State()}
}

func (p *LLVMPhaseOrderingProcess) sense(ctx context.Context, parameter string) LLVMPhaseOrderingResponse {
	if p.sim == nil {
		return LLVMPhaseOrderingResponse{Err: fmt.Errorf("llvm-phase-ordering process is not started")}
	}
	percept, err := p.sim.Sense(ctx, parameter)
	if err != nil {
		return LLVMPhaseOrderingResponse{Err: err}
	}
	return LLVMPhaseOrderingResponse{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *LLVMPhaseOrderingProcess) optimize(ctx context.Context, output []float64) LLVMPhaseOrderingResponse {
	if p.sim == nil {
		return LLVMPhaseOrderingResponse{Err: fmt.Errorf("llvm-phase-ordering process is not started")}
	}
	fitness, end, err := p.sim.Optimize(ctx, output)
	if err != nil {
		return LLVMPhaseOrderingResponse{Err: err}
	}
	return LLVMPhaseOrderingResponse{OK: true, Fitness: fitness, End: end, State: p.sim.State()}
}

func (p *LLVMPhaseOrderingProcess) state() LLVMPhaseOrderingResponse {
	if p.sim == nil {
		return LLVMPhaseOrderingResponse{Err: fmt.Errorf("llvm-phase-ordering process is not started")}
	}
	return LLVMPhaseOrderingResponse{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
