package scape

import (
	"context"
	"fmt"
	"strings"
)

type RegressionMimicMessage interface {
	isRegressionMimicMessage()
}

type RegressionMimicStartMessage struct {
	Mode string
}

func (RegressionMimicStartMessage) isRegressionMimicMessage() {}

type RegressionMimicStopMessage struct {
	Reason string
}

func (RegressionMimicStopMessage) isRegressionMimicMessage() {}

type RegressionMimicRestartMessage struct{}

func (RegressionMimicRestartMessage) isRegressionMimicMessage() {}

type RegressionMimicSenseMessage struct{}

func (RegressionMimicSenseMessage) isRegressionMimicMessage() {}

type RegressionMimicPredictMessage struct {
	Output []float64
}

func (RegressionMimicPredictMessage) isRegressionMimicMessage() {}

type RegressionMimicStateMessage struct{}

func (RegressionMimicStateMessage) isRegressionMimicMessage() {}

type RegressionMimicResponse struct {
	OK         bool
	Percept    []float64
	Fitness    Fitness
	End        bool
	State      RegressionMimicSimulatorState
	StopReason string
	Err        error
}

type RegressionMimicProcess struct {
	sim        *RegressionMimicSimulator
	mode       string
	stopped    bool
	stopReason string
}

func NewRegressionMimicProcess() *RegressionMimicProcess {
	return &RegressionMimicProcess{}
}

func (p *RegressionMimicProcess) Call(ctx context.Context, message RegressionMimicMessage) RegressionMimicResponse {
	if p == nil || message == nil {
		return RegressionMimicResponse{Err: fmt.Errorf("invalid regression-mimic process message")}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return RegressionMimicResponse{Err: err}
	}

	switch msg := message.(type) {
	case RegressionMimicStartMessage:
		return p.start(msg.Mode)
	case RegressionMimicStopMessage:
		return p.stop(msg.Reason)
	case RegressionMimicRestartMessage:
		return p.restart()
	case RegressionMimicSenseMessage:
		return p.sense(ctx)
	case RegressionMimicPredictMessage:
		return p.predict(ctx, msg.Output)
	case RegressionMimicStateMessage:
		return p.state()
	default:
		return RegressionMimicResponse{Err: fmt.Errorf("unsupported regression-mimic process message %T", message)}
	}
}

func (p *RegressionMimicProcess) start(mode string) RegressionMimicResponse {
	sim, err := NewRegressionMimicSimulator(mode)
	if err != nil {
		return RegressionMimicResponse{Err: err}
	}
	p.sim = sim
	p.mode = sim.State().Mode
	p.stopped = false
	p.stopReason = ""
	return RegressionMimicResponse{OK: true, State: sim.State()}
}

func (p *RegressionMimicProcess) stop(reason string) RegressionMimicResponse {
	if p.sim == nil {
		return RegressionMimicResponse{Err: fmt.Errorf("regression-mimic process is not started")}
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "stopped"
	}
	p.stopped = true
	p.stopReason = reason
	return RegressionMimicResponse{OK: true, End: true, State: p.sim.State(), StopReason: reason}
}

func (p *RegressionMimicProcess) restart() RegressionMimicResponse {
	if p.sim == nil {
		return p.start(p.mode)
	}
	p.sim.Reset()
	p.stopped = false
	p.stopReason = ""
	return RegressionMimicResponse{OK: true, State: p.sim.State()}
}

func (p *RegressionMimicProcess) sense(ctx context.Context) RegressionMimicResponse {
	if p.sim == nil {
		return RegressionMimicResponse{Err: fmt.Errorf("regression-mimic process is not started")}
	}
	if p.stopped {
		return RegressionMimicResponse{Err: fmt.Errorf("regression-mimic process is stopped")}
	}
	percept, err := p.sim.Sense(ctx)
	if err != nil {
		return RegressionMimicResponse{Err: err}
	}
	return RegressionMimicResponse{OK: true, Percept: percept, State: p.sim.State()}
}

func (p *RegressionMimicProcess) predict(ctx context.Context, output []float64) RegressionMimicResponse {
	if p.sim == nil {
		return RegressionMimicResponse{Err: fmt.Errorf("regression-mimic process is not started")}
	}
	if p.stopped {
		return RegressionMimicResponse{Err: fmt.Errorf("regression-mimic process is stopped")}
	}
	fitness, end, err := p.sim.Predict(ctx, output)
	if err != nil {
		return RegressionMimicResponse{Err: err}
	}
	return RegressionMimicResponse{OK: true, Fitness: fitness, End: end, State: p.sim.State()}
}

func (p *RegressionMimicProcess) state() RegressionMimicResponse {
	if p.sim == nil {
		return RegressionMimicResponse{Err: fmt.Errorf("regression-mimic process is not started")}
	}
	return RegressionMimicResponse{OK: true, State: p.sim.State(), StopReason: p.stopReason}
}
