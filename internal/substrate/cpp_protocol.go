package substrate

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

var (
	ErrCPPProcessTerminated       = errors.New("cpp process terminated")
	ErrCPPActorTerminated         = errors.New("cpp actor terminated")
	ErrUnexpectedCPPComputePID    = errors.New("unexpected cpp compute sender")
	ErrUnexpectedCPPTerminatePID  = errors.New("unexpected cpp terminate sender")
	ErrCPPVectorComputeNotSupport = errors.New("cpp vector compute not supported")
	ErrInvalidCPPMessage          = errors.New("invalid cpp message")
)

type CPPMessage interface {
	isCPPMessage()
}

// CPPComputeMessage mirrors substrate_cpp's substrate-originated compute
// messages. The current simplified runtime carries the substrate signal vector
// in Input until full coordinate-based substrate population is implemented.
type CPPComputeMessage struct {
	FromPID string
	Input   []float64
	Vector  bool
}

func (CPPComputeMessage) isCPPMessage() {}

// CPPTerminateMessage mirrors `{ExoSelfPid,terminate}`.
type CPPTerminateMessage struct {
	FromPID string
}

func (CPPTerminateMessage) isCPPMessage() {}

type CPPProcess struct {
	id           string
	substratePID string
	terminatePID string
	cpp          CPP
	parameters   map[string]float64
	terminated   bool
}

var cppProcessCounter uint64

func NewCPPProcess(id string, substratePID string, terminatePID string, cpp CPP, parameters map[string]float64) (*CPPProcess, error) {
	if cpp == nil {
		return nil, errors.New("cpp process requires cpp")
	}
	processID := strings.TrimSpace(id)
	if processID == "" {
		processID = fmt.Sprintf("cpp_%d", atomic.AddUint64(&cppProcessCounter, 1))
	}
	substratePID = strings.TrimSpace(substratePID)
	if substratePID == "" {
		substratePID = runtimeSubstrateProcessID
	}
	return &CPPProcess{
		id:           processID,
		substratePID: substratePID,
		terminatePID: strings.TrimSpace(terminatePID),
		cpp:          cpp,
		parameters:   cloneFloatMap(parameters),
	}, nil
}

func (p *CPPProcess) ID() string {
	if p == nil {
		return ""
	}
	return p.id
}

func (p *CPPProcess) HandleMessage(ctx context.Context, message CPPMessage) ([]float64, error) {
	if p == nil {
		return nil, ErrInvalidCPPMessage
	}
	switch msg := message.(type) {
	case CPPComputeMessage:
		return p.handleCompute(ctx, msg)
	case CPPTerminateMessage:
		if p.terminatePID != "" && strings.TrimSpace(msg.FromPID) != p.terminatePID {
			return nil, nil
		}
		p.terminated = true
		return nil, nil
	default:
		return nil, ErrInvalidCPPMessage
	}
}

func (p *CPPProcess) handleCompute(ctx context.Context, msg CPPComputeMessage) ([]float64, error) {
	if p.terminated {
		return nil, ErrCPPProcessTerminated
	}
	if p.substratePID != "" && strings.TrimSpace(msg.FromPID) != p.substratePID {
		return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCPPComputePID, p.substratePID, strings.TrimSpace(msg.FromPID))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if msg.Vector {
		vectorCPP, ok := p.cpp.(VectorCPP)
		if !ok {
			return nil, ErrCPPVectorComputeNotSupport
		}
		out, err := vectorCPP.ComputeVector(ctx, append([]float64(nil), msg.Input...), cloneFloatMap(p.parameters))
		if err != nil {
			return nil, err
		}
		return append([]float64(nil), out...), nil
	}
	out, err := p.cpp.Compute(ctx, append([]float64(nil), msg.Input...), cloneFloatMap(p.parameters))
	if err != nil {
		return nil, err
	}
	return []float64{out}, nil
}

type cppActorRequest struct {
	ctx     context.Context
	message CPPMessage
	reply   chan cppActorResponse
}

type cppActorResponse struct {
	output []float64
	err    error
}

type CPPActor struct {
	process *CPPProcess
	inbox   chan cppActorRequest
	done    chan struct{}
	once    sync.Once
}

func NewCPPActor(process *CPPProcess) *CPPActor {
	actor := &CPPActor{
		process: process,
		inbox:   make(chan cppActorRequest),
		done:    make(chan struct{}),
	}
	go actor.run()
	return actor
}

func (a *CPPActor) run() {
	defer close(a.done)
	for req := range a.inbox {
		output, err := a.process.HandleMessage(req.ctx, req.message)
		if req.reply != nil {
			req.reply <- cppActorResponse{
				output: output,
				err:    err,
			}
			close(req.reply)
		}
		if _, ok := req.message.(CPPTerminateMessage); ok && err == nil && a.process != nil && a.process.terminated {
			return
		}
	}
}

func (a *CPPActor) Call(ctx context.Context, message CPPMessage) ([]float64, error) {
	if a == nil || message == nil {
		return nil, ErrInvalidCPPMessage
	}
	if ctx == nil {
		ctx = context.Background()
	}
	reply := make(chan cppActorResponse, 1)
	req := cppActorRequest{
		ctx:     ctx,
		message: message,
		reply:   reply,
	}
	select {
	case <-a.done:
		return nil, ErrCPPActorTerminated
	case <-ctx.Done():
		return nil, ctx.Err()
	case a.inbox <- req:
	}
	select {
	case <-a.done:
		return nil, ErrCPPActorTerminated
	case <-ctx.Done():
		return nil, ctx.Err()
	case response := <-reply:
		return response.output, response.err
	}
}

func (a *CPPActor) ComputeFrom(ctx context.Context, fromPID string, input []float64) (float64, error) {
	output, err := a.Call(ctx, CPPComputeMessage{
		FromPID: fromPID,
		Input:   input,
	})
	if err != nil {
		return 0, err
	}
	if len(output) == 0 {
		return 0, nil
	}
	return output[0], nil
}

func (a *CPPActor) ComputeVectorFrom(ctx context.Context, fromPID string, input []float64) ([]float64, error) {
	return a.Call(ctx, CPPComputeMessage{
		FromPID: fromPID,
		Input:   input,
		Vector:  true,
	})
}

func (a *CPPActor) TerminateFrom(fromPID string) error {
	if a == nil {
		return ErrCPPActorTerminated
	}
	if a.process != nil && a.process.terminatePID != "" && strings.TrimSpace(fromPID) != a.process.terminatePID {
		return fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCPPTerminatePID, a.process.terminatePID, strings.TrimSpace(fromPID))
	}
	_, err := a.Call(context.Background(), CPPTerminateMessage{FromPID: fromPID})
	if err != nil && !errors.Is(err, ErrCPPActorTerminated) {
		return err
	}
	<-a.done
	return nil
}
