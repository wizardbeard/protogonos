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
	ErrCPPActorUninitialized      = errors.New("cpp actor uninitialized")
	ErrCPPActorAlreadyInitialized = errors.New("cpp actor already initialized")
	ErrCPPActorInitProcessNeeded  = errors.New("cpp actor init process required")
	ErrMissingCPPFanoutTarget     = errors.New("missing cpp fanout target")
	ErrUnexpectedCPPInitPID       = errors.New("unexpected cpp init sender")
	ErrUnexpectedCPPComputePID    = errors.New("unexpected cpp compute sender")
	ErrUnexpectedCPPTerminatePID  = errors.New("unexpected cpp terminate sender")
	ErrCPPVectorComputeNotSupport = errors.New("cpp vector compute not supported")
	ErrCPPCoordinateNotSupported  = errors.New("cpp coordinate compute not supported")
	ErrInvalidCPPMessage          = errors.New("invalid cpp message")
)

type CPPMessage interface {
	isCPPMessage()
}

// CPPInitMessage mirrors prep-loop state handoff from ExoSelf:
// `{ExoSelfPid,{Id,CxPid,SubstratePid,CPPName,VL,Parameters,FanoutPIds}}`.
type CPPInitMessage struct {
	FromPID      string
	ID           string
	CxPID        string
	SubstratePID string
	CPPName      string
	VL           int
	Parameters   map[string]float64
	FanoutPIDs   []string
	Process      *CPPProcess
	CPP          CPP
}

func (CPPInitMessage) isCPPMessage() {}

// CPPComputeMessage mirrors substrate_cpp's substrate-originated compute
// messages. The current simplified runtime carries the substrate signal vector
// in Input until full coordinate-based substrate population is implemented.
type CPPComputeMessage struct {
	FromPID string
	Input   []float64
	Vector  bool
}

func (CPPComputeMessage) isCPPMessage() {}

// CPPCoordinateMessage mirrors substrate_cpp coordinate compute messages:
// `{SubstratePid, PresynapticCoords, PostsynapticCoords}` and the IOW variant.
type CPPCoordinateMessage struct {
	FromPID            string
	PresynapticCoords  []float64
	PostsynapticCoords []float64
	IOW                []float64
}

func (CPPCoordinateMessage) isCPPMessage() {}

// CPPTerminateMessage mirrors `{ExoSelfPid,terminate}`.
type CPPTerminateMessage struct {
	FromPID string
}

func (CPPTerminateMessage) isCPPMessage() {}

// CPPFanoutTarget receives CPP sensory vectors as CEP-style forward messages.
type CPPFanoutTarget interface {
	CPPFanoutPID() string
	ForwardFromCPP(fromPID string, input []float64) error
}

type CPPProcess struct {
	id           string
	cxPID        string
	substratePID string
	terminatePID string
	cpp          CPP
	vl           int
	parameters   map[string]float64
	fanoutPIDs   []string
	terminated   bool
}

var cppProcessCounter uint64

func NewCPPProcess(id string, substratePID string, terminatePID string, cpp CPP, parameters map[string]float64) (*CPPProcess, error) {
	return NewCPPProcessWithFanout(id, "", substratePID, terminatePID, cpp, 0, parameters, nil)
}

func NewCPPProcessWithFanout(id string, cxPID string, substratePID string, terminatePID string, cpp CPP, vl int, parameters map[string]float64, fanoutPIDs []string) (*CPPProcess, error) {
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
		cxPID:        strings.TrimSpace(cxPID),
		substratePID: substratePID,
		terminatePID: strings.TrimSpace(terminatePID),
		cpp:          cpp,
		vl:           vl,
		parameters:   cloneFloatMap(parameters),
		fanoutPIDs:   trimCPPFanoutPIDs(fanoutPIDs),
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
	case CPPCoordinateMessage:
		return p.handleCoordinateCompute(ctx, msg)
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

func (p *CPPProcess) handleCoordinateCompute(ctx context.Context, msg CPPCoordinateMessage) ([]float64, error) {
	if p.terminated {
		return nil, ErrCPPProcessTerminated
	}
	if p.substratePID != "" && strings.TrimSpace(msg.FromPID) != p.substratePID {
		return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCPPComputePID, p.substratePID, strings.TrimSpace(msg.FromPID))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(msg.IOW) > 0 {
		cpp, ok := p.cpp.(CoordinateIOWCPP)
		if !ok {
			return nil, ErrCPPCoordinateNotSupported
		}
		out, err := cpp.ComputeCoordinatesIOW(
			ctx,
			append([]float64(nil), msg.PresynapticCoords...),
			append([]float64(nil), msg.PostsynapticCoords...),
			append([]float64(nil), msg.IOW...),
			cloneFloatMap(p.parameters),
		)
		if err != nil {
			return nil, err
		}
		return append([]float64(nil), out...), nil
	}
	cpp, ok := p.cpp.(CoordinateCPP)
	if !ok {
		return nil, ErrCPPCoordinateNotSupported
	}
	out, err := cpp.ComputeCoordinates(
		ctx,
		append([]float64(nil), msg.PresynapticCoords...),
		append([]float64(nil), msg.PostsynapticCoords...),
		cloneFloatMap(p.parameters),
	)
	if err != nil {
		return nil, err
	}
	return append([]float64(nil), out...), nil
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
	process      *CPPProcess
	initOwnerPID string
	initialized  bool
	fanouts      map[string]CPPFanoutTarget
	inbox        chan cppActorRequest
	done         chan struct{}
	once         sync.Once
}

func NewCPPActor(process *CPPProcess) *CPPActor {
	actor := &CPPActor{
		process:     process,
		initialized: process != nil,
		fanouts:     map[string]CPPFanoutTarget{},
		inbox:       make(chan cppActorRequest),
		done:        make(chan struct{}),
	}
	go actor.run()
	return actor
}

func NewCPPActorWithOwner(initOwnerPID string) *CPPActor {
	actor := &CPPActor{
		initOwnerPID: strings.TrimSpace(initOwnerPID),
		fanouts:      map[string]CPPFanoutTarget{},
		inbox:        make(chan cppActorRequest),
		done:         make(chan struct{}),
	}
	go actor.run()
	return actor
}

func (a *CPPActor) run() {
	defer close(a.done)
	for req := range a.inbox {
		output, err := a.handleActorMessage(req.ctx, req.message)
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

func (a *CPPActor) handleActorMessage(ctx context.Context, message CPPMessage) ([]float64, error) {
	switch msg := message.(type) {
	case CPPInitMessage:
		if a.initialized {
			return nil, ErrCPPActorAlreadyInitialized
		}
		if a.initOwnerPID != "" && strings.TrimSpace(msg.FromPID) != a.initOwnerPID {
			return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCPPInitPID, a.initOwnerPID, strings.TrimSpace(msg.FromPID))
		}
		process := msg.Process
		if process == nil {
			cpp := msg.CPP
			if cpp == nil {
				cppName := strings.TrimSpace(msg.CPPName)
				if cppName == "" {
					cppName = DefaultCPPName
				}
				var err error
				cpp, err = ResolveCPP(cppName)
				if err != nil {
					return nil, err
				}
			}
			var err error
			process, err = NewCPPProcessWithFanout(
				strings.TrimSpace(msg.ID),
				strings.TrimSpace(msg.CxPID),
				strings.TrimSpace(msg.SubstratePID),
				strings.TrimSpace(msg.FromPID),
				cpp,
				msg.VL,
				msg.Parameters,
				msg.FanoutPIDs,
			)
			if err != nil {
				return nil, err
			}
		}
		a.process = process
		a.initialized = true
		return nil, nil
	default:
		if !a.initialized || a.process == nil {
			return nil, ErrCPPActorUninitialized
		}
		output, err := a.process.HandleMessage(ctx, message)
		if err != nil {
			return nil, err
		}
		switch message.(type) {
		case CPPComputeMessage, CPPCoordinateMessage:
			if err := a.forwardOutput(output); err != nil {
				return nil, err
			}
		}
		return output, nil
	}
}

func (a *CPPActor) RegisterFanoutTarget(target CPPFanoutTarget) {
	if a == nil || target == nil {
		return
	}
	pid := strings.TrimSpace(target.CPPFanoutPID())
	if pid == "" {
		return
	}
	a.fanouts[pid] = target
}

func (a *CPPActor) RegisterFanoutTargets(targets ...CPPFanoutTarget) {
	for _, target := range targets {
		a.RegisterFanoutTarget(target)
	}
}

func (a *CPPActor) forwardOutput(output []float64) error {
	if a == nil || a.process == nil || len(a.process.fanoutPIDs) == 0 {
		return nil
	}
	for _, pid := range a.process.fanoutPIDs {
		target := a.fanouts[pid]
		if target == nil {
			return fmt.Errorf("%w: %s", ErrMissingCPPFanoutTarget, pid)
		}
		if err := target.ForwardFromCPP(a.process.id, output); err != nil {
			return err
		}
	}
	return nil
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

func (a *CPPActor) ComputeCoordinatesFrom(ctx context.Context, fromPID string, presynaptic []float64, postsynaptic []float64) ([]float64, error) {
	return a.Call(ctx, CPPCoordinateMessage{
		FromPID:            fromPID,
		PresynapticCoords:  presynaptic,
		PostsynapticCoords: postsynaptic,
	})
}

func (a *CPPActor) ComputeCoordinatesIOWFrom(ctx context.Context, fromPID string, presynaptic []float64, postsynaptic []float64, iow []float64) ([]float64, error) {
	return a.Call(ctx, CPPCoordinateMessage{
		FromPID:            fromPID,
		PresynapticCoords:  presynaptic,
		PostsynapticCoords: postsynaptic,
		IOW:                iow,
	})
}

func (a *CPPActor) InitFrom(ctx context.Context, msg CPPInitMessage) error {
	_, err := a.Call(ctx, msg)
	return err
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

func trimCPPFanoutPIDs(raw []string) []string {
	out := make([]string, 0, len(raw))
	for _, pid := range raw {
		trimmed := strings.TrimSpace(pid)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
