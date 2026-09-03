package io

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
)

var (
	ErrActuatorProcessTerminated       = errors.New("actuator process terminated")
	ErrActuatorActorTerminated         = errors.New("actuator actor terminated")
	ErrActuatorActorUninitialized      = errors.New("actuator actor uninitialized")
	ErrActuatorActorAlreadyInitialized = errors.New("actuator actor already initialized")
	ErrUnexpectedActuatorInitPID       = errors.New("unexpected actuator init sender")
	ErrUnexpectedActuatorForwardPID    = errors.New("unexpected actuator forward sender")
	ErrUnexpectedActuatorTerminatePID  = errors.New("unexpected actuator terminate sender")
	ErrInvalidActuatorMessage          = errors.New("invalid actuator message")
)

type ActuatorMessage interface {
	isActuatorMessage()
}

// ActuatorInitMessage mirrors actuator:prep/1 state handoff from ExoSelf:
// `{ExoSelfPid,{Id,CxPid,Scape,ActuatorName,VL,Parameters,FaninPIds,OpMode}}`.
type ActuatorInitMessage struct {
	FromPID      string
	ID           string
	CxPID        string
	Scape        string
	ActuatorName string
	VL           int
	Parameters   map[string]float64
	FaninPIDs    []string
	OpMode       string
	Process      *ActuatorProcess
	Actuator     Actuator
}

func (ActuatorInitMessage) isActuatorMessage() {}

// ActuatorForwardMessage mirrors `{FromPid,forward,Input}`.
type ActuatorForwardMessage struct {
	FromPID string
	Input   []float64
}

func (ActuatorForwardMessage) isActuatorMessage() {}

// ActuatorTerminateMessage mirrors `{ExoSelfPid,terminate}`.
type ActuatorTerminateMessage struct {
	FromPID string
}

func (ActuatorTerminateMessage) isActuatorMessage() {}

// ActuatorSyncMessage mirrors `{ActuatorPid,sync,Fitness,EndFlag}`.
type ActuatorSyncMessage struct {
	FromPID     string
	Fitness     []float64
	EndFlag     int
	GoalReached bool
}

type ActuatorFeedbackProvider interface {
	ConsumeActuatorFeedback() (ActuatorSyncMessage, bool)
}

type ActuatorProcess struct {
	id           string
	exoselfPID   string
	cxPID        string
	scape        string
	actuatorName string
	vl           int
	parameters   map[string]float64
	faninPIDs    []string
	opMode       string
	actuator     Actuator
	pendingPIDs  []string
	acc          []float64
	lastSync     *ActuatorSyncMessage
	terminated   bool
}

var actuatorProcessCounter uint64

func NewActuatorProcess(id string, exoselfPID string, cxPID string, actuator Actuator, vl int, faninPIDs []string) (*ActuatorProcess, error) {
	return NewActuatorProcessWithState(ActuatorInitMessage{
		ID:        id,
		FromPID:   exoselfPID,
		CxPID:     cxPID,
		VL:        vl,
		FaninPIDs: faninPIDs,
		Actuator:  actuator,
	})
}

func NewActuatorProcessWithState(state ActuatorInitMessage) (*ActuatorProcess, error) {
	if state.Process != nil {
		return state.Process, nil
	}
	if state.Actuator == nil {
		return nil, errors.New("actuator process requires actuator")
	}
	processID := strings.TrimSpace(state.ID)
	if processID == "" {
		processID = fmt.Sprintf("actuator_%d", atomic.AddUint64(&actuatorProcessCounter, 1))
	}
	faninPIDs := trimActuatorPIDs(state.FaninPIDs)
	return &ActuatorProcess{
		id:           processID,
		exoselfPID:   strings.TrimSpace(state.FromPID),
		cxPID:        strings.TrimSpace(state.CxPID),
		scape:        strings.TrimSpace(state.Scape),
		actuatorName: strings.TrimSpace(state.ActuatorName),
		vl:           state.VL,
		parameters:   cloneActuatorFloatMap(state.Parameters),
		faninPIDs:    faninPIDs,
		opMode:       strings.TrimSpace(state.OpMode),
		actuator:     state.Actuator,
		pendingPIDs:  append([]string(nil), faninPIDs...),
	}, nil
}

func (p *ActuatorProcess) ID() string {
	if p == nil {
		return ""
	}
	return p.id
}

func (p *ActuatorProcess) CortexPID() string {
	if p == nil {
		return ""
	}
	return p.cxPID
}

func (p *ActuatorProcess) ExoSelfPID() string {
	if p == nil {
		return ""
	}
	return p.exoselfPID
}

func (p *ActuatorProcess) HandleMessage(ctx context.Context, message ActuatorMessage) (*ActuatorSyncMessage, error) {
	if p == nil {
		return nil, ErrInvalidActuatorMessage
	}
	switch msg := message.(type) {
	case ActuatorForwardMessage:
		return p.ForwardFrom(ctx, msg.FromPID, msg.Input)
	case ActuatorTerminateMessage:
		return nil, p.TerminateFrom(msg.FromPID)
	default:
		return nil, ErrInvalidActuatorMessage
	}
}

func (p *ActuatorProcess) ForwardFrom(ctx context.Context, fromPID string, input []float64) (*ActuatorSyncMessage, error) {
	if p.terminated {
		return nil, ErrActuatorProcessTerminated
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	fromPID = strings.TrimSpace(fromPID)
	expectedPID, ok := p.nextFaninPID()
	if ok && fromPID != expectedPID {
		return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedActuatorForwardPID, expectedPID, fromPID)
	}
	p.acc = append(p.acc, input...)
	if ok {
		p.pendingPIDs = p.pendingPIDs[1:]
	}
	if len(p.pendingPIDs) > 0 {
		return nil, nil
	}
	return p.flush(ctx)
}

func (p *ActuatorProcess) TerminateFrom(fromPID string) error {
	if p.terminated {
		return ErrActuatorProcessTerminated
	}
	if p.exoselfPID != "" && strings.TrimSpace(fromPID) != p.exoselfPID {
		return fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedActuatorTerminatePID, p.exoselfPID, strings.TrimSpace(fromPID))
	}
	p.terminated = true
	return nil
}

func (p *ActuatorProcess) ConsumeActuatorFeedback() (ActuatorSyncMessage, bool) {
	if p == nil || p.lastSync == nil {
		return ActuatorSyncMessage{}, false
	}
	feedback := *p.lastSync
	feedback.Fitness = append([]float64(nil), feedback.Fitness...)
	p.lastSync = nil
	return feedback, true
}

func (p *ActuatorProcess) nextFaninPID() (string, bool) {
	if len(p.pendingPIDs) == 0 {
		if len(p.faninPIDs) == 0 {
			return "", false
		}
		p.pendingPIDs = append([]string(nil), p.faninPIDs...)
	}
	return p.pendingPIDs[0], true
}

func (p *ActuatorProcess) flush(ctx context.Context) (*ActuatorSyncMessage, error) {
	output := p.normalizedOutput()
	if err := p.actuator.Write(ctx, output); err != nil {
		return nil, err
	}
	sync := &ActuatorSyncMessage{FromPID: p.id}
	if reporter, ok := p.actuator.(ActuatorFeedbackProvider); ok {
		if feedback, ok := reporter.ConsumeActuatorFeedback(); ok {
			sync.Fitness = append([]float64(nil), feedback.Fitness...)
			sync.EndFlag = feedback.EndFlag
			sync.GoalReached = feedback.GoalReached
		}
	}
	p.acc = nil
	p.pendingPIDs = append([]string(nil), p.faninPIDs...)
	p.lastSync = &ActuatorSyncMessage{
		FromPID:     sync.FromPID,
		Fitness:     append([]float64(nil), sync.Fitness...),
		EndFlag:     sync.EndFlag,
		GoalReached: sync.GoalReached,
	}
	return sync, nil
}

func (p *ActuatorProcess) normalizedOutput() []float64 {
	out := append([]float64(nil), p.acc...)
	if p.vl <= 0 || len(out) == p.vl {
		return out
	}
	if len(out) > p.vl {
		return out[:p.vl]
	}
	return append(out, make([]float64, p.vl-len(out))...)
}

type actuatorActorRequest struct {
	ctx     context.Context
	message ActuatorMessage
	reply   chan actuatorActorResponse
}

type actuatorActorResponse struct {
	sync *ActuatorSyncMessage
	err  error
}

type ActuatorActor struct {
	ownerPID string
	mailbox  chan actuatorActorRequest
	done     chan struct{}
}

func NewActuatorActor(ownerPID string, process *ActuatorProcess) (*ActuatorActor, error) {
	return NewActuatorActorWithOwner(ownerPID, process)
}

func NewActuatorActorWithOwner(ownerPID string, process *ActuatorProcess) (*ActuatorActor, error) {
	a := &ActuatorActor{
		ownerPID: strings.TrimSpace(ownerPID),
		mailbox:  make(chan actuatorActorRequest, 16),
		done:     make(chan struct{}),
	}
	go a.loop(process)
	return a, nil
}

func (a *ActuatorActor) Call(ctx context.Context, message ActuatorMessage) (*ActuatorSyncMessage, error) {
	if a == nil {
		return nil, ErrActuatorActorTerminated
	}
	reply := make(chan actuatorActorResponse, 1)
	req := actuatorActorRequest{ctx: ctx, message: message, reply: reply}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-a.done:
		return nil, ErrActuatorActorTerminated
	case a.mailbox <- req:
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-a.done:
		return nil, ErrActuatorActorTerminated
	case res := <-reply:
		return res.sync, res.err
	}
}

func (a *ActuatorActor) InitFrom(ctx context.Context, message ActuatorInitMessage) error {
	_, err := a.Call(ctx, message)
	return err
}

func (a *ActuatorActor) ForwardFrom(ctx context.Context, fromPID string, input []float64) (*ActuatorSyncMessage, error) {
	return a.Call(ctx, ActuatorForwardMessage{FromPID: fromPID, Input: input})
}

func (a *ActuatorActor) TerminateFrom(fromPID string) error {
	_, err := a.Call(context.Background(), ActuatorTerminateMessage{FromPID: fromPID})
	return err
}

func (a *ActuatorActor) loop(process *ActuatorProcess) {
	defer close(a.done)
	initialized := process != nil
	for req := range a.mailbox {
		sync, err := a.handle(req.ctx, &process, &initialized, req.message)
		req.reply <- actuatorActorResponse{sync: sync, err: err}
		if _, ok := req.message.(ActuatorTerminateMessage); ok && err == nil {
			return
		}
	}
}

func (a *ActuatorActor) handle(ctx context.Context, process **ActuatorProcess, initialized *bool, message ActuatorMessage) (*ActuatorSyncMessage, error) {
	switch msg := message.(type) {
	case ActuatorInitMessage:
		if a.ownerPID != "" && strings.TrimSpace(msg.FromPID) != a.ownerPID {
			return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedActuatorInitPID, a.ownerPID, strings.TrimSpace(msg.FromPID))
		}
		if *initialized {
			return nil, ErrActuatorActorAlreadyInitialized
		}
		p, err := NewActuatorProcessWithState(msg)
		if err != nil {
			return nil, err
		}
		*process = p
		*initialized = true
		return nil, nil
	case ActuatorForwardMessage, ActuatorTerminateMessage:
		if !*initialized || *process == nil {
			if msg, ok := message.(ActuatorTerminateMessage); ok {
				if a.ownerPID != "" && strings.TrimSpace(msg.FromPID) != a.ownerPID {
					return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedActuatorTerminatePID, a.ownerPID, strings.TrimSpace(msg.FromPID))
				}
				return nil, nil
			}
			return nil, ErrActuatorActorUninitialized
		}
		return (*process).HandleMessage(ctx, message)
	default:
		return nil, ErrInvalidActuatorMessage
	}
}

func cloneActuatorFloatMap(in map[string]float64) map[string]float64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func trimActuatorPIDs(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, 0, len(in))
	for _, pid := range in {
		pid = strings.TrimSpace(pid)
		if pid != "" {
			out = append(out, pid)
		}
	}
	return out
}
