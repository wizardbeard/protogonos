package io

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

var (
	ErrSensorProcessTerminated       = errors.New("sensor process terminated")
	ErrSensorActorTerminated         = errors.New("sensor actor terminated")
	ErrSensorActorUninitialized      = errors.New("sensor actor uninitialized")
	ErrSensorActorAlreadyInitialized = errors.New("sensor actor already initialized")
	ErrSensorActorInitProcessNeeded  = errors.New("sensor actor init process required")
	ErrUnexpectedSensorInitPID       = errors.New("unexpected sensor init sender")
	ErrUnexpectedSensorSyncPID       = errors.New("unexpected sensor sync sender")
	ErrUnexpectedSensorTerminatePID  = errors.New("unexpected sensor terminate sender")
	ErrMissingSensorFanoutTarget     = errors.New("missing sensor fanout target")
	ErrInvalidSensorMessage          = errors.New("invalid sensor message")
)

type SensorMessage interface {
	isSensorMessage()
}

// SensorInitMessage mirrors sensor:prep/1 state handoff from ExoSelf:
// `{ExoSelfPid,{Id,CxPid,Scape,SensorName,VL,Parameters,FanoutPIds,OpMode}}`.
type SensorInitMessage struct {
	FromPID    string
	ID         string
	CxPID      string
	Scape      string
	SensorName string
	VL         int
	Parameters map[string]float64
	FanoutPIDs []string
	OpMode     string
	Process    *SensorProcess
	Sensor     Sensor
}

func (SensorInitMessage) isSensorMessage() {}

// SensorSyncMessage mirrors `{CxPid,sync}`.
type SensorSyncMessage struct {
	FromPID string
}

func (SensorSyncMessage) isSensorMessage() {}

// SensorTerminateMessage mirrors `{ExoSelfPid,terminate}`.
type SensorTerminateMessage struct {
	FromPID string
}

func (SensorTerminateMessage) isSensorMessage() {}

// SensorForwardMessage mirrors `{SensorPid,forward,SensoryVector}`.
type SensorForwardMessage struct {
	FromPID string
	Values  []float64
}

// SensorFanoutTarget receives sensor forward messages.
type SensorFanoutTarget interface {
	SensorFanoutPID() string
	ForwardFromSensor(fromPID string, values []float64) error
}

type SensorProcessCall struct {
	ProcessID  string
	ExoSelfPID string
	CortexPID  string
	Scape      string
	SensorName string
	VL         int
	Parameters map[string]float64
	OpMode     string
}

type SensorProcessReader interface {
	ReadForSensorProcess(ctx context.Context, call SensorProcessCall) ([]float64, error)
}

type SensorProcess struct {
	id           string
	exoselfPID   string
	cxPID        string
	scape        string
	sensorName   string
	vl           int
	parameters   map[string]float64
	fanoutPIDs   []string
	opMode       string
	sensor       Sensor
	fanoutTarget map[string]SensorFanoutTarget
	terminated   bool
}

var sensorProcessCounter uint64

func NewSensorProcess(id string, exoselfPID string, cxPID string, sensor Sensor, vl int, fanoutPIDs []string) (*SensorProcess, error) {
	return NewSensorProcessWithState(SensorInitMessage{
		ID:         id,
		FromPID:    exoselfPID,
		CxPID:      cxPID,
		VL:         vl,
		FanoutPIDs: fanoutPIDs,
		Sensor:     sensor,
	})
}

func NewSensorProcessWithState(state SensorInitMessage) (*SensorProcess, error) {
	if state.Process != nil {
		return state.Process, nil
	}
	if state.Sensor == nil {
		return nil, errors.New("sensor process requires sensor")
	}
	processID := strings.TrimSpace(state.ID)
	if processID == "" {
		processID = fmt.Sprintf("sensor_%d", atomic.AddUint64(&sensorProcessCounter, 1))
	}
	return &SensorProcess{
		id:           processID,
		exoselfPID:   strings.TrimSpace(state.FromPID),
		cxPID:        strings.TrimSpace(state.CxPID),
		scape:        strings.TrimSpace(state.Scape),
		sensorName:   strings.TrimSpace(state.SensorName),
		vl:           state.VL,
		parameters:   cloneSensorFloatMap(state.Parameters),
		fanoutPIDs:   trimSensorPIDs(state.FanoutPIDs),
		opMode:       strings.TrimSpace(state.OpMode),
		sensor:       state.Sensor,
		fanoutTarget: make(map[string]SensorFanoutTarget),
	}, nil
}

func (p *SensorProcess) ID() string {
	if p == nil {
		return ""
	}
	return p.id
}

func (p *SensorProcess) CortexPID() string {
	if p == nil {
		return ""
	}
	return p.cxPID
}

func (p *SensorProcess) ExoSelfPID() string {
	if p == nil {
		return ""
	}
	return p.exoselfPID
}

func (p *SensorProcess) AddFanoutTarget(target SensorFanoutTarget) error {
	if p == nil || target == nil || strings.TrimSpace(target.SensorFanoutPID()) == "" {
		return ErrMissingSensorFanoutTarget
	}
	p.fanoutTarget[strings.TrimSpace(target.SensorFanoutPID())] = target
	return nil
}

func (p *SensorProcess) HandleMessage(ctx context.Context, message SensorMessage) ([]float64, error) {
	if p == nil {
		return nil, ErrInvalidSensorMessage
	}
	switch msg := message.(type) {
	case SensorSyncMessage:
		return p.SyncFrom(ctx, msg.FromPID)
	case SensorTerminateMessage:
		return nil, p.TerminateFrom(msg.FromPID)
	default:
		return nil, ErrInvalidSensorMessage
	}
}

func (p *SensorProcess) SyncFrom(ctx context.Context, fromPID string) ([]float64, error) {
	if p.terminated {
		return nil, ErrSensorProcessTerminated
	}
	if p.cxPID != "" && strings.TrimSpace(fromPID) != p.cxPID {
		return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedSensorSyncPID, p.cxPID, strings.TrimSpace(fromPID))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	values, err := p.read(ctx)
	if err != nil {
		return nil, err
	}
	values = p.normalizeSensoryVector(values)
	if err := p.forward(values); err != nil {
		return nil, err
	}
	return append([]float64(nil), values...), nil
}

func (p *SensorProcess) read(ctx context.Context) ([]float64, error) {
	if reader, ok := p.sensor.(SensorProcessReader); ok {
		return reader.ReadForSensorProcess(ctx, SensorProcessCall{
			ProcessID:  p.id,
			ExoSelfPID: p.exoselfPID,
			CortexPID:  p.cxPID,
			Scape:      p.scape,
			SensorName: p.sensorName,
			VL:         p.vl,
			Parameters: cloneSensorFloatMap(p.parameters),
			OpMode:     p.opMode,
		})
	}
	return p.sensor.Read(ctx)
}

func (p *SensorProcess) TerminateFrom(fromPID string) error {
	if p.terminated {
		return ErrSensorProcessTerminated
	}
	if p.exoselfPID != "" && strings.TrimSpace(fromPID) != p.exoselfPID {
		return fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedSensorTerminatePID, p.exoselfPID, strings.TrimSpace(fromPID))
	}
	p.terminated = true
	return nil
}

func (p *SensorProcess) normalizeSensoryVector(values []float64) []float64 {
	if p.vl <= 0 || len(values) == p.vl {
		return append([]float64(nil), values...)
	}
	return make([]float64, p.vl)
}

func (p *SensorProcess) forward(values []float64) error {
	for _, pid := range p.fanoutPIDs {
		target := p.fanoutTarget[pid]
		if target == nil {
			continue
		}
		if err := target.ForwardFromSensor(p.id, append([]float64(nil), values...)); err != nil {
			return err
		}
	}
	return nil
}

type sensorActorRequest struct {
	ctx     context.Context
	message SensorMessage
	reply   chan sensorActorResponse
}

type sensorActorResponse struct {
	values []float64
	err    error
}

type SensorActor struct {
	ownerPID string
	mailbox  chan sensorActorRequest
	done     chan struct{}
	once     sync.Once
}

func NewSensorActor(ownerPID string, process *SensorProcess) (*SensorActor, error) {
	return NewSensorActorWithOwner(ownerPID, process)
}

func NewSensorActorWithOwner(ownerPID string, process *SensorProcess) (*SensorActor, error) {
	a := &SensorActor{
		ownerPID: strings.TrimSpace(ownerPID),
		mailbox:  make(chan sensorActorRequest, 16),
		done:     make(chan struct{}),
	}
	go a.loop(process)
	return a, nil
}

func (a *SensorActor) Call(ctx context.Context, message SensorMessage) ([]float64, error) {
	if a == nil {
		return nil, ErrSensorActorTerminated
	}
	reply := make(chan sensorActorResponse, 1)
	req := sensorActorRequest{ctx: ctx, message: message, reply: reply}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-a.done:
		return nil, ErrSensorActorTerminated
	case a.mailbox <- req:
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-a.done:
		return nil, ErrSensorActorTerminated
	case res := <-reply:
		return res.values, res.err
	}
}

func (a *SensorActor) InitFrom(ctx context.Context, message SensorInitMessage) error {
	_, err := a.Call(ctx, message)
	return err
}

func (a *SensorActor) SyncFrom(ctx context.Context, fromPID string) ([]float64, error) {
	return a.Call(ctx, SensorSyncMessage{FromPID: fromPID})
}

func (a *SensorActor) TerminateFrom(fromPID string) error {
	_, err := a.Call(context.Background(), SensorTerminateMessage{FromPID: fromPID})
	return err
}

func (a *SensorActor) loop(process *SensorProcess) {
	defer close(a.done)
	var initialized bool
	if process != nil {
		initialized = true
	}
	for req := range a.mailbox {
		values, err := a.handle(req.ctx, &process, &initialized, req.message)
		req.reply <- sensorActorResponse{values: values, err: err}
		if _, ok := req.message.(SensorTerminateMessage); ok && err == nil {
			return
		}
	}
}

func (a *SensorActor) handle(ctx context.Context, process **SensorProcess, initialized *bool, message SensorMessage) ([]float64, error) {
	switch msg := message.(type) {
	case SensorInitMessage:
		if a.ownerPID != "" && strings.TrimSpace(msg.FromPID) != a.ownerPID {
			return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedSensorInitPID, a.ownerPID, strings.TrimSpace(msg.FromPID))
		}
		if *initialized {
			return nil, ErrSensorActorAlreadyInitialized
		}
		p, err := NewSensorProcessWithState(msg)
		if err != nil {
			return nil, err
		}
		*process = p
		*initialized = true
		return nil, nil
	case SensorSyncMessage, SensorTerminateMessage:
		if !*initialized || *process == nil {
			if msg, ok := message.(SensorTerminateMessage); ok {
				if a.ownerPID != "" && strings.TrimSpace(msg.FromPID) != a.ownerPID {
					return nil, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedSensorTerminatePID, a.ownerPID, strings.TrimSpace(msg.FromPID))
				}
				return nil, nil
			}
			return nil, ErrSensorActorUninitialized
		}
		return (*process).HandleMessage(ctx, message)
	default:
		return nil, ErrInvalidSensorMessage
	}
}

func cloneSensorFloatMap(in map[string]float64) map[string]float64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func trimSensorPIDs(in []string) []string {
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
