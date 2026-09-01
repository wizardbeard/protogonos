package substrate

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
)

var (
	ErrSubstrateActorTerminated       = errors.New("substrate actor terminated")
	ErrSubstrateActorUninitialized    = errors.New("substrate actor uninitialized")
	ErrSubstrateActorAlreadyInit      = errors.New("substrate actor already initialized")
	ErrSubstrateActorInitRuntime      = errors.New("substrate actor init runtime required")
	ErrUnexpectedSubstrateInitPID     = errors.New("unexpected substrate init sender")
	ErrUnexpectedSubstrateControlPID  = errors.New("unexpected substrate control sender")
	ErrMissingSubstrateActuatorTarget = errors.New("missing substrate actuator target")
	ErrInvalidSubstrateActorMessage   = errors.New("invalid substrate actor message")
)

type SubstrateActorMessage interface {
	isSubstrateActorMessage()
}

type SubstrateActorInitState struct {
	Runtime       Runtime
	SensorPIDs    []string
	ActuatorPIDs  []string
	ActuatorVLs   []int
	FlushOnInit   bool
	FlushAfterRun bool
}

// SubstrateInitMessage mirrors `{ExoSelf,init,InitState}`.
type SubstrateInitMessage struct {
	FromPID string
	State   SubstrateActorInitState
}

func (SubstrateInitMessage) isSubstrateActorMessage() {}

// SubstrateSensorForwardMessage mirrors `{SensorPid,forward,SensorySignal}`.
type SubstrateSensorForwardMessage struct {
	FromPID string
	Signal  []float64
}

func (SubstrateSensorForwardMessage) isSubstrateActorMessage() {}

type SubstrateControlCommand string

const (
	SubstrateControlReset     SubstrateControlCommand = "reset_substrate"
	SubstrateControlBackup    SubstrateControlCommand = "backup_substrate"
	SubstrateControlRevert    SubstrateControlCommand = "revert_substrate"
	SubstrateControlTerminate SubstrateControlCommand = "terminate"
	SubstrateControlFlush     SubstrateControlCommand = "flush_buffer"
)

// SubstrateControlMessage mirrors ExoSelf control messages.
type SubstrateControlMessage struct {
	FromPID string
	Command SubstrateControlCommand
}

func (SubstrateControlMessage) isSubstrateActorMessage() {}

type SubstrateActorResponse struct {
	Ready   bool
	Outputs []float64
	Err     error
}

type SubstrateActuatorTarget interface {
	SubstrateActuatorPID() string
	ForwardFromSubstrate(fromPID string, output []float64) error
}

type substrateActorRequest struct {
	ctx     context.Context
	message SubstrateActorMessage
	reply   chan SubstrateActorResponse
}

type SubstrateActor struct {
	id              string
	ownerPID        string
	initialized     bool
	runtime         Runtime
	sensorPIDs      []string
	actuatorPIDs    []string
	actuatorVLs     []int
	actuatorTargets map[string]SubstrateActuatorTarget
	expectedSensor  int
	sensoryAcc      [][]float64
	pending         []SubstrateSensorForwardMessage
	flushAfterRun   bool
	inbox           chan substrateActorRequest
	done            chan struct{}
	terminated      bool
}

var substrateActorCounter uint64

func NewSubstrateActorWithOwner(id string, ownerPID string) *SubstrateActor {
	actorID := strings.TrimSpace(id)
	if actorID == "" {
		actorID = fmt.Sprintf("substrate_%d", atomic.AddUint64(&substrateActorCounter, 1))
	}
	actor := &SubstrateActor{
		id:              actorID,
		ownerPID:        strings.TrimSpace(ownerPID),
		actuatorTargets: map[string]SubstrateActuatorTarget{},
		inbox:           make(chan substrateActorRequest),
		done:            make(chan struct{}),
	}
	go actor.run()
	return actor
}

func (a *SubstrateActor) ID() string {
	if a == nil {
		return ""
	}
	return a.id
}

func (a *SubstrateActor) RegisterActuatorTarget(target SubstrateActuatorTarget) {
	if a == nil || target == nil {
		return
	}
	pid := strings.TrimSpace(target.SubstrateActuatorPID())
	if pid == "" {
		return
	}
	a.actuatorTargets[pid] = target
}

func (a *SubstrateActor) RegisterActuatorTargets(targets ...SubstrateActuatorTarget) {
	for _, target := range targets {
		a.RegisterActuatorTarget(target)
	}
}

func (a *SubstrateActor) run() {
	defer close(a.done)
	for req := range a.inbox {
		response := a.handle(req.ctx, req.message)
		if req.reply != nil {
			req.reply <- response
			close(req.reply)
		}
		if a.terminated {
			return
		}
	}
}

func (a *SubstrateActor) handle(ctx context.Context, message SubstrateActorMessage) SubstrateActorResponse {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return SubstrateActorResponse{Err: err}
	}
	switch msg := message.(type) {
	case SubstrateInitMessage:
		return a.handleInit(msg)
	case SubstrateSensorForwardMessage:
		if !a.initialized {
			return SubstrateActorResponse{Err: ErrSubstrateActorUninitialized}
		}
		outputs, err := a.handleSensorForward(ctx, msg)
		return SubstrateActorResponse{Outputs: outputs, Err: err}
	case SubstrateControlMessage:
		if !a.initialized && msg.Command != SubstrateControlTerminate {
			return SubstrateActorResponse{Err: ErrSubstrateActorUninitialized}
		}
		ready, err := a.handleControl(msg)
		return SubstrateActorResponse{Ready: ready, Err: err}
	default:
		return SubstrateActorResponse{Err: ErrInvalidSubstrateActorMessage}
	}
}

func (a *SubstrateActor) handleInit(msg SubstrateInitMessage) SubstrateActorResponse {
	if a.initialized {
		return SubstrateActorResponse{Err: ErrSubstrateActorAlreadyInit}
	}
	if a.ownerPID != "" && strings.TrimSpace(msg.FromPID) != a.ownerPID {
		return SubstrateActorResponse{Err: fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedSubstrateInitPID, a.ownerPID, strings.TrimSpace(msg.FromPID))}
	}
	if msg.State.Runtime == nil {
		return SubstrateActorResponse{Err: ErrSubstrateActorInitRuntime}
	}
	sensorPIDs := trimSubstrateActorPIDs(msg.State.SensorPIDs)
	if len(sensorPIDs) == 0 {
		return SubstrateActorResponse{Err: fmt.Errorf("%w: sensor pids are required", ErrInvalidSubstrateActorMessage)}
	}
	actuatorPIDs := trimSubstrateActorPIDs(msg.State.ActuatorPIDs)
	actuatorVLs := clonePositiveActuatorVLs(msg.State.ActuatorVLs, len(actuatorPIDs))
	if len(actuatorPIDs) != len(actuatorVLs) {
		return SubstrateActorResponse{Err: fmt.Errorf("%w: actuator pid/vl mismatch", ErrInvalidSubstrateActorMessage)}
	}
	a.runtime = msg.State.Runtime
	a.sensorPIDs = sensorPIDs
	a.actuatorPIDs = actuatorPIDs
	a.actuatorVLs = actuatorVLs
	a.flushAfterRun = msg.State.FlushAfterRun
	a.initialized = true
	if msg.State.FlushOnInit {
		a.flushBuffer()
	}
	return SubstrateActorResponse{Ready: true}
}

func (a *SubstrateActor) handleSensorForward(ctx context.Context, msg SubstrateSensorForwardMessage) ([]float64, error) {
	a.pending = append(a.pending, SubstrateSensorForwardMessage{
		FromPID: strings.TrimSpace(msg.FromPID),
		Signal:  append([]float64(nil), msg.Signal...),
	})
	for {
		if a.expectedSensor >= len(a.sensorPIDs) {
			a.expectedSensor = 0
		}
		expected := a.sensorPIDs[a.expectedSensor]
		idx := a.findPendingSensor(expected)
		if idx < 0 {
			return nil, nil
		}
		next := a.pending[idx]
		a.pending = append(a.pending[:idx], a.pending[idx+1:]...)
		a.sensoryAcc = append([][]float64{append([]float64(nil), next.Signal...)}, a.sensoryAcc...)
		a.expectedSensor++
		if a.expectedSensor < len(a.sensorPIDs) {
			continue
		}
		inputs := flattenSensorSignalBatches(a.sensoryAcc)
		a.sensoryAcc = nil
		a.expectedSensor = 0
		outputs, err := a.runtime.Step(ctx, inputs)
		if err != nil {
			return nil, err
		}
		if err := a.fanoutOutputs(outputs); err != nil {
			return nil, err
		}
		if a.flushAfterRun {
			a.flushBuffer()
		}
		return outputs, nil
	}
}

func (a *SubstrateActor) handleControl(msg SubstrateControlMessage) (bool, error) {
	if a.ownerPID != "" && strings.TrimSpace(msg.FromPID) != a.ownerPID {
		return false, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedSubstrateControlPID, a.ownerPID, strings.TrimSpace(msg.FromPID))
	}
	switch msg.Command {
	case SubstrateControlReset:
		if stateful, ok := a.runtime.(StatefulRuntime); ok {
			stateful.Reset()
		}
		a.expectedSensor = 0
		a.sensoryAcc = nil
		return true, nil
	case SubstrateControlBackup:
		if stateful, ok := a.runtime.(StatefulRuntime); ok {
			stateful.Backup()
		}
		a.expectedSensor = 0
		a.sensoryAcc = nil
		return true, nil
	case SubstrateControlRevert:
		if stateful, ok := a.runtime.(StatefulRuntime); ok {
			if err := stateful.Restore(); err != nil {
				return false, err
			}
		}
		a.expectedSensor = 0
		a.sensoryAcc = nil
		return true, nil
	case SubstrateControlFlush:
		a.flushBuffer()
		return true, nil
	case SubstrateControlTerminate:
		if terminable, ok := a.runtime.(TerminableRuntime); ok {
			terminable.Terminate()
		}
		a.terminated = true
		return false, nil
	default:
		return false, fmt.Errorf("%w: unsupported substrate control command %q", ErrInvalidSubstrateActorMessage, msg.Command)
	}
}

func (a *SubstrateActor) findPendingSensor(sender string) int {
	for i, msg := range a.pending {
		if msg.FromPID == sender {
			return i
		}
	}
	return -1
}

func (a *SubstrateActor) fanoutOutputs(outputs []float64) error {
	if len(a.actuatorPIDs) == 0 {
		return nil
	}
	remaining := append([]float64(nil), outputs...)
	for i, pid := range a.actuatorPIDs {
		vl := a.actuatorVLs[i]
		if len(remaining) < vl {
			return fmt.Errorf("%w: actuator %s needs %d outputs, got %d", ErrInvalidSubstrateActorMessage, pid, vl, len(remaining))
		}
		chunk := append([]float64(nil), remaining[:vl]...)
		remaining = remaining[vl:]
		target := a.actuatorTargets[pid]
		if target == nil {
			return fmt.Errorf("%w: %s", ErrMissingSubstrateActuatorTarget, pid)
		}
		if err := target.ForwardFromSubstrate(a.id, chunk); err != nil {
			return err
		}
	}
	if len(remaining) != 0 {
		return fmt.Errorf("%w: unused actuator output values=%d", ErrInvalidSubstrateActorMessage, len(remaining))
	}
	return nil
}

func (a *SubstrateActor) flushBuffer() {
	a.pending = nil
	a.sensoryAcc = nil
	a.expectedSensor = 0
}

func (a *SubstrateActor) Call(ctx context.Context, message SubstrateActorMessage) (SubstrateActorResponse, error) {
	if a == nil || message == nil {
		return SubstrateActorResponse{}, ErrInvalidSubstrateActorMessage
	}
	if ctx == nil {
		ctx = context.Background()
	}
	reply := make(chan SubstrateActorResponse, 1)
	req := substrateActorRequest{
		ctx:     ctx,
		message: message,
		reply:   reply,
	}
	select {
	case <-a.done:
		return SubstrateActorResponse{}, ErrSubstrateActorTerminated
	case <-ctx.Done():
		return SubstrateActorResponse{}, ctx.Err()
	case a.inbox <- req:
	}
	select {
	case response := <-reply:
		return response, response.Err
	case <-ctx.Done():
		return SubstrateActorResponse{}, ctx.Err()
	case <-a.done:
		select {
		case response := <-reply:
			return response, response.Err
		default:
			return SubstrateActorResponse{}, ErrSubstrateActorTerminated
		}
	}
}

func (a *SubstrateActor) Post(ctx context.Context, message SubstrateActorMessage) error {
	_, err := a.Call(ctx, message)
	return err
}

func (a *SubstrateActor) Init(ctx context.Context, fromPID string, state SubstrateActorInitState) error {
	_, err := a.Call(ctx, SubstrateInitMessage{FromPID: fromPID, State: state})
	return err
}

func (a *SubstrateActor) Forward(ctx context.Context, fromPID string, signal []float64) ([]float64, error) {
	response, err := a.Call(ctx, SubstrateSensorForwardMessage{FromPID: fromPID, Signal: signal})
	if err != nil {
		return nil, err
	}
	return append([]float64(nil), response.Outputs...), nil
}

func (a *SubstrateActor) Control(ctx context.Context, fromPID string, command SubstrateControlCommand) (bool, error) {
	response, err := a.Call(ctx, SubstrateControlMessage{FromPID: fromPID, Command: command})
	if err != nil {
		return false, err
	}
	return response.Ready, nil
}

func trimSubstrateActorPIDs(raw []string) []string {
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

func clonePositiveActuatorVLs(raw []int, actuatorCount int) []int {
	if actuatorCount == 0 {
		return nil
	}
	out := make([]int, actuatorCount)
	for i := range out {
		out[i] = 1
		if i < len(raw) && raw[i] > 0 {
			out[i] = raw[i]
		}
	}
	return out
}

func flattenSensorSignalBatches(batches [][]float64) []float64 {
	var total int
	for _, batch := range batches {
		total += len(batch)
	}
	out := make([]float64, 0, total)
	for _, batch := range batches {
		out = append(out, batch...)
	}
	return out
}
