package substrate

import (
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
)

var (
	ErrCEPProcessTerminated        = errors.New("cep process terminated")
	ErrCEPActorTerminated          = errors.New("cep actor terminated")
	ErrCEPActorNoCommandReady      = errors.New("cep actor command not ready")
	ErrCEPActorNoError             = errors.New("cep actor no error ready")
	ErrCEPActorUninitialized       = errors.New("cep actor uninitialized")
	ErrCEPActorAlreadyInitialized  = errors.New("cep actor already initialized")
	ErrCEPActorInitProcessRequired = errors.New("cep actor init process required")
	ErrUnexpectedCEPInitPID        = errors.New("unexpected cep init sender")
	ErrUnexpectedCEPForwardPID     = errors.New("unexpected cep fan-in sender")
	ErrUnexpectedCEPTerminatePID   = errors.New("unexpected cep terminate sender")
	ErrInvalidCEPOutputWidth       = errors.New("invalid cep output width")
	ErrUnsupportedCEPCommand       = errors.New("unsupported cep command")
	ErrInvalidCEPMessage           = errors.New("invalid cep message")
)

// CEPCommand mirrors the reference CEP->substrate message shape:
// {CEP_Pid, Command, Signal}.
type CEPCommand struct {
	FromPID string
	Command string
	Signal  []float64
}

// CEPMessage models the actor-style message protocol consumed by CEPProcess.
type CEPMessage interface {
	isCEPMessage()
}

// CEPForwardMessage mirrors `{FromPid,forward,Input}`.
type CEPForwardMessage struct {
	FromPID string
	Input   []float64
}

func (CEPForwardMessage) isCEPMessage() {}

// CEPTerminateMessage mirrors `{ExoSelfPid,terminate}`.
type CEPTerminateMessage struct {
	FromPID string
}

func (CEPTerminateMessage) isCEPMessage() {}

// CEPInitMessage mirrors prep-loop state handoff from ExoSelf:
// `{ExoSelfPid,{Id,CxPid,SubstratePid,CEPName,Parameters,FaninPIds}}`.
type CEPInitMessage struct {
	FromPID string
	Process *CEPProcess
}

func (CEPInitMessage) isCEPMessage() {}

// CEPSyncMessage is a mailbox barrier used to ensure previously posted
// messages have been processed before draining command/error mailboxes.
type CEPSyncMessage struct {
	SyncID uint64
}

func (CEPSyncMessage) isCEPMessage() {}

type cepActorRequest struct {
	message CEPMessage
	reply   chan cepActorResponse
}

type cepActorResponse struct {
	command CEPCommand
	ready   bool
	err     error
}

// CEPProcess is a simplified Go analog of substrate_cep actor loop semantics:
// ordered fan-in accumulation, per-cycle command emission, and explicit
// terminate behavior.
type CEPProcess struct {
	id           string
	terminatePID string
	cepName      string
	parameters   map[string]float64
	faninPIDs    []string
	expectedIdx  int
	acc          []float64
	pendingByPID map[string][][]float64
	terminated   bool
}

var cepProcessCounter uint64

func NewCEPProcess(cepName string, parameters map[string]float64, faninPIDs []string) (*CEPProcess, error) {
	return NewCEPProcessWithOwner("", "", cepName, parameters, faninPIDs)
}

func NewCEPProcessWithID(id string, cepName string, parameters map[string]float64, faninPIDs []string) (*CEPProcess, error) {
	return NewCEPProcessWithOwner(id, "", cepName, parameters, faninPIDs)
}

func NewCEPProcessWithOwner(id string, terminatePID string, cepName string, parameters map[string]float64, faninPIDs []string) (*CEPProcess, error) {
	if len(faninPIDs) == 0 {
		return nil, fmt.Errorf("fanin pids are required")
	}
	processID := strings.TrimSpace(id)
	if processID == "" {
		processID = fmt.Sprintf("cep_%d", atomic.AddUint64(&cepProcessCounter, 1))
	}
	out := &CEPProcess{
		id:           processID,
		terminatePID: strings.TrimSpace(terminatePID),
		cepName:      cepName,
		parameters:   cloneFloatMap(parameters),
		faninPIDs:    append([]string(nil), faninPIDs...),
		pendingByPID: map[string][][]float64{},
	}
	return out, nil
}

func (p *CEPProcess) ID() string {
	return p.id
}

func (p *CEPProcess) Terminate() {
	p.terminated = true
}

func (p *CEPProcess) HandleMessage(message CEPMessage) (CEPCommand, bool, error) {
	switch msg := message.(type) {
	case CEPForwardMessage:
		return p.handleForward(msg.FromPID, msg.Input)
	case CEPTerminateMessage:
		if p.terminatePID != "" && strings.TrimSpace(msg.FromPID) != p.terminatePID {
			return CEPCommand{}, false, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCEPTerminatePID, p.terminatePID, strings.TrimSpace(msg.FromPID))
		}
		p.Terminate()
		return CEPCommand{}, false, nil
	default:
		return CEPCommand{}, false, ErrInvalidCEPMessage
	}
}

func (p *CEPProcess) Forward(fromPID string, input []float64) (CEPCommand, bool, error) {
	return p.HandleMessage(CEPForwardMessage{
		FromPID: fromPID,
		Input:   input,
	})
}

func (p *CEPProcess) handleForward(fromPID string, input []float64) (CEPCommand, bool, error) {
	if p.terminated {
		return CEPCommand{}, false, ErrCEPProcessTerminated
	}
	if len(p.faninPIDs) == 0 {
		return CEPCommand{}, false, fmt.Errorf("fanin pids are required")
	}
	if !containsPID(p.faninPIDs, fromPID) {
		expected := p.faninPIDs[p.expectedIdx%len(p.faninPIDs)]
		return CEPCommand{}, false, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCEPForwardPID, expected, fromPID)
	}

	// Preserve selective receive behavior: enqueue all fan-in inputs and
	// only consume messages when the currently expected sender has pending data.
	chunk := append([]float64(nil), input...)
	p.pendingByPID[fromPID] = append(p.pendingByPID[fromPID], chunk)

	for {
		if p.expectedIdx >= len(p.faninPIDs) {
			p.expectedIdx = 0
		}
		expected := p.faninPIDs[p.expectedIdx]
		pending := p.pendingByPID[expected]
		if len(pending) == 0 {
			return CEPCommand{}, false, nil
		}
		next := pending[0]
		if len(pending) == 1 {
			delete(p.pendingByPID, expected)
		} else {
			p.pendingByPID[expected] = pending[1:]
		}

		// Preserve the reference accumulation choreography:
		// lists:append(Input, Acc), then lists:reverse(Acc) at cycle end.
		p.acc = append(next, p.acc...)
		p.expectedIdx++
		if p.expectedIdx < len(p.faninPIDs) {
			continue
		}

		proper := reverseFloatSlice(p.acc)
		p.acc = p.acc[:0]
		p.expectedIdx = 0

		command, err := BuildCEPCommand(p.cepName, proper, p.parameters)
		if err != nil {
			return CEPCommand{}, false, err
		}
		command.FromPID = p.id
		return command, true, nil
	}
}

func containsPID(pids []string, pid string) bool {
	for _, item := range pids {
		if item == pid {
			return true
		}
	}
	return false
}

// CEPActor is a mailbox-backed CEP process runner mirroring Erlang receive-loop
// control flow while preserving the same command semantics.
type CEPActor struct {
	process      *CEPProcess
	initOwnerPID string
	initialized  bool
	inbox        chan cepActorRequest
	outbox       chan CEPCommand
	errbox       chan error
	syncbox      chan uint64
	nextSyncID   uint64
	done         chan struct{}
}

func NewCEPActor(process *CEPProcess) *CEPActor {
	actor := &CEPActor{
		process:     process,
		initialized: process != nil,
		inbox:       make(chan cepActorRequest),
		outbox:      make(chan CEPCommand, 8),
		errbox:      make(chan error, 8),
		syncbox:     make(chan uint64, 8),
		done:        make(chan struct{}),
	}
	go actor.run()
	return actor
}

func NewCEPActorWithOwner(initOwnerPID string) *CEPActor {
	actor := &CEPActor{
		initOwnerPID: strings.TrimSpace(initOwnerPID),
		inbox:        make(chan cepActorRequest),
		outbox:       make(chan CEPCommand, 8),
		errbox:       make(chan error, 8),
		syncbox:      make(chan uint64, 8),
		done:         make(chan struct{}),
	}
	go actor.run()
	return actor
}

func (a *CEPActor) run() {
	defer close(a.done)
	var pending []cepActorRequest
	handle := func(req cepActorRequest) bool {
		command, ready, err := a.handleActorMessage(req.message)
		if err == nil && ready {
			a.outbox <- command
		}
		if syncMessage, ok := req.message.(CEPSyncMessage); ok && err == nil {
			a.syncbox <- syncMessage.SyncID
		}
		if err != nil {
			select {
			case a.errbox <- err:
			default:
			}
		}
		if req.reply != nil {
			req.reply <- cepActorResponse{
				command: command,
				ready:   ready,
				err:     err,
			}
			close(req.reply)
		}
		if _, ok := req.message.(CEPTerminateMessage); ok && err == nil {
			return true
		}
		return false
	}

	for {
		req := <-a.inbox
		if !a.initialized {
			if _, ok := req.message.(CEPInitMessage); !ok {
				if req.reply != nil {
					req.reply <- cepActorResponse{
						err: ErrCEPActorUninitialized,
					}
					close(req.reply)
				} else {
					pending = append(pending, req)
				}
				continue
			}
		}
		if handle(req) {
			return
		}
		if !a.initialized || len(pending) == 0 {
			continue
		}
		drainPending := pending
		pending = nil
		for _, queued := range drainPending {
			if handle(queued) {
				return
			}
		}
	}
}

func (a *CEPActor) PostSync() (uint64, error) {
	syncID := atomic.AddUint64(&a.nextSyncID, 1)
	if err := a.Post(CEPSyncMessage{SyncID: syncID}); err != nil {
		return 0, err
	}
	return syncID, nil
}

func (a *CEPActor) AwaitSync(syncID uint64) error {
	for {
		select {
		case doneID := <-a.syncbox:
			if doneID == syncID {
				return nil
			}
		case <-a.done:
			return ErrCEPActorTerminated
		}
	}
}

func (a *CEPActor) handleActorMessage(message CEPMessage) (CEPCommand, bool, error) {
	switch msg := message.(type) {
	case CEPInitMessage:
		if a.initialized {
			return CEPCommand{}, false, ErrCEPActorAlreadyInitialized
		}
		if a.initOwnerPID != "" && strings.TrimSpace(msg.FromPID) != a.initOwnerPID {
			return CEPCommand{}, false, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCEPInitPID, a.initOwnerPID, strings.TrimSpace(msg.FromPID))
		}
		if msg.Process == nil {
			return CEPCommand{}, false, ErrCEPActorInitProcessRequired
		}
		a.process = msg.Process
		a.initialized = true
		return CEPCommand{}, false, nil
	case CEPSyncMessage:
		return CEPCommand{}, false, nil
	default:
		if !a.initialized || a.process == nil {
			return CEPCommand{}, false, ErrCEPActorUninitialized
		}
		return a.process.HandleMessage(message)
	}
}

func (a *CEPActor) Post(message CEPMessage) error {
	if message == nil {
		return ErrInvalidCEPMessage
	}
	req := cepActorRequest{
		message: message,
	}
	select {
	case <-a.done:
		return ErrCEPActorTerminated
	case a.inbox <- req:
		return nil
	}
}

func (a *CEPActor) Call(message CEPMessage) (CEPCommand, bool, error) {
	if message == nil {
		return CEPCommand{}, false, ErrInvalidCEPMessage
	}
	reply := make(chan cepActorResponse, 1)
	req := cepActorRequest{
		message: message,
		reply:   reply,
	}
	select {
	case <-a.done:
		return CEPCommand{}, false, ErrCEPActorTerminated
	case a.inbox <- req:
	}
	select {
	case <-a.done:
		return CEPCommand{}, false, ErrCEPActorTerminated
	case response := <-reply:
		return response.command, response.ready, response.err
	}
}

func (a *CEPActor) NextCommand() (CEPCommand, error) {
	select {
	case command := <-a.outbox:
		return command, nil
	case <-a.done:
		return CEPCommand{}, ErrCEPActorTerminated
	default:
		return CEPCommand{}, ErrCEPActorNoCommandReady
	}
}

func (a *CEPActor) NextError() error {
	select {
	case err := <-a.errbox:
		return err
	case <-a.done:
		return ErrCEPActorTerminated
	default:
		return ErrCEPActorNoError
	}
}

func (a *CEPActor) Terminate() error {
	return a.TerminateFrom("")
}

func (a *CEPActor) TerminateFrom(fromPID string) error {
	_, _, err := a.Call(CEPTerminateMessage{FromPID: fromPID})
	if err != nil && !errors.Is(err, ErrCEPActorTerminated) {
		return err
	}
	<-a.done
	return nil
}

func BuildCEPCommand(cepName string, output []float64, parameters map[string]float64) (CEPCommand, error) {
	switch cepName {
	case SetWeightCEPName:
		if len(output) != 1 {
			return CEPCommand{}, fmt.Errorf("%w: set_weight expects 1 signal, got=%d", ErrInvalidCEPOutputWidth, len(output))
		}
		weight := cepControlValue(output[0], parameters)
		return CEPCommand{
			Command: SetWeightCEPName,
			Signal:  []float64{weight},
		}, nil
	case SetABCNCEPName:
		return CEPCommand{
			Command: SetABCNCEPName,
			Signal:  append([]float64(nil), output...),
		}, nil
	case DefaultCEPName, SetIterativeCEPName:
		if len(output) != 1 {
			return CEPCommand{}, fmt.Errorf("%w: delta_weight expects 1 signal, got=%d", ErrInvalidCEPOutputWidth, len(output))
		}
		delta := cepControlValue(output[0], parameters)
		return CEPCommand{
			Command: SetIterativeCEPName,
			Signal:  []float64{delta},
		}, nil
	default:
		return CEPCommand{}, fmt.Errorf("%w: %s", ErrUnsupportedCEPCommand, cepName)
	}
}

func ApplyCEPCommand(current float64, command CEPCommand, parameters map[string]float64) (float64, error) {
	switch strings.TrimSpace(command.Command) {
	case SetWeightCEPName:
		if len(command.Signal) != 1 {
			return 0, fmt.Errorf("%w: set_weight expects 1 signal, got=%d", ErrInvalidCEPOutputWidth, len(command.Signal))
		}
		return saturateSubstrateWeight(command.Signal[0]), nil
	case SetIterativeCEPName, DefaultCEPName:
		if len(command.Signal) != 1 {
			return 0, fmt.Errorf("%w: set_iterative expects 1 signal, got=%d", ErrInvalidCEPOutputWidth, len(command.Signal))
		}
		return saturateSubstrateWeight(current + command.Signal[0]), nil
	case SetABCNCEPName:
		if len(command.Signal) == 0 {
			return 0, fmt.Errorf("%w: set_abcn expects at least 1 signal, got=0", ErrInvalidCEPOutputWidth)
		}
		params := cloneFloatMap(parameters)
		if len(command.Signal) >= 5 {
			if params == nil {
				params = map[string]float64{}
			}
			params["A"] = command.Signal[1]
			params["B"] = command.Signal[2]
			params["C"] = command.Signal[3]
			params["N"] = command.Signal[4]
		}
		return (SetABCNCEP{}).Apply(nil, current, command.Signal[0], params)
	default:
		return 0, fmt.Errorf("%w: %s", ErrUnsupportedCEPCommand, command.Command)
	}
}

func cloneFloatMap(in map[string]float64) map[string]float64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func reverseFloatSlice(values []float64) []float64 {
	if len(values) == 0 {
		return nil
	}
	out := append([]float64(nil), values...)
	for left, right := 0, len(out)-1; left < right; left, right = left+1, right-1 {
		out[left], out[right] = out[right], out[left]
	}
	return out
}
