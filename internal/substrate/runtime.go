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
	ErrNoSubstrateBackup          = errors.New("no substrate backup available")
	ErrSubstrateRuntimeTerminated = errors.New("substrate runtime terminated")
	ErrMissingCEPActor            = errors.New("missing cep actor")
	ErrMissingSubstrateMailbox    = errors.New("missing substrate mailbox")
	ErrSubstrateMailboxTerminated = errors.New("substrate mailbox terminated")
	ErrMissingCEPFaninRelay       = errors.New("missing cep fan-in relay")
	ErrCEPFaninRelayTerminated    = errors.New("cep fan-in relay terminated")
	ErrUnexpectedCEPCommandSender = errors.New("unexpected cep command sender")
	ErrUnexpectedCEPCommandTarget = errors.New("unexpected cep command target")
)

type SimpleRuntime struct {
	cpp                 CPP
	ceps                []CEP
	cepActors           []*CEPActor
	cepActorsByWeight   [][]*CEPActor
	cepActorInits       []cepActorInit
	cepFaninRelays      [][][]*CEPFaninRelay
	substrateMailboxes  []*substrateCommandMailbox
	cepProcessFaninPIDs [][]string
	cepFaninPIDs        []string
	params              map[string]float64
	weights             []float64
	backup              []float64
	terminated          bool
}

func NewSimpleRuntime(spec Spec, weightCount int) (*SimpleRuntime, error) {
	if weightCount <= 0 {
		return nil, errors.New("weight count must be > 0")
	}
	if spec.CPPName == "" {
		spec.CPPName = DefaultCPPName
	}
	cpp, err := ResolveCPP(spec.CPPName)
	if err != nil {
		return nil, err
	}
	ceps, err := resolveCEPChain(spec)
	if err != nil {
		return nil, err
	}

	params := map[string]float64{}
	for k, v := range spec.Parameters {
		params[k] = v
	}
	cepFaninPIDsByCEP := normalizeCEPFaninPIDsByCEP(spec.CEPFaninPIDsByCEP)
	cepFaninPIDs := resolveGlobalCEPFaninPIDs(spec.CEPFaninPIDs, cepFaninPIDsByCEP)
	cepActorInits, cepProcessFaninPIDs, err := buildCEPActorInits(ceps, params, cepFaninPIDs, cepFaninPIDsByCEP)
	if err != nil {
		return nil, err
	}
	cepActorPool, err := buildCEPActorPool(cepActorInits, weightCount)
	if err != nil {
		return nil, err
	}
	cepFaninRelays := buildCEPFaninRelayPool(cepActorPool, cepProcessFaninPIDs)
	substrateMailboxes := buildSubstrateCommandMailboxPool(cepActorInits, weightCount)
	var cepActors []*CEPActor
	if len(cepActorPool) > 0 {
		cepActors = cepActorPool[0]
	}
	return &SimpleRuntime{
		cpp:                 cpp,
		ceps:                ceps,
		cepActors:           cepActors,
		cepActorsByWeight:   cepActorPool,
		cepActorInits:       cloneCEPActorInits(cepActorInits),
		cepFaninRelays:      cepFaninRelays,
		substrateMailboxes:  substrateMailboxes,
		cepProcessFaninPIDs: cepProcessFaninPIDs,
		cepFaninPIDs:        append([]string(nil), cepFaninPIDs...),
		params:              params,
		weights:             make([]float64, weightCount),
	}, nil
}

func (r *SimpleRuntime) Step(ctx context.Context, inputs []float64) ([]float64, error) {
	return r.step(ctx, inputs, nil)
}

func (r *SimpleRuntime) StepWithFanin(ctx context.Context, inputs []float64, faninSignals map[string]float64) ([]float64, error) {
	return r.step(ctx, inputs, faninSignals)
}

func (r *SimpleRuntime) step(ctx context.Context, inputs []float64, faninSignals map[string]float64) ([]float64, error) {
	if r.terminated {
		return nil, ErrSubstrateRuntimeTerminated
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	delta, err := r.cpp.Compute(ctx, inputs, r.params)
	if err != nil {
		return nil, fmt.Errorf("cpp %s compute: %w", r.cpp.Name(), err)
	}
	controlSignals, err := r.computeControlSignals(ctx, inputs, delta, faninSignals)
	if err != nil {
		return nil, err
	}
	for i := range r.weights {
		actors := r.cepActors
		if i < len(r.cepActorsByWeight) && len(r.cepActorsByWeight[i]) > 0 {
			actors = r.cepActorsByWeight[i]
		}
		expectedInits := scopeCEPActorInitsForWeight(r.cepActorInits, i)
		next := r.weights[i]
		for cepIdx, cep := range r.ceps {
			if cepIdx < len(actors) {
				actor := actors[cepIdx]
				if actor == nil {
					return nil, fmt.Errorf("cep %s process actor: %w", cep.Name(), ErrMissingCEPActor)
				}
				var relays []*CEPFaninRelay
				if i < len(r.cepFaninRelays) && cepIdx < len(r.cepFaninRelays[i]) {
					relays = r.cepFaninRelays[i][cepIdx]
				}
				faninPIDs := []string{runtimeCPPProcessID}
				if cepIdx < len(r.cepProcessFaninPIDs) && len(r.cepProcessFaninPIDs[cepIdx]) > 0 {
					faninPIDs = r.cepProcessFaninPIDs[cepIdx]
				}
				processSignals, signalErr := r.resolveProcessSignals(faninPIDs, controlSignals)
				if signalErr != nil {
					return nil, fmt.Errorf("cep %s process signals: %w", cep.Name(), signalErr)
				}
				command, ready, err := r.forwardCEPProcess(actor, relays, faninPIDs, processSignals)
				if err == nil {
					if !ready {
						continue
					}
					if cepIdx < len(expectedInits) {
						if envelopeErr := validateCEPCommandEnvelope(command, expectedInits[cepIdx]); envelopeErr != nil {
							return nil, fmt.Errorf("cep %s command envelope: %w", cep.Name(), envelopeErr)
						}
					}
					if postErr := r.postSubstrateCommand(i, command); postErr != nil {
						return nil, fmt.Errorf("cep %s mailbox post: %w", cep.Name(), postErr)
					}
					w, applyErr := r.applySubstrateMailbox(i, next)
					if applyErr != nil {
						return nil, fmt.Errorf("cep %s apply mailbox commands: %w", cep.Name(), applyErr)
					}
					next = w
					continue
				}
				if !errors.Is(err, ErrUnsupportedCEPCommand) {
					return nil, fmt.Errorf("cep %s process forward: %w", cep.Name(), err)
				}
			}

			// Keep custom CEP compatibility when a CEP name is not part of the
			// reference command surface.
			w, applyErr := cep.Apply(ctx, next, delta, r.params)
			if applyErr != nil {
				return nil, fmt.Errorf("cep %s apply: %w", cep.Name(), applyErr)
			}
			next = w
		}
		r.weights[i] = next
	}
	return r.Weights(), nil
}

func (r *SimpleRuntime) Terminate() {
	if r.terminated {
		return
	}
	r.terminated = true
	terminated := map[*CEPActor]struct{}{}
	if len(r.cepActorsByWeight) > 0 {
		for _, actors := range r.cepActorsByWeight {
			for _, actor := range actors {
				if actor == nil {
					continue
				}
				if _, exists := terminated[actor]; exists {
					continue
				}
				terminated[actor] = struct{}{}
				_ = actor.TerminateFrom(runtimeExoSelfProcessID)
			}
		}
		for _, weightRelays := range r.cepFaninRelays {
			for _, cepRelays := range weightRelays {
				for _, relay := range cepRelays {
					if relay == nil {
						continue
					}
					relay.Terminate()
				}
			}
		}
		for _, mailbox := range r.substrateMailboxes {
			if mailbox == nil {
				continue
			}
			mailbox.Terminate()
		}
		return
	}
	for _, actor := range r.cepActors {
		if actor == nil {
			continue
		}
		_ = actor.TerminateFrom(runtimeExoSelfProcessID)
	}
	for _, weightRelays := range r.cepFaninRelays {
		for _, cepRelays := range weightRelays {
			for _, relay := range cepRelays {
				if relay == nil {
					continue
				}
				relay.Terminate()
			}
		}
	}
	for _, mailbox := range r.substrateMailboxes {
		if mailbox == nil {
			continue
		}
		mailbox.Terminate()
	}
}

func (r *SimpleRuntime) Weights() []float64 {
	out := make([]float64, len(r.weights))
	copy(out, r.weights)
	return out
}

func (r *SimpleRuntime) Backup() {
	r.backup = r.Weights()
}

func (r *SimpleRuntime) Restore() error {
	if len(r.backup) == 0 {
		return ErrNoSubstrateBackup
	}
	if len(r.weights) != len(r.backup) {
		r.weights = make([]float64, len(r.backup))
	}
	copy(r.weights, r.backup)
	return nil
}

func (r *SimpleRuntime) Reset() {
	for i := range r.weights {
		r.weights[i] = 0
	}
}

const runtimeCPPProcessID = "cpp"
const runtimeCortexProcessID = "cortex"
const runtimeSubstrateProcessID = "substrate"
const runtimeExoSelfProcessID = "exoself"

func (r *SimpleRuntime) computeControlSignals(ctx context.Context, inputs []float64, scalar float64, faninSignals map[string]float64) ([]float64, error) {
	if signals, ok := r.controlSignalsFromFaninMap(faninSignals); ok {
		return signals, nil
	}

	vectorCPP, ok := r.cpp.(VectorCPP)
	if !ok {
		if len(r.cepFaninPIDs) > 1 && len(inputs) == len(r.cepFaninPIDs) && canUseInputFanInSignals(r.ceps) {
			return append([]float64(nil), inputs...), nil
		}
		return []float64{scalar}, nil
	}
	signals, err := vectorCPP.ComputeVector(ctx, inputs, r.params)
	if err != nil {
		return nil, fmt.Errorf("cpp %s compute vector: %w", r.cpp.Name(), err)
	}
	if len(signals) == 0 {
		if len(r.cepFaninPIDs) > 1 && len(inputs) == len(r.cepFaninPIDs) && canUseInputFanInSignals(r.ceps) {
			return append([]float64(nil), inputs...), nil
		}
		return []float64{scalar}, nil
	}
	return append([]float64(nil), signals...), nil
}

func (r *SimpleRuntime) controlSignalsFromFaninMap(faninSignals map[string]float64) ([]float64, bool) {
	if len(faninSignals) == 0 || len(r.cepFaninPIDs) == 0 {
		return nil, false
	}
	signals := make([]float64, 0, len(r.cepFaninPIDs))
	for _, pid := range r.cepFaninPIDs {
		value, ok := faninSignals[pid]
		if !ok {
			return nil, false
		}
		signals = append(signals, value)
	}
	return signals, true
}

func (r *SimpleRuntime) resolveProcessSignals(faninPIDs []string, controlSignals []float64) ([]float64, error) {
	if len(controlSignals) == len(faninPIDs) {
		return append([]float64(nil), controlSignals...), nil
	}
	if len(controlSignals) == 1 && len(faninPIDs) == 1 {
		return []float64{controlSignals[0]}, nil
	}

	if len(controlSignals) != len(r.cepFaninPIDs) {
		return nil, fmt.Errorf("%w: cep fan-in signal mismatch expected=%d got=%d", ErrInvalidCEPOutputWidth, len(r.cepFaninPIDs), len(controlSignals))
	}

	indexByPID := make(map[string]int, len(r.cepFaninPIDs))
	for i, pid := range r.cepFaninPIDs {
		if _, exists := indexByPID[pid]; exists {
			continue
		}
		indexByPID[pid] = i
	}

	out := make([]float64, 0, len(faninPIDs))
	for _, pid := range faninPIDs {
		idx, ok := indexByPID[pid]
		if !ok {
			return nil, fmt.Errorf("%w: missing fan-in signal for %s", ErrInvalidCEPOutputWidth, pid)
		}
		out = append(out, controlSignals[idx])
	}
	return out, nil
}

func (r *SimpleRuntime) forwardCEPProcess(actor *CEPActor, relays []*CEPFaninRelay, faninPIDs []string, signals []float64) (CEPCommand, bool, error) {
	if len(signals) != len(faninPIDs) {
		return CEPCommand{}, false, fmt.Errorf("%w: cep fan-in signal mismatch expected=%d got=%d", ErrInvalidCEPOutputWidth, len(faninPIDs), len(signals))
	}
	if actor == nil {
		return CEPCommand{}, false, ErrMissingCEPActor
	}
	var command CEPCommand
	ready := false
	if len(relays) > 0 {
		if len(relays) != len(faninPIDs) {
			return CEPCommand{}, false, fmt.Errorf("%w: fan-in relay mismatch expected=%d got=%d", ErrMissingCEPFaninRelay, len(faninPIDs), len(relays))
		}
		for i, signal := range signals {
			relay := relays[i]
			if relay == nil {
				return CEPCommand{}, false, fmt.Errorf("%w: nil fan-in relay at index=%d", ErrMissingCEPFaninRelay, i)
			}
			if err := relay.Post([]float64{signal}); err != nil {
				return CEPCommand{}, false, err
			}
		}
	} else {
		for i, signal := range signals {
			message := CEPForwardMessage{
				FromPID: faninPIDs[i],
				Input:   []float64{signal},
			}
			if err := actor.Post(message); err != nil {
				return CEPCommand{}, false, err
			}
		}
	}

	// Synchronize with the actor loop so posted fan-in messages are fully
	// processed before draining command/error mailboxes.
	syncID, err := actor.PostSync()
	if err != nil {
		return CEPCommand{}, false, err
	}
	if err := actor.AwaitSync(syncID); err != nil {
		return CEPCommand{}, false, err
	}

	for {
		nextErr := actor.NextError()
		if errors.Is(nextErr, ErrCEPActorNoError) {
			break
		}
		if nextErr != nil {
			return CEPCommand{}, false, nextErr
		}
	}

	nextCommand, err := actor.NextCommand()
	if err == nil {
		command = nextCommand
		ready = true
		return command, ready, nil
	}
	if errors.Is(err, ErrCEPActorNoCommandReady) {
		return CEPCommand{}, false, nil
	}
	return CEPCommand{}, false, err
}

type CEPFaninRelay struct {
	id      string
	fromPID string
	actor   *CEPActor
	inbox   chan cepFaninRelayRequest
	stop    chan struct{}
	done    chan struct{}
	closeMu sync.Once
}

type cepFaninRelayRequest struct {
	input []float64
	reply chan error
}

func NewCEPFaninRelay(id, fromPID string, actor *CEPActor) *CEPFaninRelay {
	relay := &CEPFaninRelay{
		id:      strings.TrimSpace(id),
		fromPID: strings.TrimSpace(fromPID),
		actor:   actor,
		inbox:   make(chan cepFaninRelayRequest),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}
	go relay.run()
	return relay
}

func (r *CEPFaninRelay) run() {
	defer close(r.done)
	for {
		select {
		case <-r.stop:
			return
		case request := <-r.inbox:
			err := r.forward(request.input)
			if request.reply != nil {
				request.reply <- err
				close(request.reply)
			}
		}
	}
}

func (r *CEPFaninRelay) ID() string {
	if r == nil {
		return ""
	}
	return r.id
}

func (r *CEPFaninRelay) FromPID() string {
	if r == nil {
		return ""
	}
	return r.fromPID
}

func (r *CEPFaninRelay) Post(input []float64) error {
	if r == nil {
		return ErrMissingCEPFaninRelay
	}
	reply := make(chan error, 1)
	req := cepFaninRelayRequest{
		input: append([]float64(nil), input...),
		reply: reply,
	}
	select {
	case <-r.stop:
		return ErrCEPFaninRelayTerminated
	case <-r.done:
		return ErrCEPFaninRelayTerminated
	case r.inbox <- req:
	}
	select {
	case <-r.done:
		return ErrCEPFaninRelayTerminated
	case err := <-reply:
		return err
	}
}

func (r *CEPFaninRelay) Terminate() {
	if r == nil {
		return
	}
	r.closeMu.Do(func() {
		close(r.stop)
		<-r.done
	})
}

func (r *CEPFaninRelay) forward(input []float64) error {
	if r.actor == nil {
		return ErrMissingCEPActor
	}
	return r.actor.Post(CEPForwardMessage{
		FromPID: r.fromPID,
		Input:   append([]float64(nil), input...),
	})
}

func trimCEPFaninPIDs(raw []string) []string {
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

func resolveCEPFaninPIDs(raw []string) []string {
	out := trimCEPFaninPIDs(raw)
	if len(out) == 0 {
		return []string{runtimeCPPProcessID}
	}
	return out
}

func normalizeCEPFaninPIDsByCEP(raw [][]string) [][]string {
	if len(raw) == 0 {
		return nil
	}
	out := make([][]string, 0, len(raw))
	for _, item := range raw {
		out = append(out, trimCEPFaninPIDs(item))
	}
	return out
}

func resolveGlobalCEPFaninPIDs(global []string, byCEP [][]string) []string {
	if trimmed := trimCEPFaninPIDs(global); len(trimmed) > 0 {
		return trimmed
	}
	for _, fanin := range byCEP {
		if len(fanin) == 0 {
			continue
		}
		return append([]string(nil), fanin...)
	}
	return []string{runtimeCPPProcessID}
}

func canUseInputFanInSignals(ceps []CEP) bool {
	if len(ceps) == 0 {
		return false
	}
	for _, cep := range ceps {
		if strings.TrimSpace(cep.Name()) != SetABCNCEPName {
			return false
		}
	}
	return true
}

type cepActorInit struct {
	id           string
	cxPID        string
	substratePID string
	cepName      string
	parameters   map[string]float64
	faninPIDs    []string
}

func buildCEPActorInits(ceps []CEP, parameters map[string]float64, faninPIDs []string, faninPIDsByCEP [][]string) ([]cepActorInit, [][]string, error) {
	inits := make([]cepActorInit, 0, len(ceps))
	processFaninPIDs := make([][]string, 0, len(ceps))
	for i, cep := range ceps {
		baseFanin := faninPIDs
		if i < len(faninPIDsByCEP) && len(faninPIDsByCEP[i]) > 0 {
			baseFanin = faninPIDsByCEP[i]
		}
		cepFaninPIDs := resolveCEPProcessFaninPIDs(cep.Name(), baseFanin)
		if len(cepFaninPIDs) == 0 {
			return nil, nil, fmt.Errorf("new cep process for %s: fanin pids are required", cep.Name())
		}
		inits = append(inits, cepActorInit{
			id:           fmt.Sprintf("cep_%d", i+1),
			cxPID:        runtimeCortexProcessID,
			substratePID: runtimeSubstrateProcessID,
			cepName:      cep.Name(),
			parameters:   cloneFloatMap(parameters),
			faninPIDs:    append([]string(nil), cepFaninPIDs...),
		})
		processFaninPIDs = append(processFaninPIDs, cepFaninPIDs)
	}
	return inits, processFaninPIDs, nil
}

func buildCEPActors(inits []cepActorInit) ([]*CEPActor, error) {
	if len(inits) == 0 {
		return nil, nil
	}
	actors := make([]*CEPActor, 0, len(inits))
	for _, init := range inits {
		actor := NewCEPActorWithOwner(runtimeExoSelfProcessID)
		if _, _, err := actor.Call(CEPInitMessage{
			FromPID:      runtimeExoSelfProcessID,
			ID:           init.id,
			CxPID:        init.cxPID,
			SubstratePID: init.substratePID,
			CEPName:      init.cepName,
			Parameters:   init.parameters,
			FaninPIDs:    init.faninPIDs,
		}); err != nil {
			return nil, fmt.Errorf("init cep actor %s: %w", init.id, err)
		}
		actors = append(actors, actor)
	}
	return actors, nil
}

func buildCEPActorPool(inits []cepActorInit, weightCount int) ([][]*CEPActor, error) {
	if len(inits) == 0 {
		return nil, nil
	}
	pool := make([][]*CEPActor, 0, weightCount)
	for weightIdx := 0; weightIdx < weightCount; weightIdx++ {
		actors, err := buildCEPActors(scopeCEPActorInitsForWeight(inits, weightIdx))
		if err != nil {
			for _, actorSet := range pool {
				for _, actor := range actorSet {
					if actor == nil {
						continue
					}
					_ = actor.TerminateFrom(runtimeExoSelfProcessID)
				}
			}
			return nil, err
		}
		pool = append(pool, actors)
	}
	return pool, nil
}

func scopeCEPActorInitsForWeight(inits []cepActorInit, weightIdx int) []cepActorInit {
	if len(inits) == 0 {
		return nil
	}
	out := make([]cepActorInit, 0, len(inits))
	for _, init := range inits {
		scoped := init
		scoped.parameters = cloneFloatMap(init.parameters)
		scoped.faninPIDs = append([]string(nil), init.faninPIDs...)
		scoped.cxPID = strings.TrimSpace(init.cxPID)
		baseSubstrateID := strings.TrimSpace(init.substratePID)
		if baseSubstrateID == "" {
			baseSubstrateID = runtimeSubstrateProcessID
		}
		scoped.substratePID = fmt.Sprintf("%s_w%d", baseSubstrateID, weightIdx+1)
		baseID := strings.TrimSpace(scoped.id)
		if baseID == "" {
			baseID = fmt.Sprintf("cep_%d", len(out)+1)
		}
		scoped.id = fmt.Sprintf("%s_w%d", baseID, weightIdx+1)
		out = append(out, scoped)
	}
	return out
}

func cloneCEPActorInits(inits []cepActorInit) []cepActorInit {
	if len(inits) == 0 {
		return nil
	}
	out := make([]cepActorInit, 0, len(inits))
	for _, init := range inits {
		cloned := init
		cloned.parameters = cloneFloatMap(init.parameters)
		cloned.faninPIDs = append([]string(nil), init.faninPIDs...)
		out = append(out, cloned)
	}
	return out
}

type substrateCommandMailbox struct {
	id          string
	inbox       chan substrateMailboxMessage
	outbox      chan CEPCommand
	syncbox     chan uint64
	nextSyncID  uint64
	pendingSync map[uint64]struct{}
	pendingMu   sync.Mutex
	stop        chan struct{}
	done        chan struct{}
	closeMu     sync.Once
}

type substrateMailboxMessage interface {
	isSubstrateMailboxMessage()
}

type substrateMailboxCommand struct {
	command CEPCommand
}

func (substrateMailboxCommand) isSubstrateMailboxMessage() {}

type substrateMailboxSync struct {
	syncID uint64
}

func (substrateMailboxSync) isSubstrateMailboxMessage() {}

func newSubstrateCommandMailbox(id string) *substrateCommandMailbox {
	mailbox := &substrateCommandMailbox{
		id:          strings.TrimSpace(id),
		inbox:       make(chan substrateMailboxMessage),
		outbox:      make(chan CEPCommand, 32),
		syncbox:     make(chan uint64, 32),
		pendingSync: map[uint64]struct{}{},
		stop:        make(chan struct{}),
		done:        make(chan struct{}),
	}
	go mailbox.run()
	return mailbox
}

func (m *substrateCommandMailbox) run() {
	defer close(m.done)
	for {
		select {
		case <-m.stop:
			return
		case message := <-m.inbox:
			switch msg := message.(type) {
			case substrateMailboxCommand:
				m.outbox <- msg.command
			case substrateMailboxSync:
				m.syncbox <- msg.syncID
			}
		}
	}
}

func (m *substrateCommandMailbox) ID() string {
	if m == nil {
		return ""
	}
	return m.id
}

func (m *substrateCommandMailbox) Post(command CEPCommand) error {
	if m == nil {
		return ErrMissingSubstrateMailbox
	}
	message := substrateMailboxCommand{command: command}
	select {
	case <-m.stop:
		return ErrSubstrateMailboxTerminated
	case <-m.done:
		return ErrSubstrateMailboxTerminated
	case m.inbox <- message:
		return nil
	}
}

func (m *substrateCommandMailbox) Drain() []CEPCommand {
	if m == nil {
		return nil
	}
	out := make([]CEPCommand, 0, 8)
	for {
		select {
		case command := <-m.outbox:
			out = append(out, command)
		default:
			return out
		}
	}
}

func (m *substrateCommandMailbox) PostSync() (uint64, error) {
	if m == nil {
		return 0, ErrMissingSubstrateMailbox
	}
	syncID := atomic.AddUint64(&m.nextSyncID, 1)
	message := substrateMailboxSync{syncID: syncID}
	select {
	case <-m.stop:
		return 0, ErrSubstrateMailboxTerminated
	case <-m.done:
		return 0, ErrSubstrateMailboxTerminated
	case m.inbox <- message:
		return syncID, nil
	}
}

func (m *substrateCommandMailbox) AwaitSync(syncID uint64) error {
	if m == nil {
		return ErrMissingSubstrateMailbox
	}
	if m.consumePendingSync(syncID) {
		return nil
	}
	for {
		select {
		case doneID := <-m.syncbox:
			if doneID == syncID {
				return nil
			}
			m.storePendingSync(doneID)
		case <-m.stop:
			return ErrSubstrateMailboxTerminated
		case <-m.done:
			return ErrSubstrateMailboxTerminated
		}
	}
}

func (m *substrateCommandMailbox) consumePendingSync(syncID uint64) bool {
	m.pendingMu.Lock()
	defer m.pendingMu.Unlock()
	if _, ok := m.pendingSync[syncID]; !ok {
		return false
	}
	delete(m.pendingSync, syncID)
	return true
}

func (m *substrateCommandMailbox) storePendingSync(syncID uint64) {
	if syncID == 0 {
		return
	}
	m.pendingMu.Lock()
	m.pendingSync[syncID] = struct{}{}
	m.pendingMu.Unlock()
}

func (m *substrateCommandMailbox) Terminate() {
	if m == nil {
		return
	}
	m.closeMu.Do(func() {
		close(m.stop)
		<-m.done
	})
}

func (m *substrateCommandMailbox) IsTerminated() bool {
	if m == nil {
		return true
	}
	select {
	case <-m.done:
		return true
	default:
		return false
	}
}

func buildSubstrateCommandMailboxPool(inits []cepActorInit, weightCount int) []*substrateCommandMailbox {
	if weightCount <= 0 {
		return nil
	}
	pool := make([]*substrateCommandMailbox, 0, weightCount)
	for weightIdx := 0; weightIdx < weightCount; weightIdx++ {
		scoped := scopeCEPActorInitsForWeight(inits, weightIdx)
		substratePID := fmt.Sprintf("%s_w%d", runtimeSubstrateProcessID, weightIdx+1)
		if len(scoped) > 0 && strings.TrimSpace(scoped[0].substratePID) != "" {
			substratePID = strings.TrimSpace(scoped[0].substratePID)
		}
		pool = append(pool, newSubstrateCommandMailbox(substratePID))
	}
	return pool
}

func buildCEPFaninRelayPool(cepActorPool [][]*CEPActor, cepProcessFaninPIDs [][]string) [][][]*CEPFaninRelay {
	if len(cepActorPool) == 0 {
		return nil
	}
	pool := make([][][]*CEPFaninRelay, 0, len(cepActorPool))
	for weightIdx, actorSet := range cepActorPool {
		weightRelays := make([][]*CEPFaninRelay, 0, len(actorSet))
		for cepIdx, actor := range actorSet {
			faninPIDs := []string{runtimeCPPProcessID}
			if cepIdx < len(cepProcessFaninPIDs) && len(cepProcessFaninPIDs[cepIdx]) > 0 {
				faninPIDs = cepProcessFaninPIDs[cepIdx]
			}
			cepRelays := make([]*CEPFaninRelay, 0, len(faninPIDs))
			for faninIdx, faninPID := range faninPIDs {
				relayID := fmt.Sprintf("fanin_%d_cep_%d_w%d", faninIdx+1, cepIdx+1, weightIdx+1)
				cepRelays = append(cepRelays, NewCEPFaninRelay(relayID, faninPID, actor))
			}
			weightRelays = append(weightRelays, cepRelays)
		}
		pool = append(pool, weightRelays)
	}
	return pool
}

func (r *SimpleRuntime) postSubstrateCommand(weightIdx int, command CEPCommand) error {
	if weightIdx < 0 || weightIdx >= len(r.substrateMailboxes) {
		return ErrMissingSubstrateMailbox
	}
	mailbox := r.substrateMailboxes[weightIdx]
	if mailbox == nil {
		return ErrMissingSubstrateMailbox
	}
	target := strings.TrimSpace(mailbox.ID())
	if target != "" && strings.TrimSpace(command.ToPID) != target {
		return fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCEPCommandTarget, target, strings.TrimSpace(command.ToPID))
	}
	return mailbox.Post(command)
}

func (r *SimpleRuntime) applySubstrateMailbox(weightIdx int, current float64) (float64, error) {
	if weightIdx < 0 || weightIdx >= len(r.substrateMailboxes) {
		return 0, ErrMissingSubstrateMailbox
	}
	mailbox := r.substrateMailboxes[weightIdx]
	if mailbox == nil {
		return 0, ErrMissingSubstrateMailbox
	}
	syncID, err := mailbox.PostSync()
	if err != nil {
		return 0, err
	}
	if err := mailbox.AwaitSync(syncID); err != nil {
		return 0, err
	}
	next := current
	for _, command := range mailbox.Drain() {
		updated, applyErr := ApplyCEPCommand(next, command, r.params)
		if applyErr != nil {
			return 0, applyErr
		}
		next = updated
	}
	return next, nil
}

func validateCEPCommandEnvelope(command CEPCommand, expected cepActorInit) error {
	expectedSender := strings.TrimSpace(expected.id)
	if expectedSender != "" && strings.TrimSpace(command.FromPID) != expectedSender {
		return fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCEPCommandSender, expectedSender, strings.TrimSpace(command.FromPID))
	}
	expectedTarget := strings.TrimSpace(expected.substratePID)
	if expectedTarget != "" && strings.TrimSpace(command.ToPID) != expectedTarget {
		return fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCEPCommandTarget, expectedTarget, strings.TrimSpace(command.ToPID))
	}
	return nil
}

func resolveCEPProcessFaninPIDs(cepName string, faninPIDs []string) []string {
	if strings.TrimSpace(cepName) == SetABCNCEPName {
		return append([]string(nil), faninPIDs...)
	}
	if len(faninPIDs) == 0 {
		return []string{runtimeCPPProcessID}
	}
	return []string{faninPIDs[0]}
}

func resolveCEPChain(spec Spec) ([]CEP, error) {
	cepNames := make([]string, 0, len(spec.CEPNames))
	for _, name := range spec.CEPNames {
		trimmed := strings.TrimSpace(name)
		if trimmed == "" {
			continue
		}
		cepNames = append(cepNames, trimmed)
	}
	if len(cepNames) == 0 {
		name := strings.TrimSpace(spec.CEPName)
		if name == "" {
			name = DefaultCEPName
		}
		cepNames = append(cepNames, name)
	}

	ceps := make([]CEP, 0, len(cepNames))
	for _, name := range cepNames {
		cep, err := ResolveCEP(name)
		if err != nil {
			return nil, err
		}
		ceps = append(ceps, cep)
	}
	return ceps, nil
}
