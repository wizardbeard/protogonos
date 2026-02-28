package substrate

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

var (
	ErrNoSubstrateBackup          = errors.New("no substrate backup available")
	ErrSubstrateRuntimeTerminated = errors.New("substrate runtime terminated")
	ErrMissingCEPActor            = errors.New("missing cep actor")
)

type SimpleRuntime struct {
	cpp                 CPP
	ceps                []CEP
	cepProcesses        []*CEPProcess
	cepActors           []*CEPActor
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
	cepProcesses, cepProcessFaninPIDs, err := buildCEPProcesses(ceps, params, cepFaninPIDs, cepFaninPIDsByCEP)
	if err != nil {
		return nil, err
	}
	cepActors, err := buildCEPActors(cepProcesses)
	if err != nil {
		return nil, err
	}
	return &SimpleRuntime{
		cpp:                 cpp,
		ceps:                ceps,
		cepProcesses:        cepProcesses,
		cepActors:           cepActors,
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
		next := r.weights[i]
		for cepIdx, cep := range r.ceps {
			if cepIdx < len(r.cepProcesses) && r.cepProcesses[cepIdx] != nil {
				if cepIdx >= len(r.cepActors) || r.cepActors[cepIdx] == nil {
					return nil, fmt.Errorf("cep %s process actor: %w", cep.Name(), ErrMissingCEPActor)
				}
				actor := r.cepActors[cepIdx]
				faninPIDs := []string{runtimeCPPProcessID}
				if cepIdx < len(r.cepProcessFaninPIDs) && len(r.cepProcessFaninPIDs[cepIdx]) > 0 {
					faninPIDs = r.cepProcessFaninPIDs[cepIdx]
				}
				processSignals, signalErr := r.resolveProcessSignals(faninPIDs, controlSignals)
				if signalErr != nil {
					return nil, fmt.Errorf("cep %s process signals: %w", cep.Name(), signalErr)
				}
				command, ready, err := r.forwardCEPProcess(actor, faninPIDs, processSignals)
				if err == nil {
					if !ready {
						continue
					}
					w, applyErr := ApplyCEPCommand(next, command, r.params)
					if applyErr != nil {
						return nil, fmt.Errorf("cep %s apply command %s: %w", cep.Name(), command.Command, applyErr)
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
	for _, actor := range r.cepActors {
		if actor == nil {
			continue
		}
		if err := actor.TerminateFrom(runtimeExoSelfProcessID); err != nil {
			continue
		}
	}
	for _, process := range r.cepProcesses {
		if process == nil || process.terminated {
			continue
		}
		process.Terminate()
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

func (r *SimpleRuntime) forwardCEPProcess(actor *CEPActor, faninPIDs []string, signals []float64) (CEPCommand, bool, error) {
	if len(signals) != len(faninPIDs) {
		return CEPCommand{}, false, fmt.Errorf("%w: cep fan-in signal mismatch expected=%d got=%d", ErrInvalidCEPOutputWidth, len(faninPIDs), len(signals))
	}
	if actor == nil {
		return CEPCommand{}, false, ErrMissingCEPActor
	}
	var command CEPCommand
	ready := false
	for i, signal := range signals {
		message := CEPForwardMessage{
			FromPID: faninPIDs[i],
			Input:   []float64{signal},
		}
		err := actor.Post(message)
		if err == nil {
			nextError := actor.NextError()
			if nextError != nil && !errors.Is(nextError, ErrCEPActorNoError) {
				err = nextError
			}
		}
		if err != nil {
			return CEPCommand{}, false, err
		}
		nextReady := false
		nextCommand, err := actor.NextCommand()
		if err == nil {
			nextReady = true
		} else if errors.Is(err, ErrCEPActorNoCommandReady) {
			err = nil
		}
		if err != nil {
			return CEPCommand{}, false, err
		}
		command = nextCommand
		ready = nextReady
	}
	return command, ready, nil
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

func buildCEPProcesses(ceps []CEP, parameters map[string]float64, faninPIDs []string, faninPIDsByCEP [][]string) ([]*CEPProcess, [][]string, error) {
	processes := make([]*CEPProcess, 0, len(ceps))
	processFaninPIDs := make([][]string, 0, len(ceps))
	for i, cep := range ceps {
		baseFanin := faninPIDs
		if i < len(faninPIDsByCEP) && len(faninPIDsByCEP[i]) > 0 {
			baseFanin = faninPIDsByCEP[i]
		}
		cepFaninPIDs := resolveCEPProcessFaninPIDs(cep.Name(), baseFanin)
		process, err := NewCEPProcessWithOwner(fmt.Sprintf("cep_%d", i+1), runtimeExoSelfProcessID, cep.Name(), parameters, cepFaninPIDs)
		if err != nil {
			return nil, nil, fmt.Errorf("new cep process for %s: %w", cep.Name(), err)
		}
		processes = append(processes, process)
		processFaninPIDs = append(processFaninPIDs, cepFaninPIDs)
	}
	return processes, processFaninPIDs, nil
}

func buildCEPActors(processes []*CEPProcess) ([]*CEPActor, error) {
	if len(processes) == 0 {
		return nil, nil
	}
	actors := make([]*CEPActor, 0, len(processes))
	for _, process := range processes {
		if process == nil {
			actors = append(actors, nil)
			continue
		}
		actor := NewCEPActorWithOwner(runtimeExoSelfProcessID)
		if _, _, err := actor.Call(CEPInitMessage{
			FromPID: runtimeExoSelfProcessID,
			Process: process,
		}); err != nil {
			return nil, fmt.Errorf("init cep actor %s: %w", process.ID(), err)
		}
		actors = append(actors, actor)
	}
	return actors, nil
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
