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
)

type SimpleRuntime struct {
	cpp          CPP
	ceps         []CEP
	cepProcesses []*CEPProcess
	cepFaninPIDs []string
	params       map[string]float64
	weights      []float64
	backup       []float64
	terminated   bool
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
	cepFaninPIDs := resolveCEPFaninPIDs(spec.CEPFaninPIDs)
	cepProcesses, err := buildCEPProcesses(ceps, params, cepFaninPIDs)
	if err != nil {
		return nil, err
	}
	return &SimpleRuntime{
		cpp:          cpp,
		ceps:         ceps,
		cepProcesses: cepProcesses,
		cepFaninPIDs: append([]string(nil), cepFaninPIDs...),
		params:       params,
		weights:      make([]float64, weightCount),
	}, nil
}

func (r *SimpleRuntime) Step(ctx context.Context, inputs []float64) ([]float64, error) {
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
	controlSignals, err := r.computeControlSignals(ctx, inputs, delta)
	if err != nil {
		return nil, err
	}
	for i := range r.weights {
		next := r.weights[i]
		for cepIdx, cep := range r.ceps {
			if cepIdx < len(r.cepProcesses) && r.cepProcesses[cepIdx] != nil {
				command, ready, err := r.forwardCEPProcess(r.cepProcesses[cepIdx], controlSignals)
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
	for _, process := range r.cepProcesses {
		if process == nil {
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

func (r *SimpleRuntime) computeControlSignals(ctx context.Context, inputs []float64, scalar float64) ([]float64, error) {
	vectorCPP, ok := r.cpp.(VectorCPP)
	if !ok {
		return []float64{scalar}, nil
	}
	signals, err := vectorCPP.ComputeVector(ctx, inputs, r.params)
	if err != nil {
		return nil, fmt.Errorf("cpp %s compute vector: %w", r.cpp.Name(), err)
	}
	if len(signals) == 0 {
		return []float64{scalar}, nil
	}
	return append([]float64(nil), signals...), nil
}

func (r *SimpleRuntime) forwardCEPProcess(process *CEPProcess, signals []float64) (CEPCommand, bool, error) {
	if len(signals) != len(r.cepFaninPIDs) {
		return CEPCommand{}, false, fmt.Errorf("%w: cep fan-in signal mismatch expected=%d got=%d", ErrInvalidCEPOutputWidth, len(r.cepFaninPIDs), len(signals))
	}
	var command CEPCommand
	ready := false
	for i, signal := range signals {
		nextCommand, nextReady, err := process.Forward(r.cepFaninPIDs[i], []float64{signal})
		if err != nil {
			return CEPCommand{}, false, err
		}
		command = nextCommand
		ready = nextReady
	}
	return command, ready, nil
}

func resolveCEPFaninPIDs(raw []string) []string {
	out := make([]string, 0, len(raw))
	for _, pid := range raw {
		trimmed := strings.TrimSpace(pid)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
	}
	if len(out) == 0 {
		return []string{runtimeCPPProcessID}
	}
	return out
}

func buildCEPProcesses(ceps []CEP, parameters map[string]float64, faninPIDs []string) ([]*CEPProcess, error) {
	processes := make([]*CEPProcess, 0, len(ceps))
	for _, cep := range ceps {
		process, err := NewCEPProcess(cep.Name(), parameters, faninPIDs)
		if err != nil {
			return nil, fmt.Errorf("new cep process for %s: %w", cep.Name(), err)
		}
		processes = append(processes, process)
	}
	return processes, nil
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
