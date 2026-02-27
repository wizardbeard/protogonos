package substrate

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

var (
	ErrNoSubstrateBackup = errors.New("no substrate backup available")
)

type SimpleRuntime struct {
	cpp     CPP
	ceps    []CEP
	params  map[string]float64
	weights []float64
	backup  []float64
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
	return &SimpleRuntime{
		cpp:     cpp,
		ceps:    ceps,
		params:  params,
		weights: make([]float64, weightCount),
	}, nil
}

func (r *SimpleRuntime) Step(ctx context.Context, inputs []float64) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	delta, err := r.cpp.Compute(ctx, inputs, r.params)
	if err != nil {
		return nil, fmt.Errorf("cpp %s compute: %w", r.cpp.Name(), err)
	}
	for i := range r.weights {
		next := r.weights[i]
		for _, cep := range r.ceps {
			command, err := BuildCEPCommand(cep.Name(), []float64{delta}, r.params)
			if err == nil {
				w, applyErr := ApplyCEPCommand(next, command, r.params)
				if applyErr != nil {
					return nil, fmt.Errorf("cep %s apply command %s: %w", cep.Name(), command.Command, applyErr)
				}
				next = w
				continue
			}
			if !errors.Is(err, ErrUnsupportedCEPCommand) {
				return nil, fmt.Errorf("cep %s command build: %w", cep.Name(), err)
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
