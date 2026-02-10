package substrate

import (
	"context"
	"errors"
	"fmt"
)

type SimpleRuntime struct {
	cpp     CPP
	cep     CEP
	params  map[string]float64
	weights []float64
}

func NewSimpleRuntime(spec Spec, weightCount int) (*SimpleRuntime, error) {
	if weightCount <= 0 {
		return nil, errors.New("weight count must be > 0")
	}
	if spec.CPPName == "" {
		spec.CPPName = DefaultCPPName
	}
	if spec.CEPName == "" {
		spec.CEPName = DefaultCEPName
	}
	cpp, err := ResolveCPP(spec.CPPName)
	if err != nil {
		return nil, err
	}
	cep, err := ResolveCEP(spec.CEPName)
	if err != nil {
		return nil, err
	}

	params := map[string]float64{}
	for k, v := range spec.Parameters {
		params[k] = v
	}
	return &SimpleRuntime{
		cpp:     cpp,
		cep:     cep,
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
		w, err := r.cep.Apply(ctx, r.weights[i], delta, r.params)
		if err != nil {
			return nil, fmt.Errorf("cep %s apply: %w", r.cep.Name(), err)
		}
		r.weights[i] = w
	}
	return r.Weights(), nil
}

func (r *SimpleRuntime) Weights() []float64 {
	out := make([]float64, len(r.weights))
	copy(out, r.weights)
	return out
}
