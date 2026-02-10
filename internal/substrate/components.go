package substrate

import (
	"context"
	"fmt"
)

const (
	DefaultCPPName = "set_weight"
	DefaultCEPName = "delta_weight"
)

type SetWeightCPP struct{}

func (SetWeightCPP) Name() string {
	return DefaultCPPName
}

func (SetWeightCPP) Compute(_ context.Context, inputs []float64, _ map[string]float64) (float64, error) {
	if len(inputs) == 0 {
		return 0, nil
	}
	var sum float64
	for _, v := range inputs {
		sum += v
	}
	return sum / float64(len(inputs)), nil
}

type DeltaWeightCEP struct{}

func (DeltaWeightCEP) Name() string {
	return DefaultCEPName
}

func (DeltaWeightCEP) Apply(_ context.Context, current float64, delta float64, params map[string]float64) (float64, error) {
	scale := 1.0
	if params != nil {
		if v, ok := params["scale"]; ok {
			scale = v
		}
	}
	return current + (delta * scale), nil
}

func init() {
	initializeDefaultComponents()
}

func initializeDefaultComponents() {
	if err := RegisterCPP(DefaultCPPName, func() CPP { return SetWeightCPP{} }); err != nil {
		panic(fmt.Errorf("register default cpp: %w", err))
	}
	if err := RegisterCEP(DefaultCEPName, func() CEP { return DeltaWeightCEP{} }); err != nil {
		panic(fmt.Errorf("register default cep: %w", err))
	}
}
