package substrate

import (
	"context"
	"fmt"
)

const (
	DefaultCPPName   = "set_weight"
	DefaultCEPName   = "delta_weight"
	SetWeightCEPName = "set_weight"
	SetABCNCEPName   = "set_abcn"
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
	return current + cepControlValue(delta, params), nil
}

type SetWeightCEP struct{}

func (SetWeightCEP) Name() string {
	return SetWeightCEPName
}

func (SetWeightCEP) Apply(_ context.Context, _ float64, delta float64, params map[string]float64) (float64, error) {
	return cepControlValue(delta, params), nil
}

type SetABCNCEP struct{}

func (SetABCNCEP) Name() string {
	return SetABCNCEPName
}

func (SetABCNCEP) Apply(_ context.Context, current float64, delta float64, params map[string]float64) (float64, error) {
	// Reference set_abcn applies a richer substrate message. In the simplified
	// scalar runtime, preserve iterative behavior by applying the same bounded
	// control signal as delta_weight.
	return current + cepControlValue(delta, params), nil
}

func cepControlValue(delta float64, params map[string]float64) float64 {
	const threshold = 0.33

	value := clamp(delta, -1, 1)
	control := 0.0
	switch {
	case value > threshold:
		control = (scaleValue(value, 1, threshold) + 1) / 2
	case value < -threshold:
		control = (scaleValue(value, -threshold, -1) - 1) / 2
	}

	if params != nil {
		if scale, ok := params["scale"]; ok {
			control *= scale
		}
	}
	return control
}

func scaleValue(value, max, min float64) float64 {
	if max == min {
		return 0
	}
	return (value*2 - (max + min)) / (max - min)
}

func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
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
	if err := RegisterCEP(SetWeightCEPName, func() CEP { return SetWeightCEP{} }); err != nil {
		panic(fmt.Errorf("register set_weight cep: %w", err))
	}
	if err := RegisterCEP(SetABCNCEPName, func() CEP { return SetABCNCEP{} }); err != nil {
		panic(fmt.Errorf("register set_abcn cep: %w", err))
	}
}
