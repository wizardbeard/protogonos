package substrate

import (
	"context"
	"fmt"
	"strings"
)

const (
	DefaultCPPName      = "set_weight"
	DefaultCEPName      = "delta_weight"
	SetIterativeCEPName = "set_iterative"
	SetWeightCEPName    = "set_weight"
	SetABCNCEPName      = "set_abcn"

	referenceSubstrateWeightLimit = 3.1415
)

var abcnParamAliases = map[string][]string{
	"a": {"abcn_a", "a"},
	"b": {"abcn_b", "b"},
	"c": {"abcn_c", "c"},
	"n": {"abcn_n", "n"},
}

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
	return saturateSubstrateWeight(current + cepControlValue(delta, params)), nil
}

type SetWeightCEP struct{}

func (SetWeightCEP) Name() string {
	return SetWeightCEPName
}

func (SetWeightCEP) Apply(_ context.Context, _ float64, delta float64, params map[string]float64) (float64, error) {
	return saturateSubstrateWeight(cepControlValue(delta, params)), nil
}

type SetABCNCEP struct{}

func (SetABCNCEP) Name() string {
	return SetABCNCEPName
}

func (SetABCNCEP) Apply(_ context.Context, current float64, delta float64, params map[string]float64) (float64, error) {
	control := cepControlValue(delta, params)
	// Reference set_abcn carries per-link [A,B,C,N] plasticity parameters.
	// In the simplified scalar runtime, support an optional coefficient-driven
	// update path when those parameters are present; otherwise keep iterative
	// delta_weight-equivalent behavior.
	if a, b, c, n, ok := readABCNParameters(params); ok {
		deltaWeight := n * (a*control*current + b*control + c*current)
		return saturateSubstrateWeight(current + deltaWeight), nil
	}
	return saturateSubstrateWeight(current + control), nil
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

func saturateSubstrateWeight(value float64) float64 {
	return clamp(value, -referenceSubstrateWeightLimit, referenceSubstrateWeightLimit)
}

func readABCNParameters(params map[string]float64) (a float64, b float64, c float64, n float64, ok bool) {
	if params == nil {
		return 0, 0, 0, 0, false
	}

	var foundA bool
	if a, foundA = findParameterValue(params, abcnParamAliases["a"]); !foundA {
		return 0, 0, 0, 0, false
	}
	var foundB bool
	if b, foundB = findParameterValue(params, abcnParamAliases["b"]); !foundB {
		return 0, 0, 0, 0, false
	}
	var foundC bool
	if c, foundC = findParameterValue(params, abcnParamAliases["c"]); !foundC {
		return 0, 0, 0, 0, false
	}
	var foundN bool
	if n, foundN = findParameterValue(params, abcnParamAliases["n"]); !foundN {
		return 0, 0, 0, 0, false
	}
	return a, b, c, n, true
}

func findParameterValue(params map[string]float64, aliases []string) (float64, bool) {
	for _, alias := range aliases {
		trimmed := strings.TrimSpace(alias)
		if trimmed == "" {
			continue
		}
		if value, ok := params[trimmed]; ok {
			return value, true
		}
		upper := strings.ToUpper(trimmed)
		if upper != trimmed {
			if value, ok := params[upper]; ok {
				return value, true
			}
		}
	}
	return 0, false
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
	if err := RegisterCEP(SetIterativeCEPName, func() CEP { return DeltaWeightCEP{} }); err != nil {
		panic(fmt.Errorf("register set_iterative cep: %w", err))
	}
	if err := RegisterCEP(SetWeightCEPName, func() CEP { return SetWeightCEP{} }); err != nil {
		panic(fmt.Errorf("register set_weight cep: %w", err))
	}
	if err := RegisterCEP(SetABCNCEPName, func() CEP { return SetABCNCEP{} }); err != nil {
		panic(fmt.Errorf("register set_abcn cep: %w", err))
	}
}
