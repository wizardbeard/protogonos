package genotype

import (
	"fmt"
	"strings"

	"protogonos/internal/model"
	"protogonos/internal/substrate"
)

// SubstrateLayerRuntimeBuildRequest carries the vector widths needed to build a
// materialized substrate runtime from genome metadata.
type SubstrateLayerRuntimeBuildRequest struct {
	Genome      model.Genome
	InputWidth  int
	OutputWidth int
}

// BuildSubstrateLayerRuntime builds a typed LayerRuntime when a genome carries
// explicit substrate layer metadata. The bool return is false when callers
// should keep using the legacy SimpleRuntime path.
func BuildSubstrateLayerRuntime(req SubstrateLayerRuntimeBuildRequest) (substrate.Runtime, bool, error) {
	cfg := req.Genome.Substrate
	if cfg == nil || !usesTypedSubstrateLayerConfig(cfg) {
		return nil, false, nil
	}
	if req.InputWidth <= 0 || req.OutputWidth <= 0 {
		return nil, false, nil
	}

	materialized, err := MaterializeSubstrate(SubstrateMaterializationRequest{
		Genome: req.Genome,
		InputSpecs: []substrate.IOCoordinateSpec{{
			Format: substrate.CoordinateFormatNoGeo,
			VL:     req.InputWidth,
		}},
		OutputSpecs: []substrate.IOCoordinateSpec{{
			Format: substrate.CoordinateFormatNoGeo,
			VL:     req.OutputWidth,
		}},
	})
	if err != nil {
		return nil, true, err
	}

	staticCPP, iterativeCPP, ceps, canPopulate, err := resolveLayerRuntimeComponents(cfg, materialized.Plasticity)
	if err != nil {
		return nil, true, err
	}
	stateMode := substrate.SubstrateStateReset
	if !canPopulate {
		switch materialized.Plasticity {
		case substrate.SubstratePlasticityNone, substrate.SubstratePlasticityABCN:
			stateMode = substrate.SubstrateStateHold
		case substrate.SubstratePlasticityIterative:
			return nil, false, nil
		}
	}

	rt, err := substrate.NewLayerRuntime(substrate.LayerRuntimeSpec{
		Plasticity:   materialized.Plasticity,
		LinkForm:     materialized.LinkForm,
		StateMode:    stateMode,
		Substrate:    materialized.Scalar,
		ABCN:         materialized.ABCN,
		StaticCPP:    staticCPP,
		IterativeCPP: iterativeCPP,
		CEPs:         ceps,
		Parameters:   cfg.Parameters,
	})
	if err != nil {
		return nil, true, err
	}
	return rt, true, nil
}

func usesTypedSubstrateLayerConfig(cfg *model.SubstrateConfig) bool {
	if cfg == nil {
		return false
	}
	if strings.TrimSpace(cfg.Plasticity) != "" || strings.TrimSpace(cfg.LinkForm) != "" {
		return true
	}
	return isSubstratePlasticityName(cfg.CPPName) || isSubstrateLinkFormName(cfg.CEPName)
}

func resolveLayerRuntimeComponents(cfg *model.SubstrateConfig, plasticity string) (substrate.CoordinateCPP, substrate.CoordinateIOWCPP, []substrate.CEP, bool, error) {
	var (
		staticCPP    substrate.CoordinateCPP
		iterativeCPP substrate.CoordinateIOWCPP
	)
	if name := layerRuntimeCPPName(cfg); name != "" {
		cpp, err := substrate.ResolveCPP(name)
		if err != nil {
			return nil, nil, nil, false, err
		}
		staticCPP = substrate.AdaptCoordinateCPP(cpp)
		iterativeCPP = substrate.AdaptCoordinateIOWCPP(cpp)
	}

	ceps, err := resolveLayerRuntimeCEPs(cfg)
	if err != nil {
		return nil, nil, nil, false, err
	}

	switch plasticity {
	case substrate.SubstratePlasticityNone:
		return staticCPP, iterativeCPP, ceps, staticCPP != nil && len(ceps) > 0, nil
	case substrate.SubstratePlasticityIterative, substrate.SubstratePlasticityABCN:
		return staticCPP, iterativeCPP, ceps, iterativeCPP != nil && len(ceps) > 0, nil
	default:
		return nil, nil, nil, false, fmt.Errorf("unsupported substrate plasticity %q", plasticity)
	}
}

func layerRuntimeCPPName(cfg *model.SubstrateConfig) string {
	if cfg == nil {
		return ""
	}
	name := strings.TrimSpace(cfg.CPPName)
	if name == "" || isSubstratePlasticityName(name) {
		return substrate.DefaultCPPName
	}
	return name
}

func resolveLayerRuntimeCEPs(cfg *model.SubstrateConfig) ([]substrate.CEP, error) {
	names := layerRuntimeCEPNames(cfg)
	if len(names) == 0 {
		return nil, nil
	}
	ceps := make([]substrate.CEP, 0, len(names))
	for _, name := range names {
		cep, err := substrate.ResolveCEP(name)
		if err != nil {
			return nil, err
		}
		ceps = append(ceps, cep)
	}
	return ceps, nil
}

func layerRuntimeCEPNames(cfg *model.SubstrateConfig) []string {
	if cfg == nil {
		return nil
	}
	names := make([]string, 0, len(cfg.CEPNames)+1)
	for _, name := range cfg.CEPNames {
		trimmed := strings.TrimSpace(name)
		if trimmed == "" || isSubstrateLinkFormName(trimmed) {
			continue
		}
		names = append(names, trimmed)
	}
	if len(names) > 0 {
		return names
	}
	if name := strings.TrimSpace(cfg.CEPName); name != "" && !isSubstrateLinkFormName(name) {
		return []string{name}
	}
	return []string{substrate.DefaultCEPName}
}

func isSubstratePlasticityName(value string) bool {
	switch strings.TrimSpace(value) {
	case substrate.SubstratePlasticityNone, substrate.SubstratePlasticityIterative, substrate.SubstratePlasticityABCN:
		return true
	default:
		return false
	}
}

func isSubstrateLinkFormName(value string) bool {
	switch strings.TrimSpace(value) {
	case substrate.LinkFormL2LFeedforward, substrate.LinkFormFullyInterconnected, substrate.LinkFormJordanRecurrent, substrate.LinkFormNeuronSelfRecurrent:
		return true
	default:
		return false
	}
}
