package genotype

import (
	"fmt"
	"strings"

	"protogonos/internal/model"
	"protogonos/internal/substrate"
)

// SubstrateMaterializationRequest carries a genome substrate config plus the
// morphology-derived IO coordinate specs needed to build concrete layers.
type SubstrateMaterializationRequest struct {
	Genome      model.Genome
	InputSpecs  []substrate.IOCoordinateSpec
	OutputSpecs []substrate.IOCoordinateSpec
}

// MaterializedSubstrate is either a scalar substrate or an ABCN substrate,
// selected by the genome's substrate plasticity setting.
type MaterializedSubstrate struct {
	Plasticity string
	LinkForm   string
	Scalar     []substrate.CoordinateHyperlayer
	ABCN       substrate.ABCNSubstrate
}

// MaterializeSubstrate builds the typed substrate layer shape represented by a
// genome. Explicit SubstrateConfig plasticity/link_form fields are preferred,
// with legacy CPPName/CEPName fallback for older constructed genomes.
func MaterializeSubstrate(req SubstrateMaterializationRequest) (MaterializedSubstrate, error) {
	cfg := req.Genome.Substrate
	if cfg == nil {
		return MaterializedSubstrate{}, fmt.Errorf("genome %s has no substrate config", req.Genome.ID)
	}

	plasticity := substrateConfigPlasticity(cfg)
	linkForm := substrateConfigLinkForm(cfg)
	if linkForm == "" {
		linkForm = substrate.LinkFormL2LFeedforward
	}

	createReq := substrate.CreateSubstrateRequest{
		InputSpecs:  append([]substrate.IOCoordinateSpec(nil), req.InputSpecs...),
		Densities:   append([]int(nil), cfg.Dimensions...),
		OutputSpecs: append([]substrate.IOCoordinateSpec(nil), req.OutputSpecs...),
		LinkForm:    linkForm,
	}

	if plasticity == substrate.SubstratePlasticityABCN {
		abcn, err := substrate.CreateABCNSubstrate(substrate.ABCNSubstrateBuildRequest{
			CreateSubstrateRequest: createReq,
			InitialA:               substrateConfigParam(cfg, "abcn_a", "a"),
			InitialB:               substrateConfigParam(cfg, "abcn_b", "b"),
			InitialC:               substrateConfigParam(cfg, "abcn_c", "c"),
			InitialN:               substrateConfigParam(cfg, "abcn_n", "n"),
		})
		if err != nil {
			return MaterializedSubstrate{}, err
		}
		return MaterializedSubstrate{
			Plasticity: plasticity,
			LinkForm:   linkForm,
			ABCN:       abcn,
		}, nil
	}

	scalar, err := substrate.CreateSubstrate(createReq)
	if err != nil {
		return MaterializedSubstrate{}, err
	}
	return MaterializedSubstrate{
		Plasticity: plasticity,
		LinkForm:   linkForm,
		Scalar:     scalar,
	}, nil
}

func substrateConfigPlasticity(cfg *model.SubstrateConfig) string {
	if cfg == nil {
		return substrate.SubstratePlasticityNone
	}
	if value := strings.TrimSpace(cfg.Plasticity); value != "" {
		return value
	}
	switch value := strings.TrimSpace(cfg.CPPName); value {
	case substrate.SubstratePlasticityABCN, substrate.SubstratePlasticityIterative, substrate.SubstratePlasticityNone:
		return value
	default:
		return substrate.SubstratePlasticityNone
	}
}

func substrateConfigLinkForm(cfg *model.SubstrateConfig) string {
	if cfg == nil {
		return ""
	}
	if value := strings.TrimSpace(cfg.LinkForm); value != "" {
		return value
	}
	return strings.TrimSpace(cfg.CEPName)
}

func substrateConfigParam(cfg *model.SubstrateConfig, names ...string) float64 {
	if cfg == nil || cfg.Parameters == nil {
		return 0
	}
	for _, name := range names {
		if value, ok := cfg.Parameters[name]; ok {
			return value
		}
		upper := strings.ToUpper(name)
		if value, ok := cfg.Parameters[upper]; ok {
			return value
		}
	}
	return 0
}
