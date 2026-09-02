package morphology

import (
	"fmt"
	"strings"

	"protogonos/internal/substrate"
)

const (
	SubstrateCPPType = "substrate_cpp"
	SubstrateCEPType = "substrate_cep"
)

const (
	CartesianCPPName              = "cartesian"
	CentripitalDistancesCPPName   = "centripital_distances"
	CartesianDistanceCPPName      = "cartesian_distance"
	CartesianCoordDiffsCPPName    = "cartesian_CoordDiffs"
	CartesianGaussedCoordDiffName = "cartesian_GaussedCoordDiffs"
	IOWCPPName                    = "iow"
	PolarCPPName                  = "polar"
	SphericalCPPName              = "spherical"
)

type SubstrateComponentSpec struct {
	Name string
	Type string
	VL   int
}

func GetInitSubstrateCPPs(dimensions int, plasticity string) ([]SubstrateComponentSpec, error) {
	cpps, err := GetSubstrateCPPs(dimensions, plasticity)
	if err != nil {
		return nil, err
	}
	if len(cpps) == 0 {
		return nil, nil
	}
	return cloneSubstrateComponentSpecs(cpps[:1]), nil
}

func GetInitSubstrateCEPs(dimensions int, plasticity string) ([]SubstrateComponentSpec, error) {
	ceps, err := GetSubstrateCEPs(dimensions, plasticity)
	if err != nil {
		return nil, err
	}
	if len(ceps) == 0 {
		return nil, nil
	}
	return cloneSubstrateComponentSpecs(ceps[:1]), nil
}

func GetSubstrateCPPs(dimensions int, plasticity string) ([]SubstrateComponentSpec, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("substrate dimensions must be positive: %d", dimensions)
	}

	switch normalizeSubstratePlasticity(plasticity) {
	case substrate.SubstratePlasticityIterative, substrate.SubstratePlasticityABCN:
		cpps := []SubstrateComponentSpec{
			{Name: CartesianCPPName, Type: SubstrateCPPType, VL: dimensions*2 + 3},
			{Name: CentripitalDistancesCPPName, Type: SubstrateCPPType, VL: 2 + 3},
			{Name: CartesianDistanceCPPName, Type: SubstrateCPPType, VL: 1 + 3},
			{Name: CartesianCoordDiffsCPPName, Type: SubstrateCPPType, VL: dimensions + 3},
			{Name: CartesianGaussedCoordDiffName, Type: SubstrateCPPType, VL: dimensions + 3},
			{Name: IOWCPPName, Type: SubstrateCPPType, VL: 3},
		}
		return appendDimensionSpecificCPPs(cpps, dimensions, dimensions*2+3), nil
	case substrate.SubstratePlasticityNone, "modular_none":
		cpps := []SubstrateComponentSpec{
			{Name: CartesianCPPName, Type: SubstrateCPPType, VL: dimensions * 2},
			{Name: CentripitalDistancesCPPName, Type: SubstrateCPPType, VL: 2},
			{Name: CartesianDistanceCPPName, Type: SubstrateCPPType, VL: 1},
			{Name: CartesianCoordDiffsCPPName, Type: SubstrateCPPType, VL: dimensions},
			{Name: CartesianGaussedCoordDiffName, Type: SubstrateCPPType, VL: dimensions},
		}
		return appendDimensionSpecificCPPs(cpps, dimensions, dimensions*2), nil
	default:
		return nil, fmt.Errorf("unsupported substrate plasticity: %s", plasticity)
	}
}

func GetSubstrateCEPs(_ int, plasticity string) ([]SubstrateComponentSpec, error) {
	switch normalizeSubstratePlasticity(plasticity) {
	case substrate.SubstratePlasticityIterative:
		return []SubstrateComponentSpec{{Name: substrate.DefaultCEPName, Type: SubstrateCEPType, VL: 1}}, nil
	case substrate.SubstratePlasticityABCN:
		return []SubstrateComponentSpec{{Name: substrate.SetABCNCEPName, Type: SubstrateCEPType, VL: 5}}, nil
	case substrate.SubstratePlasticityNone:
		return []SubstrateComponentSpec{{Name: substrate.SetWeightCEPName, Type: SubstrateCEPType, VL: 1}}, nil
	case "modular_none":
		return []SubstrateComponentSpec{{Name: substrate.WeightExpressionCEPName, Type: SubstrateCEPType, VL: 2}}, nil
	default:
		return nil, fmt.Errorf("unsupported substrate plasticity: %s", plasticity)
	}
}

func appendDimensionSpecificCPPs(cpps []SubstrateComponentSpec, dimensions, vl int) []SubstrateComponentSpec {
	switch dimensions {
	case 2:
		return append(cpps, SubstrateComponentSpec{Name: PolarCPPName, Type: SubstrateCPPType, VL: vl})
	case 3:
		return append(cpps, SubstrateComponentSpec{Name: SphericalCPPName, Type: SubstrateCPPType, VL: vl})
	default:
		return cpps
	}
}

func normalizeSubstratePlasticity(plasticity string) string {
	normalized := strings.TrimSpace(strings.ToLower(plasticity))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	if normalized == "" {
		return substrate.SubstratePlasticityNone
	}
	return normalized
}

func cloneSubstrateComponentSpecs(specs []SubstrateComponentSpec) []SubstrateComponentSpec {
	return append([]SubstrateComponentSpec(nil), specs...)
}
