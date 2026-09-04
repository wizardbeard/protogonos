package morphology

import (
	"fmt"
	"math"

	protoio "protogonos/internal/io"
	"protogonos/internal/scapeid"
)

const (
	IOTypeStandard = "standard"
	ScapePrivate   = "private"
	ScapePublic    = "public"
	FormatNoGeo    = "no_geo"
)

// IOSpec is a lightweight Go analog for the reference #sensor{} and
// #actuator{} morphology records. Name is the canonical Go runtime component;
// ReferenceName preserves the Erlang function/record name.
type IOSpec struct {
	Name          string
	ReferenceName string
	Type          string
	ScapeKind     string
	ScapeName     string
	Format        string
	VL            int
	Parameters    []string
}

func GetReferenceSensorSpecs(scapeName, profile string) ([]IOSpec, error) {
	normalized, profile := normalizeReferenceSpecRequest(scapeName, profile)
	switch normalized {
	case "xor":
		return []IOSpec{referenceSensor(protoio.XORInputLeftSensorName, protoio.XORGetInputSensorAliasName, ScapePrivate, "xor_sim", "", 2, nil)}, nil
	case "pole2-balancing":
		if profile == "" || profile == "default" {
			profile = "3"
		}
		width, err := pole2ReferenceWidth(profile)
		if err != nil {
			return nil, err
		}
		return []IOSpec{referenceSensor(protoio.Pole2CartPositionSensorName, protoio.PBGetInputSensorAliasName, ScapePrivate, "pb_sim", "", width, []string{profile})}, nil
	case "dtm":
		return []IOSpec{referenceSensor(protoio.DTMRangeFrontSensorName, protoio.DTMGetInputSensorAliasName, ScapePrivate, "dtm_sim", "", 4, []string{"all"})}, nil
	case "flatland":
		return flatlandReferenceSensorSpecs(profile), nil
	case "fx":
		return []IOSpec{referenceSensor(protoio.FXPercentChangeSensorName, protoio.FXPLISensorAliasName, ScapePrivate, "fx_sim", FormatNoGeo, 100, []string{"100", "close"})}, nil
	case "gtsa":
		return []IOSpec{referenceSensor(protoio.GTSAInputSensorName, protoio.GeneralPredictorSensorAliasName, ScapePrivate, "scape_GTSA", FormatNoGeo, 30, []string{"10"})}, nil
	case "epitopes":
		return []IOSpec{referenceSensor(protoio.EpitopesSignalSensorName, protoio.ABCPredSensorAliasName, ScapePrivate, "epitopes", FormatNoGeo, 336, epitopesReferenceParameters())}, nil
	case "llvm-phase-ordering":
		return []IOSpec{referenceSensor(protoio.LLVMRuntimeGainSensorName, protoio.BitCodeStatisticsSensorAliasName, ScapePrivate, "scape_LLVMPhaseOrdering", FormatNoGeo, 31, []string{"bzip2"})}, nil
	default:
		return nil, fmt.Errorf("unsupported reference sensor morphology: %s", scapeName)
	}
}

func GetReferenceActuatorSpecs(scapeName, profile string) ([]IOSpec, error) {
	normalized, profile := normalizeReferenceSpecRequest(scapeName, profile)
	switch normalized {
	case "xor":
		return []IOSpec{referenceActuator(protoio.XOROutputActuatorName, protoio.XORSendOutputActuatorAliasName, ScapePrivate, "xor_sim", "", 1, nil)}, nil
	case "pole2-balancing":
		return []IOSpec{referenceActuator(protoio.Pole2PushActuatorName, protoio.PBSendOutputActuatorAliasName, ScapePrivate, "pb_sim", "", 1, []string{"with_damping", "1"})}, nil
	case "dtm":
		return []IOSpec{referenceActuator(protoio.DTMMoveActuatorName, protoio.DTMSendOutputActuatorAliasName, ScapePrivate, "dtm_sim", "", 1, nil)}, nil
	case "flatland":
		return []IOSpec{referenceActuator(protoio.FlatlandTwoWheelsActuatorName, protoio.TwoWheelsActuatorAliasName, ScapePublic, "flatland", FormatNoGeo, 2, []string{"2"})}, nil
	case "fx":
		return []IOSpec{referenceActuator(protoio.FXTradeActuatorName, protoio.FXTradeActuatorAliasName, ScapePrivate, "fx_sim", FormatNoGeo, 1, nil)}, nil
	case "gtsa":
		return []IOSpec{referenceActuator(protoio.GTSAPredictActuatorName, protoio.GeneralPredictorActuatorAliasName, ScapePrivate, "scape_GTSA", FormatNoGeo, 1, []string{"1"})}, nil
	case "epitopes":
		return []IOSpec{referenceActuator(protoio.EpitopesResponseActuatorName, protoio.ABCPredActuatorAliasName, ScapePrivate, "epitopes", FormatNoGeo, 1, epitopesReferenceParameters())}, nil
	case "llvm-phase-ordering":
		return []IOSpec{referenceActuator(protoio.LLVMPhaseActuatorName, protoio.ChooseOptimizationPhaseActuatorAliasName, ScapePrivate, "scape_LLVMPhaseOrdering", FormatNoGeo, 55, []string{"bzip2"})}, nil
	default:
		return nil, fmt.Errorf("unsupported reference actuator morphology: %s", scapeName)
	}
}

func GetInitReferenceSensorSpecs(scapeName, profile string) ([]IOSpec, error) {
	specs, err := GetReferenceSensorSpecs(scapeName, profile)
	if err != nil || len(specs) == 0 {
		return specs, err
	}
	return specs[:1], nil
}

func GetInitReferenceActuatorSpecs(scapeName, profile string) ([]IOSpec, error) {
	specs, err := GetReferenceActuatorSpecs(scapeName, profile)
	if err != nil || len(specs) == 0 {
		return specs, err
	}
	return specs[:1], nil
}

func normalizeReferenceSpecRequest(scapeName, profile string) (string, string) {
	profile = normalizeMorphologyProfile(profile)
	if aliasScape, aliasProfile, ok := legacyMorphologyAlias(scapeName); ok {
		if profile == "" || profile == "default" {
			profile = aliasProfile
		}
		return aliasScape, profile
	}
	return scapeid.Normalize(scapeName), profile
}

func flatlandReferenceSensorSpecs(profile string) []IOSpec {
	spread := fmt.Sprintf("%.12g", math.Pi/2)
	params := []string{spread, "5", "0"}
	switch profile {
	case "classic", "legacy", "flatland_classic_v1":
		return []IOSpec{
			referenceSensor(protoio.FlatlandDistanceScan0SensorName, protoio.DistanceScannerSensorAliasName, ScapePublic, "flatland", FormatNoGeo, 5, params),
		}
	default:
		return []IOSpec{
			referenceSensor(protoio.FlatlandDistanceScan0SensorName, protoio.DistanceScannerSensorAliasName, ScapePublic, "flatland", FormatNoGeo, 5, params),
			referenceSensor(protoio.FlatlandColorScan0SensorName, protoio.ColorScannerSensorAliasName, ScapePublic, "flatland", FormatNoGeo, 5, params),
			referenceSensor(protoio.FlatlandEnergyScan0SensorName, protoio.EnergyScannerSensorAliasName, ScapePublic, "flatland", FormatNoGeo, 5, params),
		}
	}
}

func pole2ReferenceWidth(profile string) (int, error) {
	switch profile {
	case "2":
		return 2, nil
	case "", "3":
		return 3, nil
	case "4":
		return 4, nil
	case "6", "full":
		return 6, nil
	default:
		return 0, fmt.Errorf("unsupported pole2 reference morphology profile: %s", profile)
	}
}

func epitopesReferenceParameters() []string {
	return []string{"abc_pred16", "1121", "560", "561", "840"}
}

func referenceSensor(name, referenceName, scapeKind, scapeName, format string, vl int, parameters []string) IOSpec {
	return referenceIOSpec(name, referenceName, scapeKind, scapeName, format, vl, parameters)
}

func referenceActuator(name, referenceName, scapeKind, scapeName, format string, vl int, parameters []string) IOSpec {
	return referenceIOSpec(name, referenceName, scapeKind, scapeName, format, vl, parameters)
}

func referenceIOSpec(name, referenceName, scapeKind, scapeName, format string, vl int, parameters []string) IOSpec {
	return IOSpec{
		Name:          name,
		ReferenceName: referenceName,
		Type:          IOTypeStandard,
		ScapeKind:     scapeKind,
		ScapeName:     scapeName,
		Format:        format,
		VL:            vl,
		Parameters:    append([]string(nil), parameters...),
	}
}
