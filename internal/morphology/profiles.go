package morphology

import (
	"fmt"
	"sort"
	"strings"

	protoio "protogonos/internal/io"
	"protogonos/internal/scapeid"
)

type FlatlandClassicMorphology struct{}

func (FlatlandClassicMorphology) Name() string {
	return "flatland-classic-v1"
}

func (FlatlandClassicMorphology) Sensors() []string {
	return []string{
		protoio.FlatlandDistanceSensorName,
		protoio.FlatlandEnergySensorName,
	}
}

func (FlatlandClassicMorphology) Actuators() []string {
	return []string{protoio.FlatlandMoveActuatorName}
}

func (FlatlandClassicMorphology) Compatible(scape string) bool {
	return scape == "flatland"
}

type DTMRangeSenseMorphology struct{}

func (DTMRangeSenseMorphology) Name() string {
	return "dtm-range-sense-v1"
}

func (DTMRangeSenseMorphology) Sensors() []string {
	return []string{
		protoio.DTMRangeLeftSensorName,
		protoio.DTMRangeFrontSensorName,
		protoio.DTMRangeRightSensorName,
	}
}

func (DTMRangeSenseMorphology) Actuators() []string {
	return []string{protoio.DTMMoveActuatorName}
}

func (DTMRangeSenseMorphology) Compatible(scape string) bool {
	return scape == "dtm"
}

type DTMRewardMorphology struct{}

func (DTMRewardMorphology) Name() string {
	return "dtm-reward-v1"
}

func (DTMRewardMorphology) Sensors() []string {
	return []string{protoio.DTMRewardSensorName}
}

func (DTMRewardMorphology) Actuators() []string {
	return []string{protoio.DTMMoveActuatorName}
}

func (DTMRewardMorphology) Compatible(scape string) bool {
	return scape == "dtm"
}

type Pole2SurfaceMorphology struct {
	surface string
}

func (m Pole2SurfaceMorphology) Name() string {
	return fmt.Sprintf("pole2-balancing-%s-v1", m.surface)
}

func (m Pole2SurfaceMorphology) Sensors() []string {
	switch m.surface {
	case "2":
		return []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2Angle1SensorName,
		}
	case "3":
		return []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2Angle1SensorName,
			protoio.Pole2Angle2SensorName,
		}
	case "4":
		return []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2CartVelocitySensorName,
			protoio.Pole2Angle1SensorName,
			protoio.Pole2Velocity1SensorName,
		}
	default:
		return []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2CartVelocitySensorName,
			protoio.Pole2Angle1SensorName,
			protoio.Pole2Velocity1SensorName,
			protoio.Pole2Angle2SensorName,
			protoio.Pole2Velocity2SensorName,
		}
	}
}

func (m Pole2SurfaceMorphology) Actuators() []string {
	return []string{protoio.Pole2PushActuatorName}
}

func (m Pole2SurfaceMorphology) Compatible(scape string) bool {
	return scape == "pole2-balancing"
}

func EnsureScapeCompatibilityWithProfile(scapeName, profile string) error {
	m, err := ConstructMorphology(scapeName, profile)
	if err != nil {
		return err
	}
	scapeName = scapeid.Normalize(scapeName)
	return ValidateRegisteredComponents(scapeName, m)
}

func ConstructMorphology(scapeName, profile string) (Morphology, error) {
	scapeName = scapeid.Normalize(scapeName)
	profile = normalizeMorphologyProfile(profile)
	switch scapeName {
	case "flatland":
		switch profile {
		case "", "default", "extended", "full":
			return FlatlandMorphology{}, nil
		case "scanner", "scan", "flatland_scanner", "prey":
			return FlatlandScannerMorphology{}, nil
		case "classic", "legacy", "flatland_v1":
			return FlatlandClassicMorphology{}, nil
		default:
			return nil, fmt.Errorf("unsupported flatland morphology profile: %s", profile)
		}
	case "dtm":
		switch profile {
		case "", "default", "all", "workflow":
			return DTMMorphology{}, nil
		case "range", "range_sense":
			return DTMRangeSenseMorphology{}, nil
		case "reward":
			return DTMRewardMorphology{}, nil
		default:
			return nil, fmt.Errorf("unsupported dtm morphology profile: %s", profile)
		}
	case "pole2-balancing":
		switch profile {
		case "", "default", "all", "workflow":
			return Pole2BalancingMorphology{}, nil
		case "6", "full":
			return Pole2SurfaceMorphology{surface: "6"}, nil
		case "4":
			return Pole2SurfaceMorphology{surface: "4"}, nil
		case "3":
			return Pole2SurfaceMorphology{surface: "3"}, nil
		case "2":
			return Pole2SurfaceMorphology{surface: "2"}, nil
		default:
			return nil, fmt.Errorf("unsupported pole2 morphology profile: %s", profile)
		}
	default:
		m, ok := defaultMorphologyForScape(scapeName)
		if !ok {
			return nil, fmt.Errorf("unsupported scape morphology: %s", scapeName)
		}
		if profile != "" && profile != "default" {
			return nil, fmt.Errorf("unsupported %s morphology profile: %s", scapeName, profile)
		}
		return m, nil
	}
}

func AvailableMorphologyProfiles(scapeName string) []string {
	scapeName = scapeid.Normalize(scapeName)
	var profiles []string
	switch scapeName {
	case "flatland":
		profiles = []string{"classic", "default", "scanner"}
	case "dtm":
		profiles = []string{"default", "range_sense", "reward"}
	case "pole2-balancing":
		profiles = []string{"2", "3", "4", "6", "default"}
	default:
		if _, ok := defaultMorphologyForScape(scapeName); ok {
			profiles = []string{"default"}
		}
	}
	sort.Strings(profiles)
	return profiles
}

func normalizeMorphologyProfile(raw string) string {
	profile := strings.TrimSpace(strings.ToLower(raw))
	profile = strings.ReplaceAll(profile, "-", "_")
	return profile
}
