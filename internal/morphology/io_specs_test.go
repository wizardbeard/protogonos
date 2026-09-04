package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

func TestReferenceSensorSpecsMirrorCurrentScopeRecords(t *testing.T) {
	tests := []struct {
		scape         string
		profile       string
		wantName      string
		wantReference string
		wantScapeKind string
		wantScapeName string
		wantVL        int
	}{
		{"xor_mimic", "", protoio.XORInputLeftSensorName, protoio.XORGetInputSensorAliasName, ScapePrivate, "xor_sim", 2},
		{"pole_balancing", "", protoio.Pole2CartPositionSensorName, protoio.PBGetInputSensorAliasName, ScapePrivate, "pb_sim", 3},
		{"discrete_tmaze", "", protoio.DTMRangeFrontSensorName, protoio.DTMGetInputSensorAliasName, ScapePrivate, "dtm_sim", 4},
		{"forex_trader", "", protoio.FXPercentChangeSensorName, protoio.FXPLISensorAliasName, ScapePrivate, "fx_sim", 100},
		{"general_predictor", "", protoio.GTSAInputSensorName, protoio.GeneralPredictorSensorAliasName, ScapePrivate, "scape_GTSA", 30},
		{"epitopes", "", protoio.EpitopesSignalSensorName, protoio.ABCPredSensorAliasName, ScapePrivate, "epitopes", 336},
		{"llvm-phase-ordering", "", protoio.LLVMRuntimeGainSensorName, protoio.BitCodeStatisticsSensorAliasName, ScapePrivate, "scape_LLVMPhaseOrdering", 31},
	}

	for _, tt := range tests {
		specs, err := GetReferenceSensorSpecs(tt.scape, tt.profile)
		if err != nil {
			t.Fatalf("GetReferenceSensorSpecs(%q, %q): %v", tt.scape, tt.profile, err)
		}
		if len(specs) == 0 {
			t.Fatalf("GetReferenceSensorSpecs(%q, %q) returned no specs", tt.scape, tt.profile)
		}
		got := specs[0]
		if got.Name != tt.wantName || got.ReferenceName != tt.wantReference || got.Type != IOTypeStandard {
			t.Fatalf("unexpected sensor spec identity for %s: %+v", tt.scape, got)
		}
		if got.ScapeKind != tt.wantScapeKind || got.ScapeName != tt.wantScapeName || got.VL != tt.wantVL {
			t.Fatalf("unexpected sensor scape/VL for %s: %+v", tt.scape, got)
		}
	}
}

func TestReferenceActuatorSpecsMirrorCurrentScopeRecords(t *testing.T) {
	tests := []struct {
		scape         string
		wantName      string
		wantReference string
		wantScapeKind string
		wantScapeName string
		wantVL        int
	}{
		{"xor_mimic", protoio.XOROutputActuatorName, protoio.XORSendOutputActuatorAliasName, ScapePrivate, "xor_sim", 1},
		{"pole_balancing", protoio.Pole2PushActuatorName, protoio.PBSendOutputActuatorAliasName, ScapePrivate, "pb_sim", 1},
		{"discrete_tmaze", protoio.DTMMoveActuatorName, protoio.DTMSendOutputActuatorAliasName, ScapePrivate, "dtm_sim", 1},
		{"prey", protoio.FlatlandTwoWheelsActuatorName, protoio.TwoWheelsActuatorAliasName, ScapePublic, "flatland", 2},
		{"forex_trader", protoio.FXTradeActuatorName, protoio.FXTradeActuatorAliasName, ScapePrivate, "fx_sim", 1},
		{"general_predictor", protoio.GTSAPredictActuatorName, protoio.GeneralPredictorActuatorAliasName, ScapePrivate, "scape_GTSA", 1},
		{"epitopes", protoio.EpitopesResponseActuatorName, protoio.ABCPredActuatorAliasName, ScapePrivate, "epitopes", 1},
		{"llvm-phase-ordering", protoio.LLVMPhaseActuatorName, protoio.ChooseOptimizationPhaseActuatorAliasName, ScapePrivate, "scape_LLVMPhaseOrdering", 55},
	}

	for _, tt := range tests {
		specs, err := GetReferenceActuatorSpecs(tt.scape, "")
		if err != nil {
			t.Fatalf("GetReferenceActuatorSpecs(%q): %v", tt.scape, err)
		}
		if len(specs) != 1 {
			t.Fatalf("GetReferenceActuatorSpecs(%q) len=%d want=1", tt.scape, len(specs))
		}
		got := specs[0]
		if got.Name != tt.wantName || got.ReferenceName != tt.wantReference || got.Type != IOTypeStandard {
			t.Fatalf("unexpected actuator spec identity for %s: %+v", tt.scape, got)
		}
		if got.ScapeKind != tt.wantScapeKind || got.ScapeName != tt.wantScapeName || got.VL != tt.wantVL {
			t.Fatalf("unexpected actuator scape/VL for %s: %+v", tt.scape, got)
		}
	}
}

func TestReferenceFlatlandSensorSpecsPreserveScannerRecords(t *testing.T) {
	specs, err := GetReferenceSensorSpecs("prey", "")
	if err != nil {
		t.Fatalf("GetReferenceSensorSpecs(prey): %v", err)
	}
	if len(specs) != 3 {
		t.Fatalf("expected distance/color/energy scanner specs, got=%v", specs)
	}
	wantReferences := []string{
		protoio.DistanceScannerSensorAliasName,
		protoio.ColorScannerSensorAliasName,
		protoio.EnergyScannerSensorAliasName,
	}
	for i, want := range wantReferences {
		if specs[i].ReferenceName != want || specs[i].ScapeKind != ScapePublic || specs[i].VL != 5 {
			t.Fatalf("unexpected flatland scanner spec %d: %+v", i, specs[i])
		}
		if len(specs[i].Parameters) != 3 || specs[i].Parameters[1] != "5" {
			t.Fatalf("unexpected flatland scanner parameters %d: %+v", i, specs[i])
		}
	}
}

func TestInitReferenceSpecsReturnFirstRecord(t *testing.T) {
	sensors, err := GetInitReferenceSensorSpecs("prey", "")
	if err != nil {
		t.Fatalf("GetInitReferenceSensorSpecs(prey): %v", err)
	}
	if len(sensors) != 1 || sensors[0].ReferenceName != protoio.DistanceScannerSensorAliasName {
		t.Fatalf("unexpected init sensor specs: %+v", sensors)
	}

	actuators, err := GetInitReferenceActuatorSpecs("llvm-phase-ordering", "")
	if err != nil {
		t.Fatalf("GetInitReferenceActuatorSpecs(llvm): %v", err)
	}
	if len(actuators) != 1 || actuators[0].ReferenceName != protoio.ChooseOptimizationPhaseActuatorAliasName {
		t.Fatalf("unexpected init actuator specs: %+v", actuators)
	}
}

func TestReferenceSpecUnsupportedInputs(t *testing.T) {
	if _, err := GetReferenceSensorSpecs("unknown", ""); err == nil {
		t.Fatal("expected unsupported reference sensor morphology error")
	}
	if _, err := GetReferenceActuatorSpecs("pole2-balancing", "unsupported"); err != nil {
		t.Fatalf("actuator specs should not depend on pole2 sensor profile: %v", err)
	}
	if _, err := GetReferenceSensorSpecs("pole2-balancing", "unsupported"); err == nil {
		t.Fatal("expected unsupported pole2 sensor profile error")
	}
}
