package morphology

import (
	"testing"

	protoio "protogonos/internal/io"
)

func TestLLVMPhaseOrderingMorphologyCompatibility(t *testing.T) {
	m := LLVMPhaseOrderingMorphology{}
	if !m.Compatible("llvm-phase-ordering") {
		t.Fatal("expected llvm-phase-ordering to be compatible")
	}
	if m.Compatible("xor") {
		t.Fatal("expected xor to be incompatible")
	}
}

func TestEnsureScapeCompatibilityLLVMPhaseOrdering(t *testing.T) {
	if err := EnsureScapeCompatibility("llvm-phase-ordering"); err != nil {
		t.Fatalf("ensure compatibility: %v", err)
	}
}

func TestLLVMPhaseOrderingMorphologyDefaultSensors(t *testing.T) {
	m := LLVMPhaseOrderingMorphology{}
	sensors := m.Sensors()
	if len(sensors) != 5 {
		t.Fatalf("expected 5 llvm sensors, got %d: %#v", len(sensors), sensors)
	}
	if sensors[0] != protoio.LLVMComplexitySensorName ||
		sensors[1] != protoio.LLVMPassIndexSensorName ||
		sensors[2] != protoio.LLVMAlignmentSensorName ||
		sensors[3] != protoio.LLVMDiversitySensorName ||
		sensors[4] != protoio.LLVMRuntimeGainSensorName {
		t.Fatalf("unexpected llvm sensor order: %#v", sensors)
	}
}

func TestLLVMPhaseOrderingCoreMorphologySensors(t *testing.T) {
	m := LLVMPhaseOrderingCoreMorphology{}
	sensors := m.Sensors()
	if len(sensors) != 2 {
		t.Fatalf("expected 2 llvm core sensors, got %d: %#v", len(sensors), sensors)
	}
	if sensors[0] != protoio.LLVMComplexitySensorName ||
		sensors[1] != protoio.LLVMPassIndexSensorName {
		t.Fatalf("unexpected llvm core sensor order: %#v", sensors)
	}
	if err := ValidateRegisteredComponents("llvm-phase-ordering", m); err != nil {
		t.Fatalf("validate llvm core morphology: %v", err)
	}
}
