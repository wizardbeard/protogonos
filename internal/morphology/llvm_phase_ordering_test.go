package morphology

import "testing"

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
