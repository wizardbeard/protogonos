package morphology

import (
	"testing"

	"protogonos/internal/substrate"
)

func TestGetSubstrateCPPsForStaticPlasticity(t *testing.T) {
	cpps, err := GetSubstrateCPPs(2, "none")
	if err != nil {
		t.Fatalf("get static cpps: %v", err)
	}
	want := []SubstrateComponentSpec{
		{Name: CartesianCPPName, Type: SubstrateCPPType, VL: 4},
		{Name: CentripitalDistancesCPPName, Type: SubstrateCPPType, VL: 2},
		{Name: CartesianDistanceCPPName, Type: SubstrateCPPType, VL: 1},
		{Name: CartesianCoordDiffsCPPName, Type: SubstrateCPPType, VL: 2},
		{Name: CartesianGaussedCoordDiffName, Type: SubstrateCPPType, VL: 2},
		{Name: PolarCPPName, Type: SubstrateCPPType, VL: 4},
	}
	assertSubstrateSpecs(t, cpps, want)
}

func TestGetSubstrateCPPsForIterativePlasticity(t *testing.T) {
	cpps, err := GetSubstrateCPPs(3, "iterative")
	if err != nil {
		t.Fatalf("get iterative cpps: %v", err)
	}
	want := []SubstrateComponentSpec{
		{Name: CartesianCPPName, Type: SubstrateCPPType, VL: 9},
		{Name: CentripitalDistancesCPPName, Type: SubstrateCPPType, VL: 5},
		{Name: CartesianDistanceCPPName, Type: SubstrateCPPType, VL: 4},
		{Name: CartesianCoordDiffsCPPName, Type: SubstrateCPPType, VL: 6},
		{Name: CartesianGaussedCoordDiffName, Type: SubstrateCPPType, VL: 6},
		{Name: IOWCPPName, Type: SubstrateCPPType, VL: 3},
		{Name: SphericalCPPName, Type: SubstrateCPPType, VL: 9},
	}
	assertSubstrateSpecs(t, cpps, want)
}

func TestGetSubstrateCPPsForABCNPlasticity(t *testing.T) {
	cpps, err := GetSubstrateCPPs(1, "abcn")
	if err != nil {
		t.Fatalf("get abcn cpps: %v", err)
	}
	want := []SubstrateComponentSpec{
		{Name: CartesianCPPName, Type: SubstrateCPPType, VL: 5},
		{Name: CentripitalDistancesCPPName, Type: SubstrateCPPType, VL: 5},
		{Name: CartesianDistanceCPPName, Type: SubstrateCPPType, VL: 4},
		{Name: CartesianCoordDiffsCPPName, Type: SubstrateCPPType, VL: 4},
		{Name: CartesianGaussedCoordDiffName, Type: SubstrateCPPType, VL: 4},
		{Name: IOWCPPName, Type: SubstrateCPPType, VL: 3},
	}
	assertSubstrateSpecs(t, cpps, want)
}

func TestGetSubstrateCEPsByPlasticity(t *testing.T) {
	tests := []struct {
		plasticity string
		want       SubstrateComponentSpec
	}{
		{plasticity: "iterative", want: SubstrateComponentSpec{Name: substrate.DefaultCEPName, Type: SubstrateCEPType, VL: 1}},
		{plasticity: "abcn", want: SubstrateComponentSpec{Name: substrate.SetABCNCEPName, Type: SubstrateCEPType, VL: 5}},
		{plasticity: "none", want: SubstrateComponentSpec{Name: substrate.SetWeightCEPName, Type: SubstrateCEPType, VL: 1}},
		{plasticity: "modular-none", want: SubstrateComponentSpec{Name: substrate.WeightExpressionCEPName, Type: SubstrateCEPType, VL: 2}},
	}
	for _, tt := range tests {
		got, err := GetSubstrateCEPs(2, tt.plasticity)
		if err != nil {
			t.Fatalf("get ceps for %s: %v", tt.plasticity, err)
		}
		assertSubstrateSpecs(t, got, []SubstrateComponentSpec{tt.want})
	}
}

func TestGetInitSubstrateSelectorsReturnFirstCopiedSpec(t *testing.T) {
	cpps, err := GetInitSubstrateCPPs(2, "none")
	if err != nil {
		t.Fatalf("get init cpps: %v", err)
	}
	if len(cpps) != 1 || cpps[0].Name != CartesianCPPName {
		t.Fatalf("unexpected init cpps: %v", cpps)
	}
	cpps[0].Name = "mutated"
	again, err := GetInitSubstrateCPPs(2, "none")
	if err != nil {
		t.Fatalf("get init cpps again: %v", err)
	}
	if again[0].Name == "mutated" {
		t.Fatalf("expected copied init cpp spec, got=%v", again)
	}

	ceps, err := GetInitSubstrateCEPs(2, "modular_none")
	if err != nil {
		t.Fatalf("get init ceps: %v", err)
	}
	if len(ceps) != 1 || ceps[0].Name != substrate.WeightExpressionCEPName || ceps[0].VL != 2 {
		t.Fatalf("unexpected init ceps: %v", ceps)
	}
}

func TestGetSubstrateSelectorsValidateInputs(t *testing.T) {
	if _, err := GetSubstrateCPPs(0, "none"); err == nil {
		t.Fatal("expected dimension validation")
	}
	if _, err := GetSubstrateCPPs(2, "unknown"); err == nil {
		t.Fatal("expected cpp plasticity validation")
	}
	if _, err := GetSubstrateCEPs(2, "unknown"); err == nil {
		t.Fatal("expected cep plasticity validation")
	}
}

func assertSubstrateSpecs(t *testing.T, got, want []SubstrateComponentSpec) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("unexpected spec count: got=%v want=%v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected spec at %d: got=%+v want=%+v all=%v", i, got[i], want[i], got)
		}
	}
}
