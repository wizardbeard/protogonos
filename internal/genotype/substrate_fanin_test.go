package genotype

import (
	"reflect"
	"testing"

	"protogonos/internal/model"
)

func TestSubstrateCEPFaninPIDsOrderedUniqueFromCEPLinks(t *testing.T) {
	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CEPIDs: []string{"cep-1", "cep-2"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "n0", ActuatorID: "out"},
			{NeuronID: "n2", ActuatorID: "cep-2"},
			{NeuronID: "n2", ActuatorID: "cep-1"},
			{NeuronID: "n1", ActuatorID: "cep-1"},
			{NeuronID: "n3", ActuatorID: "cep-2"},
		},
	}

	got := SubstrateCEPFaninPIDs(genome)
	want := []string{"n2", "n1", "n3"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected cep fan-in ids: got=%v want=%v", got, want)
	}
}

func TestSubstrateCEPFaninPIDsReturnsNilWhenNoCEPEndpointLinks(t *testing.T) {
	genome := model.Genome{
		Substrate: &model.SubstrateConfig{
			CEPIDs: []string{"cep-1"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "n1", ActuatorID: "out"},
		},
	}

	if got := SubstrateCEPFaninPIDs(genome); got != nil {
		t.Fatalf("expected nil fan-in ids, got=%v", got)
	}
}
