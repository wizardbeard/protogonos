package genotype

import (
	"testing"

	"protogonos/internal/model"
)

func TestComputeGenomeSignatureIncludesTopologyLinkSummary(t *testing.T) {
	genome := model.Genome{
		ID:          "g1",
		SensorIDs:   []string{"s1", "s2"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
			{ID: "n2", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "sn1", From: "n1", To: "n2", Weight: 0.5, Enabled: true},
			{ID: "sn2", From: "n2", To: "n1", Weight: -0.25, Enabled: true, Recurrent: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: "s1", NeuronID: "n1"},
			{SensorID: "s2", NeuronID: "n2"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "n1", ActuatorID: "a1"},
		},
	}

	signature := ComputeGenomeSignature(genome)
	if signature.Fingerprint == "" {
		t.Fatal("expected non-empty fingerprint")
	}
	if signature.Summary.Type != "neural" {
		t.Fatalf("expected neural type, got=%q", signature.Summary.Type)
	}
	if signature.Summary.TotalNILs != 4 {
		t.Fatalf("expected 4 total neuron input links, got=%d", signature.Summary.TotalNILs)
	}
	if signature.Summary.TotalNOLs != 3 {
		t.Fatalf("expected 3 total neuron output links, got=%d", signature.Summary.TotalNOLs)
	}
	if signature.Summary.TotalNROs != 1 {
		t.Fatalf("expected 1 total neuron recurrent output, got=%d", signature.Summary.TotalNROs)
	}
	if signature.Summary.ActivationDistribution["identity"] != 2 {
		t.Fatalf("expected activation distribution identity=2, got=%v", signature.Summary.ActivationDistribution)
	}
	if signature.Summary.AggregatorDistribution["dot_product"] != 2 {
		t.Fatalf("expected default aggregator distribution dot_product=2, got=%v", signature.Summary.AggregatorDistribution)
	}
}

func TestComputeGenomeSignatureIncludesEncodingTypeInFingerprint(t *testing.T) {
	base := model.Genome{
		ID:          "g-base",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
		},
	}
	neural := ComputeGenomeSignature(base)
	withSubstrate := base
	withSubstrate.Substrate = &model.SubstrateConfig{
		CPPName: "set_weight",
		CEPName: "identity",
	}
	substrate := ComputeGenomeSignature(withSubstrate)
	if substrate.Summary.Type != "substrate" {
		t.Fatalf("expected substrate type, got=%q", substrate.Summary.Type)
	}
	if neural.Fingerprint == substrate.Fingerprint {
		t.Fatalf("expected different fingerprints when encoding type differs: %s", neural.Fingerprint)
	}
}
