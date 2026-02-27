package morphology

import (
	"strings"
	"testing"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestEnsureGenomeIOCompatibility(t *testing.T) {
	okGenome := model.Genome{
		ID:          "g-ok",
		SensorIDs:   []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: protoio.XORInputLeftSensorName, NeuronID: "i"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "o", ActuatorID: protoio.XOROutputActuatorName},
		},
	}
	if err := EnsureGenomeIOCompatibility("xor", okGenome); err != nil {
		t.Fatalf("expected xor genome compatibility, got err=%v", err)
	}

	badSensor := model.Genome{
		ID:          "g-bad",
		SensorIDs:   []string{protoio.ScalarInputSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
	}
	err := EnsureGenomeIOCompatibility("xor", badSensor)
	if err == nil {
		t.Fatal("expected incompatible sensor error")
	}
	if !strings.Contains(err.Error(), "incompatible") {
		t.Fatalf("expected incompatible in error, got %v", err)
	}
}

func TestEnsureGenomeIOCompatibilityValidatesLinkIntegrity(t *testing.T) {
	base := model.Genome{
		ID:          "g-links",
		SensorIDs:   []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
	}

	withUnknownSensor := base
	withUnknownSensor.SensorNeuronLinks = []model.SensorNeuronLink{{SensorID: "missing_sensor", NeuronID: "i"}}
	if err := EnsureGenomeIOCompatibility("xor", withUnknownSensor); err == nil || !strings.Contains(err.Error(), "unknown sensor") {
		t.Fatalf("expected unknown sensor-link error, got=%v", err)
	}

	withUnknownActuator := base
	withUnknownActuator.NeuronActuatorLinks = []model.NeuronActuatorLink{{NeuronID: "o", ActuatorID: "missing_actuator"}}
	if err := EnsureGenomeIOCompatibility("xor", withUnknownActuator); err == nil || !strings.Contains(err.Error(), "unknown actuator") {
		t.Fatalf("expected unknown actuator-link error, got=%v", err)
	}

	withUnknownNeuron := base
	withUnknownNeuron.SensorNeuronLinks = []model.SensorNeuronLink{{SensorID: protoio.XORInputLeftSensorName, NeuronID: "missing_neuron"}}
	if err := EnsureGenomeIOCompatibility("xor", withUnknownNeuron); err == nil || !strings.Contains(err.Error(), "unknown neuron") {
		t.Fatalf("expected unknown neuron-link error, got=%v", err)
	}
}

func TestEnsureGenomeIOCompatibilityAllowsSubstrateEndpointLinks(t *testing.T) {
	genome := model.Genome{
		ID:          "g-sub-links",
		SensorIDs:   []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Substrate: &model.SubstrateConfig{
			CPPIDs: []string{"substrate_cpp_0"},
			CEPIDs: []string{"substrate_cep_0"},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: "substrate_cpp_0", NeuronID: "i"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "o", ActuatorID: "substrate_cep_0"},
		},
	}
	if err := EnsureGenomeIOCompatibility("xor", genome); err != nil {
		t.Fatalf("expected substrate endpoint links to be accepted, got err=%v", err)
	}
}

func TestEnsureScapeCompatibilitySupportsReferenceAliases(t *testing.T) {
	aliases := []string{
		"xor_sim",
		"pb_sim",
		"dtm_sim",
		"flatland_sim",
		"scape_flatland",
		"gtsa_sim",
		"fx_sim",
		"scape_fx_sim",
		"scape_GTSA",
		"epitopes_sim",
		"scape_epitopes_sim",
		"scape_LLVMPhaseOrdering",
		"llvm_phase_ordering_sim",
	}
	for _, alias := range aliases {
		if err := EnsureScapeCompatibility(alias); err != nil {
			t.Fatalf("ensure compatibility alias=%s: %v", alias, err)
		}
	}
}
