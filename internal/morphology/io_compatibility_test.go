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
