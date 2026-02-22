package genotype

import (
	"context"
	"testing"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/storage"
)

func TestConstructSeedPopulationXOR(t *testing.T) {
	seed, err := ConstructSeedPopulation("xor", 3, 7)
	if err != nil {
		t.Fatalf("construct xor population: %v", err)
	}
	if len(seed.Genomes) != 3 {
		t.Fatalf("expected 3 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 2 || seed.InputNeuronIDs[0] != "i1" || seed.InputNeuronIDs[1] != "i2" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "o" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 2 || seed.Genomes[0].SensorIDs[0] != protoio.XORInputLeftSensorName || seed.Genomes[0].SensorIDs[1] != protoio.XORInputRightSensorName {
		t.Fatalf("unexpected xor sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.XOROutputActuatorName {
		t.Fatalf("unexpected xor actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationRegressionMimic(t *testing.T) {
	seed, err := ConstructSeedPopulation("regression-mimic", 2, 9)
	if err != nil {
		t.Fatalf("construct regression population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 1 || seed.InputNeuronIDs[0] != "i" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "o" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 1 || seed.Genomes[0].SensorIDs[0] != protoio.ScalarInputSensorName {
		t.Fatalf("unexpected regression sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.ScalarOutputActuatorName {
		t.Fatalf("unexpected regression actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationCartPoleLite(t *testing.T) {
	seed, err := ConstructSeedPopulation("cart-pole-lite", 2, 13)
	if err != nil {
		t.Fatalf("construct cart-pole-lite population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 2 || seed.InputNeuronIDs[0] != "x" || seed.InputNeuronIDs[1] != "v" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "f" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 2 || seed.Genomes[0].SensorIDs[0] != protoio.CartPolePositionSensorName || seed.Genomes[0].SensorIDs[1] != protoio.CartPoleVelocitySensorName {
		t.Fatalf("unexpected cart-pole sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.CartPoleForceActuatorName {
		t.Fatalf("unexpected cart-pole actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationFlatland(t *testing.T) {
	seed, err := ConstructSeedPopulation("flatland", 2, 19)
	if err != nil {
		t.Fatalf("construct flatland population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 2 || seed.InputNeuronIDs[0] != "d" || seed.InputNeuronIDs[1] != "e" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "m" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 2 || seed.Genomes[0].SensorIDs[0] != protoio.FlatlandDistanceSensorName || seed.Genomes[0].SensorIDs[1] != protoio.FlatlandEnergySensorName {
		t.Fatalf("unexpected flatland sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.FlatlandMoveActuatorName {
		t.Fatalf("unexpected flatland actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationGTSA(t *testing.T) {
	seed, err := ConstructSeedPopulation("gtsa", 2, 23)
	if err != nil {
		t.Fatalf("construct gtsa population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 1 || seed.InputNeuronIDs[0] != "x" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "y" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 1 || seed.Genomes[0].SensorIDs[0] != protoio.GTSAInputSensorName {
		t.Fatalf("unexpected gtsa sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.GTSAPredictActuatorName {
		t.Fatalf("unexpected gtsa actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationFX(t *testing.T) {
	seed, err := ConstructSeedPopulation("fx", 2, 29)
	if err != nil {
		t.Fatalf("construct fx population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 2 || seed.InputNeuronIDs[0] != "p" || seed.InputNeuronIDs[1] != "s" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "t" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 2 || seed.Genomes[0].SensorIDs[0] != protoio.FXPriceSensorName || seed.Genomes[0].SensorIDs[1] != protoio.FXSignalSensorName {
		t.Fatalf("unexpected fx sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.FXTradeActuatorName {
		t.Fatalf("unexpected fx actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationUnsupportedScape(t *testing.T) {
	_, err := ConstructSeedPopulation("unknown", 1, 1)
	if err == nil {
		t.Fatal("expected unsupported scape error")
	}
}

func TestCloneAgent(t *testing.T) {
	in := model.Genome{
		ID:          "g1",
		Neurons:     []model.Neuron{{ID: "n1", Activation: "identity"}},
		Synapses:    []model.Synapse{{ID: "s1", From: "n1", To: "n1", Weight: 0.5, Enabled: true}},
		SensorIDs:   []string{protoio.ScalarInputSensorName},
		ActuatorIDs: []string{protoio.ScalarOutputActuatorName},
	}

	clone := CloneAgent(in, "g2")
	if clone.ID != "g2" {
		t.Fatalf("expected cloned id g2, got %q", clone.ID)
	}
	clone.Neurons[0].Activation = "sigmoid"
	if in.Neurons[0].Activation != "identity" {
		t.Fatal("expected original genome to remain unchanged")
	}
}

func TestDeleteAgentFromPopulation(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}

	for _, id := range []string{"a1", "a2", "a3"} {
		if err := store.SaveGenome(ctx, model.Genome{ID: id}); err != nil {
			t.Fatalf("save genome %s: %v", id, err)
		}
	}

	pop := model.Population{ID: "pop-1", AgentIDs: []string{"a1", "a2", "a3"}}
	if err := store.SavePopulation(ctx, pop); err != nil {
		t.Fatalf("save population: %v", err)
	}

	if err := DeleteAgentFromPopulation(ctx, store, "pop-1", "a2"); err != nil {
		t.Fatalf("delete agent: %v", err)
	}

	updated, ok, err := store.GetPopulation(ctx, "pop-1")
	if err != nil || !ok {
		t.Fatalf("get population err=%v ok=%t", err, ok)
	}
	if len(updated.AgentIDs) != 2 || updated.AgentIDs[0] != "a1" || updated.AgentIDs[1] != "a3" {
		t.Fatalf("unexpected agent ids after delete: %#v", updated.AgentIDs)
	}
	if _, ok, err := store.GetGenome(ctx, "a2"); err != nil {
		t.Fatalf("get removed genome: %v", err)
	} else if ok {
		t.Fatal("expected removed agent genome to be deleted")
	}
	if _, ok, err := store.GetGenome(ctx, "a1"); err != nil {
		t.Fatalf("get retained genome: %v", err)
	} else if !ok {
		t.Fatal("expected retained agent genome to remain")
	}
}
