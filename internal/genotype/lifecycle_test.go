package genotype

import (
	"context"
	"math/rand"
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

func TestConstructSeedPopulationPole2Balancing(t *testing.T) {
	seed, err := ConstructSeedPopulation("pole2-balancing", 2, 17)
	if err != nil {
		t.Fatalf("construct pole2-balancing population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 9 ||
		seed.InputNeuronIDs[0] != "x" ||
		seed.InputNeuronIDs[1] != "v" ||
		seed.InputNeuronIDs[2] != "a1" ||
		seed.InputNeuronIDs[3] != "w1" ||
		seed.InputNeuronIDs[4] != "a2" ||
		seed.InputNeuronIDs[5] != "w2" ||
		seed.InputNeuronIDs[6] != "rp" ||
		seed.InputNeuronIDs[7] != "sp" ||
		seed.InputNeuronIDs[8] != "fs" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "f" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 9 ||
		seed.Genomes[0].SensorIDs[0] != protoio.Pole2CartPositionSensorName ||
		seed.Genomes[0].SensorIDs[1] != protoio.Pole2CartVelocitySensorName ||
		seed.Genomes[0].SensorIDs[2] != protoio.Pole2Angle1SensorName ||
		seed.Genomes[0].SensorIDs[3] != protoio.Pole2Velocity1SensorName ||
		seed.Genomes[0].SensorIDs[4] != protoio.Pole2Angle2SensorName ||
		seed.Genomes[0].SensorIDs[5] != protoio.Pole2Velocity2SensorName ||
		seed.Genomes[0].SensorIDs[6] != protoio.Pole2RunProgressSensorName ||
		seed.Genomes[0].SensorIDs[7] != protoio.Pole2StepProgressSensorName ||
		seed.Genomes[0].SensorIDs[8] != protoio.Pole2FitnessSignalSensorName {
		t.Fatalf("unexpected pole2 sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.Pole2PushActuatorName {
		t.Fatalf("unexpected pole2 actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
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
	wantInputs := flatlandSeedInputNeuronIDs()
	if len(seed.InputNeuronIDs) != len(wantInputs) {
		t.Fatalf("unexpected input count: got=%d want=%d ids=%#v", len(seed.InputNeuronIDs), len(wantInputs), seed.InputNeuronIDs)
	}
	for i, want := range wantInputs {
		if seed.InputNeuronIDs[i] != want {
			t.Fatalf("unexpected input ids at index %d: got=%#v want=%#v", i, seed.InputNeuronIDs, wantInputs)
		}
	}

	wantOutputs := flatlandSeedOutputNeuronIDs()
	if len(seed.OutputNeuronIDs) != len(wantOutputs) {
		t.Fatalf("unexpected output count: got=%d want=%d ids=%#v", len(seed.OutputNeuronIDs), len(wantOutputs), seed.OutputNeuronIDs)
	}
	for i, want := range wantOutputs {
		if seed.OutputNeuronIDs[i] != want {
			t.Fatalf("unexpected output ids at index %d: got=%#v want=%#v", i, seed.OutputNeuronIDs, wantOutputs)
		}
	}

	wantSensors := flatlandSeedSensorIDs()
	if len(seed.Genomes[0].SensorIDs) != len(wantSensors) {
		t.Fatalf("unexpected flatland sensor count: got=%d want=%d ids=%#v", len(seed.Genomes[0].SensorIDs), len(wantSensors), seed.Genomes[0].SensorIDs)
	}
	for i, want := range wantSensors {
		if seed.Genomes[0].SensorIDs[i] != want {
			t.Fatalf("unexpected flatland sensor ids at index %d: got=%#v want=%#v", i, seed.Genomes[0].SensorIDs, wantSensors)
		}
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.FlatlandTwoWheelsActuatorName {
		t.Fatalf("unexpected flatland actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
	if len(seed.Genomes[0].Neurons) != len(wantInputs)+len(wantOutputs) {
		t.Fatalf("unexpected flatland seed neuron count: got=%d want=%d", len(seed.Genomes[0].Neurons), len(wantInputs)+len(wantOutputs))
	}
	if len(seed.Genomes[0].Synapses) != len(wantInputs)*len(wantOutputs) {
		t.Fatalf("unexpected flatland seed synapse count: got=%d want=%d", len(seed.Genomes[0].Synapses), len(wantInputs)*len(wantOutputs))
	}
}

func TestConstructSeedPopulationFlatlandClassicProfile(t *testing.T) {
	seed, err := ConstructSeedPopulationWithOptions("flatland", 2, 19, SeedPopulationOptions{
		FlatlandProfile: FlatlandSeedProfileClassic,
	})
	if err != nil {
		t.Fatalf("construct flatland classic profile: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 2 || seed.InputNeuronIDs[0] != "d" || seed.InputNeuronIDs[1] != "e" {
		t.Fatalf("unexpected classic flatland input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "m" {
		t.Fatalf("unexpected classic flatland output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 2 ||
		seed.Genomes[0].SensorIDs[0] != protoio.FlatlandDistanceSensorName ||
		seed.Genomes[0].SensorIDs[1] != protoio.FlatlandEnergySensorName {
		t.Fatalf("unexpected classic flatland sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.FlatlandMoveActuatorName {
		t.Fatalf("unexpected classic flatland actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
	if len(seed.Genomes[0].Neurons) != 3 {
		t.Fatalf("unexpected classic flatland neuron count: %d", len(seed.Genomes[0].Neurons))
	}
	if len(seed.Genomes[0].Synapses) != 2 {
		t.Fatalf("unexpected classic flatland synapse count: %d", len(seed.Genomes[0].Synapses))
	}
}

func TestConstructSeedPopulationFlatlandScannerProfileAlias(t *testing.T) {
	seed, err := ConstructSeedPopulationWithOptions("flatland", 2, 19, SeedPopulationOptions{
		FlatlandProfile: "flatland_prey",
	})
	if err != nil {
		t.Fatalf("construct flatland scanner alias profile: %v", err)
	}
	wantInputs := flatlandSeedInputNeuronIDs()
	if len(seed.InputNeuronIDs) != len(wantInputs) {
		t.Fatalf("unexpected scanner profile input count: got=%d want=%d ids=%v", len(seed.InputNeuronIDs), len(wantInputs), seed.InputNeuronIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.FlatlandTwoWheelsActuatorName {
		t.Fatalf("unexpected scanner profile actuator ids: %v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationFlatlandScannerDensityProfileCoreMasksEdgeWeights(t *testing.T) {
	balancedSeed, err := ConstructSeedPopulationWithOptions("flatland", 1, 19, SeedPopulationOptions{
		FlatlandProfile:        FlatlandSeedProfileScanner,
		FlatlandScannerProfile: FlatlandScannerSeedProfileBalanced5,
	})
	if err != nil {
		t.Fatalf("construct balanced scanner profile: %v", err)
	}
	coreSeed, err := ConstructSeedPopulationWithOptions("flatland", 1, 19, SeedPopulationOptions{
		FlatlandProfile:        FlatlandSeedProfileScanner,
		FlatlandScannerProfile: FlatlandScannerSeedProfileCore3,
	})
	if err != nil {
		t.Fatalf("construct core scanner profile: %v", err)
	}

	balancedLeft, ok := findSynapseWeight(balancedSeed.Genomes[0], "d0", "wl")
	if !ok {
		t.Fatalf("missing d0->wl synapse in balanced scanner profile")
	}
	coreLeft, ok := findSynapseWeight(coreSeed.Genomes[0], "d0", "wl")
	if !ok {
		t.Fatalf("missing d0->wl synapse in core scanner profile")
	}
	if balancedLeft == 0 {
		t.Fatalf("expected balanced edge weight to remain active, got %f", balancedLeft)
	}
	if coreLeft != 0 {
		t.Fatalf("expected core edge weight to be masked to 0, got %f", coreLeft)
	}

	coreCenter, ok := findSynapseWeight(coreSeed.Genomes[0], "d2", "wl")
	if !ok {
		t.Fatalf("missing d2->wl synapse in core scanner profile")
	}
	if coreCenter == 0 {
		t.Fatalf("expected core center weight to remain active, got %f", coreCenter)
	}
}

func TestConstructSeedPopulationFlatlandUnsupportedScannerProfile(t *testing.T) {
	_, err := ConstructSeedPopulationWithOptions("flatland", 1, 19, SeedPopulationOptions{
		FlatlandProfile:        FlatlandSeedProfileScanner,
		FlatlandScannerProfile: "invalid-density-profile",
	})
	if err == nil {
		t.Fatal("expected unsupported flatland scanner profile error")
	}
}

func TestConstructSeedPopulationFlatlandUnsupportedProfile(t *testing.T) {
	_, err := ConstructSeedPopulationWithOptions("flatland", 1, 19, SeedPopulationOptions{
		FlatlandProfile: "invalid-profile",
	})
	if err == nil {
		t.Fatal("expected unsupported flatland profile error")
	}
}

func findSynapseWeight(genome model.Genome, from string, to string) (float64, bool) {
	for _, synapse := range genome.Synapses {
		if synapse.From == from && synapse.To == to {
			return synapse.Weight, true
		}
	}
	return 0, false
}

func TestConstructSeedPopulationDTM(t *testing.T) {
	seed, err := ConstructSeedPopulation("dtm", 2, 21)
	if err != nil {
		t.Fatalf("construct dtm population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 7 ||
		seed.InputNeuronIDs[0] != "rl" ||
		seed.InputNeuronIDs[1] != "rf" ||
		seed.InputNeuronIDs[2] != "rr" ||
		seed.InputNeuronIDs[3] != "r" ||
		seed.InputNeuronIDs[4] != "rp" ||
		seed.InputNeuronIDs[5] != "sp" ||
		seed.InputNeuronIDs[6] != "sw" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "m" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 7 ||
		seed.Genomes[0].SensorIDs[0] != protoio.DTMRangeLeftSensorName ||
		seed.Genomes[0].SensorIDs[1] != protoio.DTMRangeFrontSensorName ||
		seed.Genomes[0].SensorIDs[2] != protoio.DTMRangeRightSensorName ||
		seed.Genomes[0].SensorIDs[3] != protoio.DTMRewardSensorName ||
		seed.Genomes[0].SensorIDs[4] != protoio.DTMRunProgressSensorName ||
		seed.Genomes[0].SensorIDs[5] != protoio.DTMStepProgressSensorName ||
		seed.Genomes[0].SensorIDs[6] != protoio.DTMSwitchedSensorName {
		t.Fatalf("unexpected dtm sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.DTMMoveActuatorName {
		t.Fatalf("unexpected dtm actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
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
	if len(seed.InputNeuronIDs) != 4 || seed.InputNeuronIDs[0] != "x" || seed.InputNeuronIDs[1] != "d" || seed.InputNeuronIDs[2] != "w" || seed.InputNeuronIDs[3] != "p" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "y" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 4 ||
		seed.Genomes[0].SensorIDs[0] != protoio.GTSAInputSensorName ||
		seed.Genomes[0].SensorIDs[1] != protoio.GTSADeltaSensorName ||
		seed.Genomes[0].SensorIDs[2] != protoio.GTSAWindowMeanSensorName ||
		seed.Genomes[0].SensorIDs[3] != protoio.GTSAProgressSensorName {
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
	if len(seed.InputNeuronIDs) != 11 ||
		seed.InputNeuronIDs[0] != "p" ||
		seed.InputNeuronIDs[1] != "s" ||
		seed.InputNeuronIDs[2] != "m" ||
		seed.InputNeuronIDs[3] != "v" ||
		seed.InputNeuronIDs[4] != "n" ||
		seed.InputNeuronIDs[5] != "d" ||
		seed.InputNeuronIDs[6] != "q" ||
		seed.InputNeuronIDs[7] != "e" ||
		seed.InputNeuronIDs[8] != "pc" ||
		seed.InputNeuronIDs[9] != "ppc" ||
		seed.InputNeuronIDs[10] != "pr" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "t" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 11 ||
		seed.Genomes[0].SensorIDs[0] != protoio.FXPriceSensorName ||
		seed.Genomes[0].SensorIDs[1] != protoio.FXSignalSensorName ||
		seed.Genomes[0].SensorIDs[2] != protoio.FXMomentumSensorName ||
		seed.Genomes[0].SensorIDs[3] != protoio.FXVolatilitySensorName ||
		seed.Genomes[0].SensorIDs[4] != protoio.FXNAVSensorName ||
		seed.Genomes[0].SensorIDs[5] != protoio.FXDrawdownSensorName ||
		seed.Genomes[0].SensorIDs[6] != protoio.FXPositionSensorName ||
		seed.Genomes[0].SensorIDs[7] != protoio.FXEntrySensorName ||
		seed.Genomes[0].SensorIDs[8] != protoio.FXPercentChangeSensorName ||
		seed.Genomes[0].SensorIDs[9] != protoio.FXPrevPercentChangeSensorName ||
		seed.Genomes[0].SensorIDs[10] != protoio.FXProfitSensorName {
		t.Fatalf("unexpected fx sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.FXTradeActuatorName {
		t.Fatalf("unexpected fx actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationEpitopes(t *testing.T) {
	seed, err := ConstructSeedPopulation("epitopes", 2, 31)
	if err != nil {
		t.Fatalf("construct epitopes population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 5 || seed.InputNeuronIDs[0] != "s" || seed.InputNeuronIDs[1] != "m" || seed.InputNeuronIDs[2] != "t" || seed.InputNeuronIDs[3] != "p" || seed.InputNeuronIDs[4] != "g" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 1 || seed.OutputNeuronIDs[0] != "r" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 5 ||
		seed.Genomes[0].SensorIDs[0] != protoio.EpitopesSignalSensorName ||
		seed.Genomes[0].SensorIDs[1] != protoio.EpitopesMemorySensorName ||
		seed.Genomes[0].SensorIDs[2] != protoio.EpitopesTargetSensorName ||
		seed.Genomes[0].SensorIDs[3] != protoio.EpitopesProgressSensorName ||
		seed.Genomes[0].SensorIDs[4] != protoio.EpitopesMarginSensorName {
		t.Fatalf("unexpected epitopes sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.EpitopesResponseActuatorName {
		t.Fatalf("unexpected epitopes actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
}

func TestConstructSeedPopulationLLVMPhaseOrdering(t *testing.T) {
	seed, err := ConstructSeedPopulation("llvm-phase-ordering", 2, 37)
	if err != nil {
		t.Fatalf("construct llvm-phase-ordering population: %v", err)
	}
	if len(seed.Genomes) != 2 {
		t.Fatalf("expected 2 genomes, got %d", len(seed.Genomes))
	}
	if len(seed.InputNeuronIDs) != 5 || seed.InputNeuronIDs[0] != "c" || seed.InputNeuronIDs[1] != "p" || seed.InputNeuronIDs[2] != "a" || seed.InputNeuronIDs[3] != "d" || seed.InputNeuronIDs[4] != "r" {
		t.Fatalf("unexpected input ids: %#v", seed.InputNeuronIDs)
	}
	if len(seed.OutputNeuronIDs) != 55 || seed.OutputNeuronIDs[0] != "o00" || seed.OutputNeuronIDs[54] != "o54" {
		t.Fatalf("unexpected output ids: %#v", seed.OutputNeuronIDs)
	}
	if len(seed.Genomes[0].SensorIDs) != 5 ||
		seed.Genomes[0].SensorIDs[0] != protoio.LLVMComplexitySensorName ||
		seed.Genomes[0].SensorIDs[1] != protoio.LLVMPassIndexSensorName ||
		seed.Genomes[0].SensorIDs[2] != protoio.LLVMAlignmentSensorName ||
		seed.Genomes[0].SensorIDs[3] != protoio.LLVMDiversitySensorName ||
		seed.Genomes[0].SensorIDs[4] != protoio.LLVMRuntimeGainSensorName {
		t.Fatalf("unexpected llvm sensor ids: %#v", seed.Genomes[0].SensorIDs)
	}
	if len(seed.Genomes[0].ActuatorIDs) != 1 || seed.Genomes[0].ActuatorIDs[0] != protoio.LLVMPhaseActuatorName {
		t.Fatalf("unexpected llvm actuator ids: %#v", seed.Genomes[0].ActuatorIDs)
	}
	if len(seed.Genomes[0].Neurons) != 60 {
		t.Fatalf("expected 60 llvm neurons (5 inputs + 55 outputs), got %d", len(seed.Genomes[0].Neurons))
	}
	if len(seed.Genomes[0].Synapses) != 275 {
		t.Fatalf("expected 275 llvm synapses (5 per output), got %d", len(seed.Genomes[0].Synapses))
	}
}

func TestConstructSeedPopulationUnsupportedScape(t *testing.T) {
	_, err := ConstructSeedPopulation("unknown", 1, 1)
	if err == nil {
		t.Fatal("expected unsupported scape error")
	}
}

func TestConstructSeedPopulationSupportsReferenceScapeAliases(t *testing.T) {
	aliases := map[string]string{
		"xor_sim":                 "xor",
		"pb_sim":                  "pole2-balancing",
		"pb_sim1":                 "pole2-balancing",
		"dtm_sim":                 "dtm",
		"fx_sim":                  "fx",
		"scape_GTSA":              "gtsa",
		"scape_LLVMPhaseOrdering": "llvm-phase-ordering",
	}

	for alias, canonical := range aliases {
		seedAlias, err := ConstructSeedPopulation(alias, 2, 41)
		if err != nil {
			t.Fatalf("construct alias %s: %v", alias, err)
		}
		seedCanonical, err := ConstructSeedPopulation(canonical, 2, 41)
		if err != nil {
			t.Fatalf("construct canonical %s: %v", canonical, err)
		}
		if len(seedAlias.InputNeuronIDs) != len(seedCanonical.InputNeuronIDs) {
			t.Fatalf("input arity mismatch alias=%s canonical=%s alias_ids=%v canonical_ids=%v", alias, canonical, seedAlias.InputNeuronIDs, seedCanonical.InputNeuronIDs)
		}
		if len(seedAlias.OutputNeuronIDs) != len(seedCanonical.OutputNeuronIDs) {
			t.Fatalf("output arity mismatch alias=%s canonical=%s alias_ids=%v canonical_ids=%v", alias, canonical, seedAlias.OutputNeuronIDs, seedCanonical.OutputNeuronIDs)
		}
		if len(seedAlias.Genomes) != len(seedCanonical.Genomes) {
			t.Fatalf("genome count mismatch alias=%s canonical=%s", alias, canonical)
		}
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

func TestCloneAgentWithRemappedIDs(t *testing.T) {
	in := model.Genome{
		ID: "g1",
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "h", Activation: "tanh"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "h", Weight: 0.5, Enabled: true},
			{ID: "s2", From: "h", To: "o", Weight: 1.0, Enabled: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: protoio.ScalarInputSensorName, NeuronID: "i"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "h", ActuatorID: protoio.ScalarOutputActuatorName},
		},
		SensorIDs:   []string{protoio.ScalarInputSensorName},
		ActuatorIDs: []string{protoio.ScalarOutputActuatorName},
	}

	clone := CloneAgentWithRemappedIDs(in, "g2", []string{"i", "o"})
	if clone.ID != "g2" {
		t.Fatalf("expected clone id g2, got %q", clone.ID)
	}
	if clone.Neurons[0].ID != "i" || clone.Neurons[2].ID != "o" {
		t.Fatalf("expected preserved io neuron ids, got=%v", []string{clone.Neurons[0].ID, clone.Neurons[2].ID})
	}
	if clone.Neurons[1].ID == "h" {
		t.Fatalf("expected hidden neuron id remap, got=%q", clone.Neurons[1].ID)
	}
	if clone.Synapses[0].ID == "s1" || clone.Synapses[1].ID == "s2" {
		t.Fatalf("expected synapse id remap, got=%v", []string{clone.Synapses[0].ID, clone.Synapses[1].ID})
	}
	hiddenID := clone.Neurons[1].ID
	if clone.Synapses[0].From != "i" || clone.Synapses[0].To != hiddenID {
		t.Fatalf("unexpected remapped first synapse endpoints: %+v hidden=%s", clone.Synapses[0], hiddenID)
	}
	if clone.Synapses[1].From != hiddenID || clone.Synapses[1].To != "o" {
		t.Fatalf("unexpected remapped second synapse endpoints: %+v hidden=%s", clone.Synapses[1], hiddenID)
	}
	if clone.SensorNeuronLinks[0].NeuronID != "i" {
		t.Fatalf("expected preserved sensor link target neuron id, got=%s", clone.SensorNeuronLinks[0].NeuronID)
	}
	if clone.NeuronActuatorLinks[0].NeuronID != hiddenID {
		t.Fatalf("expected remapped actuator link neuron id=%s, got=%s", hiddenID, clone.NeuronActuatorLinks[0].NeuronID)
	}
	if in.Neurons[1].ID != "h" || in.Synapses[0].ID != "s1" {
		t.Fatal("expected original genome to remain unchanged")
	}
}

func TestCloneAgentWithRemappedIDsRemapsSubstrateEndpointIDs(t *testing.T) {
	in := model.Genome{
		ID: "g1",
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "h", Activation: "tanh"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "h", Weight: 0.5, Enabled: true},
			{ID: "s2", From: "h", To: "o", Weight: 1.0, Enabled: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: "substrate:cpp:d3:0", NeuronID: "h"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "h", ActuatorID: "substrate:cep:d3:0"},
		},
		Substrate: &model.SubstrateConfig{
			CPPName: "none",
			CEPName: "l2l_feedforward",
			CPPIDs:  []string{"substrate:cpp:d3:0"},
			CEPIDs:  []string{"substrate:cep:d3:0"},
		},
	}

	clone := CloneAgentWithRemappedIDs(in, "g2", []string{"i", "o"})
	if clone.Substrate == nil {
		t.Fatal("expected substrate config on clone")
	}
	if clone.Substrate.CPPIDs[0] == in.Substrate.CPPIDs[0] {
		t.Fatalf("expected substrate cpp id remap, original=%q clone=%q", in.Substrate.CPPIDs[0], clone.Substrate.CPPIDs[0])
	}
	if clone.Substrate.CEPIDs[0] == in.Substrate.CEPIDs[0] {
		t.Fatalf("expected substrate cep id remap, original=%q clone=%q", in.Substrate.CEPIDs[0], clone.Substrate.CEPIDs[0])
	}
	if clone.SensorNeuronLinks[0].SensorID != clone.Substrate.CPPIDs[0] {
		t.Fatalf("expected sensor link sensor id remap to cloned cpp id, link=%+v cpp=%v", clone.SensorNeuronLinks[0], clone.Substrate.CPPIDs)
	}
	if clone.NeuronActuatorLinks[0].ActuatorID != clone.Substrate.CEPIDs[0] {
		t.Fatalf("expected actuator link endpoint remap to cloned cep id, link=%+v cep=%v", clone.NeuronActuatorLinks[0], clone.Substrate.CEPIDs)
	}
	if clone.NeuronActuatorLinks[0].NeuronID != clone.Neurons[1].ID {
		t.Fatalf("expected actuator link neuron remap to hidden neuron id=%q, got=%q", clone.Neurons[1].ID, clone.NeuronActuatorLinks[0].NeuronID)
	}
	if in.SensorNeuronLinks[0].SensorID != "substrate:cpp:d3:0" || in.NeuronActuatorLinks[0].ActuatorID != "substrate:cep:d3:0" {
		t.Fatal("expected original substrate endpoint links to remain unchanged")
	}
}

func TestCloneAgentAutoID(t *testing.T) {
	in := model.Genome{
		ID: "g1",
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "n1", To: "n1", Weight: 0.5, Enabled: true},
		},
	}
	clone := CloneAgentAutoID(in, rand.New(rand.NewSource(7)))
	if clone.ID == "" || clone.ID == in.ID {
		t.Fatalf("expected generated clone id distinct from original, got=%q original=%q", clone.ID, in.ID)
	}
	if clone.Neurons[0].ID == in.Neurons[0].ID {
		t.Fatalf("expected internal neuron id remap for auto clone, got=%q", clone.Neurons[0].ID)
	}
	if clone.Synapses[0].ID == in.Synapses[0].ID {
		t.Fatalf("expected internal synapse id remap for auto clone, got=%q", clone.Synapses[0].ID)
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

func TestDeleteAgent(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}
	if err := store.SaveGenome(ctx, model.Genome{ID: "a1"}); err != nil {
		t.Fatalf("save genome: %v", err)
	}
	if err := DeleteAgent(ctx, store, "a1"); err != nil {
		t.Fatalf("delete agent: %v", err)
	}
	if _, ok, err := store.GetGenome(ctx, "a1"); err != nil {
		t.Fatalf("get genome: %v", err)
	} else if ok {
		t.Fatal("expected genome to be deleted")
	}
}

func TestDeleteAgentSafe(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}
	for _, id := range []string{"a1", "a2"} {
		if err := store.SaveGenome(ctx, model.Genome{ID: id}); err != nil {
			t.Fatalf("save genome %s: %v", id, err)
		}
	}
	if err := store.SavePopulation(ctx, model.Population{ID: "pop-safe", AgentIDs: []string{"a1", "a2"}}); err != nil {
		t.Fatalf("save population: %v", err)
	}
	if err := DeleteAgentSafe(ctx, store, "pop-safe", "a2"); err != nil {
		t.Fatalf("delete agent safe: %v", err)
	}
	pop, ok, err := store.GetPopulation(ctx, "pop-safe")
	if err != nil || !ok {
		t.Fatalf("get population err=%v ok=%t", err, ok)
	}
	if len(pop.AgentIDs) != 1 || pop.AgentIDs[0] != "a1" {
		t.Fatalf("unexpected remaining population members: %+v", pop.AgentIDs)
	}
	if _, ok, err := store.GetGenome(ctx, "a2"); err != nil {
		t.Fatalf("get removed genome: %v", err)
	} else if ok {
		t.Fatal("expected safe delete to remove genome")
	}
}
