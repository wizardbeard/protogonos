package genotype

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/scapeid"
	"protogonos/internal/storage"
)

type SeedPopulation struct {
	Genomes         []model.Genome
	InputNeuronIDs  []string
	OutputNeuronIDs []string
}

type SeedPopulationOptions struct {
	// FlatlandProfile controls the flatland seed scaffold.
	// Supported values: "scanner" (default) and "classic".
	FlatlandProfile string

	// FlatlandScannerProfile controls scanner-bin emphasis when FlatlandProfile
	// resolves to "scanner". Supported values: "balanced5" (default),
	// "core3", and "forward5".
	FlatlandScannerProfile string
}

const (
	FlatlandSeedProfileScanner          = "scanner"
	FlatlandSeedProfileClassic          = "classic"
	FlatlandScannerSeedProfileBalanced5 = "balanced5"
	FlatlandScannerSeedProfileCore3     = "core3"
	FlatlandScannerSeedProfileForward5  = "forward5"
)

func ConstructSeedPopulation(scapeName string, size int, seed int64) (SeedPopulation, error) {
	return ConstructSeedPopulationWithOptions(scapeName, size, seed, SeedPopulationOptions{})
}

func ConstructSeedPopulationWithOptions(scapeName string, size int, seed int64, options SeedPopulationOptions) (SeedPopulation, error) {
	scapeName = scapeid.Normalize(scapeName)
	switch scapeName {
	case "xor":
		return SeedPopulation{
			Genomes:         seedXORPopulation(size, seed),
			InputNeuronIDs:  []string{"i1", "i2"},
			OutputNeuronIDs: []string{"o"},
		}, nil
	case "regression-mimic":
		return SeedPopulation{
			Genomes:         seedRegressionMimicPopulation(size, seed),
			InputNeuronIDs:  []string{"i"},
			OutputNeuronIDs: []string{"o"},
		}, nil
	case "cart-pole-lite":
		return SeedPopulation{
			Genomes:         seedCartPoleLitePopulation(size, seed),
			InputNeuronIDs:  []string{"x", "v"},
			OutputNeuronIDs: []string{"f"},
		}, nil
	case "pole2-balancing":
		return SeedPopulation{
			Genomes:         seedPole2BalancingPopulation(size, seed),
			InputNeuronIDs:  []string{"x", "v", "a1", "w1", "a2", "w2", "rp", "sp", "fs"},
			OutputNeuronIDs: []string{"f"},
		}, nil
	case "flatland":
		return constructFlatlandSeedPopulation(size, seed, options)
	case "dtm":
		return SeedPopulation{
			Genomes:         seedDTMPopulation(size, seed),
			InputNeuronIDs:  []string{"rl", "rf", "rr", "r", "rp", "sp", "sw"},
			OutputNeuronIDs: []string{"m"},
		}, nil
	case "gtsa":
		return SeedPopulation{
			Genomes:         seedGTSAPopulation(size, seed),
			InputNeuronIDs:  []string{"x", "d", "w", "p"},
			OutputNeuronIDs: []string{"y"},
		}, nil
	case "fx":
		return SeedPopulation{
			Genomes:         seedFXPopulation(size, seed),
			InputNeuronIDs:  []string{"p", "s", "m", "v", "n", "d", "q", "e", "pc", "ppc", "pr"},
			OutputNeuronIDs: []string{"t"},
		}, nil
	case "epitopes":
		return SeedPopulation{
			Genomes:         seedEpitopesPopulation(size, seed),
			InputNeuronIDs:  []string{"s", "m", "t", "p", "g"},
			OutputNeuronIDs: []string{"r"},
		}, nil
	case "llvm-phase-ordering":
		return SeedPopulation{
			Genomes:         seedLLVMPhaseOrderingPopulation(size, seed),
			InputNeuronIDs:  []string{"c", "p", "a", "d", "r"},
			OutputNeuronIDs: llvmSeedOutputNeuronIDs(),
		}, nil
	default:
		return SeedPopulation{}, fmt.Errorf("unsupported scape: %s", scapeName)
	}
}

func constructFlatlandSeedPopulation(size int, seed int64, options SeedPopulationOptions) (SeedPopulation, error) {
	switch normalizeFlatlandSeedProfile(options.FlatlandProfile) {
	case FlatlandSeedProfileClassic:
		return SeedPopulation{
			Genomes:         seedFlatlandClassicPopulation(size, seed),
			InputNeuronIDs:  []string{"d", "e"},
			OutputNeuronIDs: []string{"m"},
		}, nil
	case FlatlandSeedProfileScanner:
		scannerProfile, err := resolveFlatlandScannerSeedProfile(options.FlatlandScannerProfile)
		if err != nil {
			return SeedPopulation{}, err
		}
		return SeedPopulation{
			Genomes:         seedFlatlandScannerPopulation(size, seed, scannerProfile),
			InputNeuronIDs:  flatlandSeedInputNeuronIDs(),
			OutputNeuronIDs: flatlandSeedOutputNeuronIDs(),
		}, nil
	default:
		return SeedPopulation{}, fmt.Errorf("unsupported flatland seed profile: %s", options.FlatlandProfile)
	}
}

func normalizeFlatlandSeedProfile(raw string) string {
	profile := strings.ToLower(strings.TrimSpace(raw))
	profile = strings.ReplaceAll(profile, "_", "-")
	switch profile {
	case "", "default", "scanner", "scan", "flatland-scanner", "flatland-prey", "prey":
		return FlatlandSeedProfileScanner
	case "classic", "legacy", "flatland-v1":
		return FlatlandSeedProfileClassic
	default:
		return profile
	}
}

func normalizeFlatlandScannerSeedProfile(raw string) string {
	profile := strings.ToLower(strings.TrimSpace(raw))
	profile = strings.ReplaceAll(profile, "_", "-")
	switch profile {
	case "", "default", "balanced", "balanced5", "scanner":
		return FlatlandScannerSeedProfileBalanced5
	case "core", "core3", "focused":
		return FlatlandScannerSeedProfileCore3
	case "forward", "forward5", "directional":
		return FlatlandScannerSeedProfileForward5
	default:
		return profile
	}
}

func resolveFlatlandScannerSeedProfile(raw string) (string, error) {
	profile := normalizeFlatlandScannerSeedProfile(raw)
	switch profile {
	case FlatlandScannerSeedProfileBalanced5, FlatlandScannerSeedProfileCore3, FlatlandScannerSeedProfileForward5:
		return profile, nil
	default:
		return "", fmt.Errorf("unsupported flatland scanner profile: %s", raw)
	}
}

func CloneAgent(genome model.Genome, newID string) model.Genome {
	clone := CloneGenome(genome)
	if newID != "" {
		clone.ID = newID
	}
	return clone
}

// CloneAgentAutoID mirrors genotype:clone_Agent/1 behavior by generating a new
// clone ID and remapping internal IDs.
func CloneAgentAutoID(genome model.Genome, rng *rand.Rand) model.Genome {
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	base := genome.ID
	if base == "" {
		base = "agent"
	}
	newID := fmt.Sprintf("%s-clone-%d", base, rng.Int63())
	return CloneAgentWithRemappedIDs(genome, newID, nil)
}

func CloneAgentWithRemappedIDs(genome model.Genome, newID string, preserveNeuronIDs []string) model.Genome {
	return CloneGenomeWithRemappedIDs(genome, newID, preserveNeuronIDs)
}

// DeleteAgent mirrors genotype:delete_Agent/1 semantics in the simplified model.
func DeleteAgent(ctx context.Context, store storage.Store, agentID string) error {
	if store == nil {
		return fmt.Errorf("store is required")
	}
	if agentID == "" {
		return fmt.Errorf("agent id is required")
	}
	return store.DeleteGenome(ctx, agentID)
}

// DeleteAgentSafe mirrors genotype:delete_Agent/2 safe semantics in the
// simplified model by pruning population membership and deleting the genome.
func DeleteAgentSafe(ctx context.Context, store storage.Store, populationID, agentID string) error {
	return DeleteAgentFromPopulation(ctx, store, populationID, agentID)
}

func DeleteAgentFromPopulation(ctx context.Context, store storage.Store, populationID, agentID string) error {
	if store == nil {
		return fmt.Errorf("store is required")
	}
	if populationID == "" {
		return fmt.Errorf("population id is required")
	}
	if agentID == "" {
		return fmt.Errorf("agent id is required")
	}

	pop, ok, err := store.GetPopulation(ctx, populationID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("population not found: %s", populationID)
	}

	filtered := make([]string, 0, len(pop.AgentIDs))
	found := false
	for _, id := range pop.AgentIDs {
		if id == agentID {
			found = true
			continue
		}
		filtered = append(filtered, id)
	}
	if !found {
		return fmt.Errorf("agent id %s not found in population %s", agentID, populationID)
	}

	pop.AgentIDs = filtered
	if err := store.SavePopulation(ctx, pop); err != nil {
		return err
	}
	return store.DeleteGenome(ctx, agentID)
}

func seedXORPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("xor-g0-%d", i),
			SensorIDs:       []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
			ActuatorIDs:     []string{protoio.XOROutputActuatorName},
			Neurons: []model.Neuron{
				{ID: "i1", Activation: "identity", Bias: 0},
				{ID: "i2", Activation: "identity", Bias: 0},
				{ID: "h1", Activation: "sigmoid", Bias: jitter(rng, 2)},
				{ID: "h2", Activation: "sigmoid", Bias: jitter(rng, 2)},
				{ID: "o", Activation: "sigmoid", Bias: jitter(rng, 2)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "i1", To: "h1", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s2", From: "i2", To: "h1", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s3", From: "i1", To: "h2", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s4", From: "i2", To: "h2", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s5", From: "h1", To: "o", Weight: jitter(rng, 6), Enabled: true},
				{ID: "s6", From: "h2", To: "o", Weight: jitter(rng, 6), Enabled: true},
			},
		})
	}
	return population
}

func seedRegressionMimicPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("reg-g0-%d", i),
			SensorIDs:       []string{protoio.ScalarInputSensorName},
			ActuatorIDs:     []string{protoio.ScalarOutputActuatorName},
			Neurons: []model.Neuron{
				{ID: "i", Activation: "identity", Bias: 0},
				{ID: "o", Activation: "identity", Bias: jitter(rng, 1)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "i", To: "o", Weight: jitter(rng, 2), Enabled: true},
			},
		})
	}
	return population
}

func seedCartPoleLitePopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("cp-g0-%d", i),
			SensorIDs:       []string{protoio.CartPolePositionSensorName, protoio.CartPoleVelocitySensorName},
			ActuatorIDs:     []string{protoio.CartPoleForceActuatorName},
			Neurons: []model.Neuron{
				{ID: "x", Activation: "identity", Bias: 0},
				{ID: "v", Activation: "identity", Bias: 0},
				{ID: "f", Activation: "identity", Bias: jitter(rng, 0.2)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "x", To: "f", Weight: jitter(rng, 1.5), Enabled: true},
				{ID: "s2", From: "v", To: "f", Weight: jitter(rng, 1.0), Enabled: true},
			},
		})
	}
	return population
}

func seedPole2BalancingPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("pole2-g0-%d", i),
			SensorIDs: []string{
				protoio.Pole2CartPositionSensorName,
				protoio.Pole2CartVelocitySensorName,
				protoio.Pole2Angle1SensorName,
				protoio.Pole2Velocity1SensorName,
				protoio.Pole2Angle2SensorName,
				protoio.Pole2Velocity2SensorName,
				protoio.Pole2RunProgressSensorName,
				protoio.Pole2StepProgressSensorName,
				protoio.Pole2FitnessSignalSensorName,
			},
			ActuatorIDs: []string{protoio.Pole2PushActuatorName},
			Neurons: []model.Neuron{
				{ID: "x", Activation: "identity", Bias: 0},
				{ID: "v", Activation: "identity", Bias: 0},
				{ID: "a1", Activation: "identity", Bias: 0},
				{ID: "w1", Activation: "identity", Bias: 0},
				{ID: "a2", Activation: "identity", Bias: 0},
				{ID: "w2", Activation: "identity", Bias: 0},
				{ID: "rp", Activation: "identity", Bias: 0},
				{ID: "sp", Activation: "identity", Bias: 0},
				{ID: "fs", Activation: "identity", Bias: 0},
				{ID: "f", Activation: "tanh", Bias: jitter(rng, 0.2)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "x", To: "f", Weight: -0.9 + jitter(rng, 0.2), Enabled: true},
				{ID: "s2", From: "v", To: "f", Weight: -0.5 + jitter(rng, 0.2), Enabled: true},
				{ID: "s3", From: "a1", To: "f", Weight: -4.0 + jitter(rng, 0.3), Enabled: true},
				{ID: "s4", From: "w1", To: "f", Weight: -0.8 + jitter(rng, 0.2), Enabled: true},
				{ID: "s5", From: "a2", To: "f", Weight: -5.0 + jitter(rng, 0.3), Enabled: true},
				{ID: "s6", From: "w2", To: "f", Weight: -1.0 + jitter(rng, 0.2), Enabled: true},
				{ID: "s7", From: "rp", To: "f", Weight: 0.25 + jitter(rng, 0.15), Enabled: true},
				{ID: "s8", From: "sp", To: "f", Weight: 0.2 + jitter(rng, 0.15), Enabled: true},
				{ID: "s9", From: "fs", To: "f", Weight: 0.3 + jitter(rng, 0.15), Enabled: true},
			},
		})
	}
	return population
}

func seedFlatlandClassicPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("flatland-g0-%d", i),
			SensorIDs:       []string{protoio.FlatlandDistanceSensorName, protoio.FlatlandEnergySensorName},
			ActuatorIDs:     []string{protoio.FlatlandMoveActuatorName},
			Neurons: []model.Neuron{
				{ID: "d", Activation: "identity", Bias: 0},
				{ID: "e", Activation: "identity", Bias: 0},
				{ID: "m", Activation: "tanh", Bias: jitter(rng, 0.4)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "d", To: "m", Weight: jitter(rng, 1.2), Enabled: true},
				{ID: "s2", From: "e", To: "m", Weight: jitter(rng, 1.2), Enabled: true},
			},
		})
	}
	return population
}

func seedFlatlandScannerPopulation(size int, seed int64, scannerProfile string) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	inputNeuronIDs := flatlandSeedInputNeuronIDs()
	sensorIDs := flatlandSeedSensorIDs()
	profileWeights := flatlandSeedScannerProfileWeights(scannerProfile)
	for i := 0; i < size; i++ {
		neurons := make([]model.Neuron, 0, len(inputNeuronIDs)+2)
		for _, neuronID := range inputNeuronIDs {
			neurons = append(neurons, model.Neuron{ID: neuronID, Activation: "identity", Bias: 0})
		}
		neurons = append(neurons,
			model.Neuron{ID: "wl", Activation: "tanh", Bias: jitter(rng, 0.25)},
			model.Neuron{ID: "wr", Activation: "tanh", Bias: jitter(rng, 0.25)},
		)

		synapses := make([]model.Synapse, 0, len(inputNeuronIDs)*2)
		synapseIndex := 1
		for idx, neuronID := range inputNeuronIDs {
			scale := flatlandSeedInputProfileScale(idx, profileWeights)
			leftWeight, rightWeight := flatlandSeedWheelWeights(idx)
			synapses = append(synapses,
				model.Synapse{
					ID:      fmt.Sprintf("s%d", synapseIndex),
					From:    neuronID,
					To:      "wl",
					Weight:  (leftWeight + jitter(rng, 0.2)) * scale,
					Enabled: true,
				},
				model.Synapse{
					ID:      fmt.Sprintf("s%d", synapseIndex+1),
					From:    neuronID,
					To:      "wr",
					Weight:  (rightWeight + jitter(rng, 0.2)) * scale,
					Enabled: true,
				},
			)
			synapseIndex += 2
		}

		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("flatland-g0-%d", i),
			SensorIDs:       append([]string(nil), sensorIDs...),
			ActuatorIDs:     []string{protoio.FlatlandTwoWheelsActuatorName},
			Neurons:         neurons,
			Synapses:        synapses,
		})
	}
	return population
}

func flatlandSeedScannerProfileWeights(profile string) [5]float64 {
	switch normalizeFlatlandScannerSeedProfile(profile) {
	case FlatlandScannerSeedProfileCore3:
		return [5]float64{0, 1, 1, 1, 0}
	case FlatlandScannerSeedProfileForward5:
		return [5]float64{0.25, 0.55, 0.85, 1, 1}
	default:
		return [5]float64{1, 1, 1, 1, 1}
	}
}

func flatlandSeedInputProfileScale(index int, binWeights [5]float64) float64 {
	switch {
	case index >= 0 && index < 5:
		return binWeights[index]
	case index >= 5 && index < 10:
		return binWeights[index-5]
	case index >= 10 && index < 15:
		return binWeights[index-10]
	default:
		return 1
	}
}

func flatlandSeedSensorIDs() []string {
	return []string{
		protoio.FlatlandDistanceScan0SensorName,
		protoio.FlatlandDistanceScan1SensorName,
		protoio.FlatlandDistanceScan2SensorName,
		protoio.FlatlandDistanceScan3SensorName,
		protoio.FlatlandDistanceScan4SensorName,
		protoio.FlatlandColorScan0SensorName,
		protoio.FlatlandColorScan1SensorName,
		protoio.FlatlandColorScan2SensorName,
		protoio.FlatlandColorScan3SensorName,
		protoio.FlatlandColorScan4SensorName,
		protoio.FlatlandEnergyScan0SensorName,
		protoio.FlatlandEnergyScan1SensorName,
		protoio.FlatlandEnergyScan2SensorName,
		protoio.FlatlandEnergyScan3SensorName,
		protoio.FlatlandEnergyScan4SensorName,
		protoio.FlatlandEnergySensorName,
		protoio.FlatlandPreySensorName,
		protoio.FlatlandPredatorSensorName,
		protoio.FlatlandPreyProximitySensorName,
		protoio.FlatlandPredatorProximitySensorName,
	}
}

func flatlandSeedInputNeuronIDs() []string {
	return []string{
		"d0", "d1", "d2", "d3", "d4",
		"c0", "c1", "c2", "c3", "c4",
		"es0", "es1", "es2", "es3", "es4",
		"e",
		"prey",
		"pred",
		"prey_prox",
		"pred_prox",
	}
}

func flatlandSeedOutputNeuronIDs() []string {
	return []string{"wl", "wr"}
}

func flatlandSeedWheelWeights(index int) (float64, float64) {
	binProfile := []float64{-0.8, -0.4, 0, 0.4, 0.8}
	switch {
	case index >= 0 && index < 5:
		base := binProfile[index]
		return base, -base
	case index >= 5 && index < 10:
		base := 0.35 * binProfile[index-5]
		return base, -base
	case index >= 10 && index < 15:
		base := 0.45 * binProfile[index-10]
		return base, -base
	case index == 16:
		return 0.45, -0.45
	case index == 17:
		return -0.45, 0.45
	case index == 18:
		return 0.30, 0.30
	case index == 19:
		return -0.30, -0.30
	default:
		return 0.25, 0.25
	}
}

func seedDTMPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("dtm-g0-%d", i),
			SensorIDs: []string{
				protoio.DTMRangeLeftSensorName,
				protoio.DTMRangeFrontSensorName,
				protoio.DTMRangeRightSensorName,
				protoio.DTMRewardSensorName,
				protoio.DTMRunProgressSensorName,
				protoio.DTMStepProgressSensorName,
				protoio.DTMSwitchedSensorName,
			},
			ActuatorIDs: []string{protoio.DTMMoveActuatorName},
			Neurons: []model.Neuron{
				{ID: "rl", Activation: "identity", Bias: 0},
				{ID: "rf", Activation: "identity", Bias: 0},
				{ID: "rr", Activation: "identity", Bias: 0},
				{ID: "r", Activation: "identity", Bias: 0},
				{ID: "rp", Activation: "identity", Bias: 0},
				{ID: "sp", Activation: "identity", Bias: 0},
				{ID: "sw", Activation: "identity", Bias: 0},
				{ID: "m", Activation: "tanh", Bias: jitter(rng, 0.35)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "rl", To: "m", Weight: 0.9 + jitter(rng, 0.25), Enabled: true},
				{ID: "s2", From: "rf", To: "m", Weight: -0.4 + jitter(rng, 0.25), Enabled: true},
				{ID: "s3", From: "rr", To: "m", Weight: 0.9 + jitter(rng, 0.25), Enabled: true},
				{ID: "s4", From: "r", To: "m", Weight: 0.2 + jitter(rng, 0.15), Enabled: true},
				{ID: "s5", From: "rp", To: "m", Weight: 0.3 + jitter(rng, 0.15), Enabled: true},
				{ID: "s6", From: "sp", To: "m", Weight: -0.1 + jitter(rng, 0.15), Enabled: true},
				{ID: "s7", From: "sw", To: "m", Weight: 0.2 + jitter(rng, 0.15), Enabled: true},
			},
		})
	}
	return population
}

func seedGTSAPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("gtsa-g0-%d", i),
			SensorIDs: []string{
				protoio.GTSAInputSensorName,
				protoio.GTSADeltaSensorName,
				protoio.GTSAWindowMeanSensorName,
				protoio.GTSAProgressSensorName,
			},
			ActuatorIDs: []string{protoio.GTSAPredictActuatorName},
			Neurons: []model.Neuron{
				{ID: "x", Activation: "identity", Bias: 0},
				{ID: "d", Activation: "identity", Bias: 0},
				{ID: "w", Activation: "identity", Bias: 0},
				{ID: "p", Activation: "identity", Bias: 0},
				{ID: "y", Activation: "identity", Bias: jitter(rng, 0.3)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "x", To: "y", Weight: jitter(rng, 1.0), Enabled: true},
				{ID: "s2", From: "d", To: "y", Weight: jitter(rng, 0.7), Enabled: true},
				{ID: "s3", From: "w", To: "y", Weight: jitter(rng, 0.7), Enabled: true},
				{ID: "s4", From: "p", To: "y", Weight: jitter(rng, 0.5), Enabled: true},
			},
		})
	}
	return population
}

func seedFXPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("fx-g0-%d", i),
			SensorIDs: []string{
				protoio.FXPriceSensorName,
				protoio.FXSignalSensorName,
				protoio.FXMomentumSensorName,
				protoio.FXVolatilitySensorName,
				protoio.FXNAVSensorName,
				protoio.FXDrawdownSensorName,
				protoio.FXPositionSensorName,
				protoio.FXEntrySensorName,
				protoio.FXPercentChangeSensorName,
				protoio.FXPrevPercentChangeSensorName,
				protoio.FXProfitSensorName,
			},
			ActuatorIDs: []string{protoio.FXTradeActuatorName},
			Neurons: []model.Neuron{
				{ID: "p", Activation: "identity", Bias: 0},
				{ID: "s", Activation: "identity", Bias: 0},
				{ID: "m", Activation: "identity", Bias: 0},
				{ID: "v", Activation: "identity", Bias: 0},
				{ID: "n", Activation: "identity", Bias: 0},
				{ID: "d", Activation: "identity", Bias: 0},
				{ID: "q", Activation: "identity", Bias: 0},
				{ID: "e", Activation: "identity", Bias: 0},
				{ID: "pc", Activation: "identity", Bias: 0},
				{ID: "ppc", Activation: "identity", Bias: 0},
				{ID: "pr", Activation: "identity", Bias: 0},
				{ID: "t", Activation: "tanh", Bias: jitter(rng, 0.25)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "p", To: "t", Weight: 0.35 + jitter(rng, 0.25), Enabled: true},
				{ID: "s2", From: "s", To: "t", Weight: 1.10 + jitter(rng, 0.25), Enabled: true},
				{ID: "s3", From: "m", To: "t", Weight: 0.85 + jitter(rng, 0.20), Enabled: true},
				{ID: "s4", From: "v", To: "t", Weight: -0.55 + jitter(rng, 0.20), Enabled: true},
				{ID: "s5", From: "n", To: "t", Weight: 0.25 + jitter(rng, 0.15), Enabled: true},
				{ID: "s6", From: "d", To: "t", Weight: -0.65 + jitter(rng, 0.15), Enabled: true},
				{ID: "s7", From: "q", To: "t", Weight: 0.15 + jitter(rng, 0.15), Enabled: true},
				{ID: "s8", From: "e", To: "t", Weight: -0.20 + jitter(rng, 0.10), Enabled: true},
				{ID: "s9", From: "pc", To: "t", Weight: 0.45 + jitter(rng, 0.10), Enabled: true},
				{ID: "s10", From: "ppc", To: "t", Weight: 0.30 + jitter(rng, 0.10), Enabled: true},
				{ID: "s11", From: "pr", To: "t", Weight: 0.35 + jitter(rng, 0.10), Enabled: true},
			},
		})
	}
	return population
}

func seedEpitopesPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("epitopes-g0-%d", i),
			SensorIDs: []string{
				protoio.EpitopesSignalSensorName,
				protoio.EpitopesMemorySensorName,
				protoio.EpitopesTargetSensorName,
				protoio.EpitopesProgressSensorName,
				protoio.EpitopesMarginSensorName,
			},
			ActuatorIDs: []string{protoio.EpitopesResponseActuatorName},
			Neurons: []model.Neuron{
				{ID: "s", Activation: "identity", Bias: 0},
				{ID: "m", Activation: "identity", Bias: 0},
				{ID: "t", Activation: "identity", Bias: 0},
				{ID: "p", Activation: "identity", Bias: 0},
				{ID: "g", Activation: "identity", Bias: 0},
				{ID: "r", Activation: "tanh", Bias: jitter(rng, 0.25)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "s", To: "r", Weight: 0.9 + jitter(rng, 0.2), Enabled: true},
				{ID: "s2", From: "m", To: "r", Weight: 0.7 + jitter(rng, 0.2), Enabled: true},
				{ID: "s3", From: "t", To: "r", Weight: 0.5 + jitter(rng, 0.2), Enabled: true},
				{ID: "s4", From: "p", To: "r", Weight: 0.3 + jitter(rng, 0.2), Enabled: true},
				{ID: "s5", From: "g", To: "r", Weight: 0.4 + jitter(rng, 0.2), Enabled: true},
			},
		})
	}
	return population
}

func seedLLVMPhaseOrderingPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	outputIDs := llvmSeedOutputNeuronIDs()
	surfaceSize := len(outputIDs)
	if surfaceSize <= 0 {
		surfaceSize = 1
	}

	for i := 0; i < size; i++ {
		neurons := make([]model.Neuron, 0, 5+surfaceSize)
		neurons = append(neurons,
			model.Neuron{ID: "c", Activation: "identity", Bias: 0},
			model.Neuron{ID: "p", Activation: "identity", Bias: 0},
			model.Neuron{ID: "a", Activation: "identity", Bias: 0},
			model.Neuron{ID: "d", Activation: "identity", Bias: 0},
			model.Neuron{ID: "r", Activation: "identity", Bias: 0},
		)
		synapses := make([]model.Synapse, 0, surfaceSize*5)
		for idx, outputID := range outputIDs {
			progress := float64(idx) / float64(maxIntLifecycle(1, surfaceSize-1))
			neurons = append(neurons, model.Neuron{
				ID:         outputID,
				Activation: "identity",
				Bias:       (1.0 - 2.0*progress) + jitter(rng, 0.05),
			})
			synapses = append(synapses,
				model.Synapse{
					ID:      fmt.Sprintf("s%d:p", idx),
					From:    "p",
					To:      outputID,
					Weight:  -2.2 + 4.4*progress + jitter(rng, 0.05),
					Enabled: true,
				},
				model.Synapse{
					ID:      fmt.Sprintf("s%d:c", idx),
					From:    "c",
					To:      outputID,
					Weight:  -0.35 + jitter(rng, 0.1),
					Enabled: true,
				},
				model.Synapse{
					ID:      fmt.Sprintf("s%d:a", idx),
					From:    "a",
					To:      outputID,
					Weight:  0.20 + jitter(rng, 0.08),
					Enabled: true,
				},
				model.Synapse{
					ID:      fmt.Sprintf("s%d:d", idx),
					From:    "d",
					To:      outputID,
					Weight:  0.25 + jitter(rng, 0.08),
					Enabled: true,
				},
				model.Synapse{
					ID:      fmt.Sprintf("s%d:r", idx),
					From:    "r",
					To:      outputID,
					Weight:  0.18 + jitter(rng, 0.08),
					Enabled: true,
				},
			)
		}

		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("llvm-g0-%d", i),
			SensorIDs: []string{
				protoio.LLVMComplexitySensorName,
				protoio.LLVMPassIndexSensorName,
				protoio.LLVMAlignmentSensorName,
				protoio.LLVMDiversitySensorName,
				protoio.LLVMRuntimeGainSensorName,
			},
			ActuatorIDs: []string{protoio.LLVMPhaseActuatorName},
			Neurons:     neurons,
			Synapses:    synapses,
		})
	}
	return population
}

func llvmSeedOutputNeuronIDs() []string {
	const optimizationSurface = 55
	ids := make([]string, 0, optimizationSurface)
	for i := 0; i < optimizationSurface; i++ {
		ids = append(ids, fmt.Sprintf("o%02d", i))
	}
	return ids
}

func jitter(rng *rand.Rand, amplitude float64) float64 {
	return (rng.Float64()*2 - 1) * amplitude
}

func maxIntLifecycle(a, b int) int {
	if a > b {
		return a
	}
	return b
}
