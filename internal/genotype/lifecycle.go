package genotype

import (
	"context"
	"fmt"
	"math/rand"
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

func ConstructSeedPopulation(scapeName string, size int, seed int64) (SeedPopulation, error) {
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
			InputNeuronIDs:  []string{"x", "v", "a1", "w1", "a2", "w2"},
			OutputNeuronIDs: []string{"f"},
		}, nil
	case "flatland":
		return SeedPopulation{
			Genomes:         seedFlatlandPopulation(size, seed),
			InputNeuronIDs:  []string{"d", "e"},
			OutputNeuronIDs: []string{"m"},
		}, nil
	case "dtm":
		return SeedPopulation{
			Genomes:         seedDTMPopulation(size, seed),
			InputNeuronIDs:  []string{"rl", "rf", "rr", "r"},
			OutputNeuronIDs: []string{"m"},
		}, nil
	case "gtsa":
		return SeedPopulation{
			Genomes:         seedGTSAPopulation(size, seed),
			InputNeuronIDs:  []string{"x"},
			OutputNeuronIDs: []string{"y"},
		}, nil
	case "fx":
		return SeedPopulation{
			Genomes:         seedFXPopulation(size, seed),
			InputNeuronIDs:  []string{"p", "s"},
			OutputNeuronIDs: []string{"t"},
		}, nil
	case "epitopes":
		return SeedPopulation{
			Genomes:         seedEpitopesPopulation(size, seed),
			InputNeuronIDs:  []string{"s", "m"},
			OutputNeuronIDs: []string{"r"},
		}, nil
	case "llvm-phase-ordering":
		return SeedPopulation{
			Genomes:         seedLLVMPhaseOrderingPopulation(size, seed),
			InputNeuronIDs:  []string{"c", "p"},
			OutputNeuronIDs: []string{"o"},
		}, nil
	default:
		return SeedPopulation{}, fmt.Errorf("unsupported scape: %s", scapeName)
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
			},
			ActuatorIDs: []string{protoio.Pole2PushActuatorName},
			Neurons: []model.Neuron{
				{ID: "x", Activation: "identity", Bias: 0},
				{ID: "v", Activation: "identity", Bias: 0},
				{ID: "a1", Activation: "identity", Bias: 0},
				{ID: "w1", Activation: "identity", Bias: 0},
				{ID: "a2", Activation: "identity", Bias: 0},
				{ID: "w2", Activation: "identity", Bias: 0},
				{ID: "f", Activation: "tanh", Bias: jitter(rng, 0.2)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "x", To: "f", Weight: -0.9 + jitter(rng, 0.2), Enabled: true},
				{ID: "s2", From: "v", To: "f", Weight: -0.5 + jitter(rng, 0.2), Enabled: true},
				{ID: "s3", From: "a1", To: "f", Weight: -4.0 + jitter(rng, 0.3), Enabled: true},
				{ID: "s4", From: "w1", To: "f", Weight: -0.8 + jitter(rng, 0.2), Enabled: true},
				{ID: "s5", From: "a2", To: "f", Weight: -5.0 + jitter(rng, 0.3), Enabled: true},
				{ID: "s6", From: "w2", To: "f", Weight: -1.0 + jitter(rng, 0.2), Enabled: true},
			},
		})
	}
	return population
}

func seedFlatlandPopulation(size int, seed int64) []model.Genome {
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
			},
			ActuatorIDs: []string{protoio.DTMMoveActuatorName},
			Neurons: []model.Neuron{
				{ID: "rl", Activation: "identity", Bias: 0},
				{ID: "rf", Activation: "identity", Bias: 0},
				{ID: "rr", Activation: "identity", Bias: 0},
				{ID: "r", Activation: "identity", Bias: 0},
				{ID: "m", Activation: "tanh", Bias: jitter(rng, 0.35)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "rl", To: "m", Weight: 0.9 + jitter(rng, 0.25), Enabled: true},
				{ID: "s2", From: "rf", To: "m", Weight: -0.4 + jitter(rng, 0.25), Enabled: true},
				{ID: "s3", From: "rr", To: "m", Weight: 0.9 + jitter(rng, 0.25), Enabled: true},
				{ID: "s4", From: "r", To: "m", Weight: 0.2 + jitter(rng, 0.15), Enabled: true},
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
			SensorIDs:       []string{protoio.GTSAInputSensorName},
			ActuatorIDs:     []string{protoio.GTSAPredictActuatorName},
			Neurons: []model.Neuron{
				{ID: "x", Activation: "identity", Bias: 0},
				{ID: "y", Activation: "identity", Bias: jitter(rng, 0.3)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "x", To: "y", Weight: jitter(rng, 1.0), Enabled: true},
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
			SensorIDs:       []string{protoio.FXPriceSensorName, protoio.FXSignalSensorName},
			ActuatorIDs:     []string{protoio.FXTradeActuatorName},
			Neurons: []model.Neuron{
				{ID: "p", Activation: "identity", Bias: 0},
				{ID: "s", Activation: "identity", Bias: 0},
				{ID: "t", Activation: "tanh", Bias: jitter(rng, 0.25)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "p", To: "t", Weight: jitter(rng, 1.1), Enabled: true},
				{ID: "s2", From: "s", To: "t", Weight: jitter(rng, 1.1), Enabled: true},
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
			SensorIDs:       []string{protoio.EpitopesSignalSensorName, protoio.EpitopesMemorySensorName},
			ActuatorIDs:     []string{protoio.EpitopesResponseActuatorName},
			Neurons: []model.Neuron{
				{ID: "s", Activation: "identity", Bias: 0},
				{ID: "m", Activation: "identity", Bias: 0},
				{ID: "r", Activation: "tanh", Bias: jitter(rng, 0.25)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "s", To: "r", Weight: 0.9 + jitter(rng, 0.2), Enabled: true},
				{ID: "s2", From: "m", To: "r", Weight: 0.7 + jitter(rng, 0.2), Enabled: true},
			},
		})
	}
	return population
}

func seedLLVMPhaseOrderingPopulation(size int, seed int64) []model.Genome {
	rng := rand.New(rand.NewSource(seed))
	population := make([]model.Genome, 0, size)
	for i := 0; i < size; i++ {
		population = append(population, model.Genome{
			VersionedRecord: model.VersionedRecord{SchemaVersion: storage.CurrentSchemaVersion, CodecVersion: storage.CurrentCodecVersion},
			ID:              fmt.Sprintf("llvm-g0-%d", i),
			SensorIDs:       []string{protoio.LLVMComplexitySensorName, protoio.LLVMPassIndexSensorName},
			ActuatorIDs:     []string{protoio.LLVMPhaseActuatorName},
			Neurons: []model.Neuron{
				{ID: "c", Activation: "identity", Bias: 0},
				{ID: "p", Activation: "identity", Bias: 0},
				{ID: "o", Activation: "identity", Bias: 1 + jitter(rng, 0.1)},
			},
			Synapses: []model.Synapse{
				{ID: "s1", From: "p", To: "o", Weight: -2 + jitter(rng, 0.2), Enabled: true},
				{ID: "s2", From: "c", To: "o", Weight: -0.2 + jitter(rng, 0.15), Enabled: true},
			},
		})
	}
	return population
}

func jitter(rng *rand.Rand, amplitude float64) float64 {
	return (rng.Float64()*2 - 1) * amplitude
}
