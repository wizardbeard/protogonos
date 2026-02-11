package genotype

import (
	"context"
	"fmt"
	"math/rand"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/storage"
)

type SeedPopulation struct {
	Genomes         []model.Genome
	InputNeuronIDs  []string
	OutputNeuronIDs []string
}

func ConstructSeedPopulation(scapeName string, size int, seed int64) (SeedPopulation, error) {
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
	return store.SavePopulation(ctx, pop)
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

func jitter(rng *rand.Rand, amplitude float64) float64 {
	return (rng.Float64()*2 - 1) * amplitude
}
