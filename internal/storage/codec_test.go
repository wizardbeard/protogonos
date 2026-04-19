package storage

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"protogonos/internal/model"
)

func TestDecodeGenomeFixture(t *testing.T) {
	genome := decodeGenomeFixture(t, "minimal_genome_v1.json")
	if genome.ID != "genome-minimal-1" {
		t.Fatalf("unexpected genome id: %s", genome.ID)
	}
}

func TestDecodeGenomeFixtureWithExplicitIOLinks(t *testing.T) {
	genome := decodeGenomeFixture(t, "io_links_genome_v1.json")
	if genome.ID != "genome-io-links-1" {
		t.Fatalf("unexpected genome id: %s", genome.ID)
	}
	if len(genome.SensorNeuronLinks) != 3 {
		t.Fatalf("unexpected sensor-neuron links: %+v", genome.SensorNeuronLinks)
	}
	if len(genome.NeuronActuatorLinks) != 3 {
		t.Fatalf("unexpected neuron-actuator links: %+v", genome.NeuronActuatorLinks)
	}
	if genome.SensorLinks != len(genome.SensorNeuronLinks) {
		t.Fatalf("sensor link counter mismatch: count=%d explicit=%d", genome.SensorLinks, len(genome.SensorNeuronLinks))
	}
	if genome.ActuatorLinks != len(genome.NeuronActuatorLinks) {
		t.Fatalf("actuator link counter mismatch: count=%d explicit=%d", genome.ActuatorLinks, len(genome.NeuronActuatorLinks))
	}
}

func TestDecodeGenomeFixtureWithVectorIO(t *testing.T) {
	genome := decodeGenomeFixture(t, "vector_io_genome_v1.json")
	if genome.ID != "genome-vector-io-1" {
		t.Fatalf("unexpected genome id: %s", genome.ID)
	}
	if len(genome.SensorIDs) != 1 || genome.SensorIDs[0] != "vector_input" {
		t.Fatalf("unexpected vector sensor ids: %+v", genome.SensorIDs)
	}
	if len(genome.ActuatorIDs) != 1 || genome.ActuatorIDs[0] != "vector_output" {
		t.Fatalf("unexpected vector actuator ids: %+v", genome.ActuatorIDs)
	}
	if len(genome.SensorNeuronLinks) != 1 || genome.SensorNeuronLinks[0].SensorID != "vector_input" {
		t.Fatalf("unexpected vector sensor-neuron links: %+v", genome.SensorNeuronLinks)
	}
	if len(genome.NeuronActuatorLinks) != 1 || genome.NeuronActuatorLinks[0].ActuatorID != "vector_output" {
		t.Fatalf("unexpected vector neuron-actuator links: %+v", genome.NeuronActuatorLinks)
	}
}

func TestDecodeGenomeFixtureWithSubstrateIO(t *testing.T) {
	genome := decodeGenomeFixture(t, "substrate_io_genome_v1.json")
	if genome.ID != "genome-substrate-io-1" {
		t.Fatalf("unexpected genome id: %s", genome.ID)
	}
	if genome.Substrate == nil {
		t.Fatal("expected substrate config")
	}
	if len(genome.Substrate.CPPIDs) != 1 || genome.Substrate.CPPIDs[0] != "substrate_cpp_0" {
		t.Fatalf("unexpected substrate cpp ids: %+v", genome.Substrate.CPPIDs)
	}
	if len(genome.Substrate.CEPIDs) != 1 || genome.Substrate.CEPIDs[0] != "substrate_cep_0" {
		t.Fatalf("unexpected substrate cep ids: %+v", genome.Substrate.CEPIDs)
	}
	if len(genome.SensorNeuronLinks) != 1 || genome.SensorNeuronLinks[0].SensorID != "substrate_cpp_0" {
		t.Fatalf("unexpected substrate sensor-neuron links: %+v", genome.SensorNeuronLinks)
	}
	if len(genome.NeuronActuatorLinks) != 1 || genome.NeuronActuatorLinks[0].ActuatorID != "substrate_cep_0" {
		t.Fatalf("unexpected substrate neuron-actuator links: %+v", genome.NeuronActuatorLinks)
	}
}

func TestDecodeAgentFixture(t *testing.T) {
	path := fixturePath("minimal_agent_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	agent, err := DecodeAgent(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if agent.ID != "agent-minimal-1" {
		t.Fatalf("unexpected agent id: %s", agent.ID)
	}
	if agent.GenomeID != "genome-minimal-1" {
		t.Fatalf("unexpected genome id: %s", agent.GenomeID)
	}
}

func TestDecodePopulationFixture(t *testing.T) {
	path := fixturePath("minimal_population_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	population, err := DecodePopulation(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if population.ID != "population-minimal-1" {
		t.Fatalf("unexpected population id: %s", population.ID)
	}
	if len(population.AgentIDs) != 1 || population.AgentIDs[0] != "agent-minimal-1" {
		t.Fatalf("unexpected population agent ids: %+v", population.AgentIDs)
	}
}

func TestDecodeScapeSummaryFixture(t *testing.T) {
	path := fixturePath("minimal_scape_summary_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	summary, err := DecodeScapeSummary(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if summary.Name != "xor" {
		t.Fatalf("unexpected scape name: %s", summary.Name)
	}
	if summary.BestFitness != 0.75 {
		t.Fatalf("unexpected best fitness: %f", summary.BestFitness)
	}
}

func TestGenomeCodecRoundTrip(t *testing.T) {
	input := model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		ID:              "g1",
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity", Bias: 0},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "n1", To: "n1", Weight: 1, Enabled: true, Recurrent: false},
		},
		SensorIDs:   []string{"sensor:input"},
		ActuatorIDs: []string{"actuator:output"},
	}

	encoded, err := EncodeGenome(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	decoded, err := DecodeGenome(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if decoded.ID != input.ID {
		t.Fatalf("id mismatch: got=%s want=%s", decoded.ID, input.ID)
	}
	if len(decoded.Neurons) != len(input.Neurons) {
		t.Fatalf("neuron count mismatch: got=%d want=%d", len(decoded.Neurons), len(input.Neurons))
	}
}

func TestAgentCodecRoundTrip(t *testing.T) {
	input := model.Agent{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		ID:              "a1",
		GenomeID:        "g1",
	}

	encoded, err := EncodeAgent(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	decoded, err := DecodeAgent(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if decoded.ID != input.ID || decoded.GenomeID != input.GenomeID {
		t.Fatalf("decoded agent mismatch: got=%+v want=%+v", decoded, input)
	}
}

func TestGenomeCodecRoundTripFixtureEqualityWithExplicitIOLinks(t *testing.T) {
	expected := decodeGenomeFixture(t, "io_links_genome_v1.json")

	encoded, err := EncodeGenome(expected)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	actual, err := DecodeGenome(encoded)
	if err != nil {
		t.Fatalf("decode roundtrip: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("roundtrip mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestGenomeCodecRoundTripFixtureEqualityWithVectorIO(t *testing.T) {
	expected := decodeGenomeFixture(t, "vector_io_genome_v1.json")

	encoded, err := EncodeGenome(expected)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	actual, err := DecodeGenome(encoded)
	if err != nil {
		t.Fatalf("decode roundtrip: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("roundtrip mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestGenomeCodecRoundTripFixtureEqualityWithSubstrateIO(t *testing.T) {
	expected := decodeGenomeFixture(t, "substrate_io_genome_v1.json")

	encoded, err := EncodeGenome(expected)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	actual, err := DecodeGenome(encoded)
	if err != nil {
		t.Fatalf("decode roundtrip: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("roundtrip mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestAgentCodecRoundTripFixtureEquality(t *testing.T) {
	path := fixturePath("minimal_agent_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	expected, err := DecodeAgent(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	encoded, err := EncodeAgent(expected)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	actual, err := DecodeAgent(encoded)
	if err != nil {
		t.Fatalf("decode roundtrip: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("roundtrip mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestPopulationCodecRoundTrip(t *testing.T) {
	input := model.Population{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		ID:              "p1",
		AgentIDs:        []string{"a1", "a2"},
		Generation:      3,
	}

	encoded, err := EncodePopulation(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	decoded, err := DecodePopulation(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if decoded.ID != input.ID || decoded.Generation != input.Generation {
		t.Fatalf("decoded population mismatch: got=%+v want=%+v", decoded, input)
	}
}

func TestPopulationCodecRoundTripFixtureEquality(t *testing.T) {
	path := fixturePath("minimal_population_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	expected, err := DecodePopulation(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	encoded, err := EncodePopulation(expected)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	actual, err := DecodePopulation(encoded)
	if err != nil {
		t.Fatalf("decode roundtrip: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("roundtrip mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestScapeSummaryCodecRoundTrip(t *testing.T) {
	input := model.ScapeSummary{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		Name:            "xor",
		Description:     "toy xor benchmark",
		BestFitness:     0.95,
	}

	encoded, err := EncodeScapeSummary(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	decoded, err := DecodeScapeSummary(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if decoded.Name != input.Name || decoded.BestFitness != input.BestFitness {
		t.Fatalf("decoded summary mismatch: got=%+v want=%+v", decoded, input)
	}
}

func TestScapeSummaryCodecRoundTripFixtureEquality(t *testing.T) {
	path := fixturePath("minimal_scape_summary_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	expected, err := DecodeScapeSummary(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	encoded, err := EncodeScapeSummary(expected)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	actual, err := DecodeScapeSummary(encoded)
	if err != nil {
		t.Fatalf("decode roundtrip: %v", err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("roundtrip mismatch\nactual=%+v\nexpected=%+v", actual, expected)
	}
}

func TestLineageCodecRoundTrip(t *testing.T) {
	input := []model.LineageRecord{
		{
			VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
			GenomeID:        "g1",
			ParentID:        "",
			Generation:      0,
			Operation:       "seed",
			Events: []model.EvoHistoryEvent{
				{Mutation: "seed"},
				{Mutation: "add_link", IDs: []string{"L0:n0", "L1:n1"}},
			},
			Fingerprint: "fp1",
			Summary: model.LineageSummary{
				TotalNeurons:           3,
				TotalSynapses:          2,
				TotalRecurrentSynapses: 0,
				TotalSensors:           1,
				TotalActuators:         1,
				ActivationDistribution: map[string]int{"identity": 2, "sigmoid": 1},
				AggregatorDistribution: map[string]int{"dot_product": 3},
			},
		},
	}

	encoded, err := EncodeLineage(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, err := DecodeLineage(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(decoded, input) {
		t.Fatalf("decoded lineage mismatch: got=%+v want=%+v", decoded, input)
	}
}

func TestLineageCodecVersionMismatch(t *testing.T) {
	input := []model.LineageRecord{
		{
			VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion + 1},
			GenomeID:        "g1",
		},
	}
	encoded, err := EncodeLineage(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	_, err = DecodeLineage(encoded)
	if !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("expected ErrVersionMismatch, got: %v", err)
	}
}

func TestFitnessHistoryCodecRoundTrip(t *testing.T) {
	input := []float64{0.1, 0.4, 0.8}
	encoded, err := EncodeFitnessHistory(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, err := DecodeFitnessHistory(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(decoded, input) {
		t.Fatalf("decoded history mismatch: got=%+v want=%+v", decoded, input)
	}
}

func TestGenerationDiagnosticsCodecRoundTrip(t *testing.T) {
	input := []model.GenerationDiagnostics{
		{Generation: 1, BestFitness: 0.8, MeanFitness: 0.6, MinFitness: 0.2, SpeciesCount: 2, FingerprintDiversity: 2},
		{Generation: 2, BestFitness: 0.9, MeanFitness: 0.7, MinFitness: 0.3, SpeciesCount: 3, FingerprintDiversity: 3},
	}
	encoded, err := EncodeGenerationDiagnostics(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, err := DecodeGenerationDiagnostics(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(decoded, input) {
		t.Fatalf("decoded diagnostics mismatch: got=%+v want=%+v", decoded, input)
	}
}

func TestSpeciesHistoryCodecRoundTrip(t *testing.T) {
	input := []model.SpeciesGeneration{
		{
			Generation: 1,
			Species:    []model.SpeciesMetrics{{Key: "sp-1", Size: 3, MeanFitness: 0.7, BestFitness: 0.9}},
			NewSpecies: []string{"sp-1"},
		},
		{
			Generation:     2,
			Species:        []model.SpeciesMetrics{{Key: "sp-1", Size: 2, MeanFitness: 0.8, BestFitness: 0.95}},
			ExtinctSpecies: []string{"sp-2"},
		},
	}
	encoded, err := EncodeSpeciesHistory(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, err := DecodeSpeciesHistory(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(decoded, input) {
		t.Fatalf("decoded species history mismatch: got=%+v want=%+v", decoded, input)
	}
}

func TestTopGenomesCodecRoundTrip(t *testing.T) {
	input := []model.TopGenomeRecord{
		{Rank: 1, Fitness: 0.9, Genome: model.Genome{ID: "g1"}},
		{Rank: 2, Fitness: 0.8, Genome: model.Genome{ID: "g2"}},
	}
	encoded, err := EncodeTopGenomes(input)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, err := DecodeTopGenomes(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(decoded, input) {
		t.Fatalf("decoded top genomes mismatch: got=%+v want=%+v", decoded, input)
	}
}

func TestDecodeGenomeVersionMismatch(t *testing.T) {
	genome := decodeGenomeFixture(t, "minimal_genome_v1.json")
	genome.CodecVersion++

	encoded, err := EncodeGenome(genome)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	_, err = DecodeGenome(encoded)
	if !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("expected ErrVersionMismatch, got: %v", err)
	}
}

func TestDecodeAgentVersionMismatch(t *testing.T) {
	path := fixturePath("minimal_agent_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	agent, err := DecodeAgent(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	agent.CodecVersion++

	encoded, err := EncodeAgent(agent)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	_, err = DecodeAgent(encoded)
	if !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("expected ErrVersionMismatch, got: %v", err)
	}
}

func TestDecodePopulationVersionMismatch(t *testing.T) {
	path := fixturePath("minimal_population_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	population, err := DecodePopulation(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	population.SchemaVersion++

	encoded, err := EncodePopulation(population)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	_, err = DecodePopulation(encoded)
	if !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("expected ErrVersionMismatch, got: %v", err)
	}
}

func TestDecodeScapeSummaryVersionMismatch(t *testing.T) {
	path := fixturePath("minimal_scape_summary_v1.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	summary, err := DecodeScapeSummary(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	summary.CodecVersion++

	encoded, err := EncodeScapeSummary(summary)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	_, err = DecodeScapeSummary(encoded)
	if !errors.Is(err, ErrVersionMismatch) {
		t.Fatalf("expected ErrVersionMismatch, got: %v", err)
	}
}

func fixturePath(name string) string {
	return filepath.Join("..", "..", "testdata", "fixtures", name)
}

func decodeGenomeFixture(t *testing.T, name string) model.Genome {
	t.Helper()

	path := fixturePath(name)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	genome, err := DecodeGenome(data)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	return genome
}
