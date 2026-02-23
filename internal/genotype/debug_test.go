package genotype

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/storage"
)

func TestFormatGenomeIncludesCoreSections(t *testing.T) {
	genome := model.Genome{
		ID:          "g-debug",
		SensorIDs:   []string{protoio.XORInputLeftSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "sigmoid", Aggregator: "dot_product"},
		},
		Synapses: []model.Synapse{
			{ID: "s1", From: "i", To: "o", Weight: 0.5, Enabled: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: protoio.XORInputLeftSensorName, NeuronID: "i"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "o", ActuatorID: protoio.XOROutputActuatorName},
		},
		Substrate: &model.SubstrateConfig{
			CPPName:     "cpp-basic",
			CEPName:     "cep-basic",
			CPPIDs:      []string{"cpp-2", "cpp-1"},
			CEPIDs:      []string{"cep-1"},
			Dimensions:  []int{3, 4},
			Parameters:  map[string]float64{"alpha": 0.5, "beta": 0.25},
			WeightCount: 6,
		},
	}

	out := FormatGenome(genome)
	required := []string{
		"genome: g-debug",
		"sensors:",
		"actuators:",
		"neurons: 2",
		"synapses: 1",
		"sensor_links: 1",
		"actuator_links: 1",
		"synapse s1 i->o",
		"substrate: cpp=cpp-basic cep=cep-basic weight_count=6 dimensions=[3 4]",
		"substrate_cpp_ids: [cpp-1 cpp-2]",
		"substrate_cep_ids: [cep-1]",
		"substrate_parameters: {alpha=0.5, beta=0.25}",
	}
	for _, token := range required {
		if !strings.Contains(out, token) {
			t.Fatalf("expected formatted genome to contain %q, got:\n%s", token, out)
		}
	}
}

func TestFormatGenomeListFormDeterministic(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s-right", "s-left"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "h", Activation: "tanh"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s2", From: "h", To: "o", Weight: -1, Enabled: true},
			{ID: "s3", From: "i", To: "o", Weight: 0.25, Enabled: true},
			{ID: "s1", From: "i", To: "h", Weight: 0.5, Enabled: true},
		},
		SensorNeuronLinks: []model.SensorNeuronLink{
			{SensorID: "s-left", NeuronID: "h"},
			{SensorID: "s-right", NeuronID: "o"},
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "o", ActuatorID: "a1"},
			{NeuronID: "h", ActuatorID: "a1"},
		},
	}

	got := FormatGenomeListForm(genome)
	const want = "" +
		"s-left:\n" +
		"s-right:\n" +
		"h: s-left# i# 0.5\n" +
		"i:\n" +
		"o: s-right# h# -1 i# 0.25\n" +
		"a1: h o\n"
	if got != want {
		t.Fatalf("unexpected list form:\nwant:\n%s\ngot:\n%s", want, got)
	}
}

func TestWriteGenomeListForm(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
		},
	}
	path := filepath.Join(t.TempDir(), "agent.list")
	if err := WriteGenomeListForm(path, genome); err != nil {
		t.Fatalf("write list form: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read list form file: %v", err)
	}
	want := FormatGenomeListForm(genome)
	if string(data) != want {
		t.Fatalf("unexpected file contents:\nwant:\n%s\ngot:\n%s", want, string(data))
	}
}

func TestWriteGenomeListFormDefault(t *testing.T) {
	genome := model.Genome{
		ID:          "g-default",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
	}
	path, err := WriteGenomeListFormDefault(genome, t.TempDir())
	if err != nil {
		t.Fatalf("write default list form: %v", err)
	}
	if filepath.Base(path) != "g-default.agent" {
		t.Fatalf("expected default file name g-default.agent, got %q", filepath.Base(path))
	}
}

func TestPrintAndPrintListFormFromStore(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}

	genome := model.Genome{
		ID:          "g-store-print",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
		Neurons: []model.Neuron{
			{ID: "n1", Activation: "identity"},
		},
	}
	if err := store.SaveGenome(ctx, genome); err != nil {
		t.Fatalf("save genome: %v", err)
	}

	verbose, err := Print(ctx, store, genome.ID)
	if err != nil {
		t.Fatalf("print: %v", err)
	}
	if !strings.Contains(verbose, "genome: g-store-print") {
		t.Fatalf("unexpected print output:\n%s", verbose)
	}

	listForm, err := PrintListForm(ctx, store, genome.ID)
	if err != nil {
		t.Fatalf("print list form: %v", err)
	}
	if !strings.Contains(listForm, "n1:") {
		t.Fatalf("unexpected list form output:\n%s", listForm)
	}
}

func TestWriteListFormForGenomeID(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}

	genome := model.Genome{
		ID:          "g-store-file",
		SensorIDs:   []string{"s1"},
		ActuatorIDs: []string{"a1"},
	}
	if err := store.SaveGenome(ctx, genome); err != nil {
		t.Fatalf("save genome: %v", err)
	}

	dir := t.TempDir()
	path, err := WriteListFormForGenomeID(ctx, store, genome.ID, dir)
	if err != nil {
		t.Fatalf("write list form for id: %v", err)
	}
	if filepath.Base(path) != "g-store-file.agent" {
		t.Fatalf("unexpected output filename: %s", filepath.Base(path))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read written file: %v", err)
	}
	if string(data) != FormatGenomeListForm(genome) {
		t.Fatalf("unexpected output file contents:\nwant:\n%s\ngot:\n%s", FormatGenomeListForm(genome), string(data))
	}
}
