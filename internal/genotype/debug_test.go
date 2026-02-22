package genotype

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
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
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "o", ActuatorID: "a1"},
			{NeuronID: "h", ActuatorID: "a1"},
		},
	}

	got := FormatGenomeListForm(genome)
	const want = "" +
		"s-left:\n" +
		"s-right:\n" +
		"h: i# 0.5\n" +
		"i:\n" +
		"o: h# -1 i# 0.25\n" +
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
