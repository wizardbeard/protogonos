package genotype

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"protogonos/internal/model"
)

// FormatGenome returns a human-readable multiline dump of the genome.
func FormatGenome(genome model.Genome) string {
	var b strings.Builder
	fmt.Fprintf(&b, "genome: %s\n", genome.ID)
	fmt.Fprintf(&b, "sensors: %v\n", uniqueSortedStrings(genome.SensorIDs))
	fmt.Fprintf(&b, "actuators: %v\n", uniqueSortedStrings(genome.ActuatorIDs))
	fmt.Fprintf(&b, "neurons: %d\n", len(genome.Neurons))
	for _, neuron := range sortedNeurons(genome.Neurons) {
		aggregator := neuron.Aggregator
		if aggregator == "" {
			aggregator = "dot_product"
		}
		fmt.Fprintf(&b, "  neuron %s act=%s aggr=%s bias=%g\n", neuron.ID, neuron.Activation, aggregator, neuron.Bias)
	}
	fmt.Fprintf(&b, "synapses: %d\n", len(genome.Synapses))
	for _, synapse := range sortedSynapses(genome.Synapses) {
		fmt.Fprintf(&b, "  synapse %s %s->%s w=%g enabled=%t recurrent=%t\n", synapse.ID, synapse.From, synapse.To, synapse.Weight, synapse.Enabled, synapse.Recurrent)
	}
	if len(genome.SensorNeuronLinks) > 0 {
		fmt.Fprintf(&b, "sensor_links: %d\n", len(genome.SensorNeuronLinks))
		for _, link := range sortedSensorNeuronLinks(genome.SensorNeuronLinks) {
			fmt.Fprintf(&b, "  %s->%s\n", link.SensorID, link.NeuronID)
		}
	}
	if len(genome.NeuronActuatorLinks) > 0 {
		fmt.Fprintf(&b, "actuator_links: %d\n", len(genome.NeuronActuatorLinks))
		for _, link := range sortedNeuronActuatorLinks(genome.NeuronActuatorLinks) {
			fmt.Fprintf(&b, "  %s->%s\n", link.NeuronID, link.ActuatorID)
		}
	}
	return b.String()
}

// FormatGenomeListForm returns a compact adjacency-list style representation
// analogous to genotype:print_ListForm/1.
func FormatGenomeListForm(genome model.Genome) string {
	var b strings.Builder

	for _, sensorID := range uniqueSortedStrings(genome.SensorIDs) {
		fmt.Fprintf(&b, "%s:\n", sensorID)
	}

	neuronIDs := make([]string, 0, len(genome.Neurons))
	for _, neuron := range genome.Neurons {
		if neuron.ID == "" {
			continue
		}
		neuronIDs = append(neuronIDs, neuron.ID)
	}
	sort.Strings(neuronIDs)
	seenNeuron := make(map[string]struct{}, len(neuronIDs))
	for _, neuronID := range neuronIDs {
		if _, ok := seenNeuron[neuronID]; ok {
			continue
		}
		seenNeuron[neuronID] = struct{}{}

		fmt.Fprintf(&b, "%s:", neuronID)
		incoming := incomingSynapses(genome.Synapses, neuronID)
		for _, synapse := range incoming {
			fmt.Fprintf(&b, " %s# %g", synapse.From, synapse.Weight)
		}
		b.WriteString("\n")
	}

	for _, actuatorID := range uniqueSortedStrings(genome.ActuatorIDs) {
		fmt.Fprintf(&b, "%s:", actuatorID)
		fanin := actuatorFaninNeurons(genome.NeuronActuatorLinks, actuatorID)
		for _, neuronID := range fanin {
			fmt.Fprintf(&b, " %s", neuronID)
		}
		b.WriteString("\n")
	}

	return b.String()
}

// WriteGenomeListForm writes FormatGenomeListForm output to path.
func WriteGenomeListForm(path string, genome model.Genome) error {
	return os.WriteFile(path, []byte(FormatGenomeListForm(genome)), 0o644)
}

func uniqueSortedStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func sortedNeurons(neurons []model.Neuron) []model.Neuron {
	out := append([]model.Neuron(nil), neurons...)
	sort.Slice(out, func(i, j int) bool {
		return out[i].ID < out[j].ID
	})
	return out
}

func sortedSynapses(synapses []model.Synapse) []model.Synapse {
	out := append([]model.Synapse(nil), synapses...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].ID != out[j].ID {
			return out[i].ID < out[j].ID
		}
		if out[i].From != out[j].From {
			return out[i].From < out[j].From
		}
		return out[i].To < out[j].To
	})
	return out
}

func sortedSensorNeuronLinks(links []model.SensorNeuronLink) []model.SensorNeuronLink {
	out := append([]model.SensorNeuronLink(nil), links...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].SensorID != out[j].SensorID {
			return out[i].SensorID < out[j].SensorID
		}
		return out[i].NeuronID < out[j].NeuronID
	})
	return out
}

func sortedNeuronActuatorLinks(links []model.NeuronActuatorLink) []model.NeuronActuatorLink {
	out := append([]model.NeuronActuatorLink(nil), links...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].ActuatorID != out[j].ActuatorID {
			return out[i].ActuatorID < out[j].ActuatorID
		}
		return out[i].NeuronID < out[j].NeuronID
	})
	return out
}

func incomingSynapses(synapses []model.Synapse, toNeuronID string) []model.Synapse {
	out := make([]model.Synapse, 0, len(synapses))
	for _, synapse := range synapses {
		if synapse.To != toNeuronID {
			continue
		}
		out = append(out, synapse)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].From != out[j].From {
			return out[i].From < out[j].From
		}
		return out[i].ID < out[j].ID
	})
	return out
}

func actuatorFaninNeurons(links []model.NeuronActuatorLink, actuatorID string) []string {
	fanin := make([]string, 0, len(links))
	for _, link := range links {
		if link.ActuatorID != actuatorID {
			continue
		}
		fanin = append(fanin, link.NeuronID)
	}
	return uniqueSortedStrings(fanin)
}
