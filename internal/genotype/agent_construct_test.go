package genotype

import (
	"math/rand"
	"testing"
)

func TestConstructCortexSupportsXORMimicAlias(t *testing.T) {
	constraint := DefaultConstructConstraint()
	constraint.Morphology = "xor_mimic"

	out, err := ConstructCortex(
		"agent-1",
		0,
		constraint,
		"neural",
		"none",
		"l2l_feedforward",
		rand.New(rand.NewSource(3)),
	)
	if err != nil {
		t.Fatalf("construct cortex: %v", err)
	}
	if out.Genome.ID != "agent-1" {
		t.Fatalf("unexpected genome id: %s", out.Genome.ID)
	}
	if len(out.Genome.SensorIDs) != 2 || len(out.Genome.ActuatorIDs) != 1 {
		t.Fatalf("unexpected io scaffold: sensors=%v actuators=%v", out.Genome.SensorIDs, out.Genome.ActuatorIDs)
	}
	if out.Genome.Substrate != nil {
		t.Fatalf("expected neural encoding to keep substrate nil, got=%+v", out.Genome.Substrate)
	}
	if len(out.InputNeuronIDs) != 2 || len(out.OutputNeuronIDs) != 1 {
		t.Fatalf("unexpected input/output neuron ids: in=%v out=%v", out.InputNeuronIDs, out.OutputNeuronIDs)
	}
}

func TestConstructAgentMaterializesStrategyAndSubstrate(t *testing.T) {
	constraint := DefaultConstructConstraint()
	constraint.Morphology = "xor"
	constraint.AgentEncodingTypes = []string{"substrate"}
	constraint.SubstratePlasticities = []string{"none"}
	constraint.SubstrateLinkforms = []string{"l2l_feedforward"}
	constraint.TuningSelectionFs = []string{"dynamic_random"}
	constraint.AnnealingParameters = []float64{0.75}
	constraint.PerturbationRanges = []float64{1.25}
	constraint.HeredityTypes = []string{"lamarckian"}
	constraint.TotTopologicalMutationsFs = []TopologicalMutationOption{
		{Name: "ncount_exponential", Param: 0.6},
	}

	agent, err := ConstructAgent("sp-1", "agent-42", constraint, rand.New(rand.NewSource(11)))
	if err != nil {
		t.Fatalf("construct agent: %v", err)
	}
	if agent.SpecieID != "sp-1" {
		t.Fatalf("unexpected specie id: %s", agent.SpecieID)
	}
	if agent.EncodingType != "substrate" {
		t.Fatalf("expected substrate encoding, got=%s", agent.EncodingType)
	}
	if agent.Genome.Substrate == nil {
		t.Fatal("expected substrate config for substrate encoding")
	}
	if agent.Genome.Substrate.CPPName != "none" || agent.Genome.Substrate.CEPName != "l2l_feedforward" {
		t.Fatalf("unexpected substrate metadata: %+v", agent.Genome.Substrate)
	}
	if len(agent.Genome.Substrate.Dimensions) != 1 || agent.Genome.Substrate.Dimensions[0] <= 0 {
		t.Fatalf("expected positive substrate dimension, got=%v", agent.Genome.Substrate.Dimensions)
	}
	if agent.Genome.Strategy == nil {
		t.Fatal("expected strategy config to be materialized")
	}
	if agent.Genome.Strategy.TuningSelection != "dynamic_random" {
		t.Fatalf("unexpected tuning selection: %s", agent.Genome.Strategy.TuningSelection)
	}
	if agent.Genome.Strategy.AnnealingFactor != 0.75 {
		t.Fatalf("unexpected annealing parameter: %f", agent.Genome.Strategy.AnnealingFactor)
	}
	if agent.Genome.Strategy.TopologicalMode != "ncount_exponential" || agent.Genome.Strategy.TopologicalParam != 0.6 {
		t.Fatalf("unexpected topological config: %+v", agent.Genome.Strategy)
	}
	if agent.Genome.Strategy.HeredityType != "lamarckian" {
		t.Fatalf("unexpected heredity type: %s", agent.Genome.Strategy.HeredityType)
	}
	if agent.PerturbationRange != 1.25 {
		t.Fatalf("unexpected perturbation range: %f", agent.PerturbationRange)
	}
}

func TestConstructAgentValidatesAgentID(t *testing.T) {
	if _, err := ConstructAgent("sp", "", DefaultConstructConstraint(), rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("expected validation error for empty agent id")
	}
}

func TestConstructCortexRejectsUnknownMorphology(t *testing.T) {
	constraint := DefaultConstructConstraint()
	constraint.Morphology = "unknown-scope"
	if _, err := ConstructCortex("agent", 0, constraint, "neural", "none", "l2l_feedforward", rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("expected error for unknown morphology")
	}
}
