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
	if len(agent.Genome.Substrate.Dimensions) < 3 {
		t.Fatalf("expected substrate density vector length >= 3, got=%v", agent.Genome.Substrate.Dimensions)
	}
	if agent.Genome.Substrate.Dimensions[0] != 1 || agent.Genome.Substrate.Dimensions[1] != 1 {
		t.Fatalf("expected substrate density prefix [1,1], got=%v", agent.Genome.Substrate.Dimensions)
	}
	if got := agent.Genome.Substrate.Parameters["dimensions"]; got < 3 {
		t.Fatalf("expected stored optimal substrate dimension >= 3, got=%f", got)
	}
	if got := agent.Genome.Substrate.Parameters["cpp_count"]; got != 2 {
		t.Fatalf("expected cpp_count=2 (xor sensors), got=%f", got)
	}
	if got := agent.Genome.Substrate.Parameters["cep_count"]; got != 1 {
		t.Fatalf("expected cep_count=1 (xor actuators), got=%f", got)
	}
	if got := agent.Genome.Substrate.Parameters["seed_cpp_links"]; got != 2 {
		t.Fatalf("expected seed_cpp_links=2 for xor substrate scaffold, got=%f", got)
	}
	if got := agent.Genome.Substrate.Parameters["seed_cep_links"]; got != 1 {
		t.Fatalf("expected seed_cep_links=1 for xor substrate scaffold, got=%f", got)
	}
	if len(agent.SubstrateCPPIDs) != 2 || len(agent.SubstrateCEPIDs) != 1 {
		t.Fatalf("expected substrate endpoint IDs in constructed agent, cpp=%v cep=%v", agent.SubstrateCPPIDs, agent.SubstrateCEPIDs)
	}
	if len(agent.SubstrateCPPIDs) > 0 && len(agent.SubstrateCPPIDs[0]) == 0 {
		t.Fatalf("expected non-empty cpp id values: %v", agent.SubstrateCPPIDs)
	}
	if len(agent.Genome.SensorIDs) != 2 || len(agent.Genome.ActuatorIDs) != 1 {
		t.Fatalf("expected external morphology io ids (2 sensors/1 actuator), got sensors=%v actuators=%v", agent.Genome.SensorIDs, agent.Genome.ActuatorIDs)
	}
	if len(agent.Genome.SensorNeuronLinks) != 0 || len(agent.Genome.NeuronActuatorLinks) != 0 {
		t.Fatalf("expected no direct io endpoint links on substrate genome, got sensor_links=%v actuator_links=%v", agent.Genome.SensorNeuronLinks, agent.Genome.NeuronActuatorLinks)
	}
	if agent.Genome.SensorLinks != 0 || agent.Genome.ActuatorLinks != 0 {
		t.Fatalf("expected zero direct io link counters for substrate genome, got sensor_links=%d actuator_links=%d", agent.Genome.SensorLinks, agent.Genome.ActuatorLinks)
	}
	if len(agent.SubstrateSensorNeuronLinks) != 2 || len(agent.SubstrateNeuronActuatorLinks) != 1 {
		t.Fatalf("expected stored substrate seed links (2 sensor,1 actuator), got sensor=%v actuator=%v", agent.SubstrateSensorNeuronLinks, agent.SubstrateNeuronActuatorLinks)
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

func TestConstructCortexSupportsActuatorVectorLengths(t *testing.T) {
	constraint := DefaultConstructConstraint()
	constraint.Morphology = "xor"
	constraint.ActuatorVectorLengths = map[string]int{
		"xor_output": 3,
	}
	out, err := ConstructCortex(
		"agent-vl",
		0,
		constraint,
		"neural",
		"none",
		"l2l_feedforward",
		rand.New(rand.NewSource(15)),
	)
	if err != nil {
		t.Fatalf("construct cortex with actuator vl: %v", err)
	}
	if len(out.OutputNeuronIDs) != 3 {
		t.Fatalf("expected 3 output neurons for xor_output vl=3, got=%v", out.OutputNeuronIDs)
	}
	if len(out.Genome.NeuronActuatorLinks) != 3 {
		t.Fatalf("expected 3 neuron-actuator links, got=%v", out.Genome.NeuronActuatorLinks)
	}
	for _, link := range out.Genome.NeuronActuatorLinks {
		if link.ActuatorID != "xor_output" {
			t.Fatalf("expected all output links to xor_output, got=%+v", out.Genome.NeuronActuatorLinks)
		}
	}
}

func TestDefaultSubstrateDensities(t *testing.T) {
	got := defaultSubstrateDensities(5)
	want := []int{1, 1, 5, 5, 5}
	if len(got) != len(want) {
		t.Fatalf("expected len=%d, got=%d (%v)", len(want), len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected densities at index %d: got=%v want=%v", i, got, want)
		}
	}
}

func TestConstructSubstrateEndpointIDs(t *testing.T) {
	cpp, cep := constructSubstrateEndpointIDs(4, 2, 3)
	if len(cpp) != 2 || len(cep) != 3 {
		t.Fatalf("expected cpp=2 cep=3, got cpp=%v cep=%v", cpp, cep)
	}
	if cpp[0] != "substrate:cpp:d4:0" || cep[0] != "substrate:cep:d4:0" {
		t.Fatalf("unexpected substrate endpoint naming, cpp=%v cep=%v", cpp, cep)
	}

	defaultCPP, defaultCEP := constructSubstrateEndpointIDs(0, 0, 0)
	if len(defaultCPP) != 1 || len(defaultCEP) != 1 {
		t.Fatalf("expected default endpoint counts 1/1, got cpp=%v cep=%v", defaultCPP, defaultCEP)
	}
}
