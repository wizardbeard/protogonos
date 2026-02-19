package map2rec

import (
	"errors"
	"math"
	"testing"
)

func TestConvertUnsupportedKind(t *testing.T) {
	_, err := Convert("unknown", map[string]any{})
	if !errors.Is(err, ErrUnsupportedKind) {
		t.Fatalf("expected ErrUnsupportedKind, got %v", err)
	}
}

func TestConvertDispatchesSensorAndActuatorKinds(t *testing.T) {
	gotSensor, err := Convert("sensor", map[string]any{"name": "s"})
	if err != nil {
		t.Fatalf("convert sensor: %v", err)
	}
	sensor, ok := gotSensor.(SensorRecord)
	if !ok || sensor.Name != "s" {
		t.Fatalf("unexpected sensor dispatch result: %#v", gotSensor)
	}

	gotActuator, err := Convert("actuator", map[string]any{"name": "a"})
	if err != nil {
		t.Fatalf("convert actuator: %v", err)
	}
	actuator, ok := gotActuator.(ActuatorRecord)
	if !ok || actuator.Name != "a" {
		t.Fatalf("unexpected actuator dispatch result: %#v", gotActuator)
	}

	gotNeuron, err := Convert("neuron", map[string]any{"af": "tanh"})
	if err != nil {
		t.Fatalf("convert neuron: %v", err)
	}
	neuron, ok := gotNeuron.(NeuronRecord)
	if !ok || neuron.ActivationFunction != "tanh" {
		t.Fatalf("unexpected neuron dispatch result: %#v", gotNeuron)
	}

	gotAgent, err := Convert("agent", map[string]any{"encoding_type": "neural"})
	if err != nil {
		t.Fatalf("convert agent: %v", err)
	}
	agent, ok := gotAgent.(AgentRecord)
	if !ok || agent.EncodingType != "neural" {
		t.Fatalf("unexpected agent dispatch result: %#v", gotAgent)
	}

	gotCortex, err := Convert("cortex", map[string]any{"id": "c1"})
	if err != nil {
		t.Fatalf("convert cortex: %v", err)
	}
	if cortex, ok := gotCortex.(CortexRecord); !ok || cortex.ID != "c1" {
		t.Fatalf("unexpected cortex dispatch result: %#v", gotCortex)
	}

	gotSubstrate, err := Convert("substrate", map[string]any{"id": "sub1"})
	if err != nil {
		t.Fatalf("convert substrate: %v", err)
	}
	if substrate, ok := gotSubstrate.(SubstrateRecord); !ok || substrate.ID != "sub1" {
		t.Fatalf("unexpected substrate dispatch result: %#v", gotSubstrate)
	}

	gotPolis, err := Convert("polis", map[string]any{"id": "p1"})
	if err != nil {
		t.Fatalf("convert polis: %v", err)
	}
	if polis, ok := gotPolis.(PolisRecord); !ok || polis.ID != "p1" {
		t.Fatalf("unexpected polis dispatch result: %#v", gotPolis)
	}

	gotScape, err := Convert("scape", map[string]any{"id": "sc1"})
	if err != nil {
		t.Fatalf("convert scape: %v", err)
	}
	if scape, ok := gotScape.(ScapeRecord); !ok || scape.ID != "sc1" {
		t.Fatalf("unexpected scape dispatch result: %#v", gotScape)
	}

	gotSector, err := Convert("sector", map[string]any{"id": "sec1"})
	if err != nil {
		t.Fatalf("convert sector: %v", err)
	}
	if sector, ok := gotSector.(SectorRecord); !ok || sector.ID != "sec1" {
		t.Fatalf("unexpected sector dispatch result: %#v", gotSector)
	}

	gotAvatar, err := Convert("avatar", map[string]any{"id": "av1"})
	if err != nil {
		t.Fatalf("convert avatar: %v", err)
	}
	if avatar, ok := gotAvatar.(AvatarRecord); !ok || avatar.ID != "av1" {
		t.Fatalf("unexpected avatar dispatch result: %#v", gotAvatar)
	}

	gotSpecie, err := Convert("specie", map[string]any{"id": "sp1"})
	if err != nil {
		t.Fatalf("convert specie: %v", err)
	}
	if specie, ok := gotSpecie.(SpecieRecord); !ok || specie.ID != "sp1" {
		t.Fatalf("unexpected specie dispatch result: %#v", gotSpecie)
	}

	gotPopulation, err := Convert("population", map[string]any{"id": "pop1"})
	if err != nil {
		t.Fatalf("convert population: %v", err)
	}
	if population, ok := gotPopulation.(PopulationRecord); !ok || population.ID != "pop1" {
		t.Fatalf("unexpected population dispatch result: %#v", gotPopulation)
	}

	gotTrace, err := Convert("trace", map[string]any{"step_size": 100})
	if err != nil {
		t.Fatalf("convert trace: %v", err)
	}
	if trace, ok := gotTrace.(TraceRecord); !ok || trace.StepSize != 100 {
		t.Fatalf("unexpected trace dispatch result: %#v", gotTrace)
	}

	gotStat, err := Convert("stat", map[string]any{"avg_fitness": 0.4})
	if err != nil {
		t.Fatalf("convert stat: %v", err)
	}
	if stat, ok := gotStat.(StatRecord); !ok || stat.AvgFitness != 0.4 {
		t.Fatalf("unexpected stat dispatch result: %#v", gotStat)
	}

	gotTopo, err := Convert("topology_summary", map[string]any{"tot_neurons": 4})
	if err != nil {
		t.Fatalf("convert topology_summary: %v", err)
	}
	if topo, ok := gotTopo.(TopologySummaryRecord); !ok || topo.TotalNeurons != 4 {
		t.Fatalf("unexpected topology_summary dispatch result: %#v", gotTopo)
	}

	gotSignature, err := Convert("signature", map[string]any{"generalized_Pattern": []any{"p"}})
	if err != nil {
		t.Fatalf("convert signature: %v", err)
	}
	if sig, ok := gotSignature.(SignatureRecord); !ok || sig.GeneralizedPattern == nil {
		t.Fatalf("unexpected signature dispatch result: %#v", gotSignature)
	}

	gotChampion, err := Convert("champion", map[string]any{"fitness": 0.9})
	if err != nil {
		t.Fatalf("convert champion: %v", err)
	}
	if champion, ok := gotChampion.(ChampionRecord); !ok || champion.Fitness != 0.9 {
		t.Fatalf("unexpected champion dispatch result: %#v", gotChampion)
	}
}

func TestConvertConstraintOverridesKnownFieldsAndIgnoresUnknown(t *testing.T) {
	in := map[string]any{
		"morphology":             "gtsa_v1",
		"population_selection_f": "hof_competition",
		"tuning_duration_f":      []any{"nsize_proportional", 0.25},
		"mutation_operators":     []any{[]any{"add_neuron", 10.0}, map[string]any{"name": "add_inlink", "weight": 6.0}},
		"tot_topological_mutations_fs": []any{
			[]any{"ncount_linear", 1.0},
		},
		"unknown_field": 123,
	}

	out := ConvertConstraint(in)
	if out.Morphology != "gtsa_v1" {
		t.Fatalf("unexpected morphology: %s", out.Morphology)
	}
	if out.PopulationSelectionF != "hof_competition" {
		t.Fatalf("unexpected selection: %s", out.PopulationSelectionF)
	}
	if out.TuningDurationF.Name != "nsize_proportional" || out.TuningDurationF.Param != 0.25 {
		t.Fatalf("unexpected tuning duration: %+v", out.TuningDurationF)
	}
	if len(out.MutationOperators) != 2 {
		t.Fatalf("unexpected mutation operators: %+v", out.MutationOperators)
	}
	if len(out.TotTopologicalMutationsFs) != 1 || out.TotTopologicalMutationsFs[0].Name != "ncount_linear" {
		t.Fatalf("unexpected topological policies: %+v", out.TotTopologicalMutationsFs)
	}
}

func TestConvertConstraintKeepsDefaultsOnInvalidFieldShape(t *testing.T) {
	in := map[string]any{
		"neural_afs": []any{"tanh", 7},
	}
	out := ConvertConstraint(in)
	if len(out.NeuralAFs) != 3 {
		t.Fatalf("expected default neural_afs to be retained, got %+v", out.NeuralAFs)
	}
}

func TestConvertPMPMapsFields(t *testing.T) {
	in := map[string]any{
		"op_mode":             "gt",
		"population_id":       "pop-1",
		"survival_percentage": 0.75,
		"specie_size_limit":   12,
		"init_specie_size":    24,
		"polis_id":            "polis-a",
		"generation_limit":    300,
		"evaluations_limit":   9000,
		"fitness_goal":        "inf",
		"benchmarker_pid":     "bench-1",
		"committee_pid":       "committee-1",
	}
	out := ConvertPMP(in)
	if out.PopulationID != "pop-1" || out.GenerationLimit != 300 {
		t.Fatalf("unexpected pmp conversion: %+v", out)
	}
	if !math.IsInf(out.FitnessGoal, 1) {
		t.Fatalf("expected infinite fitness goal, got %f", out.FitnessGoal)
	}
}

func TestConvertSensorMapsFields(t *testing.T) {
	in := map[string]any{
		"id":            "sensor-1",
		"name":          "rangefinder",
		"type":          "distance",
		"cx_id":         "cx-1",
		"scape":         []any{"public", "flatland"},
		"vl":            9,
		"fanout_ids":    []any{"n-1", "n-2"},
		"generation":    3,
		"format":        map[string]any{"kind": "geo"},
		"parameters":    []any{1, 2},
		"gt_parameters": map[string]any{"noise": 0.1},
		"phys_rep":      map[string]any{"radius": 1.2},
		"vis_rep":       map[string]any{"color": "green"},
		"pre_f":         "identity",
		"post_f":        "clip",
	}
	out := ConvertSensor(in)
	if out.Name != "rangefinder" || out.Type != "distance" || out.VL != 9 {
		t.Fatalf("unexpected sensor map2rec output: %+v", out)
	}
	if len(out.FanoutIDs) != 2 || out.PreF != "identity" || out.PostF != "clip" {
		t.Fatalf("unexpected sensor io fields: %+v", out)
	}
}

func TestConvertSensorMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertSensor(map[string]any{"fanout_ids": "bad"})
	if len(out.FanoutIDs) != 0 {
		t.Fatalf("expected default empty fanout IDs, got %+v", out.FanoutIDs)
	}
}

func TestConvertActuatorMapsFields(t *testing.T) {
	in := map[string]any{
		"id":            "act-1",
		"name":          "thrust",
		"type":          "scalar",
		"cx_id":         "cx-2",
		"scape":         []any{"private", "cart-pole-lite"},
		"vl":            1,
		"fanin_ids":     []any{"n-3"},
		"generation":    4,
		"format":        map[string]any{"kind": "raw"},
		"parameters":    map[string]any{"limit": 1.0},
		"gt_parameters": []any{0.2, 0.4},
		"phys_rep":      "none",
		"vis_rep":       map[string]any{"shape": "bar"},
		"pre_f":         "scale",
		"post_f":        "identity",
	}
	out := ConvertActuator(in)
	if out.Name != "thrust" || out.Type != "scalar" || out.VL != 1 {
		t.Fatalf("unexpected actuator map2rec output: %+v", out)
	}
	if len(out.FaninIDs) != 1 || out.PreF != "scale" || out.PostF != "identity" {
		t.Fatalf("unexpected actuator io fields: %+v", out)
	}
}

func TestConvertActuatorMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertActuator(map[string]any{"fanin_ids": map[string]any{"x": 1}})
	if len(out.FaninIDs) != 0 {
		t.Fatalf("expected default empty fanin IDs, got %+v", out.FaninIDs)
	}
}

func TestConvertNeuronMapsFields(t *testing.T) {
	in := map[string]any{
		"id":                    "n-1",
		"generation":            2,
		"cx_id":                 "cx-1",
		"pre_processor":         "identity",
		"signal_integrator":     "sum",
		"af":                    "tanh",
		"post_processor":        "identity",
		"pf":                    []any{"hebbian", 0.1},
		"aggr_f":                "dot_product",
		"input_idps":            []any{"a", "b"},
		"input_idps_modulation": []any{"m"},
		"output_ids":            []any{"o1"},
		"ro_ids":                []any{"r1"},
	}
	out := ConvertNeuron(in)
	if out.ActivationFunction != "tanh" || out.SignalIntegrator != "sum" {
		t.Fatalf("unexpected neuron conversion: %+v", out)
	}
	if len(out.InputIDPs) != 2 || len(out.RecurrentOutputIDs) != 1 {
		t.Fatalf("unexpected neuron io conversion: %+v", out)
	}
}

func TestConvertNeuronMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertNeuron(map[string]any{"output_ids": "bad"})
	if len(out.OutputIDs) != 0 {
		t.Fatalf("expected default output IDs, got %+v", out.OutputIDs)
	}
}

func TestConvertAgentMapsFields(t *testing.T) {
	in := map[string]any{
		"id":                          "a-1",
		"encoding_type":               "neural",
		"generation":                  4,
		"population_id":               "pop-1",
		"specie_id":                   "sp-1",
		"cx_id":                       "cx-1",
		"fingerprint":                 "fp-1",
		"constraint":                  map[string]any{"morphology": "xor"},
		"evo_hist":                    []any{"m1"},
		"fitness":                     0.9,
		"innovation_factor":           []any{0, 1},
		"pattern":                     []any{"p1"},
		"tuning_selection_f":          "dynamic_random",
		"annealing_parameter":         0.5,
		"tuning_duration_f":           []any{"const", 10},
		"perturbation_range":          1.0,
		"mutation_operators":          []any{[]any{"add_neuron", 10}},
		"tot_topological_mutations_f": []any{"ncount_exponential", 0.5},
		"heredity_type":               "darwinian",
		"substrate_id":                "sub-1",
		"offspring_ids":               []any{"a-2"},
		"parent_ids":                  []any{"a-0"},
		"champion_flag":               []any{true},
		"evolvability":                0.2,
		"brittleness":                 0.1,
		"robustness":                  0.3,
		"evolutionary_capacitance":    0.4,
		"behavioral_trace":            map[string]any{"steps": 100},
		"fs":                          1.5,
		"main_fitness":                0.8,
	}
	out := ConvertAgent(in)
	if out.EncodingType != "neural" || out.Generation != 4 || out.TuningSelectionF != "dynamic_random" {
		t.Fatalf("unexpected agent conversion: %+v", out)
	}
	if len(out.OffspringIDs) != 1 || len(out.ParentIDs) != 1 || out.MainFitness != 0.8 {
		t.Fatalf("unexpected agent lineage conversion: %+v", out)
	}
}

func TestConvertAgentMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertAgent(map[string]any{"offspring_ids": "bad", "fs": "bad"})
	if len(out.OffspringIDs) != 0 {
		t.Fatalf("expected default offspring IDs, got %+v", out.OffspringIDs)
	}
	if out.FS != 1 {
		t.Fatalf("expected default fs=1, got %f", out.FS)
	}
}

func TestConvertCortexMapsFields(t *testing.T) {
	in := map[string]any{
		"id":           "cx-1",
		"agent_id":     "a-1",
		"neuron_ids":   []any{"n1", "n2"},
		"sensor_ids":   []any{"s1"},
		"actuator_ids": []any{"ac1"},
	}
	out := ConvertCortex(in)
	if out.ID != "cx-1" || out.AgentID != "a-1" {
		t.Fatalf("unexpected cortex conversion: %+v", out)
	}
	if len(out.NeuronIDs) != 2 || len(out.SensorIDs) != 1 || len(out.ActuatorIDs) != 1 {
		t.Fatalf("unexpected cortex lists: %+v", out)
	}
}

func TestConvertSubstrateMapsFields(t *testing.T) {
	in := map[string]any{
		"id":         "sub-1",
		"agent_id":   "a-1",
		"densities":  map[string]any{"i": 4, "h": 8, "o": 2},
		"linkform":   "l2l_feedforward",
		"plasticity": "hebbian",
		"cpp_ids":    []any{"cpp-1"},
		"cep_ids":    []any{"cep-1", "cep-2"},
	}
	out := ConvertSubstrate(in)
	if out.ID != "sub-1" || out.AgentID != "a-1" {
		t.Fatalf("unexpected substrate conversion identity: %+v", out)
	}
	if out.Linkform != "l2l_feedforward" || out.Plasticity != "hebbian" {
		t.Fatalf("unexpected substrate conversion behavior fields: %+v", out)
	}
	if len(out.CPPIDs) != 1 || len(out.CEPIDs) != 2 {
		t.Fatalf("unexpected substrate conversion component IDs: %+v", out)
	}
}

func TestConvertSubstrateMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertSubstrate(map[string]any{"cpp_ids": "bad", "cep_ids": map[string]any{"id": "cep-1"}})
	if out.Plasticity != "none" {
		t.Fatalf("expected default plasticity none, got %#v", out.Plasticity)
	}
	if len(out.CPPIDs) != 0 || len(out.CEPIDs) != 0 {
		t.Fatalf("expected default substrate component IDs, got cpp=%+v cep=%+v", out.CPPIDs, out.CEPIDs)
	}
}

func TestConvertPolisMapsFields(t *testing.T) {
	in := map[string]any{
		"id":             "polis-1",
		"scape_ids":      []any{"xor", "flatland"},
		"population_ids": []any{"pop-1"},
		"specie_ids":     []any{"sp-1"},
		"dx_ids":         []any{"dx-1"},
		"parameters":     []any{map[string]any{"key": "value"}},
	}
	out := ConvertPolis(in)
	if out.ID != "polis-1" {
		t.Fatalf("unexpected polis conversion identity: %+v", out)
	}
	if len(out.ScapeIDs) != 2 || len(out.PopulationIDs) != 1 || len(out.Parameters) != 1 {
		t.Fatalf("unexpected polis conversion collections: %+v", out)
	}
}

func TestConvertPolisMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertPolis(map[string]any{"scape_ids": "bad", "parameters": "bad"})
	if len(out.ScapeIDs) != 0 || len(out.Parameters) != 0 {
		t.Fatalf("expected default polis collections, got scapes=%+v params=%+v", out.ScapeIDs, out.Parameters)
	}
}

func TestConvertScapeMapsFields(t *testing.T) {
	in := map[string]any{
		"id":             "scape-1",
		"type":           "flatland",
		"physics":        map[string]any{"g": 9.8},
		"metabolics":     map[string]any{"decay": 0.1},
		"sector2avatars": map[string]any{"s1": []any{"a1"}},
		"avatars":        []any{"a1", "a2"},
		"plants":         []any{"p1"},
		"walls":          []any{"w1"},
		"pillars":        []any{"pi1"},
		"laws":           []any{"l1"},
		"anomolies":      []any{"n1"},
		"artifacts":      []any{"ar1"},
		"objects":        []any{"o1"},
		"elements":       []any{"e1"},
		"atoms":          []any{"at1"},
		"scheduler":      42,
	}
	out := ConvertScape(in)
	if out.ID != "scape-1" || out.Type != "flatland" || out.Scheduler != 42 {
		t.Fatalf("unexpected scape conversion identity/scheduler: %+v", out)
	}
	if len(out.Avatars) != 2 || len(out.Objects) != 1 || len(out.Atoms) != 1 {
		t.Fatalf("unexpected scape conversion collections: %+v", out)
	}
}

func TestConvertScapeMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertScape(map[string]any{"avatars": "bad", "scheduler": "bad"})
	if len(out.Avatars) != 0 {
		t.Fatalf("expected default avatars collection, got %+v", out.Avatars)
	}
	if out.Scheduler != 0 {
		t.Fatalf("expected default scheduler=0, got %d", out.Scheduler)
	}
}

func TestConvertSectorMapsFields(t *testing.T) {
	in := map[string]any{
		"id":             "sector-1",
		"type":           "arena",
		"scape_pid":      "scape-1",
		"sector_size":    []any{100, 100},
		"physics":        map[string]any{"drag": 0.2},
		"metabolics":     map[string]any{"burn": 0.05},
		"sector2avatars": map[string]any{"zone-a": []any{"a1"}},
		"avatars":        []any{"a1"},
		"plants":         []any{"p1"},
		"walls":          []any{"w1"},
		"pillars":        []any{"pi1"},
		"laws":           []any{"l1"},
		"anomolies":      []any{"n1"},
		"artifacts":      []any{"ar1"},
		"objects":        []any{"o1"},
		"elements":       []any{"e1"},
		"atoms":          []any{"at1"},
	}
	out := ConvertSector(in)
	if out.ID != "sector-1" || out.Type != "arena" || out.ScapePID != "scape-1" {
		t.Fatalf("unexpected sector identity mapping: %+v", out)
	}
	if len(out.Avatars) != 1 || len(out.Objects) != 1 || len(out.Atoms) != 1 {
		t.Fatalf("unexpected sector collection mapping: %+v", out)
	}
}

func TestConvertSectorMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertSector(map[string]any{"avatars": "bad", "objects": "bad"})
	if len(out.Avatars) != 0 || len(out.Objects) != 0 {
		t.Fatalf("expected default sector collections, got avatars=%+v objects=%+v", out.Avatars, out.Objects)
	}
}

func TestConvertAvatarMapsFields(t *testing.T) {
	in := map[string]any{
		"id":         "avatar-1",
		"sector":     "sector-1",
		"morphology": "flatland",
		"type":       "agent",
		"specie":     "sp-1",
		"energy":     10.5,
		"health":     8.5,
		"food":       2.0,
		"age":        4,
		"kills":      1,
		"loc":        []any{1, 2},
		"direction":  []any{0, 1},
		"r":          0.5,
		"mass":       7.2,
		"objects":    []any{"obj-1"},
		"vis":        []any{"ray-1"},
		"state":      "active",
		"stats":      map[string]any{"fitness": 0.7},
		"actuators":  []any{"act-1"},
		"sensors":    []any{"sens-1"},
		"sound":      "quiet",
		"gestalt":    map[string]any{"mode": "search"},
		"spear":      "none",
	}
	out := ConvertAvatar(in)
	if out.ID != "avatar-1" || out.Sector != "sector-1" || out.Morphology != "flatland" {
		t.Fatalf("unexpected avatar identity mapping: %+v", out)
	}
	if out.Energy != 10.5 || out.Health != 8.5 || out.Age != 4 || out.Kills != 1 {
		t.Fatalf("unexpected avatar numeric mapping: %+v", out)
	}
	if len(out.Vis) != 1 || out.State != "active" || out.Sound != "quiet" {
		t.Fatalf("unexpected avatar misc mapping: %+v", out)
	}
}

func TestConvertAvatarMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertAvatar(map[string]any{"energy": "bad", "vis": "bad"})
	if out.Energy != 0 {
		t.Fatalf("expected default energy=0, got %f", out.Energy)
	}
	if len(out.Vis) != 0 {
		t.Fatalf("expected default vis collection, got %+v", out.Vis)
	}
}

func TestConvertSpecieMapsFields(t *testing.T) {
	in := map[string]any{
		"id":                    "sp-1",
		"population_id":         "pop-1",
		"fingerprint":           "fp-1",
		"constraint":            map[string]any{"morphology": "xor"},
		"all_agent_ids":         []any{"a1", "a2"},
		"agent_ids":             []any{"a2"},
		"dead_pool":             []any{"a0"},
		"champion_ids":          []any{"a2"},
		"fitness":               0.8,
		"innovation_factor":     []any{1, 2},
		"stats":                 []any{"st"},
		"seed_agent_ids":        []any{"a1"},
		"hof_distinguishers":    []any{"tot_n"},
		"specie_distinguishers": []any{"pattern"},
		"hall_of_fame":          []any{"champ"},
	}
	out := ConvertSpecie(in)
	if out.ID != "sp-1" || out.PopulationID != "pop-1" {
		t.Fatalf("unexpected specie conversion: %+v", out)
	}
	if len(out.AllAgentIDs) != 2 || len(out.HallOfFame) != 1 {
		t.Fatalf("unexpected specie collections: %+v", out)
	}
}

func TestConvertPopulationMapsFields(t *testing.T) {
	in := map[string]any{
		"id":                      "pop-1",
		"polis_id":                "polis-1",
		"specie_ids":              []any{"sp1"},
		"morphologies":            []any{"xor"},
		"innovation_factor":       []any{1, 2},
		"evo_alg_f":               "generational",
		"fitness_postprocessor_f": "size_proportional",
		"selection_f":             "hof_competition",
		"trace":                   map[string]any{"tot_evaluations": 10},
		"seed_agent_ids":          []any{"a1"},
		"seed_specie_ids":         []any{"sp1"},
	}
	out := ConvertPopulation(in)
	if out.ID != "pop-1" || out.PolisID != "polis-1" {
		t.Fatalf("unexpected population conversion: %+v", out)
	}
	if out.EvoAlgF != "generational" || out.SelectionF != "hof_competition" {
		t.Fatalf("unexpected population policies: %+v", out)
	}
	if len(out.SeedAgentIDs) != 1 || len(out.SeedSpecieIDs) != 1 {
		t.Fatalf("unexpected population seeds: %+v", out)
	}
}

func TestConvertTraceMapsFields(t *testing.T) {
	in := map[string]any{
		"stats":           []any{"s1"},
		"tot_evaluations": 42,
		"step_size":       250,
	}
	out := ConvertTrace(in)
	if len(out.Stats) != 1 || out.TotalEvaluations != 42 || out.StepSize != 250 {
		t.Fatalf("unexpected trace conversion: %+v", out)
	}
}

func TestConvertTraceMalformedKnownFieldKeepsDefault(t *testing.T) {
	out := ConvertTrace(map[string]any{"stats": "bad", "step_size": "bad"})
	if len(out.Stats) != 0 {
		t.Fatalf("expected default trace stats, got %+v", out.Stats)
	}
	if out.StepSize != 500 {
		t.Fatalf("expected default step size 500, got %d", out.StepSize)
	}
}

func TestConvertStatMapsFields(t *testing.T) {
	in := map[string]any{
		"morphology":         "xor",
		"specie_id":          "sp-1",
		"avg_neurons":        3.5,
		"std_neurons":        1.1,
		"avg_fitness":        0.4,
		"std_fitness":        0.05,
		"max_fitness":        0.8,
		"min_fitness":        0.1,
		"validation_fitness": 0.75,
		"test_fitness":       0.70,
		"avg_diversity":      0.2,
		"evaluations":        123,
		"time_stamp":         "2026-02-12T00:00:00Z",
	}
	out := ConvertStat(in)
	if out.AvgFitness != 0.4 || out.Evaluations != 123 || out.ValidationFitness != 0.75 {
		t.Fatalf("unexpected stat conversion: %+v", out)
	}
}

func TestConvertTopologySummaryMapsFields(t *testing.T) {
	in := map[string]any{
		"type":            "feedforward",
		"tot_neurons":     12,
		"tot_n_ils":       3,
		"tot_n_ols":       2,
		"tot_n_ros":       1,
		"af_distribution": map[string]any{"tanh": 8},
	}
	out := ConvertTopologySummary(in)
	if out.TotalNeurons != 12 || out.TotalNILs != 3 || out.TotalNROs != 1 {
		t.Fatalf("unexpected topology summary conversion: %+v", out)
	}
}

func TestConvertSignatureMapsFields(t *testing.T) {
	in := map[string]any{
		"generalized_Pattern":   []any{"p1"},
		"generalized_EvoHist":   []any{"h1"},
		"generalized_Sensors":   []any{"s1"},
		"generalized_Actuators": []any{"a1"},
		"topology_summary":      map[string]any{"tot_neurons": 3},
	}
	out := ConvertSignature(in)
	if out.GeneralizedPattern == nil || out.TopologySummary == nil {
		t.Fatalf("unexpected signature conversion: %+v", out)
	}
}

func TestConvertChampionMapsFields(t *testing.T) {
	in := map[string]any{
		"hof_fingerprint":        "hof-1",
		"id":                     "c-1",
		"fitness":                0.9,
		"validation_fitness":     0.85,
		"test_fitness":           0.83,
		"main_fitness":           0.8,
		"tot_n":                  12,
		"evolvability":           0.3,
		"robustness":             0.2,
		"brittleness":            0.1,
		"generation":             7,
		"behavioral_differences": []any{"b1"},
		"fs":                     1.2,
	}
	out := ConvertChampion(in)
	if out.Fitness != 0.9 || out.TotalNeurons != 12 || out.Generation != 7 || out.FS != 1.2 {
		t.Fatalf("unexpected champion conversion: %+v", out)
	}
}
