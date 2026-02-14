package evo

import (
	"context"
	"errors"
	"math/rand"
	"strings"
	"testing"
	"time"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/scape"
)

type namedNoopMutation struct {
	name string
}

func (o namedNoopMutation) Name() string { return o.name }

func (o namedNoopMutation) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	return genome, nil
}

type failingMutation struct {
	name string
}

func (o failingMutation) Name() string { return o.name }

func (o failingMutation) Apply(_ context.Context, _ model.Genome) (model.Genome, error) {
	return model.Genome{}, errors.New("forced failure")
}

type captureSpeciesSelector struct {
	gotSpeciesByGenomeID map[string]string
}

func (captureSpeciesSelector) Name() string { return "capture_species_selector" }

func (captureSpeciesSelector) PickParent(_ *rand.Rand, _ []ScoredGenome, _ int) (model.Genome, error) {
	return model.Genome{}, errors.New("unexpected non-generation selector path")
}

func (captureSpeciesSelector) PickParentForGeneration(_ *rand.Rand, _ []ScoredGenome, _ int, _ int) (model.Genome, error) {
	return model.Genome{}, errors.New("unexpected generation selector path")
}

func (s *captureSpeciesSelector) PickParentForGenerationWithSpecies(_ *rand.Rand, ranked []ScoredGenome, eliteCount, _ int, speciesByGenomeID map[string]string) (model.Genome, error) {
	if eliteCount <= 0 || eliteCount > len(ranked) {
		return model.Genome{}, errors.New("invalid elite count")
	}
	s.gotSpeciesByGenomeID = make(map[string]string, len(speciesByGenomeID))
	for k, v := range speciesByGenomeID {
		s.gotSpeciesByGenomeID[k] = v
	}
	return ranked[0].Genome, nil
}

type oneDimScape struct{}

func (oneDimScape) Name() string { return "one-dim" }

func (oneDimScape) Evaluate(ctx context.Context, a scape.Agent) (scape.Fitness, scape.Trace, error) {
	runner, ok := a.(scape.StepAgent)
	if !ok {
		return 0, nil, context.Canceled
	}

	out, err := runner.RunStep(ctx, []float64{1.0})
	if err != nil {
		return 0, nil, err
	}
	if len(out) != 1 {
		return 0, nil, context.Canceled
	}

	target := 1.0
	delta := out[0] - target
	mse := delta * delta
	fitness := 1.0 - mse
	return scape.Fitness(fitness), scape.Trace{"mse": mse, "prediction": out[0]}, nil
}

func TestPopulationMonitorImprovesFitness(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
		newLinearGenome("g4", -0.2),
		newLinearGenome("g5", 0.0),
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        PerturbWeightAt{Index: 0, Delta: 0.2},
		PopulationSize:  len(initial),
		EliteCount:      2,
		Generations:     6,
		Workers:         3,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	if len(result.BestByGeneration) != 6 {
		t.Fatalf("expected 6 generations, got %d", len(result.BestByGeneration))
	}
	if len(result.GenerationDiagnostics) != 6 {
		t.Fatalf("expected 6 generation diagnostics, got %d", len(result.GenerationDiagnostics))
	}
	if len(result.SpeciesHistory) != 6 {
		t.Fatalf("expected 6 species history records, got %d", len(result.SpeciesHistory))
	}
	if result.GenerationDiagnostics[0].Generation != 1 {
		t.Fatalf("expected first diagnostics generation=1, got %d", result.GenerationDiagnostics[0].Generation)
	}
	if len(result.SpeciesHistory[0].Species) == 0 {
		t.Fatal("expected first species history generation to include species entries")
	}
	if result.GenerationDiagnostics[0].SpeciationThreshold <= 0 {
		t.Fatalf("expected speciation threshold diagnostics, got %f", result.GenerationDiagnostics[0].SpeciationThreshold)
	}
	if result.GenerationDiagnostics[0].TargetSpeciesCount <= 0 {
		t.Fatalf("expected target species diagnostics, got %d", result.GenerationDiagnostics[0].TargetSpeciesCount)
	}
	if len(result.FinalPopulation) != len(initial) {
		t.Fatalf("final population size mismatch: got=%d want=%d", len(result.FinalPopulation), len(initial))
	}
	if len(result.Lineage) == 0 {
		t.Fatal("expected lineage records")
	}
	for _, rec := range result.Lineage {
		if rec.Fingerprint == "" {
			t.Fatalf("expected lineage fingerprint for genome %s", rec.GenomeID)
		}
	}

	first := result.BestByGeneration[0]
	last := result.BestByGeneration[len(result.BestByGeneration)-1]
	if last <= first {
		t.Fatalf("expected improvement across generations: first=%f last=%f", first, last)
	}
}

func TestPopulationMonitorStopsAtFitnessGoal(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", 1.0),
		newLinearGenome("g1", 0.8),
		newLinearGenome("g2", 0.6),
		newLinearGenome("g3", 0.4),
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        namedNoopMutation{name: "noop"},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     6,
		FitnessGoal:     0.99,
		Workers:         2,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(result.BestByGeneration) != 1 {
		t.Fatalf("expected early stop after first generation, got %d generations", len(result.BestByGeneration))
	}
}

func TestPopulationMonitorStopsAtEvaluationLimit(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:            oneDimScape{},
		Mutation:         namedNoopMutation{name: "noop"},
		PopulationSize:   len(initial),
		EliteCount:       1,
		Generations:      6,
		EvaluationsLimit: len(initial),
		Workers:          2,
		Seed:             1,
		InputNeuronIDs:   []string{"i"},
		OutputNeuronIDs:  []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(result.BestByGeneration) != 1 {
		t.Fatalf("expected stop after first generation due to evaluation limit, got %d generations", len(result.BestByGeneration))
	}
}

func TestPopulationMonitorMixedMutationPolicyLineage(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
		newLinearGenome("g4", -0.2),
		newLinearGenome("g5", 0.0),
		newLinearGenome("g6", 0.2),
		newLinearGenome("g7", 0.4),
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		MutationPolicy:  []WeightedMutation{{Operator: namedNoopMutation{name: "op_a"}, Weight: 1}, {Operator: namedNoopMutation{name: "op_b"}, Weight: 1}},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     2,
		Workers:         2,
		Seed:            2,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	seenA := false
	seenB := false
	for _, record := range result.Lineage {
		if record.Operation == "op_a" {
			seenA = true
		}
		if record.Operation == "op_b" {
			seenB = true
		}
	}
	if !seenA || !seenB {
		t.Fatalf("expected both mutation operators in lineage, seenA=%t seenB=%t", seenA, seenB)
	}
}

func TestPopulationMonitorMutationPolicyValidation(t *testing.T) {
	_, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		MutationPolicy:  []WeightedMutation{{Operator: nil, Weight: 1}},
		PopulationSize:  4,
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err == nil {
		t.Fatal("expected mutation policy operator validation error")
	}

	_, err = NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		MutationPolicy:  []WeightedMutation{{Operator: namedNoopMutation{name: "x"}, Weight: -1}},
		PopulationSize:  4,
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err == nil {
		t.Fatal("expected mutation policy weight validation error")
	}

	_, err = NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		MutationPolicy:  []WeightedMutation{{Operator: namedNoopMutation{name: "x"}, Weight: 0}},
		PopulationSize:  4,
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err == nil {
		t.Fatal("expected at least one positive policy weight")
	}

	_, err = NewPopulationMonitor(MonitorConfig{
		Scape:            oneDimScape{},
		Mutation:         namedNoopMutation{name: "noop"},
		PopulationSize:   4,
		EliteCount:       1,
		Generations:      1,
		EvaluationsLimit: -1,
		Workers:          1,
		Seed:             1,
		InputNeuronIDs:   []string{"i"},
		OutputNeuronIDs:  []string{"o"},
	})
	if err == nil {
		t.Fatal("expected evaluations limit validation error")
	}

	_, err = NewPopulationMonitor(MonitorConfig{
		Scape:              oneDimScape{},
		Mutation:           namedNoopMutation{name: "noop"},
		PopulationSize:     4,
		EliteCount:         1,
		SurvivalPercentage: 1.1,
		Generations:        1,
		Workers:            1,
		Seed:               1,
		InputNeuronIDs:     []string{"i"},
		OutputNeuronIDs:    []string{"o"},
	})
	if err == nil {
		t.Fatal("expected survival percentage validation error")
	}

	_, err = NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        namedNoopMutation{name: "noop"},
		PopulationSize:  4,
		EliteCount:      1,
		SpecieSizeLimit: -1,
		Generations:     1,
		Workers:         1,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err == nil {
		t.Fatal("expected specie size limit validation error")
	}
}

func TestLimitSpeciesParentPool(t *testing.T) {
	ranked := []ScoredGenome{
		{Genome: model.Genome{ID: "a1"}, Fitness: 10},
		{Genome: model.Genome{ID: "a2"}, Fitness: 9},
		{Genome: model.Genome{ID: "b1"}, Fitness: 8},
		{Genome: model.Genome{ID: "a3"}, Fitness: 7},
		{Genome: model.Genome{ID: "b2"}, Fitness: 6},
	}
	speciesByGenomeID := map[string]string{
		"a1": "A",
		"a2": "A",
		"a3": "A",
		"b1": "B",
		"b2": "B",
	}

	limited := limitSpeciesParentPool(ranked, speciesByGenomeID, 1)
	if len(limited) != 2 {
		t.Fatalf("expected 2 genomes after per-species limit, got %d", len(limited))
	}
	if limited[0].Genome.ID != "a1" || limited[1].Genome.ID != "b1" {
		t.Fatalf("expected top genome per species preserved by rank order, got %+v", limited)
	}

	unlimited := limitSpeciesParentPool(ranked, speciesByGenomeID, 0)
	if len(unlimited) != len(ranked) {
		t.Fatalf("expected unlimited parent pool size %d, got %d", len(ranked), len(unlimited))
	}
}

func TestPopulationMonitorPauseContinueControl(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
	}
	control := make(chan MonitorCommand, 4)
	control <- CommandPause

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        namedNoopMutation{name: "noop"},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     2,
		Workers:         2,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
		Control:         control,
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	done := make(chan RunResult, 1)
	errs := make(chan error, 1)
	go func() {
		result, runErr := monitor.Run(context.Background(), initial)
		if runErr != nil {
			errs <- runErr
			return
		}
		done <- result
	}()

	select {
	case <-done:
		t.Fatal("expected run to pause before spawning next generation")
	case runErr := <-errs:
		t.Fatalf("run failed while paused: %v", runErr)
	case <-time.After(30 * time.Millisecond):
	}

	control <- CommandContinue
	select {
	case runErr := <-errs:
		t.Fatalf("run failed after continue: %v", runErr)
	case result := <-done:
		if len(result.BestByGeneration) != 2 {
			t.Fatalf("expected full run after continue, got %d generations", len(result.BestByGeneration))
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for run completion after continue")
	}
}

func TestPopulationMonitorStopControl(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
	}
	control := make(chan MonitorCommand, 1)
	control <- CommandStop

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        namedNoopMutation{name: "noop"},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     4,
		Workers:         2,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
		Control:         control,
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(result.BestByGeneration) != 0 {
		t.Fatalf("expected immediate stop before evaluation, got %d generations", len(result.BestByGeneration))
	}
}

func TestPopulationMonitorDerivesEliteCountFromSurvivalPercentage(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
		newLinearGenome("g4", -0.2),
		newLinearGenome("g5", 0.0),
	}
	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:              oneDimScape{},
		Mutation:           namedNoopMutation{name: "noop"},
		PopulationSize:     len(initial),
		EliteCount:         0,
		SurvivalPercentage: 0.5,
		Generations:        1,
		Workers:            2,
		Seed:               1,
		InputNeuronIDs:     []string{"i"},
		OutputNeuronIDs:    []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}
	if monitor.cfg.EliteCount != 3 {
		t.Fatalf("expected derived elite count 3, got %d", monitor.cfg.EliteCount)
	}
	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	eliteClones := 0
	for _, rec := range result.Lineage {
		if rec.Operation == "elite_clone" && rec.Generation == 1 {
			eliteClones++
		}
	}
	if eliteClones != 3 {
		t.Fatalf("expected 3 elite clones from survival percentage, got %d", eliteClones)
	}
}

func TestPopulationMonitorMutationPolicyFallback(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        PerturbWeightAt{Index: 0, Delta: 0.1},
		MutationPolicy:  []WeightedMutation{{Operator: failingMutation{name: "fail_op"}, Weight: 1}},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     2,
		Workers:         2,
		Seed:            3,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	seenFallback := false
	for _, record := range result.Lineage {
		if record.Operation == "perturb_weight_at(fallback)" {
			seenFallback = true
			break
		}
	}
	if !seenFallback {
		t.Fatal("expected fallback mutation lineage records")
	}
}

func TestPopulationMonitorMutationPolicyCustomWeights(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
		newLinearGenome("g4", -0.2),
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:    oneDimScape{},
		Mutation: PerturbWeightAt{Index: 0, Delta: 0.1},
		MutationPolicy: []WeightedMutation{
			{Operator: namedNoopMutation{name: "op_a"}, Weight: 1},
			{Operator: namedNoopMutation{name: "op_b"}, Weight: 0},
		},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     2,
		Workers:         2,
		Seed:            3,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	for _, record := range result.Lineage {
		if record.Operation == "op_b" {
			t.Fatalf("unexpected op_b in lineage with zero weight: %+v", result.Lineage)
		}
	}
}

func TestPopulationMonitorUsesRegisteredIOForRegressionMimic(t *testing.T) {
	initial := []model.Genome{
		{
			VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1},
			ID:              "reg-0",
			SensorIDs:       []string{protoio.ScalarInputSensorName},
			ActuatorIDs:     []string{protoio.ScalarOutputActuatorName},
			Neurons: []model.Neuron{
				{ID: "i", Activation: "identity", Bias: 0},
				{ID: "o", Activation: "identity", Bias: 0},
			},
			Synapses: []model.Synapse{
				{ID: "s", From: "i", To: "o", Weight: 1, Enabled: true, Recurrent: false},
			},
		},
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           scape.RegressionMimicScape{},
		Mutation:        PerturbWeightAt{Index: 0, Delta: 0},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	if len(result.BestByGeneration) != 1 {
		t.Fatalf("expected one generation result, got=%d", len(result.BestByGeneration))
	}
	if result.BestByGeneration[0] < 0.999999 {
		t.Fatalf("expected near-perfect fitness with identity network, got=%f", result.BestByGeneration[0])
	}
}

func TestPopulationMonitorBuildsSubstrateFromGenomeConfig(t *testing.T) {
	initial := []model.Genome{
		{
			VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1},
			ID:              "sub-0",
			Neurons: []model.Neuron{
				{ID: "i", Activation: "identity", Bias: 0},
				{ID: "o", Activation: "identity", Bias: 0},
			},
			Synapses: []model.Synapse{
				{ID: "s", From: "i", To: "o", Weight: 1, Enabled: true, Recurrent: false},
			},
			Substrate: &model.SubstrateConfig{
				CPPName:     "set_weight",
				CEPName:     "delta_weight",
				WeightCount: 1,
				Parameters:  map[string]float64{"scale": 1.0},
			},
		},
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        PerturbWeightAt{Index: 0, Delta: 0},
		PopulationSize:  1,
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(result.BestByGeneration) != 1 {
		t.Fatalf("expected one generation result, got=%d", len(result.BestByGeneration))
	}
}

func TestPopulationMonitorSizeProportionalPostprocessorChangesWinner(t *testing.T) {
	complexBestRaw := newComplexLinearGenome("complex", 1.0)
	simpleSecondRaw := newLinearGenome("simple", 0.95)
	initial := []model.Genome{complexBestRaw, simpleSecondRaw}

	baseline, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        PerturbWeightAt{Index: 0, Delta: 0},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            10,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("baseline monitor: %v", err)
	}
	baselineResult, err := baseline.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("baseline run: %v", err)
	}
	if baselineResult.FinalPopulation[0].Genome.ID != "complex" {
		t.Fatalf("expected complex to win raw fitness, got: %s", baselineResult.FinalPopulation[0].Genome.ID)
	}

	withSizePenalty, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        PerturbWeightAt{Index: 0, Delta: 0},
		Postprocessor:   SizeProportionalPostprocessor{},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            10,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("size monitor: %v", err)
	}
	sizeResult, err := withSizePenalty.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("size run: %v", err)
	}
	if sizeResult.FinalPopulation[0].Genome.ID != "simple" {
		t.Fatalf("expected simple to win size-proportional fitness, got: %s", sizeResult.FinalPopulation[0].Genome.ID)
	}
}

func TestPopulationMonitorTournamentSelectorCanPickNonEliteParent(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", 1.0), // top fitness
		newLinearGenome("g1", 0.8),
		newLinearGenome("g2", 0.6),
		newLinearGenome("g3", 0.4),
		newLinearGenome("g4", 0.2),
		newLinearGenome("g5", 0.0),
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        namedNoopMutation{name: "noop"},
		Selector:        TournamentSelector{PoolSize: 4, TournamentSize: 2},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     2,
		Workers:         2,
		Seed:            7,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	seenNonTopParent := false
	for _, record := range result.Lineage {
		if record.Operation != "noop" {
			continue
		}
		if record.ParentID != "g0" {
			seenNonTopParent = true
			break
		}
	}
	if !seenNonTopParent {
		t.Fatalf("expected tournament selector to pick non-top parent at least once: %+v", result.Lineage)
	}
}

func TestPopulationMonitorPassesAdaptiveSpeciesAssignmentsToSelector(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", 1.0),
		newLinearGenome("g1", 0.8),
		newComplexLinearGenome("g2", 0.6),
		newComplexLinearGenome("g3", 0.4),
	}
	selector := &captureSpeciesSelector{}
	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        namedNoopMutation{name: "noop"},
		Selector:        selector,
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            77,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	_, err = monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(selector.gotSpeciesByGenomeID) != len(initial) {
		t.Fatalf("expected species assignments for all genomes, got=%d want=%d", len(selector.gotSpeciesByGenomeID), len(initial))
	}
}

func TestBuildSpeciesOffspringPlanAllocatesBySharedFitness(t *testing.T) {
	ranked := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.90},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.80},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.20},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.10},
	}
	speciesByGenomeID := map[string]string{
		"a0": "sp-a",
		"a1": "sp-a",
		"b0": "sp-b",
		"b1": "sp-b",
	}

	plan := buildSpeciesOffspringPlan(ranked, speciesByGenomeID, 6)
	got := map[string]int{}
	total := 0
	for _, item := range plan {
		got[item.SpeciesKey] = item.Count
		total += item.Count
	}
	if total != 6 {
		t.Fatalf("expected total offspring=6, got=%d", total)
	}
	if got["sp-a"] <= got["sp-b"] {
		t.Fatalf("expected fitter species to receive more offspring, got sp-a=%d sp-b=%d", got["sp-a"], got["sp-b"])
	}
}

func TestNextGenerationRespectsSpeciesOffspringPlan(t *testing.T) {
	ranked := []ScoredGenome{
		{Genome: newLinearGenome("a0", 1), Fitness: 0.95},
		{Genome: newLinearGenome("a1", 1), Fitness: 0.85},
		{Genome: newComplexLinearGenome("b0", 1), Fitness: 0.20},
		{Genome: newComplexLinearGenome("b1", 1), Fitness: 0.10},
	}
	speciesByGenomeID := map[string]string{
		"a0": "sp-a",
		"a1": "sp-a",
		"b0": "sp-b",
		"b1": "sp-b",
	}
	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        namedNoopMutation{name: "noop"},
		PopulationSize:  4,
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            9,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	next, _, err := monitor.nextGeneration(context.Background(), ranked, speciesByGenomeID, 0)
	if err != nil {
		t.Fatalf("next generation: %v", err)
	}
	if len(next) != 4 {
		t.Fatalf("expected population size 4, got=%d", len(next))
	}

	fromA := 0
	fromB := 0
	for _, genome := range next {
		if strings.HasPrefix(genome.ID, "a") {
			fromA++
		}
		if strings.HasPrefix(genome.ID, "b") {
			fromB++
		}
	}
	if fromA <= fromB {
		t.Fatalf("expected more offspring from fitter species, got fromA=%d fromB=%d", fromA, fromB)
	}
}

func TestPopulationMonitorSkipsNoSynapseMutationError(t *testing.T) {
	initial := []model.Genome{
		{
			VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1},
			ID:              "g0",
			Neurons: []model.Neuron{
				{ID: "i", Activation: "identity"},
				{ID: "o", Activation: "identity"},
			},
		},
	}
	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        &PerturbRandomWeight{Rand: rand.New(rand.NewSource(1)), MaxDelta: 1.0},
		PopulationSize:  1,
		EliteCount:      1,
		Generations:     1,
		Workers:         1,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}
	if _, err := monitor.Run(context.Background(), initial); err != nil {
		t.Fatalf("run should not fail on ErrNoSynapses, got: %v", err)
	}
}

func TestPopulationMonitorGatesIncompatibleContextualMutations(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
	}
	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:    oneDimScape{},
		Mutation: namedNoopMutation{name: "noop"},
		MutationPolicy: []WeightedMutation{
			{Operator: &PerturbPlasticityRate{Rand: rand.New(rand.NewSource(1)), MaxDelta: 0.2}, Weight: 10},
			{Operator: &PerturbSubstrateParameter{Rand: rand.New(rand.NewSource(2)), MaxDelta: 0.2}, Weight: 10},
			{Operator: namedNoopMutation{name: "noop"}, Weight: 1},
		},
		PopulationSize:  len(initial),
		EliteCount:      1,
		Generations:     2,
		Workers:         1,
		Seed:            99,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}
	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	for _, rec := range result.Lineage {
		if strings.Contains(rec.Operation, "perturb_plasticity_rate") || strings.Contains(rec.Operation, "perturb_substrate_parameter") {
			t.Fatalf("expected incompatible contextual operators to be gated, got operation %q", rec.Operation)
		}
	}
}

func TestNoveltyPostprocessorIsNoopForReferenceParity(t *testing.T) {
	scored := []ScoredGenome{
		{Genome: newLinearGenome("a", 1), Fitness: 1},
		{Genome: newLinearGenome("b", 1), Fitness: 1},
		{Genome: newComplexLinearGenome("c", 1), Fitness: 1},
	}
	out := NoveltyProportionalPostprocessor{}.Process(scored)

	if len(out) != len(scored) {
		t.Fatalf("unexpected output length: got=%d want=%d", len(out), len(scored))
	}
	for i := range out {
		if out[i].Fitness != scored[i].Fitness {
			t.Fatalf("expected no-op novelty postprocessor at index %d: got=%f want=%f", i, out[i].Fitness, scored[i].Fitness)
		}
	}
}

func TestEliteSelectorValidation(t *testing.T) {
	_, err := (EliteSelector{}).PickParent(nil, []ScoredGenome{}, 1)
	if err == nil {
		t.Fatal("expected error when random source is nil")
	}

	_, err = (EliteSelector{}).PickParent(rand.New(rand.NewSource(1)), []ScoredGenome{}, 1)
	if err == nil {
		t.Fatal("expected error when elite count exceeds population")
	}
}

func newLinearGenome(id string, weight float64) model.Genome {
	return model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1},
		ID:              id,
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity", Bias: 0},
			{ID: "o", Activation: "identity", Bias: 0},
		},
		Synapses: []model.Synapse{
			{ID: "s", From: "i", To: "o", Weight: weight, Enabled: true, Recurrent: false},
		},
	}
}

func newComplexLinearGenome(id string, weight float64) model.Genome {
	g := newLinearGenome(id, weight)
	g.Neurons = append(g.Neurons,
		model.Neuron{ID: "h1", Activation: "identity", Bias: 0},
		model.Neuron{ID: "h2", Activation: "identity", Bias: 0},
		model.Neuron{ID: "h3", Activation: "identity", Bias: 0},
		model.Neuron{ID: "h4", Activation: "identity", Bias: 0},
	)
	g.Synapses = append(g.Synapses,
		model.Synapse{ID: "s1", From: "h1", To: "h2", Weight: 0.1, Enabled: true},
		model.Synapse{ID: "s2", From: "h2", To: "h3", Weight: 0.1, Enabled: true},
		model.Synapse{ID: "s3", From: "h3", To: "h4", Weight: 0.1, Enabled: true},
		model.Synapse{ID: "s4", From: "h4", To: "h1", Weight: 0.1, Enabled: true, Recurrent: true},
	)
	return g
}
