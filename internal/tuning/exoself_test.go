package tuning

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"testing"

	"protogonos/internal/model"
)

func TestExoselfImprovesFitness(t *testing.T) {
	genome := model.Genome{
		ID: "g",
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{{ID: "s", From: "i", To: "o", Weight: -2, Enabled: true}},
	}

	tuner := &Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 6, StepSize: 0.4}
	fitnessFn := func(_ context.Context, g model.Genome) (float64, error) {
		delta := g.Synapses[0].Weight - 1
		return 1 - delta*delta, nil
	}

	before, _ := fitnessFn(context.Background(), genome)
	tuned, err := tuner.Tune(context.Background(), genome, 40, fitnessFn)
	if err != nil {
		t.Fatalf("tune: %v", err)
	}
	after, _ := fitnessFn(context.Background(), tuned)

	if after <= before {
		t.Fatalf("expected tuned fitness > baseline: before=%f after=%f", before, after)
	}
}

func TestExoselfNoSynapsesNoop(t *testing.T) {
	tuner := &Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 4, StepSize: 0.2}
	genome := model.Genome{ID: "g"}

	out, err := tuner.Tune(context.Background(), genome, 10, func(context.Context, model.Genome) (float64, error) {
		return 0, nil
	})
	if err != nil {
		t.Fatalf("tune: %v", err)
	}
	if out.ID != genome.ID {
		t.Fatalf("unexpected genome mutation")
	}
}

func TestExoselfInputValidation(t *testing.T) {
	genome := model.Genome{Synapses: []model.Synapse{{ID: "s", Weight: 0}}}
	fitnessFn := func(context.Context, model.Genome) (float64, error) { return 0, nil }

	if _, err := (&Exoself{}).Tune(context.Background(), genome, 1, fitnessFn); err == nil {
		t.Fatal("expected rand validation error")
	}
	if _, err := (&Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 0, StepSize: 1}).Tune(context.Background(), genome, 1, fitnessFn); err == nil {
		t.Fatal("expected steps validation error")
	}
	if _, err := (&Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 1, StepSize: 0}).Tune(context.Background(), genome, 1, fitnessFn); err == nil {
		t.Fatal("expected step size validation error")
	}
	if _, err := (&Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 1, StepSize: 1, PerturbationRange: -1}).Tune(context.Background(), genome, 1, fitnessFn); err == nil {
		t.Fatal("expected perturbation range validation error")
	}
	if _, err := (&Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 1, StepSize: 1, AnnealingFactor: -1}).Tune(context.Background(), genome, 1, fitnessFn); err == nil {
		t.Fatal("expected annealing factor validation error")
	}
	if _, err := (&Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 1, StepSize: 1}).Tune(context.Background(), genome, 1, nil); err == nil {
		t.Fatal("expected fitness validation error")
	}
	if _, err := (&Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 1, StepSize: 1, MinImprovement: -0.1}).Tune(context.Background(), genome, 1, fitnessFn); err == nil {
		t.Fatal("expected min improvement validation error")
	}
	if _, err := (&Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 1, StepSize: 1, CandidateSelection: "unknown"}).Tune(context.Background(), genome, 1, fitnessFn); err == nil {
		t.Fatal("expected candidate selection validation error")
	}
}

func TestExoselfMinImprovementBlocksSmallGains(t *testing.T) {
	genome := model.Genome{
		ID:       "g",
		Synapses: []model.Synapse{{ID: "s", Weight: 0.0, Enabled: true}},
	}
	tuner := &Exoself{
		Rand:           rand.New(rand.NewSource(3)),
		Steps:          6,
		StepSize:       0.25,
		MinImprovement: 0.5,
	}
	fitnessFn := func(_ context.Context, g model.Genome) (float64, error) {
		delta := math.Abs(g.Synapses[0].Weight - 0.2)
		return -delta, nil
	}

	tuned, err := tuner.Tune(context.Background(), genome, 40, fitnessFn)
	if err != nil {
		t.Fatalf("tune: %v", err)
	}
	if tuned.Synapses[0].Weight != genome.Synapses[0].Weight {
		t.Fatalf("expected unchanged weight when gains are below threshold: got=%f want=%f", tuned.Synapses[0].Weight, genome.Synapses[0].Weight)
	}
}

func TestExoselfAttemptsZeroReturnsClone(t *testing.T) {
	genome := model.Genome{ID: "g", Synapses: []model.Synapse{{ID: "s", Weight: 1}}}
	tuner := &Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 2, StepSize: 0.5}

	out, err := tuner.Tune(context.Background(), genome, 0, func(context.Context, model.Genome) (float64, error) {
		return math.Pi, nil
	})
	if err != nil {
		t.Fatalf("tune: %v", err)
	}
	if out.Synapses[0].Weight != genome.Synapses[0].Weight {
		t.Fatalf("weight changed unexpectedly")
	}
}

func TestExoselfTuneWithReportProvidesTelemetry(t *testing.T) {
	genome := model.Genome{
		ID:       "g",
		Synapses: []model.Synapse{{ID: "s", Weight: 0.2, Enabled: true}},
	}
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(41)),
		Steps:              2,
		StepSize:           0.2,
		CandidateSelection: CandidateSelectOriginal,
	}
	fitnessFn := func(_ context.Context, g model.Genome) (float64, error) {
		return g.Synapses[0].Weight, nil
	}
	_, report, err := tuner.TuneWithReport(context.Background(), genome, 2, fitnessFn)
	if err != nil {
		t.Fatalf("tune with report: %v", err)
	}
	if report.AttemptsPlanned != 2 || report.AttemptsExecuted == 0 {
		t.Fatalf("unexpected attempt telemetry: %+v", report)
	}
	if report.CandidateEvaluations == 0 {
		t.Fatalf("expected candidate evaluations in report: %+v", report)
	}
}

func TestExoselfDynamicRandomSelectionSupported(t *testing.T) {
	genome := model.Genome{
		ID:       "g",
		Synapses: []model.Synapse{{ID: "s", Weight: 0.2, Enabled: true}},
	}
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(7)),
		Steps:              3,
		StepSize:           0.15,
		CandidateSelection: CandidateSelectDynamic,
	}
	fitnessFn := func(_ context.Context, g model.Genome) (float64, error) {
		return g.Synapses[0].Weight, nil
	}
	if _, err := tuner.Tune(context.Background(), genome, 8, fitnessFn); err != nil {
		t.Fatalf("tune with dynamic_random selection: %v", err)
	}
}

func TestExoselfExtendedSelectionModesSupported(t *testing.T) {
	genome := model.Genome{
		ID:       "g",
		Synapses: []model.Synapse{{ID: "s", Weight: 0.2, Enabled: true}},
	}
	fitnessFn := func(_ context.Context, g model.Genome) (float64, error) {
		return g.Synapses[0].Weight, nil
	}
	modes := []string{
		CandidateSelectDynamicA,
		CandidateSelectAll,
		CandidateSelectAllRandom,
		CandidateSelectActive,
		CandidateSelectActiveRnd,
		CandidateSelectRecent,
		CandidateSelectRecentRnd,
		CandidateSelectCurrent,
		CandidateSelectCurrentRd,
		CandidateSelectLastGen,
		CandidateSelectLastGenRd,
	}
	for i, mode := range modes {
		tuner := &Exoself{
			Rand:               rand.New(rand.NewSource(int64(100 + i))),
			Steps:              3,
			StepSize:           0.15,
			CandidateSelection: mode,
		}
		if _, err := tuner.Tune(context.Background(), genome, 8, fitnessFn); err != nil {
			t.Fatalf("tune with mode=%s: %v", mode, err)
		}
	}
}

func TestNormalizeCandidateSelectionName(t *testing.T) {
	cases := map[string]string{
		"":                 CandidateSelectBestSoFar,
		"best_so_far":      CandidateSelectBestSoFar,
		"original":         CandidateSelectOriginal,
		"dynamic":          CandidateSelectDynamicA,
		"dynamic_random":   CandidateSelectDynamic,
		"all":              CandidateSelectAll,
		"all_random":       CandidateSelectAllRandom,
		"active":           CandidateSelectActive,
		"active_random":    CandidateSelectActiveRnd,
		"recent":           CandidateSelectRecent,
		"recent_random":    CandidateSelectRecentRnd,
		"current":          CandidateSelectCurrent,
		"current_random":   CandidateSelectCurrentRd,
		"lastgen":          CandidateSelectLastGen,
		"lastgen_random":   CandidateSelectLastGenRd,
		"unknown_mode_xyz": "unknown_mode_xyz",
	}
	for in, want := range cases {
		if got := NormalizeCandidateSelectionName(in); got != want {
			t.Fatalf("normalize selection %q: got=%q want=%q", in, got, want)
		}
	}
}

func TestExoselfConcurrentTuneSafe(t *testing.T) {
	genome := model.Genome{
		ID: "g",
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity"},
		},
		Synapses: []model.Synapse{{ID: "s", From: "i", To: "o", Weight: 0.1, Enabled: true}},
	}
	tuner := &Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 4, StepSize: 0.2}
	fitnessFn := func(_ context.Context, g model.Genome) (float64, error) {
		return g.Synapses[0].Weight, nil
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 32)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := tuner.Tune(context.Background(), genome, 8, fitnessFn); err != nil {
				errCh <- err
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatalf("unexpected tuning error: %v", err)
	}
}

func TestExoselfRandomSelectionModesReturnNonEmptyPool(t *testing.T) {
	tuner := &Exoself{
		Rand: rand.New(rand.NewSource(11)),
	}
	best := model.Genome{ID: "best"}
	original := model.Genome{ID: "original"}
	recent := model.Genome{ID: "recent"}

	randomModes := []string{
		CandidateSelectDynamic,
		CandidateSelectAllRandom,
		CandidateSelectActiveRnd,
		CandidateSelectRecentRnd,
		CandidateSelectCurrentRd,
		CandidateSelectLastGenRd,
	}
	for _, mode := range randomModes {
		tuner.CandidateSelection = mode
		pool, err := tuner.candidateBases(best, original, recent)
		if err != nil {
			t.Fatalf("candidateBases(%s): %v", mode, err)
		}
		if len(pool) == 0 {
			t.Fatalf("candidateBases(%s) returned empty pool", mode)
		}
	}
}

func TestExoselfStopsEarlyWhenGoalReached(t *testing.T) {
	genome := model.Genome{
		ID:       "g",
		Synapses: []model.Synapse{{ID: "s", Weight: 0.5, Enabled: true}},
	}
	calls := 0
	tuner := &Exoself{
		Rand:        rand.New(rand.NewSource(19)),
		Steps:       4,
		StepSize:    0.2,
		GoalFitness: 0.9,
	}
	fitnessFn := func(_ context.Context, _ model.Genome) (float64, error) {
		calls++
		return 1.0, nil
	}
	if _, err := tuner.Tune(context.Background(), genome, 25, fitnessFn); err != nil {
		t.Fatalf("tune: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected one fitness evaluation due to goal short-circuit, got %d", calls)
	}
}

func TestExoselfPerturbationRangeAffectsDelta(t *testing.T) {
	genome := model.Genome{
		ID:       "g",
		Synapses: []model.Synapse{{ID: "s", Weight: 0, Enabled: true}},
	}
	fitnessFn := func(_ context.Context, g model.Genome) (float64, error) {
		return g.Synapses[0].Weight, nil
	}

	base := Exoself{
		Rand:               rand.New(rand.NewSource(23)),
		Steps:              1,
		StepSize:           0.25,
		CandidateSelection: CandidateSelectOriginal,
	}
	tunedBase, err := base.Tune(context.Background(), genome, 1, fitnessFn)
	if err != nil {
		t.Fatalf("base tune: %v", err)
	}

	ranged := Exoself{
		Rand:               rand.New(rand.NewSource(23)),
		Steps:              1,
		StepSize:           0.25,
		PerturbationRange:  2.0,
		CandidateSelection: CandidateSelectOriginal,
	}
	tunedRanged, err := ranged.Tune(context.Background(), genome, 1, fitnessFn)
	if err != nil {
		t.Fatalf("ranged tune: %v", err)
	}

	baseDelta := math.Abs(tunedBase.Synapses[0].Weight - genome.Synapses[0].Weight)
	rangedDelta := math.Abs(tunedRanged.Synapses[0].Weight - genome.Synapses[0].Weight)
	if rangedDelta <= baseDelta {
		t.Fatalf("expected perturbation range to increase magnitude: base=%f ranged=%f", baseDelta, rangedDelta)
	}
}

func TestFilterCandidatesByAgeUsesInferredGeneration(t *testing.T) {
	pool := []model.Genome{
		{ID: "g0-root"},
		{ID: "xor-g2-3"},
		{ID: "xor-g5-9"},
	}
	currentOnly := filterCandidatesByAge(pool, 0)
	if len(currentOnly) != 1 || currentOnly[0].ID != "xor-g5-9" {
		ids := make([]string, 0, len(currentOnly))
		for _, g := range currentOnly {
			ids = append(ids, g.ID)
		}
		t.Fatalf("expected current generation candidate only, got=%v", ids)
	}

	active := filterCandidatesByAge(pool, 3)
	if len(active) != 2 {
		t.Fatalf("expected two active candidates, got=%d", len(active))
	}
}

func TestInferGenomeGeneration(t *testing.T) {
	if g, ok := inferGenomeGeneration("xor-g12-77"); !ok || g != 12 {
		t.Fatalf("expected parse generation 12, got=%d ok=%t", g, ok)
	}
	if _, ok := inferGenomeGeneration("no-generation-here"); ok {
		t.Fatal("expected missing generation parse failure")
	}
}

func TestExoselfLastGenAliasUsesCurrentGenerationPool(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(29)),
		CandidateSelection: CandidateSelectLastGen,
	}
	best := model.Genome{ID: "xor-g5-best"}
	original := model.Genome{ID: "xor-g2-original"}
	recent := model.Genome{ID: "xor-g4-recent"}

	pool, err := tuner.candidateBases(best, original, recent)
	if err != nil {
		t.Fatalf("candidateBases(lastgen): %v", err)
	}
	if len(pool) != 1 || pool[0].ID != best.ID {
		t.Fatalf("expected lastgen to resolve to current-generation pool; got=%v", pool)
	}
}

func TestExoselfAllSelectionUsesEntireCandidatePool(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(31)),
		CandidateSelection: CandidateSelectAll,
	}
	best := model.Genome{ID: "xor-g5-best"}
	original := model.Genome{ID: "xor-g1-original"}
	recent := model.Genome{ID: "xor-g4-recent"}

	pool, err := tuner.candidateBases(best, original, recent)
	if err != nil {
		t.Fatalf("candidateBases(all): %v", err)
	}
	if len(pool) != 3 {
		t.Fatalf("expected all mode to keep full pool, got=%d", len(pool))
	}
	seen := map[string]bool{}
	for _, g := range pool {
		seen[g.ID] = true
	}
	for _, id := range []string{best.ID, original.ID, recent.ID} {
		if !seen[id] {
			t.Fatalf("expected all mode pool to contain %q", id)
		}
	}
}

func TestExoselfAllRandomCanSelectOlderCandidates(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(37)),
		CandidateSelection: CandidateSelectAllRandom,
	}
	best := model.Genome{ID: "xor-g5-best"}
	original := model.Genome{ID: "xor-g1-original"}
	recent := model.Genome{ID: "xor-g4-recent"}

	seen := map[string]bool{}
	for i := 0; i < 128; i++ {
		pool, err := tuner.candidateBases(best, original, recent)
		if err != nil {
			t.Fatalf("candidateBases(all_random): %v", err)
		}
		for _, g := range pool {
			seen[g.ID] = true
		}
	}
	if !seen[original.ID] {
		t.Fatalf("expected all_random to include older candidate %q over repeated draws", original.ID)
	}
}

func TestSelectedNeuronPerturbTargetsActiveUsesAgeAnnealing(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(43)),
		CandidateSelection: CandidateSelectActive,
	}
	genome := model.Genome{
		ID: "xor-g5-i0",
		Neurons: []model.Neuron{
			{ID: "n-new", Generation: 5, Activation: "identity"},
			{ID: "n-mid", Generation: 3, Activation: "identity"},
			{ID: "n-old", Generation: 1, Activation: "identity"},
		},
	}

	targets := tuner.selectedNeuronPerturbTargets(genome, 1.0, 0.5)
	if len(targets) != 2 {
		t.Fatalf("expected active mode to include age<=3 neurons only, got=%d", len(targets))
	}
	if targets[0].neuronID != "n-new" {
		t.Fatalf("expected first target n-new, got=%s", targets[0].neuronID)
	}
	if math.Abs(targets[0].spread-math.Pi) > 1e-9 {
		t.Fatalf("expected age-0 spread=pi, got=%f", targets[0].spread)
	}
	if targets[1].neuronID != "n-mid" {
		t.Fatalf("expected second target n-mid, got=%s", targets[1].neuronID)
	}
	if math.Abs(targets[1].spread-(math.Pi*0.25)) > 1e-9 {
		t.Fatalf("expected age-2 spread=pi*0.25, got=%f", targets[1].spread)
	}
}

func TestExoselfPerturbCandidateCurrentModeTargetsCurrentGeneration(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(47)),
		Steps:              8,
		StepSize:           0.2,
		CandidateSelection: CandidateSelectCurrent,
	}
	base := model.Genome{
		ID: "xor-g5-i0",
		Neurons: []model.Neuron{
			{ID: "n-old", Generation: 2, Activation: "identity"},
			{ID: "n-new", Generation: 5, Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s-old", From: "n-old", To: "n-old", Weight: 0.0, Enabled: true},
			{ID: "s-new", From: "n-new", To: "n-new", Weight: 0.0, Enabled: true},
		},
	}

	mutated, err := tuner.perturbCandidate(context.Background(), base, 1.0, 0.5)
	if err != nil {
		t.Fatalf("perturbCandidate: %v", err)
	}
	if mutated.Synapses[0].Weight != base.Synapses[0].Weight {
		t.Fatal("expected older-neuron synapse to remain unchanged in current mode")
	}
	if mutated.Synapses[1].Weight == base.Synapses[1].Weight {
		t.Fatal("expected current-neuron synapse to mutate in current mode")
	}
}

func TestSelectedNeuronPerturbTargetsIncludesActuatorDrivenTargets(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(53)),
		CandidateSelection: CandidateSelectCurrent,
	}
	genome := model.Genome{
		ID: "xor-g5-i0",
		Neurons: []model.Neuron{
			{ID: "n-g0-old", Generation: 0, Activation: "identity"},
			{ID: "n-g0-act", Generation: 0, Activation: "identity"},
		},
		ActuatorIDs: []string{"a-current"},
		ActuatorGenerations: map[string]int{
			"a-current": 5,
		},
		NeuronActuatorLinks: []model.NeuronActuatorLink{
			{NeuronID: "n-g0-act", ActuatorID: "a-current"},
		},
	}

	targets := tuner.selectedNeuronPerturbTargets(genome, 1.0, 0.5)
	if len(targets) != 1 {
		t.Fatalf("expected one actuator-driven target, got=%d", len(targets))
	}
	if targets[0].neuronID != "n-g0-act" {
		t.Fatalf("expected actuator-linked neuron target, got=%s", targets[0].neuronID)
	}
	if targets[0].sourceKind != tuningElementActuator || targets[0].sourceID != "a-current" {
		t.Fatalf("expected actuator source metadata, got kind=%s id=%s", targets[0].sourceKind, targets[0].sourceID)
	}
	if math.Abs(targets[0].spread-math.Pi) > 1e-9 {
		t.Fatalf("expected age-0 spread=pi from current actuator, got=%f", targets[0].spread)
	}
}
