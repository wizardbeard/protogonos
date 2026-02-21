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
		ID: "g",
		Neurons: []model.Neuron{
			{ID: "n", Activation: "identity"},
		},
		Synapses: []model.Synapse{{ID: "s", From: "n", To: "n", Weight: 0, Enabled: true}},
	}
	base := &Exoself{
		Rand:               rand.New(rand.NewSource(23)),
		Steps:              1,
		StepSize:           0.25,
		CandidateSelection: CandidateSelectOriginal,
	}
	tunedBase, err := base.perturbCandidate(context.Background(), genome, 1.0, 1.0)
	if err != nil {
		t.Fatalf("base perturb: %v", err)
	}

	ranged := &Exoself{
		Rand:               rand.New(rand.NewSource(23)),
		Steps:              1,
		StepSize:           0.25,
		PerturbationRange:  2.0,
		CandidateSelection: CandidateSelectOriginal,
	}
	tunedRanged, err := ranged.perturbCandidate(context.Background(), genome, 2.0, 1.0)
	if err != nil {
		t.Fatalf("ranged perturb: %v", err)
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

func TestSelectedNeuronPerturbTargetsIncludesDirectActuatorTargets(t *testing.T) {
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
	}

	targets := tuner.selectedNeuronPerturbTargets(genome, 1.0, 0.5)
	if len(targets) != 1 {
		t.Fatalf("expected one actuator target, got=%d", len(targets))
	}
	if targets[0].neuronID != "" {
		t.Fatalf("expected direct actuator target without neuron projection, got=%s", targets[0].neuronID)
	}
	if targets[0].sourceKind != tuningElementActuator || targets[0].sourceID != "a-current" {
		t.Fatalf("expected actuator source metadata, got kind=%s id=%s", targets[0].sourceKind, targets[0].sourceID)
	}
	if math.Abs(targets[0].spread-math.Pi) > 1e-9 {
		t.Fatalf("expected age-0 spread=pi from current actuator, got=%f", targets[0].spread)
	}
}

func TestExoselfPerturbCandidateMutatesActuatorTunablesForActuatorTargets(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(67)),
		Steps:              4,
		StepSize:           0.2,
		CandidateSelection: CandidateSelectCurrent,
	}
	base := model.Genome{
		ID:          "xor-g5-i0",
		ActuatorIDs: []string{"a-current"},
		ActuatorGenerations: map[string]int{
			"a-current": 5,
		},
	}

	mutated, err := tuner.perturbCandidate(context.Background(), base, 1.0, 0.5)
	if err != nil {
		t.Fatalf("perturbCandidate: %v", err)
	}
	if mutated.ActuatorTunables == nil {
		t.Fatal("expected actuator tunables to be created")
	}
	if mutated.ActuatorTunables["a-current"] == 0 {
		t.Fatal("expected actuator-local tunable to be perturbed")
	}
}

func TestSelectedNeuronPerturbTargetsActiveHasNoFallbackWhenPoolEmpty(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(59)),
		CandidateSelection: CandidateSelectActive,
	}
	genome := model.Genome{
		ID: "xor-g5-i0",
		Neurons: []model.Neuron{
			{ID: "n-g0-a", Generation: 0, Activation: "identity"},
			{ID: "n-g0-b", Generation: 0, Activation: "identity"},
		},
	}
	targets := tuner.selectedNeuronPerturbTargets(genome, 1.0, 0.5)
	if len(targets) != 0 {
		t.Fatalf("expected active mode empty target pool when no young elements, got=%d", len(targets))
	}
}

func TestExoselfPerturbCandidateActiveEmptyPoolNoOp(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(63)),
		Steps:              6,
		StepSize:           0.4,
		CandidateSelection: CandidateSelectActive,
	}
	base := model.Genome{
		ID: "xor-g5-i0",
		Neurons: []model.Neuron{
			{ID: "n-g0-a", Generation: 0, Activation: "identity"},
			{ID: "n-g0-b", Generation: 0, Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{ID: "s-a", From: "n-g0-a", To: "n-g0-a", Weight: 0.3, Enabled: true},
			{ID: "s-b", From: "n-g0-b", To: "n-g0-b", Weight: -0.2, Enabled: true},
		},
	}

	mutated, err := tuner.perturbCandidate(context.Background(), base, 1.0, 0.5)
	if err != nil {
		t.Fatalf("perturbCandidate: %v", err)
	}
	if mutated.Synapses[0].Weight != base.Synapses[0].Weight || mutated.Synapses[1].Weight != base.Synapses[1].Weight {
		t.Fatalf("expected active-empty perturbation to be no-op, before=%+v after=%+v", base.Synapses, mutated.Synapses)
	}
}

func TestSelectedNeuronPerturbTargetsActiveRandomFallsBackToFirstElement(t *testing.T) {
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(61)),
		CandidateSelection: CandidateSelectActiveRnd,
	}
	genome := model.Genome{
		ID: "xor-g5-i0",
		Neurons: []model.Neuron{
			{ID: "n-g0-a", Generation: 0, Activation: "identity"},
			{ID: "n-g0-b", Generation: 0, Activation: "identity"},
		},
	}
	targets := tuner.selectedNeuronPerturbTargets(genome, 1.0, 0.5)
	if len(targets) != 1 {
		t.Fatalf("expected active_random fallback to one target, got=%d", len(targets))
	}
	if targets[0].neuronID != "n-g0-a" {
		t.Fatalf("expected fallback to first element target n-g0-a, got=%s", targets[0].neuronID)
	}
	if math.Abs(targets[0].spread-math.Pi) > 1e-9 {
		t.Fatalf("expected fallback spread=pi, got=%f", targets[0].spread)
	}
}

func TestScalarFitnessDominatesUsesRelativeThreshold(t *testing.T) {
	if scalarFitnessDominates(11.0, 10.0, 0.1) {
		t.Fatal("expected strict greater-than check at relative threshold boundary")
	}
	if !scalarFitnessDominates(11.0001, 10.0, 0.1) {
		t.Fatal("expected dominance when candidate exceeds incumbent*(1+mip)")
	}
	if !scalarFitnessDominates(-0.95, -1.0, 0.1) {
		t.Fatal("expected dominance for monotonic improvement above relative threshold")
	}
	if scalarFitnessDominates(-1.11, -1.0, 0.1) {
		t.Fatal("expected no dominance for worse candidate below negative threshold")
	}
}

func TestVectorFitnessDominatesUsesPerElementRelativeThreshold(t *testing.T) {
	if !vectorFitnessDominates([]float64{1.2, 2.3}, []float64{1.0, 2.0}, 0.1) {
		t.Fatal("expected full vector dominance when every element exceeds threshold")
	}
	if vectorFitnessDominates([]float64{1.2, 2.1}, []float64{1.0, 2.0}, 0.1) {
		t.Fatal("expected no dominance when any element fails threshold")
	}
	if vectorFitnessDominates([]float64{1.0}, []float64{1.0, 2.0}, 0.1) {
		t.Fatal("expected length mismatch to fail dominance")
	}
}

func TestExoselfAttemptsStopAtConsecutiveNoImprovementBudget(t *testing.T) {
	genome := model.Genome{
		ID:       "g",
		Synapses: []model.Synapse{{ID: "s", Weight: 0.2, Enabled: true}},
	}
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(71)),
		Steps:              1,
		StepSize:           0.2,
		CandidateSelection: CandidateSelectOriginal,
	}
	_, report, err := tuner.TuneWithReport(context.Background(), genome, 3, func(context.Context, model.Genome) (float64, error) {
		return 1.0, nil
	})
	if err != nil {
		t.Fatalf("tune with report: %v", err)
	}
	if report.AttemptsExecuted != 3 {
		t.Fatalf("expected attempts to stop at planned consecutive no-improvement budget, got=%d", report.AttemptsExecuted)
	}
}

func TestExoselfAttemptsResetOnImprovement(t *testing.T) {
	genome := model.Genome{
		ID:       "g",
		Synapses: []model.Synapse{{ID: "s", Weight: 0.2, Enabled: true}},
	}
	tuner := &Exoself{
		Rand:               rand.New(rand.NewSource(73)),
		Steps:              1,
		StepSize:           0.2,
		GoalFitness:        4.0,
		CandidateSelection: CandidateSelectOriginal,
	}
	evals := 0
	_, report, err := tuner.TuneWithReport(context.Background(), genome, 2, func(context.Context, model.Genome) (float64, error) {
		evals++
		return float64(evals), nil
	})
	if err != nil {
		t.Fatalf("tune with report: %v", err)
	}
	if !report.GoalReached {
		t.Fatalf("expected goal to be reached, report=%+v", report)
	}
	if report.AttemptsExecuted <= report.AttemptsPlanned {
		t.Fatalf("expected improvement resets to allow attempts beyond planned budget, report=%+v", report)
	}
}

func TestTransposeVectorsUsesShortestLength(t *testing.T) {
	in := [][]float64{
		{1, 2, 3},
		{4, 5},
		{6, 7, 8, 9},
	}
	out := transposeVectors(in)
	if len(out) != 2 {
		t.Fatalf("expected transpose length 2 from shortest vector, got=%d", len(out))
	}
	if len(out[0]) != 3 || out[0][0] != 1 || out[0][1] != 4 || out[0][2] != 6 {
		t.Fatalf("unexpected first transpose column: %+v", out[0])
	}
	if len(out[1]) != 3 || out[1][0] != 2 || out[1][1] != 5 || out[1][2] != 7 {
		t.Fatalf("unexpected second transpose column: %+v", out[1])
	}
}

func TestVectorAvgComputesColumnMeans(t *testing.T) {
	in := [][]float64{
		{1, 2, 3},
		{4, 5},
		{6, 7, 8, 9},
	}
	avg := vectorAvg(in)
	if len(avg) != 2 {
		t.Fatalf("expected avg length 2, got=%d", len(avg))
	}
	if math.Abs(avg[0]-11.0/3.0) > 1e-12 {
		t.Fatalf("unexpected avg[0]=%f", avg[0])
	}
	if math.Abs(avg[1]-14.0/3.0) > 1e-12 {
		t.Fatalf("unexpected avg[1]=%f", avg[1])
	}
}

func TestVectorBasicStatsMatchesReferenceSemantics(t *testing.T) {
	in := [][]float64{
		{1, 3},
		{2, 1},
		{0, 5},
	}
	maxV, minV, avgV, stdV := vectorBasicStats(in)

	if len(maxV) != 2 || maxV[0] != 2 || maxV[1] != 1 {
		t.Fatalf("unexpected lexicographic max vector: %+v", maxV)
	}
	if len(minV) != 2 || minV[0] != 0 || minV[1] != 5 {
		t.Fatalf("unexpected lexicographic min vector: %+v", minV)
	}
	if len(avgV) != 2 || math.Abs(avgV[0]-1.0) > 1e-12 || math.Abs(avgV[1]-3.0) > 1e-12 {
		t.Fatalf("unexpected avg vector: %+v", avgV)
	}
	if len(stdV) != 2 || math.Abs(stdV[0]-math.Sqrt(2.0/3.0)) > 1e-12 || math.Abs(stdV[1]-math.Sqrt(8.0/3.0)) > 1e-12 {
		t.Fatalf("unexpected std vector: %+v", stdV)
	}
}

func TestVectorBasicStatsEmptyInput(t *testing.T) {
	maxV, minV, avgV, stdV := vectorBasicStats(nil)
	if maxV != nil || minV != nil || avgV != nil || stdV != nil {
		t.Fatalf("expected nil stats for empty input, got max=%+v min=%+v avg=%+v std=%+v", maxV, minV, avgV, stdV)
	}
}
