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
