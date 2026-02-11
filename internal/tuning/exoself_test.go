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
	if _, err := (&Exoself{Rand: rand.New(rand.NewSource(1)), Steps: 1, StepSize: 1}).Tune(context.Background(), genome, 1, nil); err == nil {
		t.Fatal("expected fitness validation error")
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
