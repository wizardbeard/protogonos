package evo

import (
	"context"
	"math/rand"
	"strings"
	"testing"

	"protogonos/internal/model"
)

func TestTopologicalMutationPolicies(t *testing.T) {
	genome := newComplexLinearGenome("g", 1.0)

	c, err := (ConstTopologicalMutations{Count: 3}).MutationCount(genome, 0, nil)
	if err != nil || c != 3 {
		t.Fatalf("const policy mismatch count=%d err=%v", c, err)
	}

	l, err := (NCountLinearTopologicalMutations{Multiplier: 0.5, MaxCount: 10}).MutationCount(genome, 0, nil)
	if err != nil || l < 1 {
		t.Fatalf("linear policy invalid count=%d err=%v", l, err)
	}

	e, err := (NCountExponentialTopologicalMutations{Power: 0.5, MaxCount: 10}).MutationCount(genome, 0, nil)
	if err != nil || e < 1 {
		t.Fatalf("exponential policy invalid count=%d err=%v", e, err)
	}
}

func TestNCountExponentialTopologicalMutationsRandomRange(t *testing.T) {
	genome := newComplexLinearGenome("g", 1.0)
	maxCount := int(float64(len(genome.Neurons)))
	policy := NCountExponentialTopologicalMutations{Power: 1.0, MaxCount: 0}

	seen := map[int]struct{}{}
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 128; i++ {
		count, err := policy.MutationCount(genome, 0, rng)
		if err != nil {
			t.Fatalf("mutation count: %v", err)
		}
		if count < 1 || count > maxCount {
			t.Fatalf("count out of expected range: got=%d want=[1,%d]", count, maxCount)
		}
		seen[count] = struct{}{}
	}
	if len(seen) < 2 && maxCount > 1 {
		t.Fatalf("expected stochastic distribution with at least 2 distinct counts, got=%v", seen)
	}
}

func TestPopulationMonitorAppliesMultipleMutationsPerChild(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
		newLinearGenome("g2", -0.6),
		newLinearGenome("g3", -0.4),
	}

	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:                oneDimScape{},
		Mutation:             namedNoopMutation{name: "noop"},
		TopologicalMutations: ConstTopologicalMutations{Count: 3},
		PopulationSize:       len(initial),
		EliteCount:           1,
		Generations:          1,
		Workers:              1,
		Seed:                 1,
		InputNeuronIDs:       []string{"i"},
		OutputNeuronIDs:      []string{"o"},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	result, err := monitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	seenTriplet := false
	for _, record := range result.Lineage {
		if strings.Count(record.Operation, "noop") == 3 {
			seenTriplet = true
			break
		}
	}
	if !seenTriplet {
		t.Fatalf("expected operation lineage containing three mutation names: %+v", result.Lineage)
	}
}
