package evo

import (
	"context"
	"math/rand"
	"testing"

	"protogonos/internal/model"
	"protogonos/internal/tuning"
)

func TestPopulationMonitorTuningImprovesFirstGeneration(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -2.0),
		newLinearGenome("g1", -1.8),
		newLinearGenome("g2", -1.6),
		newLinearGenome("g3", -1.4),
	}

	baselineMonitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        PerturbWeightAt{Index: 0, Delta: 0},
		PopulationSize:  len(initial),
		EliteCount:      2,
		Generations:     1,
		Workers:         2,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
	})
	if err != nil {
		t.Fatalf("baseline monitor: %v", err)
	}

	withTuningMonitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:           oneDimScape{},
		Mutation:        PerturbWeightAt{Index: 0, Delta: 0},
		PopulationSize:  len(initial),
		EliteCount:      2,
		Generations:     1,
		Workers:         2,
		Seed:            1,
		InputNeuronIDs:  []string{"i"},
		OutputNeuronIDs: []string{"o"},
		Tuner: &tuning.Exoself{
			Rand:     rand.New(rand.NewSource(7)),
			Steps:    8,
			StepSize: 0.5,
		},
		TuneAttempts: 20,
	})
	if err != nil {
		t.Fatalf("tuned monitor: %v", err)
	}

	baseline, err := baselineMonitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("baseline run: %v", err)
	}
	withTuning, err := withTuningMonitor.Run(context.Background(), initial)
	if err != nil {
		t.Fatalf("tuned run: %v", err)
	}

	if len(baseline.BestByGeneration) != 1 || len(withTuning.BestByGeneration) != 1 {
		t.Fatalf("unexpected history length")
	}
	if withTuning.BestByGeneration[0] <= baseline.BestByGeneration[0] {
		t.Fatalf("expected tuning to improve generation-1 best: baseline=%f tuned=%f", baseline.BestByGeneration[0], withTuning.BestByGeneration[0])
	}
}
