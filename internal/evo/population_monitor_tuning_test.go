package evo

import (
	"context"
	"math/rand"
	"sync"
	"testing"

	"protogonos/internal/model"
	"protogonos/internal/tuning"
)

type recordingTuner struct {
	mu       sync.Mutex
	attempts []int
}

func (r *recordingTuner) Name() string {
	return "recording_tuner"
}

func (r *recordingTuner) Tune(_ context.Context, genome model.Genome, attempts int, _ tuning.FitnessFn) (model.Genome, error) {
	r.mu.Lock()
	r.attempts = append(r.attempts, attempts)
	r.mu.Unlock()
	return genome, nil
}

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

func TestPopulationMonitorTuneAttemptPolicy(t *testing.T) {
	initial := []model.Genome{
		newLinearGenome("g0", -1.0),
		newLinearGenome("g1", -0.8),
	}

	rec := &recordingTuner{}
	monitor, err := NewPopulationMonitor(MonitorConfig{
		Scape:             oneDimScape{},
		Mutation:          PerturbWeightAt{Index: 0, Delta: 0},
		PopulationSize:    len(initial),
		EliteCount:        1,
		Generations:       3,
		Workers:           1,
		Seed:              1,
		InputNeuronIDs:    []string{"i"},
		OutputNeuronIDs:   []string{"o"},
		Tuner:             rec,
		TuneAttempts:      4,
		TuneAttemptPolicy: tuning.LinearDecayAttemptPolicy{MinAttempts: 1},
	})
	if err != nil {
		t.Fatalf("new monitor: %v", err)
	}

	if _, err := monitor.Run(context.Background(), initial); err != nil {
		t.Fatalf("run: %v", err)
	}

	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.attempts) == 0 {
		t.Fatal("expected recorded attempts")
	}
	seen4 := false
	seen2 := false
	seen1 := false
	for _, a := range rec.attempts {
		if a == 4 {
			seen4 = true
		}
		if a == 2 {
			seen2 = true
		}
		if a == 1 {
			seen1 = true
		}
	}
	if !seen4 || !seen2 || !seen1 {
		t.Fatalf("expected attempts to include 4,2,1; got=%v", rec.attempts)
	}
}
