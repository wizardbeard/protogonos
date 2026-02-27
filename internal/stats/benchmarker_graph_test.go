package stats

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"protogonos/internal/model"
)

func TestBuildBenchmarkerGraphs(t *testing.T) {
	base := t.TempDir()
	exp := BenchmarkExperiment{
		ID:     "exp-graphs",
		RunIDs: []string{"exp-graphs-run-001", "exp-graphs-run-002"},
	}

	traceAcc := []TraceGeneration{
		{
			Generation: 1,
			Stats: []TraceStatEntry{
				{
					SpeciesKey:       "sp-a",
					ChampionGenomeID: "g-a",
					ChampionGenome: model.Genome{
						ID:      "g-a",
						Neurons: []model.Neuron{{ID: "n1"}, {ID: "n2"}},
					},
					BestFitness:       0.20,
					ValidationFitness: float64Ptr(0.15),
				},
			},
		},
		{
			Generation: 2,
			Stats: []TraceStatEntry{
				{
					SpeciesKey:       "sp-a",
					ChampionGenomeID: "g-a2",
					ChampionGenome: model.Genome{
						ID:      "g-a2",
						Neurons: []model.Neuron{{ID: "n1"}, {ID: "n2"}, {ID: "n3"}},
					},
					BestFitness:       0.45,
					ValidationFitness: float64Ptr(0.30),
				},
			},
		},
	}

	for _, runID := range exp.RunIDs {
		runDir := filepath.Join(base, runID)
		if _, err := WriteRunArtifacts(base, RunArtifacts{
			Config: RunConfig{
				RunID:          runID,
				Scape:          "xor",
				PopulationSize: 6,
				Generations:    2,
				Seed:           1,
			},
			BestByGeneration: []float64{0.2, 0.45},
			TraceAcc:         traceAcc,
		}); err != nil {
			t.Fatalf("write run artifacts for %s: %v", runID, err)
		}
		if err := WriteBenchmarkSeries(runDir, []float64{0.2, 0.45}); err != nil {
			t.Fatalf("write benchmark series for %s: %v", runID, err)
		}
	}

	graphs, err := BuildBenchmarkerGraphs(base, exp)
	if err != nil {
		t.Fatalf("build benchmarker graphs: %v", err)
	}
	if len(graphs) != 1 {
		t.Fatalf("expected one morphology graph, got %d", len(graphs))
	}
	graph := graphs[0]
	if graph.Morphology != "xor" {
		t.Fatalf("unexpected morphology: %+v", graph)
	}
	if len(graph.EvaluationIndex) != 2 {
		t.Fatalf("unexpected graph series length: %+v", graph)
	}
	if len(graph.AvgFitness) != 2 || graph.AvgFitness[0] <= 0 || graph.AvgFitness[1] <= 0 {
		t.Fatalf("unexpected avg fitness series: %+v", graph.AvgFitness)
	}
	if len(graph.ValidationFitness) != 2 {
		t.Fatalf("expected validation series for both generations: %+v", graph.ValidationFitness)
	}
}

func TestWriteBenchmarkerGraphs(t *testing.T) {
	base := t.TempDir()
	graphs := []BenchmarkerGraph{
		{
			Morphology:           "xor",
			EvaluationIndex:      []int{500, 1000},
			AvgFitness:           []float64{0.2, 0.4},
			FitnessStd:           []float64{0.01, 0.02},
			AvgNeurons:           []float64{2, 3},
			NeuronsStd:           []float64{0, 0},
			AvgDiversity:         []float64{1, 1},
			DiversityStd:         []float64{0, 0},
			MaxFitness:           []float64{0.2, 0.4},
			MaxAvgFitness:        []float64{0.2, 0.4},
			MinAvgFitness:        []float64{0.2, 0.4},
			Evaluations:          []float64{6, 12},
			ValidationFitness:    []float64{0.15, 0.3},
			ValidationFitnessStd: []float64{0, 0},
			ValidationMaxFitness: []float64{0.15, 0.3},
			ValidationMinFitness: []float64{0.15, 0.3},
		},
	}
	paths, err := WriteBenchmarkerGraphs(base, "exp-graphs", "report_Graphs", graphs)
	if err != nil {
		t.Fatalf("write benchmarker graphs: %v", err)
	}
	if len(paths) != 1 {
		t.Fatalf("expected one graph output, got %d", len(paths))
	}
	data, err := os.ReadFile(paths[0])
	if err != nil {
		t.Fatalf("read graph output: %v", err)
	}
	text := string(data)
	if !strings.Contains(text, "#Avg Fitness Vs Evaluations, Morphology:xor") {
		t.Fatalf("expected avg fitness section, got:\n%s", text)
	}
	if !strings.Contains(text, "#Validation Avg Fitness Vs Evaluations, Morphology:xor") {
		t.Fatalf("expected validation section, got:\n%s", text)
	}
}
