package stats

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"protogonos/internal/nn"
)

type BenchmarkerGraph struct {
	Morphology           string    `json:"morphology"`
	AvgNeurons           []float64 `json:"avg_neurons"`
	NeuronsStd           []float64 `json:"neurons_std"`
	AvgFitness           []float64 `json:"avg_fitness"`
	FitnessStd           []float64 `json:"fitness_std"`
	MaxFitness           []float64 `json:"max_fitness"`
	MinFitness           []float64 `json:"min_fitness"`
	MaxAvgFitness        []float64 `json:"maxavg_fitness"`
	MaxAvgFitnessStd     []float64 `json:"maxavg_fitness_std"`
	MinAvgFitness        []float64 `json:"minavg_fitness"`
	AvgDiversity         []float64 `json:"avg_diversity"`
	DiversityStd         []float64 `json:"diversity_std"`
	Evaluations          []float64 `json:"evaluations"`
	ValidationFitness    []float64 `json:"validation_fitness"`
	ValidationFitnessStd []float64 `json:"validation_fitness_std"`
	ValidationMaxFitness []float64 `json:"validationmax_fitness"`
	ValidationMinFitness []float64 `json:"validationmin_fitness"`
	EvaluationIndex      []int     `json:"evaluation_index"`
}

type benchmarkRunGraphData struct {
	populationSize int
	series         []float64
	traceAcc       []TraceGeneration
}

func BuildBenchmarkerGraphs(baseDir string, exp BenchmarkExperiment) ([]BenchmarkerGraph, error) {
	if len(exp.RunIDs) == 0 {
		return []BenchmarkerGraph{}, nil
	}
	runsByMorphology := make(map[string][]benchmarkRunGraphData)
	for _, runID := range exp.RunIDs {
		cfg, ok, err := ReadRunConfig(baseDir, runID)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("run config not found for run id: %s", runID)
		}
		series, ok, err := ReadBenchmarkSeries(baseDir, runID)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("benchmark series not found for run id: %s", runID)
		}
		traceAcc, ok, err := ReadTraceAcc(baseDir, runID)
		if err != nil {
			return nil, err
		}
		if !ok {
			traceAcc = nil
		}
		morphology := strings.TrimSpace(cfg.Scape)
		if morphology == "" {
			morphology = "unknown"
		}
		runsByMorphology[morphology] = append(runsByMorphology[morphology], benchmarkRunGraphData{
			populationSize: cfg.PopulationSize,
			series:         series,
			traceAcc:       traceAcc,
		})
	}

	morphologies := make([]string, 0, len(runsByMorphology))
	for morphology := range runsByMorphology {
		morphologies = append(morphologies, morphology)
	}
	sort.Strings(morphologies)

	graphs := make([]BenchmarkerGraph, 0, len(morphologies))
	for _, morphology := range morphologies {
		graphs = append(graphs, buildGraphForMorphology(morphology, runsByMorphology[morphology]))
	}
	return graphs, nil
}

func WriteBenchmarkerGraphs(baseDir, experimentID, graphPostfix string, graphs []BenchmarkerGraph) ([]string, error) {
	if experimentID == "" {
		return nil, fmt.Errorf("graph experiment id is required")
	}
	reportDir := filepath.Join(baseDir, benchmarkExperimentsDir, experimentID)
	return WriteBenchmarkerGraphsToDir(reportDir, graphPostfix, graphs)
}

func WriteBenchmarkerGraphsToDir(outputDir, graphPostfix string, graphs []BenchmarkerGraph) ([]string, error) {
	if graphPostfix == "" {
		graphPostfix = "report_Graphs"
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(graphs))
	for _, graph := range graphs {
		name := "graph_" + sanitizeGraphToken(graph.Morphology) + "_" + graphPostfix
		path := filepath.Join(outputDir, name)
		if err := writeBenchmarkerGraphFile(path, graph); err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths, nil
}

func ReadTraceAccFile(path string) ([]TraceGeneration, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var traceAcc []TraceGeneration
	if err := json.Unmarshal(data, &traceAcc); err != nil {
		return nil, err
	}
	return traceAcc, nil
}

func BuildBenchmarkerGraphFromTrace(traceAcc []TraceGeneration, morphology string) BenchmarkerGraph {
	series := make([]float64, len(traceAcc))
	for i, generation := range traceAcc {
		hasValue := false
		best := 0.0
		for _, entry := range generation.Stats {
			if !hasValue || entry.BestFitness > best {
				best = entry.BestFitness
				hasValue = true
			}
		}
		if hasValue {
			series[i] = best
		}
	}
	if strings.TrimSpace(morphology) == "" {
		morphology = "trace"
	}
	return buildGraphForMorphology(morphology, []benchmarkRunGraphData{
		{
			populationSize: 1,
			series:         series,
			traceAcc:       traceAcc,
		},
	})
}

func buildGraphForMorphology(morphology string, runs []benchmarkRunGraphData) BenchmarkerGraph {
	graph := BenchmarkerGraph{
		Morphology: morphology,
	}
	maxGeneration := 0
	for _, run := range runs {
		if len(run.series) > maxGeneration {
			maxGeneration = len(run.series)
		}
	}
	graph.EvaluationIndex = make([]int, 0, maxGeneration)
	for generation := 0; generation < maxGeneration; generation++ {
		graph.EvaluationIndex = append(graph.EvaluationIndex, 500*(generation+1))

		fitnessVals := make([]float64, 0, len(runs))
		evaluationVals := make([]float64, 0, len(runs))
		neuronVals := make([]float64, 0, len(runs))
		diversityVals := make([]float64, 0, len(runs))
		validationAvgVals := make([]float64, 0, len(runs))
		validationMaxVals := make([]float64, 0, len(runs))
		validationMinVals := make([]float64, 0, len(runs))

		for _, run := range runs {
			if generation < len(run.series) {
				fitnessVals = append(fitnessVals, run.series[generation])
				populationSize := run.populationSize
				if populationSize <= 0 {
					populationSize = 1
				}
				evaluationVals = append(evaluationVals, float64(populationSize*(generation+1)))
			}
			if generation < len(run.traceAcc) {
				stats := run.traceAcc[generation].Stats
				if len(stats) > 0 {
					neuronCountVals := make([]float64, 0, len(stats))
					validationVals := make([]float64, 0, len(stats))
					for _, stat := range stats {
						if len(stat.ChampionGenome.Neurons) > 0 {
							neuronCountVals = append(neuronCountVals, float64(len(stat.ChampionGenome.Neurons)))
						}
						if stat.ValidationFitness != nil {
							validationVals = append(validationVals, *stat.ValidationFitness)
						}
					}
					if len(neuronCountVals) > 0 {
						avg, _ := nn.Avg(neuronCountVals)
						neuronVals = append(neuronVals, avg)
					}
					diversityVals = append(diversityVals, float64(len(stats)))
					if len(validationVals) > 0 {
						avg, _ := nn.Avg(validationVals)
						validationAvgVals = append(validationAvgVals, avg)
						validationMaxVals = append(validationMaxVals, maxFloat(validationVals))
						validationMinVals = append(validationMinVals, minFloat(validationVals))
					}
				}
			}
		}

		avgFitness, fitnessStd := avgStd(fitnessVals)
		maxFitness := maxOrZero(fitnessVals)
		minFitness := minOrZero(fitnessVals)
		avgEvaluations, _ := avgStd(evaluationVals)
		avgNeurons, neuronStd := avgStd(neuronVals)
		avgDiversity, diversityStd := avgStd(diversityVals)
		validationAvg, validationStd := avgStd(validationAvgVals)
		validationMax := maxOrZero(validationMaxVals)
		validationMin := minOrZero(validationMinVals)

		graph.AvgFitness = append(graph.AvgFitness, avgFitness)
		graph.FitnessStd = append(graph.FitnessStd, fitnessStd)
		graph.MaxFitness = append(graph.MaxFitness, maxFitness)
		graph.MinFitness = append(graph.MinFitness, minFitness)
		graph.MaxAvgFitness = append(graph.MaxAvgFitness, avgFitness)
		graph.MaxAvgFitnessStd = append(graph.MaxAvgFitnessStd, fitnessStd)
		graph.MinAvgFitness = append(graph.MinAvgFitness, minFitness)
		graph.Evaluations = append(graph.Evaluations, avgEvaluations)
		graph.AvgNeurons = append(graph.AvgNeurons, avgNeurons)
		graph.NeuronsStd = append(graph.NeuronsStd, neuronStd)
		graph.AvgDiversity = append(graph.AvgDiversity, avgDiversity)
		graph.DiversityStd = append(graph.DiversityStd, diversityStd)
		graph.ValidationFitness = append(graph.ValidationFitness, validationAvg)
		graph.ValidationFitnessStd = append(graph.ValidationFitnessStd, validationStd)
		graph.ValidationMaxFitness = append(graph.ValidationMaxFitness, validationMax)
		graph.ValidationMinFitness = append(graph.ValidationMinFitness, validationMin)
	}
	return graph
}

func writeBenchmarkerGraphFile(path string, graph BenchmarkerGraph) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	if _, err := fmt.Fprintf(file, "#Avg Fitness Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeriesWithStd(file, graph.EvaluationIndex, graph.AvgFitness, graph.FitnessStd); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n#Avg Neurons Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeriesWithStd(file, graph.EvaluationIndex, graph.AvgNeurons, graph.NeuronsStd); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n#Avg Diversity Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeriesWithStd(file, graph.EvaluationIndex, graph.AvgDiversity, graph.DiversityStd); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n# Max Fitness Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeries(file, graph.EvaluationIndex, graph.MaxFitness); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n#Avg. Max Fitness Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeries(file, graph.EvaluationIndex, graph.MaxAvgFitness); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n#Avg. Min Fitness Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeries(file, graph.EvaluationIndex, graph.MinAvgFitness); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n#Specie-Population Turnover Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeries(file, graph.EvaluationIndex, graph.Evaluations); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n#Validation Avg Fitness Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeriesWithStd(file, graph.EvaluationIndex, graph.ValidationFitness, graph.ValidationFitnessStd); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n#Validation Max Fitness Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	if err := writeSeries(file, graph.EvaluationIndex, graph.ValidationMaxFitness); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(file, "\n\n#Validation Min Fitness Vs Evaluations, Morphology:%s\n", graph.Morphology); err != nil {
		return err
	}
	return writeSeries(file, graph.EvaluationIndex, graph.ValidationMinFitness)
}

func writeSeriesWithStd(file *os.File, index []int, values, std []float64) error {
	length := minInt(len(index), minInt(len(values), len(std)))
	for i := 0; i < length; i++ {
		if _, err := fmt.Fprintf(file, "%d %g %g\n", index[i], values[i], std[i]); err != nil {
			return err
		}
	}
	return nil
}

func writeSeries(file *os.File, index []int, values []float64) error {
	length := minInt(len(index), len(values))
	for i := 0; i < length; i++ {
		if _, err := fmt.Fprintf(file, "%d %g\n", index[i], values[i]); err != nil {
			return err
		}
	}
	return nil
}

func sanitizeGraphToken(value string) string {
	var b strings.Builder
	for _, r := range value {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			continue
		}
		b.WriteByte('_')
	}
	token := strings.Trim(b.String(), "_")
	if token == "" {
		return "unknown"
	}
	return token
}

func avgStd(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	avg, _ := nn.Avg(values)
	std, _ := nn.Std(values)
	return avg, std
}

func maxFloat(values []float64) float64 {
	max := values[0]
	for _, value := range values[1:] {
		if value > max {
			max = value
		}
	}
	return max
}

func minFloat(values []float64) float64 {
	min := values[0]
	for _, value := range values[1:] {
		if value < min {
			min = value
		}
	}
	return min
}

func maxOrZero(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	return maxFloat(values)
}

func minOrZero(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	return minFloat(values)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
