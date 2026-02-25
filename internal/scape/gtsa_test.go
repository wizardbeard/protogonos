package scape

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestGTSAScapeScoresBetterForSignalAwarePolicy(t *testing.T) {
	scape := GTSAScape{}
	zero := scriptedStepAgent{
		id: "zero",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	zeroFitness, _, err := scape.Evaluate(context.Background(), zero)
	if err != nil {
		t.Fatalf("evaluate zero: %v", err)
	}
	copyFitness, _, err := scape.Evaluate(context.Background(), copyInput)
	if err != nil {
		t.Fatalf("evaluate copy: %v", err)
	}
	if copyFitness <= zeroFitness {
		t.Fatalf("expected signal-aware policy to outperform zero, got copy=%f zero=%f", copyFitness, zeroFitness)
	}
}

func TestGTSAScapeEvaluateModeUsesConfiguredWindow(t *testing.T) {
	scape := GTSAScape{}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	gtFitness, gtTrace, err := scape.EvaluateMode(context.Background(), copyInput, "gt")
	if err != nil {
		t.Fatalf("evaluate gt mode: %v", err)
	}
	validationFitness, validationTrace, err := scape.EvaluateMode(context.Background(), copyInput, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if gtMode, _ := gtTrace["mode"].(string); gtMode != "gt" {
		t.Fatalf("expected gt mode trace marker, got %+v", gtTrace)
	}
	if validationMode, _ := validationTrace["mode"].(string); validationMode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	if gtFitness == validationFitness {
		t.Fatalf("expected mode windows to produce distinct fitness values, got gt=%f validation=%f", gtFitness, validationFitness)
	}
}

func TestGTSAScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.GTSAInputSensorName},
		ActuatorIDs: []string{protoio.GTSAPredictActuatorName},
		Neurons: []model.Neuron{
			{ID: "input", Activation: "identity"},
			{ID: "predict", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "input", To: "predict", Weight: 1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.GTSAInputSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.GTSAPredictActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"gtsa-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"input"},
		[]string{"predict"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := GTSAScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["mse"].(float64); !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
}

func TestGTSAScapeEvaluateWithExtendedIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.GTSAInputSensorName,
			protoio.GTSADeltaSensorName,
			protoio.GTSAWindowMeanSensorName,
			protoio.GTSAProgressSensorName,
		},
		ActuatorIDs: []string{protoio.GTSAPredictActuatorName},
		Neurons: []model.Neuron{
			{ID: "input", Activation: "identity"},
			{ID: "delta", Activation: "identity"},
			{ID: "mean", Activation: "identity"},
			{ID: "progress", Activation: "identity"},
			{ID: "predict", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "input", To: "predict", Weight: 0.7, Enabled: true},
			{From: "delta", To: "predict", Weight: 0.2, Enabled: true},
			{From: "mean", To: "predict", Weight: 0.2, Enabled: true},
			{From: "progress", To: "predict", Weight: -0.1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.GTSAInputSensorName:      protoio.NewScalarInputSensor(0),
		protoio.GTSADeltaSensorName:      protoio.NewScalarInputSensor(0),
		protoio.GTSAWindowMeanSensorName: protoio.NewScalarInputSensor(0),
		protoio.GTSAProgressSensorName:   protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.GTSAPredictActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"gtsa-agent-extended-io",
		genome,
		sensors,
		actuators,
		[]string{"input", "delta", "mean", "progress"},
		[]string{"predict"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := GTSAScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["mean_abs_delta"].(float64); !ok {
		t.Fatalf("trace missing mean_abs_delta: %+v", trace)
	}
	if _, ok := trace["last_progress"].(float64); !ok {
		t.Fatalf("trace missing last_progress: %+v", trace)
	}
}

func TestGTSAScapeTraceIncludesPredictionDiagnostics(t *testing.T) {
	scape := GTSAScape{}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	_, trace, err := scape.EvaluateMode(context.Background(), copyInput, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if _, ok := trace["mae"].(float64); !ok {
		t.Fatalf("trace missing mae: %+v", trace)
	}
	accuracy, ok := trace["direction_accuracy"].(float64)
	if !ok {
		t.Fatalf("trace missing direction_accuracy: %+v", trace)
	}
	if accuracy < 0 || accuracy > 1 {
		t.Fatalf("direction_accuracy out of range: %f", accuracy)
	}
}

func TestGTSAScapeTraceIncludesTableWindowState(t *testing.T) {
	scape := GTSAScape{}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	_, trace, err := scape.EvaluateMode(context.Background(), copyInput, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if table, ok := trace["table_name"].(string); !ok || table == "" {
		t.Fatalf("trace missing table_name: %+v", trace)
	}
	if rows, ok := trace["window_rows"].(int); !ok || rows <= 0 {
		t.Fatalf("trace missing window_rows: %+v", trace)
	}
	if length, ok := trace["window_length"].(int); !ok || length <= 0 {
		t.Fatalf("trace missing window_length: %+v", trace)
	}
	if width, ok := trace["feature_width"].(int); !ok || width <= 0 {
		t.Fatalf("trace missing feature_width: %+v", trace)
	}
	start, sok := trace["index_start"].(int)
	current, cok := trace["index_current"].(int)
	end, eok := trace["index_end"].(int)
	if !sok || !cok || !eok {
		t.Fatalf("trace missing index window fields: %+v", trace)
	}
	if start <= 0 || end < start {
		t.Fatalf("invalid index window in trace: %+v", trace)
	}
	if current < start || current > end {
		t.Fatalf("index_current out of bounds in trace: %+v", trace)
	}
}

func TestGTSAScapeLoadTableCSV(t *testing.T) {
	ResetGTSATableSource()
	t.Cleanup(ResetGTSATableSource)

	path := filepath.Join(t.TempDir(), "gtsa_custom.csv")
	var builder strings.Builder
	builder.WriteString("t,value\n")
	for i := 0; i < 96; i++ {
		fmt.Fprintf(&builder, "%d,%0.8f\n", i, gtsaSeries(i)+0.03)
	}
	if err := os.WriteFile(path, []byte(builder.String()), 0o644); err != nil {
		t.Fatalf("write csv: %v", err)
	}

	if err := LoadGTSATableCSV(path, GTSATableBounds{
		TrainEnd:      24,
		ValidationEnd: 48,
		TestEnd:       96,
	}); err != nil {
		t.Fatalf("load csv table: %v", err)
	}

	scape := GTSAScape{}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	_, trace, err := scape.EvaluateMode(context.Background(), copyInput, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	tableName, ok := trace["table_name"].(string)
	if !ok || !strings.Contains(tableName, "gtsa_custom.csv") {
		t.Fatalf("expected loaded csv table in trace, got %+v", trace)
	}
	start, sok := trace["index_start"].(int)
	end, eok := trace["index_end"].(int)
	if !sok || !eok || start != 49 || end != 96 {
		t.Fatalf("expected csv bounds in trace start=49 end=96, got %+v", trace)
	}
}

func TestGTSAScapeLoadTableCSVRejectsInvalidBounds(t *testing.T) {
	ResetGTSATableSource()
	t.Cleanup(ResetGTSATableSource)

	path := filepath.Join(t.TempDir(), "gtsa_invalid.csv")
	if err := os.WriteFile(path, []byte("1\n2\n3\n4\n5\n"), 0o644); err != nil {
		t.Fatalf("write csv: %v", err)
	}

	err := LoadGTSATableCSV(path, GTSATableBounds{
		TrainEnd:      4,
		ValidationEnd: 3,
		TestEnd:       5,
	})
	if err == nil {
		t.Fatal("expected invalid bounds error")
	}

	scape := GTSAScape{}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(input []float64) []float64 {
			return []float64{input[0]}
		},
	}

	_, trace, evalErr := scape.EvaluateMode(context.Background(), copyInput, "gt")
	if evalErr != nil {
		t.Fatalf("evaluate gt mode: %v", evalErr)
	}
	if table, _ := trace["table_name"].(string); table != "gtsa.synthetic.v2" {
		t.Fatalf("expected default table after rejected load, got %+v", trace)
	}
}
