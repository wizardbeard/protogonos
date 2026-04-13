package scape

import (
	"context"
	"fmt"
	"math"
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
	if surface, _ := gtTrace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected gt step_input sensor surface, got %+v", gtTrace)
	}
	if surface, _ := gtTrace["control_surface"].(string); surface != "step_output" {
		t.Fatalf("expected gt step_output control surface, got %+v", gtTrace)
	}
	if width, _ := gtTrace["sensor_width"].(int); width <= 0 {
		t.Fatalf("expected gt step sensor width > 0, got %+v", gtTrace)
	}
	if validationMode, _ := validationTrace["mode"].(string); validationMode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	if surface, _ := validationTrace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected validation step_input sensor surface, got %+v", validationTrace)
	}
	if surface, _ := validationTrace["control_surface"].(string); surface != "step_output" {
		t.Fatalf("expected validation step_output control surface, got %+v", validationTrace)
	}
	if width, _ := validationTrace["sensor_width"].(int); width <= 0 {
		t.Fatalf("expected validation step sensor width > 0, got %+v", validationTrace)
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
	if surface, _ := trace["sensor_surface"].(string); surface != "core" {
		t.Fatalf("expected core sensor surface, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.GTSAPredictActuatorName {
		t.Fatalf("expected control surface %s, got %+v", protoio.GTSAPredictActuatorName, trace)
	}
	if width, _ := trace["sensor_width"].(int); width != 1 {
		t.Fatalf("expected core sensor width 1, got %+v", trace)
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
	if surface, _ := trace["sensor_surface"].(string); surface != "extended" {
		t.Fatalf("expected extended sensor surface, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.GTSAPredictActuatorName {
		t.Fatalf("expected control surface %s, got %+v", protoio.GTSAPredictActuatorName, trace)
	}
	if width, _ := trace["sensor_width"].(int); width != 4 {
		t.Fatalf("expected extended sensor width 4, got %+v", trace)
	}
}

func TestGTSAScapeEvaluateWithTickSensorsAndWriteOnlyActuator(t *testing.T) {
	agent := scriptedTickAgent{
		id: "gtsa-tick-write-only",
		sensors: map[string]protoio.Sensor{
			protoio.GTSAInputSensorName: protoio.NewScalarInputSensor(0),
		},
		actuators: map[string]protoio.Actuator{
			protoio.GTSAPredictActuatorName: &writeOnlyActuator{name: protoio.GTSAPredictActuatorName},
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			input, err := sensors[protoio.GTSAInputSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			if len(input) == 0 {
				return []float64{0}, nil
			}
			return []float64{input[0]}, nil
		},
	}

	scape := GTSAScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent with write-only actuator: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "core" {
		t.Fatalf("expected core sensor surface, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.GTSAPredictActuatorName {
		t.Fatalf("expected control surface %s, got %+v", protoio.GTSAPredictActuatorName, trace)
	}
	if width, _ := trace["sensor_width"].(int); width != 1 {
		t.Fatalf("expected core sensor width 1, got %+v", trace)
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
	if surface, _ := trace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected step_input sensor surface, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != "step_output" {
		t.Fatalf("expected step_output control surface, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width <= 0 {
		t.Fatalf("trace missing step sensor_width: %+v", trace)
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
	progress, pok := trace["last_progress"].(float64)
	if !pok {
		t.Fatalf("trace missing last_progress: %+v", trace)
	}
	wantProgress := float64(current-start) / float64(maxGTSA(1, end-start))
	if math.Abs(progress-wantProgress) > 1e-9 {
		t.Fatalf("expected last_progress=%f from final index window, got %+v", wantProgress, trace)
	}
}

func TestGTSAScapeCountsFinalScoredPrediction(t *testing.T) {
	cfg := gtsaModeConfig{
		mode:       "gt",
		startIndex: 1,
		scoreSteps: 1,
		windowRows: 1,
	}
	ctx := context.WithValue(context.Background(), gtsaDataSourceContextKey{}, gtsaTable{
		info: gtsaInfo{
			name:   "unit",
			ivl:    1,
			ovl:    1,
			trnEnd: 2,
			valEnd: 2,
			tstEnd: 2,
		},
		values: []float64{0, 2, 3},
	})

	fitness, trace, err := evaluateGTSA(
		ctx,
		cfg,
		func(_ context.Context, percept gtsaPercept) (float64, error) {
			return percept.current, nil
		},
	)
	if err != nil {
		t.Fatalf("evaluate unit gtsa: %v", err)
	}
	if steps, ok := trace["steps"].(int); !ok || steps != 1 {
		t.Fatalf("expected single scored step, got %+v", trace)
	}
	if mse, ok := trace["mse"].(float64); !ok || mse != 9 {
		t.Fatalf("expected final scored mse=9, got %+v", trace)
	}
	if mae, ok := trace["mae"].(float64); !ok || mae != 3 {
		t.Fatalf("expected final scored mae=3, got %+v", trace)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness from counted terminal prediction, got %f trace=%+v", fitness, trace)
	}
}

func TestGTSAScapeZeroScoreTraceUsesFinalProgress(t *testing.T) {
	cfg := gtsaModeConfig{
		mode:        "gt",
		startIndex:  1,
		warmupSteps: 1,
		scoreSteps:  0,
		windowRows:  1,
	}
	ctx := context.WithValue(context.Background(), gtsaDataSourceContextKey{}, gtsaTable{
		info: gtsaInfo{
			name:   "unit",
			ivl:    1,
			ovl:    1,
			trnEnd: 2,
			valEnd: 2,
			tstEnd: 2,
		},
		values: []float64{0, 2, 3},
	})

	_, trace, err := evaluateGTSA(
		ctx,
		cfg,
		func(_ context.Context, percept gtsaPercept) (float64, error) {
			return percept.current, nil
		},
	)
	if err != nil {
		t.Fatalf("evaluate zero-score gtsa: %v", err)
	}
	if warmup, ok := trace["warmup_steps"].(int); !ok || warmup != 1 {
		t.Fatalf("expected warmup_steps=1, got %+v", trace)
	}
	if steps, ok := trace["steps"].(int); !ok || steps != 0 {
		t.Fatalf("expected steps=0 for zero-score path, got %+v", trace)
	}
	start, sok := trace["index_start"].(int)
	current, cok := trace["index_current"].(int)
	end, eok := trace["index_end"].(int)
	if !sok || !cok || !eok {
		t.Fatalf("trace missing index window fields: %+v", trace)
	}
	progress, pok := trace["last_progress"].(float64)
	if !pok {
		t.Fatalf("trace missing last_progress: %+v", trace)
	}
	wantProgress := float64(current-start) / float64(maxGTSA(1, end-start))
	if math.Abs(progress-wantProgress) > 1e-9 {
		t.Fatalf("expected zero-score last_progress=%f from final index window, got %+v", wantProgress, trace)
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
