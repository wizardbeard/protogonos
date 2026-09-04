package scape

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestEpitopesScapeRewardsMemoryAwarePolicy(t *testing.T) {
	scape := EpitopesScape{}
	signalOnly := scriptedStepAgent{
		id: "signal-only",
		fn: func(in []float64) []float64 {
			if len(in) == 0 {
				return []float64{0}
			}
			return []float64{in[0]}
		},
	}
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	signalFitness, _, err := scape.Evaluate(context.Background(), signalOnly)
	if err != nil {
		t.Fatalf("evaluate signal-only: %v", err)
	}
	memoryFitness, _, err := scape.Evaluate(context.Background(), memoryAware)
	if err != nil {
		t.Fatalf("evaluate memory-aware: %v", err)
	}
	if memoryFitness <= signalFitness {
		t.Fatalf("expected memory-aware policy to outperform signal-only, got memory=%f signal=%f", memoryFitness, signalFitness)
	}
}

func TestEpitopesScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.EpitopesSignalSensorName,
			protoio.EpitopesMemorySensorName,
		},
		ActuatorIDs: []string{protoio.EpitopesResponseActuatorName},
		Neurons: []model.Neuron{
			{ID: "s", Activation: "identity"},
			{ID: "m", Activation: "identity"},
			{ID: "r", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "s", To: "r", Weight: 1.0, Enabled: true},
			{From: "m", To: "r", Weight: 0.7, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.EpitopesSignalSensorName: protoio.NewScalarInputSensor(0),
		protoio.EpitopesMemorySensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.EpitopesResponseActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("epitopes-agent-io", genome, sensors, actuators, []string{"s", "m"}, []string{"r"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := EpitopesScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness < 0.75 {
		t.Fatalf("expected fitness >= 0.75, got %f", fitness)
	}
	if _, ok := trace["accuracy"].(float64); !ok {
		t.Fatalf("trace missing accuracy: %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "core" {
		t.Fatalf("expected sensor_surface=core, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 2 {
		t.Fatalf("expected sensor_width=2, got %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface != protoio.EpitopesResponseActuatorName {
		t.Fatalf("expected control_surface=%s, got %+v", protoio.EpitopesResponseActuatorName, trace)
	}
}

func TestEpitopesScapeEvaluateWithExtendedIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.EpitopesSignalSensorName,
			protoio.EpitopesMemorySensorName,
			protoio.EpitopesTargetSensorName,
			protoio.EpitopesProgressSensorName,
			protoio.EpitopesMarginSensorName,
		},
		ActuatorIDs: []string{protoio.EpitopesResponseActuatorName},
		Neurons: []model.Neuron{
			{ID: "s", Activation: "identity"},
			{ID: "m", Activation: "identity"},
			{ID: "t", Activation: "identity"},
			{ID: "p", Activation: "identity"},
			{ID: "g", Activation: "identity"},
			{ID: "r", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "s", To: "r", Weight: 0.2, Enabled: true},
			{From: "m", To: "r", Weight: 0.2, Enabled: true},
			{From: "t", To: "r", Weight: 1.2, Enabled: true},
			{From: "p", To: "r", Weight: 0.05, Enabled: true},
			{From: "g", To: "r", Weight: 0.1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.EpitopesSignalSensorName:   protoio.NewScalarInputSensor(0),
		protoio.EpitopesMemorySensorName:   protoio.NewScalarInputSensor(0),
		protoio.EpitopesTargetSensorName:   protoio.NewScalarInputSensor(0),
		protoio.EpitopesProgressSensorName: protoio.NewScalarInputSensor(0),
		protoio.EpitopesMarginSensorName:   protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.EpitopesResponseActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"epitopes-agent-extended-io",
		genome,
		sensors,
		actuators,
		[]string{"s", "m", "t", "p", "g"},
		[]string{"r"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := EpitopesScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness < 0.9 {
		t.Fatalf("expected high fitness with target-aware channels, got %f", fitness)
	}
	if _, ok := trace["mean_target"].(float64); !ok {
		t.Fatalf("trace missing mean_target: %+v", trace)
	}
	if _, ok := trace["mean_progress"].(float64); !ok {
		t.Fatalf("trace missing mean_progress: %+v", trace)
	}
	if _, ok := trace["mean_decision_margin"].(float64); !ok {
		t.Fatalf("trace missing mean_decision_margin: %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "extended" {
		t.Fatalf("expected sensor_surface=extended, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 5 {
		t.Fatalf("expected sensor_width=5, got %+v", trace)
	}
}

func TestEpitopesScapeEvaluateWithTickSensorsAndNoActuatorSnapshot(t *testing.T) {
	agent := scriptedTickAgent{
		id: "epitopes-tick-no-snapshot",
		sensors: map[string]protoio.Sensor{
			protoio.EpitopesSignalSensorName: protoio.NewScalarInputSensor(0),
			protoio.EpitopesMemorySensorName: protoio.NewScalarInputSensor(0),
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			signal, err := sensors[protoio.EpitopesSignalSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			memory, err := sensors[protoio.EpitopesMemorySensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			signalValue := 0.0
			if len(signal) > 0 {
				signalValue = signal[0]
			}
			memoryValue := 0.0
			if len(memory) > 0 {
				memoryValue = memory[0]
			}
			return []float64{signalValue + 0.7*memoryValue}, nil
		},
	}

	scape := EpitopesScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent without snapshot actuator: %v", err)
	}
	if fitness < 0.75 {
		t.Fatalf("expected fitness >= 0.75, got %f", fitness)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "core" {
		t.Fatalf("expected sensor_surface=core, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 2 {
		t.Fatalf("expected sensor_width=2, got %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface != protoio.EpitopesResponseActuatorName {
		t.Fatalf("expected control_surface=%s, got %+v", protoio.EpitopesResponseActuatorName, trace)
	}
}

func TestEpitopesScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := EpitopesScape{}
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	_, trace, err := scape.EvaluateMode(context.Background(), memoryAware, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := trace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", trace)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected validation sensor_surface=step_input, got %+v", trace)
	}
	featureWidth, fok := trace["feature_width"].(int)
	if width, _ := trace["sensor_width"].(int); !fok || width != featureWidth {
		t.Fatalf("expected validation sensor_width to match feature_width, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != "step_output" {
		t.Fatalf("expected validation control_surface=step_output, got %+v", trace)
	}
	if startIndex, ok := trace["start_index"].(int); !ok || startIndex <= 0 {
		t.Fatalf("expected positive start_index in validation mode, got %+v", trace)
	}
}

func TestEpitopesScapeStepPerceptIncludesSequenceFeatures(t *testing.T) {
	scape := EpitopesScape{}
	maxPerceptWidth := 0
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) > maxPerceptWidth {
				maxPerceptWidth = len(in)
			}
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), memoryAware)
	if err != nil {
		t.Fatalf("evaluate memory-aware: %v", err)
	}
	if maxPerceptWidth <= 2 {
		t.Fatalf("expected step percept to include sequence features, got width=%d", maxPerceptWidth)
	}
	if width, ok := trace["feature_width"].(int); !ok || width <= 2 {
		t.Fatalf("expected feature_width > 2 in trace, got %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "step_input" {
		t.Fatalf("expected sensor_surface=step_input, got %+v", trace)
	}
	featureWidth, fok := trace["feature_width"].(int)
	if width, ok := trace["sensor_width"].(int); !ok || !fok || width != featureWidth {
		t.Fatalf("expected sensor_width to match feature_width, got %+v", trace)
	}
	if seqLen, ok := trace["sequence_length"].(int); !ok || seqLen <= 0 {
		t.Fatalf("expected positive sequence_length in trace, got %+v", trace)
	}
}

func TestEpitopesScapeTraceIncludesTableWindowState(t *testing.T) {
	scape := EpitopesScape{}
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	_, trace, err := scape.EvaluateMode(context.Background(), memoryAware, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark mode: %v", err)
	}
	if opMode, ok := trace["op_mode"].(string); !ok || opMode != "benchmark" {
		t.Fatalf("expected benchmark op_mode, got %+v", trace)
	}
	if _, ok := trace["table_name"].(string); !ok {
		t.Fatalf("trace missing table_name: %+v", trace)
	}
	start, sok := trace["start_index"].(int)
	end, eok := trace["end_index"].(int)
	if !sok || !eok || start <= 0 || end < start {
		t.Fatalf("trace missing or invalid window bounds: %+v", trace)
	}
	if total, ok := trace["total"].(int); !ok || total <= 0 {
		t.Fatalf("trace missing total sample count: %+v", trace)
	}
	if progress, ok := trace["mean_progress"].(float64); !ok || progress < 0 || progress > 1 {
		t.Fatalf("trace missing mean_progress in [0,1]: %+v", trace)
	}
	if current, ok := trace["index_current"].(int); !ok || current != 0 {
		t.Fatalf("expected halted benchmark session to report index_current=0, got %+v", trace)
	}
	if progress, _ := trace["mean_progress"].(float64); progress >= 1 {
		t.Fatalf("expected mean_progress to reflect post-classify session state, got %+v", trace)
	}
}

func TestEpitopesScapeLoadTableCSV(t *testing.T) {
	ResetEpitopesTableSource()
	t.Cleanup(ResetEpitopesTableSource)

	path := filepath.Join(t.TempDir(), "epitopes_custom.csv")
	var builder strings.Builder
	builder.WriteString("signal,memory,class\n")
	for i := 0; i < 220; i++ {
		signal := 0.7 * math.Sin(float64(i)*0.11)
		memory := 0.5 * math.Sin(float64(i-1)*0.11)
		classification := 0
		if signal+0.7*memory >= 0 {
			classification = 1
		}
		fmt.Fprintf(&builder, "%0.6f,%0.6f,%d\n", signal, memory, classification)
	}
	if err := os.WriteFile(path, []byte(builder.String()), 0o644); err != nil {
		t.Fatalf("write epitopes csv: %v", err)
	}

	if err := LoadEpitopesTableCSV(path, EpitopesTableBounds{
		GTStart:         1,
		GTEnd:           64,
		ValidationStart: 65,
		ValidationEnd:   96,
		TestStart:       97,
		TestEnd:         128,
		BenchmarkStart:  129,
		BenchmarkEnd:    220,
	}); err != nil {
		t.Fatalf("load epitopes csv: %v", err)
	}

	scape := EpitopesScape{}
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	fitness, trace, err := scape.EvaluateMode(context.Background(), memoryAware, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark mode: %v", err)
	}
	if fitness < 0.8 {
		t.Fatalf("expected high fitness on loaded epitopes table, got %f", fitness)
	}
	table, ok := trace["table_name"].(string)
	if !ok || !strings.Contains(table, "epitopes_custom.csv") {
		t.Fatalf("expected loaded table in trace, got %+v", trace)
	}
	start, sok := trace["start_index"].(int)
	end, eok := trace["end_index"].(int)
	if !sok || !eok || start != 129 || end != 220 {
		t.Fatalf("expected custom benchmark window 129..220, got %+v", trace)
	}
}

func TestEpitopesScapeLoadTableCSVRejectsInvalidBounds(t *testing.T) {
	ResetEpitopesTableSource()
	t.Cleanup(ResetEpitopesTableSource)

	path := filepath.Join(t.TempDir(), "epitopes_invalid.csv")
	if err := os.WriteFile(path, []byte("0.1,0.0,1\n0.2,0.1,1\n0.3,0.2,1\n"), 0o644); err != nil {
		t.Fatalf("write epitopes csv: %v", err)
	}

	err := LoadEpitopesTableCSV(path, EpitopesTableBounds{
		GTStart: 4,
		GTEnd:   3,
	})
	if err == nil {
		t.Fatal("expected invalid bounds error")
	}

	scape := EpitopesScape{}
	memoryAware := scriptedStepAgent{
		id: "memory-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{in[0] + 0.7*in[1]}
		},
	}

	_, trace, evalErr := scape.Evaluate(context.Background(), memoryAware)
	if evalErr != nil {
		t.Fatalf("evaluate default epitopes table: %v", evalErr)
	}
	if table, _ := trace["table_name"].(string); table != "abc_pred16" {
		t.Fatalf("expected default epitopes table after rejected load, got %+v", trace)
	}
}

func TestSelectEpitopesTableSourceSwitchesBuiltInTable(t *testing.T) {
	ResetEpitopesTableSource()
	t.Cleanup(ResetEpitopesTableSource)

	if err := SelectEpitopesTableSource("abc_pred10", EpitopesTableBounds{}); err != nil {
		t.Fatalf("select epitopes table source: %v", err)
	}

	scape := EpitopesScape{}
	copyInput := scriptedStepAgent{
		id: "copy",
		fn: func(in []float64) []float64 {
			if len(in) == 0 {
				return []float64{0}
			}
			return []float64{in[0]}
		},
	}
	_, trace, err := scape.EvaluateMode(context.Background(), copyInput, "benchmark")
	if err != nil {
		t.Fatalf("evaluate selected epitopes table: %v", err)
	}
	if table, _ := trace["table_name"].(string); table != "abc_pred10" {
		t.Fatalf("expected selected table abc_pred10, got %+v", trace)
	}
	if seqLen, _ := trace["sequence_length"].(int); seqLen != 10 {
		t.Fatalf("expected selected sequence_length=10, got %+v", trace)
	}
}

func TestSelectEpitopesTableSourceRejectsUnknownTable(t *testing.T) {
	if err := SelectEpitopesTableSource("abc_pred999", EpitopesTableBounds{}); err == nil {
		t.Fatal("expected unknown built-in table error")
	}
}

func TestAvailableEpitopesTableNamesIncludesReferenceDefaults(t *testing.T) {
	got := AvailableEpitopesTableNames()
	want := []string{"abc_pred10", "abc_pred12", "abc_pred14", "abc_pred16", "abc_pred18", "abc_pred20"}
	if len(got) != len(want) {
		t.Fatalf("unexpected built-in epitopes table count: got=%d want=%d tables=%v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected built-in epitopes table order: got=%v want=%v", got, want)
		}
	}
}

func TestEpitopesTableDBLifecycle(t *testing.T) {
	if stopped := StopEpitopesTableDB(); stopped {
		t.Cleanup(func() {
			_ = StopEpitopesTableDB()
		})
	}
	db := StartEpitopesTableDB()
	if db == nil {
		t.Fatal("expected non-nil epitopes table db")
	}
	if !db.Running() {
		t.Fatal("expected running epitopes table db after start")
	}
	if _, ok := DefaultEpitopesTableDB(); !ok {
		t.Fatal("expected default epitopes table db to be registered")
	}
	if len(AvailableEpitopesTableNames()) == 0 {
		t.Fatal("expected built-in table catalog to be loaded")
	}

	db.Terminate()
	select {
	case <-db.Done():
	case <-time.After(250 * time.Millisecond):
		t.Fatal("expected epitopes table db termination to complete")
	}
	if db.Running() {
		t.Fatal("expected epitopes table db to be stopped")
	}
	if _, ok := DefaultEpitopesTableDB(); ok {
		t.Fatal("expected default epitopes table db to be cleared")
	}
}

func TestStartEpitopesTableDBIsIdempotentWhileRunning(t *testing.T) {
	if stopped := StopEpitopesTableDB(); stopped {
		t.Cleanup(func() {
			_ = StopEpitopesTableDB()
		})
	}
	first := StartEpitopesTableDB()
	if first == nil {
		t.Fatal("expected first db process")
	}
	second := StartEpitopesTableDB()
	if second == nil {
		t.Fatal("expected second db process")
	}
	if first != second {
		t.Fatal("expected start to reuse the running default epitopes db process")
	}
	first.Terminate()
	<-first.Done()
}

func TestEpitopesSimulatorSenseClassifyFlowResetsAfterHalt(t *testing.T) {
	sim, err := NewEpitopesSimulator(context.Background(), "gt", EpitopesSimParameters{
		TableName:  "abc_pred16",
		StartIndex: 1,
		EndIndex:   2,
	})
	if err != nil {
		t.Fatalf("new epitopes simulator: %v", err)
	}

	state := sim.State()
	if state.StartIndex != 1 || state.EndIndex != 2 || state.IndexCurrent != 0 || state.Halted {
		t.Fatalf("unexpected initial simulator state: %+v", state)
	}

	if _, err := sim.Sense(); err != nil {
		t.Fatalf("sense #1: %v", err)
	}
	if reward, halt, err := sim.Classify([]float64{1}); err != nil {
		t.Fatalf("classify #1: %v", err)
	} else if halt {
		t.Fatalf("expected halt=false on first classify, reward=%d", reward)
	}
	afterFirst := sim.State()
	if afterFirst.ClassificationCount != 1 || afterFirst.LastPredictedClass != 1 || afterFirst.LastClassifiedIndex != 1 {
		t.Fatalf("expected first classification diagnostics, got %+v", afterFirst)
	}
	if afterFirst.LastExpectedClass != afterFirst.LastReward || afterFirst.Accuracy != float64(afterFirst.CorrectCount) {
		t.Fatalf("unexpected first classification score state: %+v", afterFirst)
	}
	if _, err := sim.Sense(); err != nil {
		t.Fatalf("sense #2: %v", err)
	}
	if _, halt, err := sim.Classify([]float64{1}); err != nil {
		t.Fatalf("classify #2: %v", err)
	} else if !halt {
		t.Fatal("expected halt=true on second classify at end index")
	}

	haltedState := sim.State()
	if !haltedState.Halted || haltedState.IndexCurrent != 0 {
		t.Fatalf("expected halted simulator state after end index classify, got %+v", haltedState)
	}
	if haltedState.TerminationReason != "end_index" || haltedState.ClassificationCount != 2 || haltedState.LastClassifiedIndex != 2 {
		t.Fatalf("expected terminal classification diagnostics, got %+v", haltedState)
	}

	if _, err := sim.Sense(); err != nil {
		t.Fatalf("sense after halt reset: %v", err)
	}
	resetState := sim.State()
	if resetState.Halted || resetState.IndexCurrent != 1 {
		t.Fatalf("expected sense to reset simulator index to start, got %+v", resetState)
	}
	if resetState.TerminationReason != "" || resetState.ClassificationCount != 0 || resetState.CorrectCount != 0 {
		t.Fatalf("expected halt-reset sense to clear diagnostics, got %+v", resetState)
	}
}

func TestEpitopesSimulatorBenchmarkUsesBenchmarkWindow(t *testing.T) {
	sim, err := NewEpitopesSimulator(context.Background(), "benchmark", EpitopesSimParameters{
		TableName:           "abc_pred16",
		StartIndex:          1,
		EndIndex:            64,
		StartBenchmarkIndex: 129,
		EndBenchmarkIndex:   130,
	})
	if err != nil {
		t.Fatalf("new epitopes simulator benchmark: %v", err)
	}

	state := sim.State()
	if state.StartIndex != 129 || state.EndIndex != 130 || state.OpMode != "benchmark" {
		t.Fatalf("unexpected benchmark simulator window state: %+v", state)
	}
	if _, err := sim.Sense(); err != nil {
		t.Fatalf("benchmark sense: %v", err)
	}
	if state := sim.State(); state.IndexCurrent != 129 {
		t.Fatalf("expected benchmark sense to start at benchmark window index, got %+v", state)
	}
}

func TestNewEpitopesSimulatorRejectsUnknownTable(t *testing.T) {
	if _, err := NewEpitopesSimulator(context.Background(), "gt", EpitopesSimParameters{
		TableName: "abc_pred999",
	}); err == nil {
		t.Fatal("expected unknown epitopes simulator table error")
	}
}

func TestEpitopesProcessCommandWrapper(t *testing.T) {
	process := NewEpitopesProcess()
	ctx := context.Background()
	params := EpitopesSimParameters{
		TableName:  "abc_pred16",
		StartIndex: 1,
		EndIndex:   2,
	}

	if response := process.Call(ctx, EpitopesSenseMessage{}); response.Err == nil {
		t.Fatal("expected sense before start to fail")
	}
	start := process.Call(ctx, EpitopesStartMessage{OpMode: "gt", Params: params})
	if start.Err != nil || !start.OK {
		t.Fatalf("start response=%+v", start)
	}
	if start.State.OpMode != "gt" || start.State.TableName != "abc_pred16" || start.State.Halted {
		t.Fatalf("unexpected start state=%+v", start.State)
	}

	sense := process.Call(ctx, EpitopesSenseMessage{})
	if sense.Err != nil || !sense.OK {
		t.Fatalf("sense response=%+v", sense)
	}
	if len(sense.Percept) == 0 || sense.State.IndexCurrent != 1 {
		t.Fatalf("unexpected sense response=%+v", sense)
	}

	classify := process.Call(ctx, EpitopesClassifyMessage{Output: []float64{1}})
	if classify.Err != nil || !classify.OK {
		t.Fatalf("classify response=%+v", classify)
	}
	if classify.End {
		t.Fatalf("expected first classify to remain active, got %+v", classify)
	}
	if classify.State.ClassificationCount != 1 || classify.State.LastPredictedClass != 1 || classify.State.LastClassifiedIndex != 1 {
		t.Fatalf("expected process classification diagnostics, got %+v", classify.State)
	}

	state := process.Call(ctx, EpitopesStateMessage{})
	if state.Err != nil || !state.OK || state.State.IndexCurrent != 2 {
		t.Fatalf("state response=%+v", state)
	}

	stop := process.Call(ctx, EpitopesStopMessage{Reason: "normal"})
	if stop.Err != nil || !stop.OK || !stop.End {
		t.Fatalf("stop response=%+v", stop)
	}
	if stop.StopReason != "normal" || !stop.State.Halted {
		t.Fatalf("unexpected stop response=%+v", stop)
	}
	if stop.State.TerminationReason != "normal" {
		t.Fatalf("expected stop reason in state, got %+v", stop.State)
	}
	if response := process.Call(ctx, EpitopesSenseMessage{}); response.Err == nil {
		t.Fatal("expected sense after stop to fail")
	}

	restart := process.Call(ctx, EpitopesRestartMessage{})
	if restart.Err != nil || !restart.OK {
		t.Fatalf("restart response=%+v", restart)
	}
	if restart.State.Halted || restart.State.OpMode != "gt" || restart.State.IndexCurrent != 0 {
		t.Fatalf("unexpected restart response=%+v", restart)
	}
	if restart.State.TerminationReason != "" || restart.State.ClassificationCount != 0 {
		t.Fatalf("expected restart to clear process diagnostics, got %+v", restart.State)
	}
}

func TestEpitopesProcessIOAdapters(t *testing.T) {
	sensorIDs := []string{
		protoio.EpitopesSignalSensorName,
		protoio.EpitopesMemorySensorName,
		protoio.EpitopesTargetSensorName,
		protoio.EpitopesProgressSensorName,
		protoio.EpitopesMarginSensorName,
	}
	actuatorIDs := []string{protoio.EpitopesResponseActuatorName}

	sensors, actuators, err := NewEpitopesProcessIO("gt", sensorIDs, actuatorIDs)
	if err != nil {
		t.Fatalf("new epitopes process io: %v", err)
	}
	values := make(map[string]float64, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		reader, ok := sensors[sensorID].(protoio.SensorProcessReader)
		if !ok {
			t.Fatalf("sensor %s does not expose process reader", sensorID)
		}
		value, err := reader.ReadForSensorProcess(context.Background(), protoio.SensorProcessCall{
			Scape:      "epitopes",
			SensorName: protoio.ABCPredSensorAliasName,
			VL:         1,
			OpMode:     "gt",
		})
		if err != nil {
			t.Fatalf("read process sensor %s: %v", sensorID, err)
		}
		if len(value) != 1 {
			t.Fatalf("expected one scalar from %s, got %+v", sensorID, value)
		}
		values[sensorID] = value[0]
	}
	margin := values[protoio.EpitopesSignalSensorName] + 0.7*values[protoio.EpitopesMemorySensorName]
	if math.Abs(values[protoio.EpitopesMarginSensorName]-margin) > 1e-9 {
		t.Fatalf("expected derived margin %f, got %f", margin, values[protoio.EpitopesMarginSensorName])
	}

	writer, ok := actuators[protoio.EpitopesResponseActuatorName].(protoio.ActuatorProcessWriter)
	if !ok {
		t.Fatal("epitopes response actuator does not expose process writer")
	}
	sync, err := writer.WriteForActuatorProcess(context.Background(), protoio.ActuatorProcessCall{
		Scape:        "epitopes",
		ActuatorName: protoio.ABCPredActuatorAliasName,
		OpMode:       "gt",
		Output:       []float64{values[protoio.EpitopesTargetSensorName]},
	})
	if err != nil {
		t.Fatalf("write process actuator: %v", err)
	}
	if sync.EndFlag != 0 || len(sync.Fitness) != 1 || sync.Fitness[0] != 1 {
		t.Fatalf("expected non-terminal correct classification sync, got %+v", sync)
	}
}

func TestEpitopesProcessIOStartConfigUsesModeWindows(t *testing.T) {
	source := currentEpitopesSource(context.Background())
	opMode, params := epitopesProcessStartConfig("validation")
	if opMode != "benchmark" {
		t.Fatalf("expected validation to use benchmark simulator mode, got %s", opMode)
	}
	if params.StartBenchmarkIndex != source.windows.validationStart || params.EndBenchmarkIndex != source.windows.validationEnd {
		t.Fatalf("expected validation benchmark bounds %+v, got %+v", source.windows, params)
	}

	opMode, params = epitopesProcessStartConfig("test")
	if opMode != "benchmark" {
		t.Fatalf("expected test to use benchmark simulator mode, got %s", opMode)
	}
	if params.StartBenchmarkIndex != source.windows.testStart || params.EndBenchmarkIndex != source.windows.testEnd {
		t.Fatalf("expected test benchmark bounds %+v, got %+v", source.windows, params)
	}
}

func TestEpitopesScapeEvaluateWithActorProcessIO(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.EpitopesSignalSensorName,
			protoio.EpitopesMemorySensorName,
			protoio.EpitopesTargetSensorName,
			protoio.EpitopesProgressSensorName,
			protoio.EpitopesMarginSensorName,
		},
		ActuatorIDs: []string{protoio.EpitopesResponseActuatorName},
		Neurons: []model.Neuron{
			{ID: "s", Activation: "identity"},
			{ID: "m", Activation: "identity"},
			{ID: "t", Activation: "identity"},
			{ID: "p", Activation: "identity"},
			{ID: "g", Activation: "identity"},
			{ID: "r", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "s", To: "r", Weight: 0.2, Enabled: true},
			{From: "m", To: "r", Weight: 0.2, Enabled: true},
			{From: "t", To: "r", Weight: 1.2, Enabled: true},
			{From: "p", To: "r", Weight: 0.05, Enabled: true},
			{From: "g", To: "r", Weight: 0.1, Enabled: true},
		},
	}

	sensors, actuators, err := NewEpitopesProcessIO("gt", genome.SensorIDs, genome.ActuatorIDs)
	if err != nil {
		t.Fatalf("new epitopes process io: %v", err)
	}
	cortex, err := agent.NewCortex(
		"epitopes-agent-actor-process-io",
		genome,
		sensors,
		actuators,
		[]string{"s", "m", "t", "p", "g"},
		[]string{"r"},
		nil,
		agent.WithIOProcessContext("epitopes", "gt"),
		agent.WithIOActors(),
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	t.Cleanup(cortex.Terminate)

	fitness, trace, err := EpitopesScape{}.EvaluateMode(context.Background(), cortex, "gt")
	if err != nil {
		t.Fatalf("evaluate actor process io: %v", err)
	}
	if fitness < 0.9 {
		t.Fatalf("expected high actor-process epitopes fitness, got %f trace=%+v", fitness, trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "extended" {
		t.Fatalf("expected extended epitopes sensor surface, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 5 {
		t.Fatalf("expected epitopes sensor_width=5, got %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface != protoio.EpitopesResponseActuatorName {
		t.Fatalf("expected control_surface=%s, got %+v", protoio.EpitopesResponseActuatorName, trace)
	}
}
