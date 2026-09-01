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

func TestFXScapeRewardsSignalFollowingPolicy(t *testing.T) {
	scape := FXScape{}
	flat := scriptedStepAgent{
		id: "flat",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	follow := scriptedStepAgent{
		id: "follow",
		fn: fxFollowSignalAction,
	}

	flatFitness, _, err := scape.Evaluate(context.Background(), flat)
	if err != nil {
		t.Fatalf("evaluate flat: %v", err)
	}
	followFitness, _, err := scape.Evaluate(context.Background(), follow)
	if err != nil {
		t.Fatalf("evaluate follow: %v", err)
	}
	if followFitness <= flatFitness {
		t.Fatalf("expected signal-following strategy to outperform flat, got follow=%f flat=%f", followFitness, flatFitness)
	}
}

func TestFXScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := FXScape{}
	follow := scriptedStepAgent{
		id: "follow",
		fn: fxFollowSignalAction,
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), follow, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	if surface, _ := validationTrace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected validation step_input sensor surface, got %+v", validationTrace)
	}
	if surface, _ := validationTrace["control_surface"].(string); surface != "step_output" {
		t.Fatalf("expected validation step_output control surface, got %+v", validationTrace)
	}
	if width, _ := validationTrace["sensor_width"].(int); width <= 2 {
		t.Fatalf("expected validation step sensor width > 2, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), follow, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	if surface, _ := testTrace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected test step_input sensor surface, got %+v", testTrace)
	}
	if surface, _ := testTrace["control_surface"].(string); surface != "step_output" {
		t.Fatalf("expected test step_output control surface, got %+v", testTrace)
	}
	if width, _ := testTrace["sensor_width"].(int); width <= 2 {
		t.Fatalf("expected test step sensor width > 2, got %+v", testTrace)
	}
}

func TestFXScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FXPriceSensorName,
			protoio.FXSignalSensorName,
		},
		ActuatorIDs: []string{protoio.FXTradeActuatorName},
		Neurons: []model.Neuron{
			{ID: "price", Activation: "identity"},
			{ID: "signal", Activation: "identity"},
			{ID: "trade", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "signal", To: "trade", Weight: 1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FXPriceSensorName:  protoio.NewScalarInputSensor(0),
		protoio.FXSignalSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FXTradeActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"fx-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"price", "signal"},
		[]string{"trade"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := FXScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["equity"].(float64); !ok {
		t.Fatalf("trace missing equity: %+v", trace)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "market" {
		t.Fatalf("expected market sensor surface, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FXTradeActuatorName {
		t.Fatalf("expected control surface %s, got %+v", protoio.FXTradeActuatorName, trace)
	}
	if width, _ := trace["sensor_width"].(int); width != 2 {
		t.Fatalf("expected market sensor width 2, got %+v", trace)
	}
}

func TestFXScapeEvaluateWithExtendedIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FXPriceSensorName,
			protoio.FXSignalSensorName,
			protoio.FXMomentumSensorName,
			protoio.FXVolatilitySensorName,
			protoio.FXNAVSensorName,
			protoio.FXDrawdownSensorName,
			protoio.FXPositionSensorName,
			protoio.FXEntrySensorName,
			protoio.FXPercentChangeSensorName,
			protoio.FXPrevPercentChangeSensorName,
			protoio.FXProfitSensorName,
		},
		ActuatorIDs: []string{protoio.FXTradeActuatorName},
		Neurons: []model.Neuron{
			{ID: "price", Activation: "identity"},
			{ID: "signal", Activation: "identity"},
			{ID: "mom", Activation: "identity"},
			{ID: "vol", Activation: "identity"},
			{ID: "nav", Activation: "identity"},
			{ID: "dd", Activation: "identity"},
			{ID: "pos", Activation: "identity"},
			{ID: "entry", Activation: "identity"},
			{ID: "pc", Activation: "identity"},
			{ID: "ppc", Activation: "identity"},
			{ID: "profit", Activation: "identity"},
			{ID: "trade", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "signal", To: "trade", Weight: 1.1, Enabled: true},
			{From: "mom", To: "trade", Weight: 0.8, Enabled: true},
			{From: "vol", To: "trade", Weight: -0.5, Enabled: true},
			{From: "nav", To: "trade", Weight: 0.25, Enabled: true},
			{From: "dd", To: "trade", Weight: -0.6, Enabled: true},
			{From: "pos", To: "trade", Weight: 0.15, Enabled: true},
			{From: "entry", To: "trade", Weight: -0.2, Enabled: true},
			{From: "pc", To: "trade", Weight: 0.45, Enabled: true},
			{From: "ppc", To: "trade", Weight: 0.35, Enabled: true},
			{From: "profit", To: "trade", Weight: 0.35, Enabled: true},
			{From: "price", To: "trade", Weight: 0.2, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FXPriceSensorName:             protoio.NewScalarInputSensor(0),
		protoio.FXSignalSensorName:            protoio.NewScalarInputSensor(0),
		protoio.FXMomentumSensorName:          protoio.NewScalarInputSensor(0),
		protoio.FXVolatilitySensorName:        protoio.NewScalarInputSensor(0),
		protoio.FXNAVSensorName:               protoio.NewScalarInputSensor(0),
		protoio.FXDrawdownSensorName:          protoio.NewScalarInputSensor(0),
		protoio.FXPositionSensorName:          protoio.NewScalarInputSensor(0),
		protoio.FXEntrySensorName:             protoio.NewScalarInputSensor(0),
		protoio.FXPercentChangeSensorName:     protoio.NewScalarInputSensor(0),
		protoio.FXPrevPercentChangeSensorName: protoio.NewScalarInputSensor(0),
		protoio.FXProfitSensorName:            protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FXTradeActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"fx-agent-io-extended",
		genome,
		sensors,
		actuators,
		[]string{"price", "signal", "mom", "vol", "nav", "dd", "pos", "entry", "pc", "ppc", "profit"},
		[]string{"trade"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := FXScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if width, ok := trace["feature_width"].(int); !ok || width < 10 {
		t.Fatalf("expected extended feature width in trace, got %+v", trace)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "extended" {
		t.Fatalf("expected extended sensor surface, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FXTradeActuatorName {
		t.Fatalf("expected control surface %s, got %+v", protoio.FXTradeActuatorName, trace)
	}
	if width, _ := trace["sensor_width"].(int); width != 11 {
		t.Fatalf("expected extended sensor width 11, got %+v", trace)
	}
}

func TestFXScapeEvaluateWithTickSensorsAndNoActuatorSnapshot(t *testing.T) {
	agent := scriptedTickAgent{
		id: "fx-tick-no-snapshot",
		sensors: map[string]protoio.Sensor{
			protoio.FXPriceSensorName:  protoio.NewScalarInputSensor(0),
			protoio.FXSignalSensorName: protoio.NewScalarInputSensor(0),
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			signal, err := sensors[protoio.FXSignalSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			if len(signal) == 0 {
				return []float64{0}, nil
			}
			return []float64{signal[0]}, nil
		},
	}

	scape := FXScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent without snapshot actuator: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "market" {
		t.Fatalf("expected market sensor surface, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FXTradeActuatorName {
		t.Fatalf("expected control surface %s, got %+v", protoio.FXTradeActuatorName, trace)
	}
	if width, _ := trace["sensor_width"].(int); width != 2 {
		t.Fatalf("expected market sensor width 2, got %+v", trace)
	}
}

func TestFXScapeTraceIncludesAccountLifecycle(t *testing.T) {
	scape := FXScape{}
	follow := scriptedStepAgent{
		id: "follow",
		fn: fxFollowSignalAction,
	}

	_, trace, err := scape.Evaluate(context.Background(), follow)
	if err != nil {
		t.Fatalf("evaluate follow: %v", err)
	}
	if _, ok := trace["net_worth"].(float64); !ok {
		t.Fatalf("trace missing net_worth: %+v", trace)
	}
	if opened, ok := trace["orders_opened"].(int); !ok || opened <= 0 {
		t.Fatalf("expected positive orders_opened in trace, got %+v", trace)
	}
	if _, ok := trace["realized_pl"].(float64); !ok {
		t.Fatalf("trace missing realized_pl: %+v", trace)
	}
	if _, ok := trace["margin_call"].(bool); !ok {
		t.Fatalf("trace missing margin_call flag: %+v", trace)
	}
	if _, ok := trace["percentage_change"].(float64); !ok {
		t.Fatalf("trace missing percentage_change: %+v", trace)
	}
	if _, ok := trace["prev_percentage_change"].(float64); !ok {
		t.Fatalf("trace missing prev_percentage_change: %+v", trace)
	}
	if _, ok := trace["profit"].(float64); !ok {
		t.Fatalf("trace missing profit: %+v", trace)
	}
}

func TestFXScapeTraceStepsReflectEarlyMarginCall(t *testing.T) {
	ResetFXSeriesSource()
	t.Cleanup(ResetFXSeriesSource)

	fxSeriesSourceMu.Lock()
	fxSeriesSource = fxSeries{
		name: "fx.test.crash",
		values: []float64{
			1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
		},
	}
	fxSeriesSourceMu.Unlock()

	scape := FXScape{}
	longOnly := scriptedStepAgent{
		id: "long-only",
		fn: func(_ []float64) []float64 { return []float64{1} },
	}

	_, trace, err := scape.Evaluate(context.Background(), longOnly)
	if err != nil {
		t.Fatalf("evaluate long-only: %v", err)
	}
	if marginCall, _ := trace["margin_call"].(bool); !marginCall {
		t.Fatalf("expected forced margin call, got %+v", trace)
	}
	steps, ok := trace["steps"].(int)
	if !ok {
		t.Fatalf("trace missing steps: %+v", trace)
	}
	if steps <= 0 || steps >= 64 {
		t.Fatalf("expected early-terminated step count, got %+v", trace)
	}
	if position, ok := trace["position"].(float64); !ok || position != 0 {
		t.Fatalf("expected margin call to liquidate open position, got %+v", trace)
	}
	if closed, ok := trace["orders_closed"].(int); !ok || closed <= 0 {
		t.Fatalf("expected margin call liquidation to count as closed order, got %+v", trace)
	}
}

func TestFXScapeStepPerceptIncludesMarketInternals(t *testing.T) {
	scape := FXScape{}
	maxPerceptWidth := 0
	maxPrevPercentChange := 0.0
	follow := scriptedStepAgent{
		id: "follow",
		fn: func(input []float64) []float64 {
			if len(input) > maxPerceptWidth {
				maxPerceptWidth = len(input)
			}
			if len(input) > 12 && math.Abs(input[12]) > maxPrevPercentChange {
				maxPrevPercentChange = math.Abs(input[12])
			}
			return fxFollowSignalAction(input)
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), follow)
	if err != nil {
		t.Fatalf("evaluate follow: %v", err)
	}
	if maxPerceptWidth <= 2 {
		t.Fatalf("expected fx step percept to include internal features, got width=%d", maxPerceptWidth)
	}
	if width, ok := trace["feature_width"].(int); !ok || width <= 2 {
		t.Fatalf("expected feature_width > 2 in trace, got %+v", trace)
	}
	if width, ok := trace["feature_width"].(int); !ok || width < 14 {
		t.Fatalf("expected feature_width >= 14 with prev percentage channel, got %+v", trace)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected step_input sensor surface, got %+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != "step_output" {
		t.Fatalf("expected step_output control surface, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width < 14 {
		t.Fatalf("expected sensor_width >= 14 with prev percentage channel, got %+v", trace)
	}
	if _, ok := trace["prev_percentage_change"].(float64); !ok {
		t.Fatalf("expected prev_percentage_change in trace, got %+v", trace)
	}
	if maxPrevPercentChange <= 0 {
		t.Fatalf("expected non-zero previous percentage-change signal during episode, got max=%f", maxPrevPercentChange)
	}
}

func TestFXSimulatorSenseTradeInternalsAndRestart(t *testing.T) {
	sim, err := NewFXSimulator(context.Background(), "test")
	if err != nil {
		t.Fatalf("new fx simulator: %v", err)
	}

	initial := sim.State()
	if initial.Mode != "test" || initial.CurrentStep != initial.StartStep || initial.Halted {
		t.Fatalf("unexpected initial simulator state: %+v", initial)
	}

	percept, err := sim.Sense(context.Background())
	if err != nil {
		t.Fatalf("sense: %v", err)
	}
	if len(percept) != fxPerceptWidth {
		t.Fatalf("expected percept width %d, got %d", fxPerceptWidth, len(percept))
	}

	fitness, end, err := sim.Trade(context.Background(), 1)
	if err != nil {
		t.Fatalf("trade open: %v", err)
	}
	if end || fitness != 0 {
		t.Fatalf("expected non-terminal trade response, fitness=%f end=%v", fitness, end)
	}
	afterOpen := sim.State()
	if afterOpen.CurrentStep != initial.CurrentStep+1 {
		t.Fatalf("expected trade to advance one step, before=%+v after=%+v", initial, afterOpen)
	}
	if afterOpen.Position != 1 || afterOpen.OrdersOpened != 1 {
		t.Fatalf("expected opened long position, got %+v", afterOpen)
	}

	internals, err := sim.Internals(context.Background())
	if err != nil {
		t.Fatalf("internals: %v", err)
	}
	if len(internals) != 3 || internals[0] != 1 || internals[1] <= 0 {
		t.Fatalf("unexpected internals after open: %v", internals)
	}

	for {
		_, err := sim.Sense(context.Background())
		if err != nil {
			t.Fatalf("sense during run: %v", err)
		}
		fitness, end, err = sim.Trade(context.Background(), 0)
		if err != nil {
			t.Fatalf("trade close/hold: %v", err)
		}
		if end {
			break
		}
	}
	finalState := sim.State()
	if !finalState.Halted {
		t.Fatalf("expected terminal state, got %+v", finalState)
	}
	if fitness <= 0 {
		t.Fatalf("expected terminal total-profit fitness, got %f", fitness)
	}
	if finalState.Position != 0 || finalState.OrdersClosed <= 0 {
		t.Fatalf("expected closed final position, got %+v", finalState)
	}

	sim.Restart()
	restarted := sim.State()
	if restarted.Halted || restarted.CurrentStep != restarted.StartStep || restarted.OrdersOpened != 0 || restarted.Position != 0 {
		t.Fatalf("unexpected restarted state: %+v", restarted)
	}
}

func TestFXProcessCommandWrapper(t *testing.T) {
	process := NewFXProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FXSenseMessage{}); response.Err == nil {
		t.Fatal("expected sense before start to fail")
	}
	start := process.Call(ctx, FXStartMessage{Mode: "test"})
	if start.Err != nil || !start.OK {
		t.Fatalf("start response=%+v", start)
	}
	if start.State.Mode != "test" || start.State.Halted {
		t.Fatalf("unexpected start state=%+v", start.State)
	}

	sense := process.Call(ctx, FXSenseMessage{})
	if sense.Err != nil || !sense.OK {
		t.Fatalf("sense response=%+v", sense)
	}
	if len(sense.Percept) != fxPerceptWidth {
		t.Fatalf("expected percept width %d, got response=%+v", fxPerceptWidth, sense)
	}

	trade := process.Call(ctx, FXTradeMessage{Action: 1})
	if trade.Err != nil || !trade.OK {
		t.Fatalf("trade response=%+v", trade)
	}
	if trade.End || trade.Fitness != 0 {
		t.Fatalf("expected non-terminal trade response, got %+v", trade)
	}
	if trade.State.Position != 1 || trade.State.OrdersOpened != 1 {
		t.Fatalf("expected opened long position, got %+v", trade.State)
	}

	internals := process.Call(ctx, FXInternalsMessage{})
	if internals.Err != nil || !internals.OK {
		t.Fatalf("internals response=%+v", internals)
	}
	if len(internals.Internals) != 3 || internals.Internals[0] != 1 {
		t.Fatalf("unexpected internals response=%+v", internals)
	}

	state := process.Call(ctx, FXStateMessage{})
	if state.Err != nil || !state.OK || state.State.OrdersOpened != 1 {
		t.Fatalf("state response=%+v", state)
	}

	stop := process.Call(ctx, FXStopMessage{Reason: "normal"})
	if stop.Err != nil || !stop.OK || !stop.End {
		t.Fatalf("stop response=%+v", stop)
	}
	if stop.StopReason != "normal" || !stop.State.Halted {
		t.Fatalf("unexpected stop response=%+v", stop)
	}
	if response := process.Call(ctx, FXSenseMessage{}); response.Err == nil {
		t.Fatal("expected sense after stop to fail")
	}

	restart := process.Call(ctx, FXRestartMessage{})
	if restart.Err != nil || !restart.OK {
		t.Fatalf("restart response=%+v", restart)
	}
	if restart.State.Halted || restart.State.CurrentStep != restart.State.StartStep || restart.State.OrdersOpened != 0 {
		t.Fatalf("unexpected restart response=%+v", restart)
	}
}

func TestFXScapeLoadSeriesCSV(t *testing.T) {
	ResetFXSeriesSource()
	t.Cleanup(ResetFXSeriesSource)

	path := filepath.Join(t.TempDir(), "fx_custom.csv")
	var builder strings.Builder
	builder.WriteString("t,close\n")
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&builder, "%d,%0.6f\n", i, 1.02+0.00045*float64(i))
	}
	if err := os.WriteFile(path, []byte(builder.String()), 0o644); err != nil {
		t.Fatalf("write fx csv: %v", err)
	}

	if err := LoadFXSeriesCSV(path); err != nil {
		t.Fatalf("load fx csv: %v", err)
	}

	scape := FXScape{}
	follow := scriptedStepAgent{
		id: "follow",
		fn: fxFollowSignalAction,
	}

	_, trace, err := scape.EvaluateMode(context.Background(), follow, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	seriesName, ok := trace["series_name"].(string)
	if !ok || !strings.Contains(seriesName, "fx_custom.csv") {
		t.Fatalf("expected loaded fx csv series in trace, got %+v", trace)
	}
	if points, ok := trace["series_points"].(int); !ok || points != 400 {
		t.Fatalf("expected series_points=400, got %+v", trace)
	}
}

func TestFXScapeLoadSeriesCSVRejectsInvalidPrice(t *testing.T) {
	ResetFXSeriesSource()
	t.Cleanup(ResetFXSeriesSource)

	path := filepath.Join(t.TempDir(), "fx_invalid.csv")
	data := "close\n1.020\n1.021\n-1.000\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write fx csv: %v", err)
	}

	if err := LoadFXSeriesCSV(path); err == nil {
		t.Fatal("expected invalid-price error")
	}

	scape := FXScape{}
	follow := scriptedStepAgent{
		id: "follow",
		fn: fxFollowSignalAction,
	}

	_, trace, err := scape.Evaluate(context.Background(), follow)
	if err != nil {
		t.Fatalf("evaluate default series: %v", err)
	}
	if series, _ := trace["series_name"].(string); series != "fx.synthetic.v2" {
		t.Fatalf("expected default fx series after rejected load, got %+v", trace)
	}
}

func fxFollowSignalAction(input []float64) []float64 {
	if len(input) < 2 {
		return []float64{0}
	}
	switch {
	case input[1] > 0:
		return []float64{1}
	case input[1] < 0:
		return []float64{-1}
	default:
		return []float64{0}
	}
}
