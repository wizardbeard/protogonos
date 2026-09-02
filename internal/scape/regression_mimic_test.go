package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestRegressionMimicScapeEvaluateWithIdentityAgent(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0},
		},
		Synapses: []model.Synapse{
			{From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}

	cortex, err := agent.NewCortex("reg-agent", genome, nil, nil, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := RegressionMimicScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}

	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	if mse > 1e-9 {
		t.Fatalf("expected mse ~0, got %f", mse)
	}
	if fitness < 0.999999 {
		t.Fatalf("expected near-perfect fitness, got %f", fitness)
	}
}

func TestRegressionMimicScapeEvaluateWithScalarIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.ScalarInputSensorName},
		ActuatorIDs: []string{protoio.ScalarOutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0},
		},
		Synapses: []model.Synapse{
			{From: "i", To: "o", Weight: 1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.ScalarInputSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.ScalarOutputActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("reg-agent-io", genome, sensors, actuators, []string{"i"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := RegressionMimicScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}

	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	if mse > 1e-9 {
		t.Fatalf("expected mse ~0, got %f", mse)
	}
	if fitness < 0.999999 {
		t.Fatalf("expected near-perfect fitness, got %f", fitness)
	}
}

func TestRegressionMimicScapeEvaluateWithTickSensorsAndWriteOnlyActuator(t *testing.T) {
	agent := scriptedTickAgent{
		id: "reg-tick-write-only",
		sensors: map[string]protoio.Sensor{
			protoio.ScalarInputSensorName: protoio.NewScalarInputSensor(0),
		},
		actuators: map[string]protoio.Actuator{
			protoio.ScalarOutputActuatorName: &writeOnlyActuator{name: protoio.ScalarOutputActuatorName},
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			input, err := sensors[protoio.ScalarInputSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			if len(input) == 0 {
				return []float64{0}, nil
			}
			return []float64{input[0]}, nil
		},
	}

	scape := RegressionMimicScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent with write-only actuator: %v", err)
	}
	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	if mse > 1e-9 {
		t.Fatalf("expected mse ~0, got %f", mse)
	}
	if fitness < 0.999999 {
		t.Fatalf("expected near-perfect fitness, got %f", fitness)
	}
}

func TestRegressionMimicScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := RegressionMimicScape{}
	identity := scriptedStepAgent{
		id: "identity",
		fn: func(input []float64) []float64 {
			if len(input) == 0 {
				return []float64{0}
			}
			return []float64{input[0]}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), identity, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), identity, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	if validationSamples, vok := validationTrace["samples"].(int); vok {
		if testSamples, tok := testTrace["samples"].(int); tok && testSamples == validationSamples {
			t.Fatalf("expected distinct regression mode sample windows, got validation=%d test=%d", validationSamples, testSamples)
		}
	}
}

func TestRegressionMimicSimulatorSensePredictAndReset(t *testing.T) {
	sim, err := NewRegressionMimicSimulator("gt")
	if err != nil {
		t.Fatalf("new regression-mimic simulator: %v", err)
	}
	state := sim.State()
	if state.Mode != "gt" || state.Samples != 5 || state.SampleIndex != 0 {
		t.Fatalf("unexpected initial simulator state: %+v", state)
	}

	for i := 0; i < state.Samples; i++ {
		percept, err := sim.Sense(context.Background())
		if err != nil {
			t.Fatalf("sense %d: %v", i, err)
		}
		if len(percept) != 1 {
			t.Fatalf("expected scalar percept, got %v", percept)
		}
		fitness, end, err := sim.Predict(context.Background(), []float64{percept[0]})
		if err != nil {
			t.Fatalf("predict %d: %v", i, err)
		}
		if i < state.Samples-1 {
			if end || fitness != 0 {
				t.Fatalf("expected non-terminal predict at %d, fitness=%f end=%t", i, fitness, end)
			}
			continue
		}
		if !end || fitness < 0.999999 {
			t.Fatalf("expected terminal near-perfect fitness, fitness=%f end=%t", fitness, end)
		}
	}
	state = sim.State()
	if state.SampleIndex != 0 || state.ErrAcc != 0 || state.LastMSE != 0 || state.LastFitness < 0.999999 || len(state.Predictions) != 0 {
		t.Fatalf("unexpected terminal reset state: %+v", state)
	}
	if state.TerminationReason != "completed" {
		t.Fatalf("expected completed terminal reason, got %+v", state)
	}

	if _, _, err := sim.Predict(context.Background(), []float64{}); err == nil {
		t.Fatal("expected empty prediction error")
	}
	sim.Reset()
	state = sim.State()
	if state.SampleIndex != 0 || state.LastFitness != 0 || state.LastMSE != 0 || state.TerminationReason != "" {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}

func TestRegressionMimicProcessCommandWrapper(t *testing.T) {
	process := NewRegressionMimicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, RegressionMimicSenseMessage{}); response.Err == nil {
		t.Fatal("expected sense before start to fail")
	}
	start := process.Call(ctx, RegressionMimicStartMessage{Mode: "validation"})
	if start.Err != nil || !start.OK {
		t.Fatalf("start response=%+v", start)
	}
	if start.State.Mode != "validation" || start.State.SampleIndex != 0 {
		t.Fatalf("unexpected start state=%+v", start.State)
	}

	sense := process.Call(ctx, RegressionMimicSenseMessage{})
	if sense.Err != nil || !sense.OK {
		t.Fatalf("sense response=%+v", sense)
	}
	if len(sense.Percept) != 1 {
		t.Fatalf("expected scalar percept, got response=%+v", sense)
	}

	predict := process.Call(ctx, RegressionMimicPredictMessage{Output: []float64{sense.Percept[0]}})
	if predict.Err != nil || !predict.OK {
		t.Fatalf("predict response=%+v", predict)
	}
	if predict.End || predict.Fitness != 0 || predict.State.SampleIndex != 1 {
		t.Fatalf("unexpected non-terminal predict response=%+v", predict)
	}

	state := process.Call(ctx, RegressionMimicStateMessage{})
	if state.Err != nil || !state.OK || state.State.SampleIndex != 1 {
		t.Fatalf("state response=%+v", state)
	}

	stop := process.Call(ctx, RegressionMimicStopMessage{Reason: "normal"})
	if stop.Err != nil || !stop.OK || !stop.End {
		t.Fatalf("stop response=%+v", stop)
	}
	if stop.StopReason != "normal" {
		t.Fatalf("unexpected stop response=%+v", stop)
	}
	if stop.State.TerminationReason != "normal" {
		t.Fatalf("expected stop reason in state, got %+v", stop.State)
	}
	if response := process.Call(ctx, RegressionMimicSenseMessage{}); response.Err == nil {
		t.Fatal("expected sense after stop to fail")
	}

	restart := process.Call(ctx, RegressionMimicRestartMessage{})
	if restart.Err != nil || !restart.OK {
		t.Fatalf("restart response=%+v", restart)
	}
	if restart.State.SampleIndex != 0 || restart.State.Mode != "validation" {
		t.Fatalf("unexpected restart response=%+v", restart)
	}
	if restart.State.TerminationReason != "" {
		t.Fatalf("expected restart to clear terminal reason, got %+v", restart.State)
	}
}
