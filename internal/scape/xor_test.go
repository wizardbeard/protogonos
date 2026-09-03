package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestXORScapeEvaluateWithHandBuiltAgent(t *testing.T) {
	// Hidden-layer sigmoid network that approximates XOR.
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "h1", Activation: "sigmoid", Bias: -10},
			{ID: "h2", Activation: "sigmoid", Bias: 30},
			{ID: "o", Activation: "sigmoid", Bias: -30},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "h1", Weight: 20, Enabled: true},
			{From: "i2", To: "h1", Weight: 20, Enabled: true},
			{From: "i1", To: "h2", Weight: -20, Enabled: true},
			{From: "i2", To: "h2", Weight: -20, Enabled: true},
			{From: "h1", To: "o", Weight: 20, Enabled: true},
			{From: "h2", To: "o", Weight: 20, Enabled: true},
		},
	}

	cortex, err := agent.NewCortex("xor-agent", genome, nil, nil, []string{"i1", "i2"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	xor := XORScape{}
	fitness, trace, err := xor.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}

	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	sse, ok := trace["sse"].(float64)
	if !ok {
		t.Fatalf("trace missing sse: %+v", trace)
	}
	if mse > 0.05 {
		t.Fatalf("expected mse <= 0.05, got %f", mse)
	}
	wantFitness := Fitness(1.0 / (sse + 0.000001))
	if diff := float64(fitness - wantFitness); diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("expected reciprocal-sse fitness %f, got %f (trace=%+v)", wantFitness, fitness, trace)
	}
}

func TestXORScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "h1", Activation: "sigmoid", Bias: -10},
			{ID: "h2", Activation: "sigmoid", Bias: 30},
			{ID: "o", Activation: "sigmoid", Bias: -30},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "h1", Weight: 20, Enabled: true},
			{From: "i2", To: "h1", Weight: 20, Enabled: true},
			{From: "i1", To: "h2", Weight: -20, Enabled: true},
			{From: "i2", To: "h2", Weight: -20, Enabled: true},
			{From: "h1", To: "o", Weight: 20, Enabled: true},
			{From: "h2", To: "o", Weight: 20, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.XORInputLeftSensorName:  protoio.NewScalarInputSensor(0),
		protoio.XORInputRightSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.XOROutputActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("xor-agent-io", genome, sensors, actuators, []string{"i1", "i2"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	xor := XORScape{}
	fitness, trace, err := xor.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}

	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	sse, ok := trace["sse"].(float64)
	if !ok {
		t.Fatalf("trace missing sse: %+v", trace)
	}
	if mse > 0.05 {
		t.Fatalf("expected mse <= 0.05, got %f", mse)
	}
	wantFitness := Fitness(1.0 / (sse + 0.000001))
	if diff := float64(fitness - wantFitness); diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("expected reciprocal-sse fitness %f, got %f (trace=%+v)", wantFitness, fitness, trace)
	}
}

func TestXORScapeEvaluateWithTickSensorsAndNoActuatorSnapshot(t *testing.T) {
	agent := scriptedTickAgent{
		id: "xor-tick-no-snapshot",
		sensors: map[string]protoio.Sensor{
			protoio.XORInputLeftSensorName:  protoio.NewScalarInputSensor(0),
			protoio.XORInputRightSensorName: protoio.NewScalarInputSensor(0),
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			left, err := sensors[protoio.XORInputLeftSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			right, err := sensors[protoio.XORInputRightSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			leftValue := 0
			if len(left) > 0 {
				leftValue = int(left[0])
			}
			rightValue := 0
			if len(right) > 0 {
				rightValue = int(right[0])
			}
			if leftValue^rightValue == 1 {
				return []float64{1}, nil
			}
			return []float64{0}, nil
		},
	}

	xor := XORScape{}
	fitness, trace, err := xor.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent without snapshot actuator: %v", err)
	}
	mse, ok := trace["mse"].(float64)
	if !ok {
		t.Fatalf("trace missing mse: %+v", trace)
	}
	if mse > 1e-9 {
		t.Fatalf("expected mse ~0, got %f", mse)
	}
	if fitness < 1000 {
		t.Fatalf("expected strong xor fitness, got %f", fitness)
	}
}

func TestXORScapeEvaluateModeAnnotatesMode(t *testing.T) {
	xor := XORScape{}
	parity := scriptedStepAgent{
		id: "xor-parity",
		fn: func(input []float64) []float64 {
			if len(input) < 2 {
				return []float64{0}
			}
			if int(input[0])^int(input[1]) == 1 {
				return []float64{1}
			}
			return []float64{0}
		},
	}

	_, validationTrace, err := xor.EvaluateMode(context.Background(), parity, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := xor.EvaluateMode(context.Background(), parity, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	if validationCases, vok := validationTrace["cases"].(int); vok {
		if testCases, tok := testTrace["cases"].(int); tok && testCases == validationCases {
			t.Fatalf("expected distinct xor mode case windows, got validation=%d test=%d", validationCases, testCases)
		}
	}
}

func TestXORSimulatorSensePredictAndReset(t *testing.T) {
	sim, err := NewXORSimulator("gt")
	if err != nil {
		t.Fatalf("new xor simulator: %v", err)
	}
	state := sim.State()
	if state.Mode != "gt" || state.Cases != 4 || state.CaseIndex != 0 {
		t.Fatalf("unexpected initial simulator state: %+v", state)
	}

	expected := []float64{0, 1, 1, 0}
	for i, want := range expected {
		percept, err := sim.Sense(context.Background())
		if err != nil {
			t.Fatalf("sense %d: %v", i, err)
		}
		if len(percept) != 2 {
			t.Fatalf("expected xor percept width 2, got %v", percept)
		}
		fitness, end, err := sim.Predict(context.Background(), []float64{want})
		if err != nil {
			t.Fatalf("predict %d: %v", i, err)
		}
		if i < len(expected)-1 {
			if end || fitness != 0 {
				t.Fatalf("expected non-terminal predict at %d, fitness=%f end=%t", i, fitness, end)
			}
			continue
		}
		if !end || fitness < 1000 {
			t.Fatalf("expected terminal high-fitness xor response, fitness=%f end=%t", fitness, end)
		}
	}
	state = sim.State()
	if state.CaseIndex != 0 || state.ErrAcc != 0 || state.LastSSE != 0 || state.LastFitness < 1000 || len(state.Predictions) != 0 {
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
	if state.CaseIndex != 0 || state.LastFitness != 0 || state.LastSSE != 0 || state.TerminationReason != "" {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}

func TestXORProcessCommandWrapper(t *testing.T) {
	process := NewXORProcess()
	ctx := context.Background()

	if response := process.Call(ctx, XORSenseMessage{}); response.Err == nil {
		t.Fatal("expected sense before start to fail")
	}
	start := process.Call(ctx, XORStartMessage{Mode: "validation"})
	if start.Err != nil || !start.OK {
		t.Fatalf("start response=%+v", start)
	}
	if start.State.Mode != "validation" || start.State.CaseIndex != 0 {
		t.Fatalf("unexpected start state=%+v", start.State)
	}

	sense := process.Call(ctx, XORSenseMessage{})
	if sense.Err != nil || !sense.OK {
		t.Fatalf("sense response=%+v", sense)
	}
	if len(sense.Percept) != 2 {
		t.Fatalf("expected xor percept width 2, got response=%+v", sense)
	}

	predict := process.Call(ctx, XORPredictMessage{Output: []float64{1}})
	if predict.Err != nil || !predict.OK {
		t.Fatalf("predict response=%+v", predict)
	}
	if predict.End || predict.Fitness != 0 || predict.State.CaseIndex != 1 {
		t.Fatalf("unexpected non-terminal predict response=%+v", predict)
	}

	state := process.Call(ctx, XORStateMessage{})
	if state.Err != nil || !state.OK || state.State.CaseIndex != 1 {
		t.Fatalf("state response=%+v", state)
	}

	stop := process.Call(ctx, XORStopMessage{Reason: "normal"})
	if stop.Err != nil || !stop.OK || !stop.End {
		t.Fatalf("stop response=%+v", stop)
	}
	if stop.StopReason != "normal" {
		t.Fatalf("unexpected stop response=%+v", stop)
	}
	if stop.State.TerminationReason != "normal" {
		t.Fatalf("expected stop reason in state, got %+v", stop.State)
	}
	if response := process.Call(ctx, XORSenseMessage{}); response.Err == nil {
		t.Fatal("expected sense after stop to fail")
	}

	restart := process.Call(ctx, XORRestartMessage{})
	if restart.Err != nil || !restart.OK {
		t.Fatalf("restart response=%+v", restart)
	}
	if restart.State.CaseIndex != 0 || restart.State.Mode != "validation" {
		t.Fatalf("unexpected restart response=%+v", restart)
	}
	if restart.State.TerminationReason != "" {
		t.Fatalf("expected restart to clear terminal reason, got %+v", restart.State)
	}
}

func TestXORProcessIOAdaptersUseSharedScapeSession(t *testing.T) {
	sensors, actuators, err := NewXORProcessIO(
		"gt",
		[]string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		[]string{protoio.XOROutputActuatorName},
	)
	if err != nil {
		t.Fatalf("new xor process io: %v", err)
	}

	left, ok := sensors[protoio.XORInputLeftSensorName].(protoio.SensorProcessReader)
	if !ok {
		t.Fatalf("left sensor does not implement SensorProcessReader: %T", sensors[protoio.XORInputLeftSensorName])
	}
	right, ok := sensors[protoio.XORInputRightSensorName].(protoio.SensorProcessReader)
	if !ok {
		t.Fatalf("right sensor does not implement SensorProcessReader: %T", sensors[protoio.XORInputRightSensorName])
	}
	output, ok := actuators[protoio.XOROutputActuatorName].(protoio.ActuatorProcessWriter)
	if !ok {
		t.Fatalf("output actuator does not implement ActuatorProcessWriter: %T", actuators[protoio.XOROutputActuatorName])
	}

	ctx := context.Background()
	for i, want := range []float64{0, 1, 1, 0} {
		leftValue, err := left.ReadForSensorProcess(ctx, protoio.SensorProcessCall{
			Scape:      "xor",
			SensorName: protoio.XORGetInputSensorAliasName,
			VL:         1,
			OpMode:     "gt",
		})
		if err != nil {
			t.Fatalf("left read %d: %v", i, err)
		}
		rightValue, err := right.ReadForSensorProcess(ctx, protoio.SensorProcessCall{
			Scape:      "xor",
			SensorName: protoio.XORGetInputSensorAliasName,
			VL:         1,
			OpMode:     "gt",
		})
		if err != nil {
			t.Fatalf("right read %d: %v", i, err)
		}
		if len(leftValue) != 1 || len(rightValue) != 1 {
			t.Fatalf("expected scalar percepts at %d, left=%v right=%v", i, leftValue, rightValue)
		}

		sync, err := output.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{
			Scape:        "xor",
			ActuatorName: protoio.XORSendOutputActuatorAliasName,
			Output:       []float64{want},
			OpMode:       "gt",
		})
		if err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
		if i < 3 {
			if sync.EndFlag != 0 || len(sync.Fitness) != 1 || sync.Fitness[0] != 0 {
				t.Fatalf("expected non-terminal sync at %d, got %+v", i, sync)
			}
			continue
		}
		if sync.EndFlag != 1 || len(sync.Fitness) != 1 || sync.Fitness[0] < 1000 {
			t.Fatalf("expected terminal high-fitness sync, got %+v", sync)
		}
	}
}

func TestXORScapeEvaluateWithActorProcessIO(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.XORInputLeftSensorName, protoio.XORInputRightSensorName},
		ActuatorIDs: []string{protoio.XOROutputActuatorName},
		Neurons: []model.Neuron{
			{ID: "i1", Activation: "identity"},
			{ID: "i2", Activation: "identity"},
			{ID: "h1", Activation: "sigmoid", Bias: -10},
			{ID: "h2", Activation: "sigmoid", Bias: 30},
			{ID: "o", Activation: "sigmoid", Bias: -30},
		},
		Synapses: []model.Synapse{
			{From: "i1", To: "h1", Weight: 20, Enabled: true},
			{From: "i2", To: "h1", Weight: 20, Enabled: true},
			{From: "i1", To: "h2", Weight: -20, Enabled: true},
			{From: "i2", To: "h2", Weight: -20, Enabled: true},
			{From: "h1", To: "o", Weight: 20, Enabled: true},
			{From: "h2", To: "o", Weight: 20, Enabled: true},
		},
	}

	sensors, actuators, err := NewXORProcessIO("gt", genome.SensorIDs, genome.ActuatorIDs)
	if err != nil {
		t.Fatalf("new xor process io: %v", err)
	}
	cortex, err := agent.NewCortex(
		"xor-agent-actor-process-io",
		genome,
		sensors,
		actuators,
		[]string{"i1", "i2"},
		[]string{"o"},
		nil,
		agent.WithIOProcessContext("xor", "gt"),
		agent.WithIOActors(),
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	t.Cleanup(func() {
		cortex.Terminate()
	})

	fitness, trace, err := XORScape{}.EvaluateMode(context.Background(), cortex, "gt")
	if err != nil {
		t.Fatalf("evaluate actor process io: %v", err)
	}
	if cases, ok := trace["cases"].(int); !ok || cases != 4 {
		t.Fatalf("expected 4 xor cases, got trace=%+v", trace)
	}
	if fitness < 1000 {
		t.Fatalf("expected high actor-process xor fitness, got %f trace=%+v", fitness, trace)
	}
}
