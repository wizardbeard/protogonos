package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestCartPoleLiteScapeEvaluateWithHandBuiltAgent(t *testing.T) {
	genome := model.Genome{
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "v", Activation: "identity"},
			{ID: "f", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -1.2, Enabled: true},
			{From: "v", To: "f", Weight: -0.6, Enabled: true},
		},
	}

	cortex, err := agent.NewCortex("cp-agent", genome, nil, nil, []string{"x", "v"}, []string{"f"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := CartPoleLiteScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	avgReward, ok := trace["avg_reward"].(float64)
	if !ok {
		t.Fatalf("trace missing avg_reward: %+v", trace)
	}
	if avgReward <= 0.5 {
		t.Fatalf("expected avg_reward > 0.5, got %f", avgReward)
	}
	if fitness <= 0.5 {
		t.Fatalf("expected fitness > 0.5, got %f", fitness)
	}
}

func TestCartPoleLiteScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.CartPolePositionSensorName, protoio.CartPoleVelocitySensorName},
		ActuatorIDs: []string{protoio.CartPoleForceActuatorName},
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "v", Activation: "identity"},
			{ID: "f", Activation: "identity"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -1.2, Enabled: true},
			{From: "v", To: "f", Weight: -0.6, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.CartPolePositionSensorName: protoio.NewScalarInputSensor(0),
		protoio.CartPoleVelocitySensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.CartPoleForceActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("cp-agent-io", genome, sensors, actuators, []string{"x", "v"}, []string{"f"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := CartPoleLiteScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	avgReward, ok := trace["avg_reward"].(float64)
	if !ok {
		t.Fatalf("trace missing avg_reward: %+v", trace)
	}
	if avgReward <= 0.5 {
		t.Fatalf("expected avg_reward > 0.5, got %f", avgReward)
	}
	if fitness <= 0.5 {
		t.Fatalf("expected fitness > 0.5, got %f", fitness)
	}
}

func TestCartPoleLiteScapeEvaluateWithTickSensorsAndNoActuatorSnapshot(t *testing.T) {
	agent := scriptedTickAgent{
		id: "cp-tick-no-snapshot",
		sensors: map[string]protoio.Sensor{
			protoio.CartPolePositionSensorName: protoio.NewScalarInputSensor(0),
			protoio.CartPoleVelocitySensorName: protoio.NewScalarInputSensor(0),
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			position, err := sensors[protoio.CartPolePositionSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			velocity, err := sensors[protoio.CartPoleVelocitySensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			x := 0.0
			if len(position) > 0 {
				x = position[0]
			}
			v := 0.0
			if len(velocity) > 0 {
				v = velocity[0]
			}
			return []float64{-1.2*x - 0.6*v}, nil
		},
	}

	scape := CartPoleLiteScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent without snapshot actuator: %v", err)
	}
	avgReward, ok := trace["avg_reward"].(float64)
	if !ok {
		t.Fatalf("trace missing avg_reward: %+v", trace)
	}
	if avgReward <= 0.5 {
		t.Fatalf("expected avg_reward > 0.5, got %f", avgReward)
	}
	if fitness <= 0.5 {
		t.Fatalf("expected fitness > 0.5, got %f", fitness)
	}
}

func TestCartPoleLiteScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := CartPoleLiteScape{}
	stabilizer := scriptedStepAgent{
		id: "stabilizer",
		fn: func(input []float64) []float64 {
			if len(input) < 2 {
				return []float64{0}
			}
			return []float64{-1.2*input[0] - 0.6*input[1]}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), stabilizer, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), stabilizer, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
}

func TestCartPoleLiteSimulatorSensePushStateAndReset(t *testing.T) {
	sim, err := NewCartPoleLiteSimulator("validation")
	if err != nil {
		t.Fatalf("new simulator: %v", err)
	}

	state := sim.State()
	if state.Mode != "validation" || state.Episodes != 4 || state.StepsPerEpisode != 48 {
		t.Fatalf("unexpected initial state: %+v", state)
	}
	if state.Position != -1.0 || state.Velocity != 0 {
		t.Fatalf("unexpected initial observation: %+v", state)
	}

	percept, err := sim.Sense(context.Background())
	if err != nil {
		t.Fatalf("sense: %v", err)
	}
	if len(percept) != 2 || percept[0] != -1.0 || percept[1] != 0 {
		t.Fatalf("unexpected percept: %v", percept)
	}

	fitness, end, err := sim.Push(context.Background(), []float64{0})
	if err != nil {
		t.Fatalf("push: %v", err)
	}
	if end {
		t.Fatalf("expected first push to continue")
	}
	if fitness <= 0 {
		t.Fatalf("expected positive step fitness, got %f", fitness)
	}
	state = sim.State()
	if state.StepsSurvived != 1 || state.LastStepReward <= 0 || state.TotalReward <= 0 {
		t.Fatalf("unexpected post-push state: %+v", state)
	}

	sim.Reset()
	state = sim.State()
	if state.StepsSurvived != 0 || state.StepIndex != 0 || state.Position != -1.0 || state.Halted {
		t.Fatalf("unexpected reset state: %+v", state)
	}

	for !end {
		fitness, end, err = sim.Push(context.Background(), []float64{0})
		if err != nil {
			t.Fatalf("push to completion: %v", err)
		}
	}
	state = sim.State()
	if !state.Halted || state.TerminationReason == "" {
		t.Fatalf("expected terminal state with reason: %+v", state)
	}
	if fitness <= 0 || state.LastFitness != fitness {
		t.Fatalf("expected final averaged fitness, got fitness=%f state=%+v", fitness, state)
	}
}

func TestCartPoleLiteProcessCommandWrapper(t *testing.T) {
	process := NewCartPoleLiteProcess()
	ctx := context.Background()

	start := process.Call(ctx, CartPoleLiteStartMessage{Mode: "test"})
	if start.Err != nil || !start.OK {
		t.Fatalf("start failed: %+v", start)
	}
	if start.State.Mode != "test" || start.State.Episodes != 5 {
		t.Fatalf("unexpected start state: %+v", start.State)
	}

	sense := process.Call(ctx, CartPoleLiteSenseMessage{})
	if sense.Err != nil || !sense.OK {
		t.Fatalf("sense failed: %+v", sense)
	}
	if len(sense.Percept) != 2 {
		t.Fatalf("unexpected percept width: %v", sense.Percept)
	}

	push := process.Call(ctx, CartPoleLitePushMessage{Output: []float64{0.25}})
	if push.Err != nil || !push.OK || push.End {
		t.Fatalf("push failed or ended early: %+v", push)
	}
	if push.Fitness <= 0 || push.State.StepsSurvived != 1 {
		t.Fatalf("unexpected push response: %+v", push)
	}

	restart := process.Call(ctx, CartPoleLiteRestartMessage{})
	if restart.Err != nil || !restart.OK {
		t.Fatalf("restart failed: %+v", restart)
	}
	if restart.State.StepsSurvived != 0 || restart.State.Position != -1.2 {
		t.Fatalf("unexpected restart state: %+v", restart.State)
	}

	stop := process.Call(ctx, CartPoleLiteStopMessage{Reason: "done"})
	if stop.Err != nil || !stop.OK || !stop.End || stop.StopReason != "done" {
		t.Fatalf("stop failed: %+v", stop)
	}
	if !stop.State.Halted || stop.State.TerminationReason != "done" {
		t.Fatalf("unexpected stopped state: %+v", stop.State)
	}
}
