package scape

import (
	"context"
	"math"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestPole2BalancingScapeEvaluatesStepPolicies(t *testing.T) {
	scape := Pole2BalancingScape{}
	thrash := scriptedStepAgent{
		id: "thrash",
		fn: func(_ []float64) []float64 { return []float64{1} },
	}
	stabilize := scriptedStepAgent{
		id: "stabilize",
		fn: func(in []float64) []float64 {
			if len(in) < 6 {
				return []float64{0}
			}
			force := -(0.9*in[0] + 0.6*in[1] + 8.0*in[2] + 1.4*in[3] + 10.0*in[4] + 1.8*in[5])
			return []float64{force}
		},
	}

	thrashFitness, thrashTrace, err := scape.Evaluate(context.Background(), thrash)
	if err != nil {
		t.Fatalf("evaluate thrash: %v", err)
	}
	stabilizeFitness, stabilizeTrace, err := scape.Evaluate(context.Background(), stabilize)
	if err != nil {
		t.Fatalf("evaluate stabilize: %v", err)
	}
	if thrashFitness <= 0 || stabilizeFitness <= 0 {
		t.Fatalf("expected positive fitness for evaluated policies, got stabilize=%f thrash=%f", stabilizeFitness, thrashFitness)
	}
	if _, ok := thrashTrace["steps_survived"].(int); !ok {
		t.Fatalf("thrash trace missing steps_survived: %+v", thrashTrace)
	}
	if _, ok := stabilizeTrace["steps_survived"].(int); !ok {
		t.Fatalf("stabilize trace missing steps_survived: %+v", stabilizeTrace)
	}
}

func TestPole2SimulatorSensePushStateAndReset(t *testing.T) {
	sim, err := NewPole2Simulator("validation")
	if err != nil {
		t.Fatalf("new pole2 simulator: %v", err)
	}
	initial := sim.State()
	if initial.Mode != "validation" || initial.StepsSurvived != 0 || initial.Halted {
		t.Fatalf("unexpected initial simulator state: %+v", initial)
	}

	sense3, err := sim.Sense(context.Background(), "3")
	if err != nil {
		t.Fatalf("sense 3: %v", err)
	}
	if len(sense3) != 3 {
		t.Fatalf("expected 3-channel sense vector, got %v", sense3)
	}
	angle1, err := sim.Sense(context.Background(), "pangle1")
	if err != nil {
		t.Fatalf("sense pangle1: %v", err)
	}
	if len(angle1) != 1 || math.Abs(angle1[0]-sense3[1]) > 1e-12 {
		t.Fatalf("expected pangle1 to match second 3-surface channel, angle1=%v sense3=%v", angle1, sense3)
	}

	fitness, end, err := sim.Push(context.Background(), []float64{0.25, -1, -1})
	if err != nil {
		t.Fatalf("push vector control: %v", err)
	}
	if end || fitness <= 0 {
		t.Fatalf("expected non-terminal positive push fitness, fitness=%f end=%v", fitness, end)
	}
	afterPush := sim.State()
	if afterPush.StepsSurvived != 1 || afterPush.FitnessAcc <= 0 || afterPush.LastStepFitness <= 0 {
		t.Fatalf("expected advanced simulator state, got %+v", afterPush)
	}
	if afterPush.VectorControlSteps != 1 || afterPush.DampingOffSteps != 1 || afterPush.SinglePoleSteps != 1 {
		t.Fatalf("expected vector-control counters, got %+v", afterPush)
	}
	if afterPush.RunProgress <= 0 || afterPush.StepProgress <= 0 {
		t.Fatalf("expected positive progress after push, got %+v", afterPush)
	}

	if _, err := sim.Sense(context.Background(), "bad"); err == nil {
		t.Fatal("expected unsupported sense parameter error")
	}

	for !sim.State().Halted {
		_, end, err = sim.Push(context.Background(), []float64{1})
		if err != nil {
			t.Fatalf("push until terminal: %v", err)
		}
		if end {
			break
		}
	}
	terminal := sim.State()
	if !terminal.Halted || terminal.TerminationReason == "" {
		t.Fatalf("expected terminal simulator state, got %+v", terminal)
	}
	if _, err := sim.Sense(context.Background(), "6"); err == nil {
		t.Fatal("expected halted simulator sense error")
	}

	sim.Reset()
	restarted := sim.State()
	if restarted.Halted || restarted.StepsSurvived != 0 || restarted.FitnessAcc != 0 || restarted.VectorControlSteps != 0 {
		t.Fatalf("unexpected reset simulator state: %+v", restarted)
	}
}

func TestPole2ProcessCommandWrapper(t *testing.T) {
	process := NewPole2Process()
	ctx := context.Background()

	if response := process.Call(ctx, Pole2SenseMessage{Parameter: "3"}); response.Err == nil {
		t.Fatal("expected sense before start to fail")
	}
	start := process.Call(ctx, Pole2StartMessage{Mode: "validation"})
	if start.Err != nil || !start.OK {
		t.Fatalf("start response=%+v", start)
	}
	if start.State.Mode != "validation" || start.State.Halted {
		t.Fatalf("unexpected start state=%+v", start.State)
	}

	sense := process.Call(ctx, Pole2SenseMessage{Parameter: "3"})
	if sense.Err != nil || !sense.OK {
		t.Fatalf("sense response=%+v", sense)
	}
	if len(sense.Percept) != 3 {
		t.Fatalf("expected 3-channel percept, got response=%+v", sense)
	}

	push := process.Call(ctx, Pole2PushMessage{Output: []float64{0.25, -1, -1}})
	if push.Err != nil || !push.OK {
		t.Fatalf("push response=%+v", push)
	}
	if push.End || push.Fitness <= 0 {
		t.Fatalf("expected non-terminal positive push response, got %+v", push)
	}
	if push.State.StepsSurvived != 1 || push.State.VectorControlSteps != 1 || push.State.DampingOffSteps != 1 || push.State.SinglePoleSteps != 1 {
		t.Fatalf("unexpected push state=%+v", push.State)
	}

	state := process.Call(ctx, Pole2StateMessage{})
	if state.Err != nil || !state.OK || state.State.StepsSurvived != 1 {
		t.Fatalf("state response=%+v", state)
	}

	stop := process.Call(ctx, Pole2StopMessage{Reason: "normal"})
	if stop.Err != nil || !stop.OK || !stop.End {
		t.Fatalf("stop response=%+v", stop)
	}
	if stop.StopReason != "normal" || !stop.State.Halted || stop.State.TerminationReason != "normal" {
		t.Fatalf("unexpected stop response=%+v", stop)
	}
	if response := process.Call(ctx, Pole2SenseMessage{Parameter: "6"}); response.Err == nil {
		t.Fatal("expected sense after stop to fail")
	}

	restart := process.Call(ctx, Pole2RestartMessage{})
	if restart.Err != nil || !restart.OK {
		t.Fatalf("restart response=%+v", restart)
	}
	if restart.State.Halted || restart.State.StepsSurvived != 0 || restart.State.FitnessAcc != 0 || restart.State.Mode != "validation" {
		t.Fatalf("unexpected restart response=%+v", restart)
	}
}

func TestPole2ProcessIOAdaptersUseSharedScapeSession(t *testing.T) {
	sensorIDs := []string{
		protoio.Pole2CartPositionSensorName,
		protoio.Pole2CartVelocitySensorName,
		protoio.Pole2Angle1SensorName,
		protoio.Pole2Velocity1SensorName,
		protoio.Pole2Angle2SensorName,
		protoio.Pole2Velocity2SensorName,
		protoio.Pole2RunProgressSensorName,
		protoio.Pole2StepProgressSensorName,
		protoio.Pole2FitnessSignalSensorName,
	}
	sensors, actuators, err := NewPole2ProcessIO("validation", sensorIDs, []string{protoio.Pole2PushActuatorName})
	if err != nil {
		t.Fatalf("new pole2 process io: %v", err)
	}

	ctx := context.Background()
	for _, sensorID := range sensorIDs {
		reader, ok := sensors[sensorID].(protoio.SensorProcessReader)
		if !ok {
			t.Fatalf("sensor %s does not implement SensorProcessReader: %T", sensorID, sensors[sensorID])
		}
		value, err := reader.ReadForSensorProcess(ctx, protoio.SensorProcessCall{
			Scape:      "pole2-balancing",
			SensorName: protoio.PBGetInputSensorAliasName,
			VL:         1,
			OpMode:     "validation",
		})
		if err != nil {
			t.Fatalf("read %s: %v", sensorID, err)
		}
		if len(value) != 1 {
			t.Fatalf("expected scalar value for %s, got %v", sensorID, value)
		}
	}

	writer, ok := actuators[protoio.Pole2PushActuatorName].(protoio.ActuatorProcessWriter)
	if !ok {
		t.Fatalf("push actuator does not implement ActuatorProcessWriter: %T", actuators[protoio.Pole2PushActuatorName])
	}
	sync, err := writer.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{
		Scape:        "pole2-balancing",
		ActuatorName: protoio.PBSendOutputActuatorAliasName,
		Output:       []float64{0.25, -1, -1},
		OpMode:       "validation",
	})
	if err != nil {
		t.Fatalf("write push: %v", err)
	}
	if sync.EndFlag != 0 || len(sync.Fitness) != 1 || sync.Fitness[0] <= 0 {
		t.Fatalf("expected non-terminal positive-fitness push sync, got %+v", sync)
	}
}

func TestPole2BalancingScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2CartVelocitySensorName,
			protoio.Pole2Angle1SensorName,
			protoio.Pole2Velocity1SensorName,
			protoio.Pole2Angle2SensorName,
			protoio.Pole2Velocity2SensorName,
			protoio.Pole2RunProgressSensorName,
			protoio.Pole2StepProgressSensorName,
			protoio.Pole2FitnessSignalSensorName,
		},
		ActuatorIDs: []string{protoio.Pole2PushActuatorName},
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "v", Activation: "identity"},
			{ID: "a1", Activation: "identity"},
			{ID: "w1", Activation: "identity"},
			{ID: "a2", Activation: "identity"},
			{ID: "w2", Activation: "identity"},
			{ID: "rp", Activation: "identity"},
			{ID: "sp", Activation: "identity"},
			{ID: "fs", Activation: "identity"},
			{ID: "f", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -0.8, Enabled: true},
			{From: "v", To: "f", Weight: -0.2, Enabled: true},
			{From: "a1", To: "f", Weight: -4.5, Enabled: true},
			{From: "w1", To: "f", Weight: -0.9, Enabled: true},
			{From: "a2", To: "f", Weight: -6.0, Enabled: true},
			{From: "w2", To: "f", Weight: -1.1, Enabled: true},
			{From: "rp", To: "f", Weight: 0.2, Enabled: true},
			{From: "sp", To: "f", Weight: 0.15, Enabled: true},
			{From: "fs", To: "f", Weight: 0.25, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.Pole2CartPositionSensorName:  protoio.NewScalarInputSensor(0),
		protoio.Pole2CartVelocitySensorName:  protoio.NewScalarInputSensor(0),
		protoio.Pole2Angle1SensorName:        protoio.NewScalarInputSensor(0),
		protoio.Pole2Velocity1SensorName:     protoio.NewScalarInputSensor(0),
		protoio.Pole2Angle2SensorName:        protoio.NewScalarInputSensor(0),
		protoio.Pole2Velocity2SensorName:     protoio.NewScalarInputSensor(0),
		protoio.Pole2RunProgressSensorName:   protoio.NewScalarInputSensor(0),
		protoio.Pole2StepProgressSensorName:  protoio.NewScalarInputSensor(0),
		protoio.Pole2FitnessSignalSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.Pole2PushActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"pole2-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"x", "v", "a1", "w1", "a2", "w2", "rp", "sp", "fs"},
		[]string{"f"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := Pole2BalancingScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["steps_survived"].(int); !ok {
		t.Fatalf("trace missing steps_survived: %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "6" {
		t.Fatalf("expected full pole2 sensor_surface=6, got %+v", trace)
	}
	if surface, ok := trace["workflow_surface"].(string); !ok || surface != "all" {
		t.Fatalf("expected pole2 workflow_surface=all, got %+v", trace)
	}
	if _, ok := trace["mean_run_progress"].(float64); !ok {
		t.Fatalf("expected mean_run_progress in trace, got %+v", trace)
	}
	if _, ok := trace["mean_step_progress"].(float64); !ok {
		t.Fatalf("expected mean_step_progress in trace, got %+v", trace)
	}
	if _, ok := trace["mean_fitness_signal"].(float64); !ok {
		t.Fatalf("expected mean_fitness_signal in trace, got %+v", trace)
	}
}

func TestPole2BalancingScapeEvaluateWithActorProcessIO(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2CartVelocitySensorName,
			protoio.Pole2Angle1SensorName,
			protoio.Pole2Velocity1SensorName,
			protoio.Pole2Angle2SensorName,
			protoio.Pole2Velocity2SensorName,
			protoio.Pole2RunProgressSensorName,
			protoio.Pole2StepProgressSensorName,
			protoio.Pole2FitnessSignalSensorName,
		},
		ActuatorIDs: []string{protoio.Pole2PushActuatorName},
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "v", Activation: "identity"},
			{ID: "a1", Activation: "identity"},
			{ID: "w1", Activation: "identity"},
			{ID: "a2", Activation: "identity"},
			{ID: "w2", Activation: "identity"},
			{ID: "rp", Activation: "identity"},
			{ID: "sp", Activation: "identity"},
			{ID: "fs", Activation: "identity"},
			{ID: "f", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -0.8, Enabled: true},
			{From: "v", To: "f", Weight: -0.2, Enabled: true},
			{From: "a1", To: "f", Weight: -4.5, Enabled: true},
			{From: "w1", To: "f", Weight: -0.9, Enabled: true},
			{From: "a2", To: "f", Weight: -6.0, Enabled: true},
			{From: "w2", To: "f", Weight: -1.1, Enabled: true},
			{From: "rp", To: "f", Weight: 0.2, Enabled: true},
			{From: "sp", To: "f", Weight: 0.15, Enabled: true},
			{From: "fs", To: "f", Weight: 0.25, Enabled: true},
		},
	}

	sensors, actuators, err := NewPole2ProcessIO("validation", genome.SensorIDs, genome.ActuatorIDs)
	if err != nil {
		t.Fatalf("new pole2 process io: %v", err)
	}
	cortex, err := agent.NewCortex(
		"pole2-agent-actor-process-io",
		genome,
		sensors,
		actuators,
		[]string{"x", "v", "a1", "w1", "a2", "w2", "rp", "sp", "fs"},
		[]string{"f"},
		nil,
		agent.WithIOProcessContext("pole2-balancing", "validation"),
		agent.WithIOActors(),
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	t.Cleanup(cortex.Terminate)

	fitness, trace, err := Pole2BalancingScape{}.EvaluateMode(context.Background(), cortex, "validation")
	if err != nil {
		t.Fatalf("evaluate actor process io: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive actor-process pole2 fitness, got %f trace=%+v", fitness, trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "6" {
		t.Fatalf("expected full pole2 sensor_surface=6, got %+v", trace)
	}
	if surface, ok := trace["workflow_surface"].(string); !ok || surface != "all" {
		t.Fatalf("expected pole2 workflow_surface=all, got %+v", trace)
	}
}

func TestPole2BalancingScapeEvaluateWithReducedIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.Pole2CartPositionSensorName,
			protoio.Pole2Angle1SensorName,
			protoio.Pole2Angle2SensorName,
		},
		ActuatorIDs: []string{protoio.Pole2PushActuatorName},
		Neurons: []model.Neuron{
			{ID: "x", Activation: "identity"},
			{ID: "a1", Activation: "identity"},
			{ID: "a2", Activation: "identity"},
			{ID: "f", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "x", To: "f", Weight: -0.7, Enabled: true},
			{From: "a1", To: "f", Weight: -3.8, Enabled: true},
			{From: "a2", To: "f", Weight: -5.4, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.Pole2CartPositionSensorName: protoio.NewScalarInputSensor(0),
		protoio.Pole2Angle1SensorName:       protoio.NewScalarInputSensor(0),
		protoio.Pole2Angle2SensorName:       protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.Pole2PushActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"pole2-agent-io-reduced",
		genome,
		sensors,
		actuators,
		[]string{"x", "a1", "a2"},
		[]string{"f"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := Pole2BalancingScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "3" {
		t.Fatalf("expected reduced pole2 sensor_surface=3, got %+v", trace)
	}
	if surface, ok := trace["workflow_surface"].(string); !ok || surface != "none" {
		t.Fatalf("expected reduced pole2 workflow_surface=none, got %+v", trace)
	}
}

func TestPole2BalancingScapeEvaluateWithTickSensorsAndWriteOnlyActuator(t *testing.T) {
	agent := scriptedTickAgent{
		id: "pole2-tick-write-only",
		sensors: map[string]protoio.Sensor{
			protoio.Pole2CartPositionSensorName: protoio.NewScalarInputSensor(0),
			protoio.Pole2Angle1SensorName:       protoio.NewScalarInputSensor(0),
			protoio.Pole2Angle2SensorName:       protoio.NewScalarInputSensor(0),
		},
		actuators: map[string]protoio.Actuator{
			protoio.Pole2PushActuatorName: &writeOnlyActuator{name: protoio.Pole2PushActuatorName},
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			position, err := sensors[protoio.Pole2CartPositionSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			angle1, err := sensors[protoio.Pole2Angle1SensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			angle2, err := sensors[protoio.Pole2Angle2SensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			x := 0.0
			if len(position) > 0 {
				x = position[0]
			}
			a1 := 0.0
			if len(angle1) > 0 {
				a1 = angle1[0]
			}
			a2 := 0.0
			if len(angle2) > 0 {
				a2 = angle2[0]
			}
			return []float64{-(0.7*x + 3.8*a1 + 5.4*a2)}, nil
		},
	}

	scape := Pole2BalancingScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent with write-only actuator: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "3" {
		t.Fatalf("expected reduced pole2 sensor_surface=3, got %+v", trace)
	}
	if surface, ok := trace["workflow_surface"].(string); !ok || surface != "none" {
		t.Fatalf("expected reduced pole2 workflow_surface=none, got %+v", trace)
	}
}

func TestPole2BalancingScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := Pole2BalancingScape{}
	stabilize := scriptedStepAgent{
		id: "stabilize",
		fn: func(in []float64) []float64 {
			if len(in) < 6 {
				return []float64{0}
			}
			force := -(0.9*in[0] + 0.6*in[1] + 8.0*in[2] + 1.4*in[3] + 10.0*in[4] + 1.8*in[5])
			return []float64{force}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), stabilize, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	if maxSteps, ok := validationTrace["max_steps"].(int); !ok || maxSteps <= 0 {
		t.Fatalf("expected positive max_steps in validation trace, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), stabilize, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	if validationInit, vok := validationTrace["init_angle2"].(float64); vok {
		if testInit, tok := testTrace["init_angle2"].(float64); tok && validationInit == testInit {
			t.Fatalf("expected distinct mode initialization for pole2 windows, got validation=%f test=%f", validationInit, testInit)
		}
	}
}

func TestPole2BalancingScapeTraceIncludesTerminationAccounting(t *testing.T) {
	scape := Pole2BalancingScape{}
	thrash := scriptedStepAgent{
		id: "thrash",
		fn: func(_ []float64) []float64 { return []float64{1} },
	}

	_, trace, err := scape.Evaluate(context.Background(), thrash)
	if err != nil {
		t.Fatalf("evaluate thrash: %v", err)
	}
	if _, ok := trace["termination_reason"].(string); !ok {
		t.Fatalf("trace missing termination_reason: %+v", trace)
	}
	if _, ok := trace["fitness_acc"].(float64); !ok {
		t.Fatalf("trace missing fitness_acc: %+v", trace)
	}
	if _, ok := trace["avg_step_fitness"].(float64); !ok {
		t.Fatalf("trace missing avg_step_fitness: %+v", trace)
	}
	if _, ok := trace["goal_steps"].(int); !ok {
		t.Fatalf("trace missing goal_steps: %+v", trace)
	}
	if goalSteps, ok := trace["goal_steps"].(int); !ok || goalSteps != 100000 {
		t.Fatalf("expected reference goal_steps=100000 in gt mode, got %+v", trace)
	}
	if _, ok := trace["terminated_by_bounds"].(bool); !ok {
		t.Fatalf("trace missing terminated_by_bounds: %+v", trace)
	}
	if _, ok := trace["default_damping"].(bool); !ok {
		t.Fatalf("trace missing default_damping: %+v", trace)
	}
	if _, ok := trace["default_double_pole"].(bool); !ok {
		t.Fatalf("trace missing default_double_pole: %+v", trace)
	}
	if vectorSteps, ok := trace["vector_control_steps"].(int); !ok || vectorSteps != 0 {
		t.Fatalf("expected scalar thrash policy to report zero vector_control_steps, got %+v", trace)
	}
	if _, ok := trace["damping_off_steps"].(int); !ok {
		t.Fatalf("trace missing damping_off_steps: %+v", trace)
	}
	if _, ok := trace["single_pole_steps"].(int); !ok {
		t.Fatalf("trace missing single_pole_steps: %+v", trace)
	}
	if _, ok := trace["feature_width"].(int); !ok {
		t.Fatalf("trace missing feature_width: %+v", trace)
	}
	if _, ok := trace["mean_run_progress"].(float64); !ok {
		t.Fatalf("trace missing mean_run_progress: %+v", trace)
	}
	if _, ok := trace["mean_step_progress"].(float64); !ok {
		t.Fatalf("trace missing mean_step_progress: %+v", trace)
	}
	if _, ok := trace["mean_fitness_signal"].(float64); !ok {
		t.Fatalf("trace missing mean_fitness_signal: %+v", trace)
	}
	if _, ok := trace["last_run_progress"].(float64); !ok {
		t.Fatalf("trace missing last_run_progress: %+v", trace)
	}
	if _, ok := trace["last_step_progress"].(float64); !ok {
		t.Fatalf("trace missing last_step_progress: %+v", trace)
	}
	if _, ok := trace["last_fitness_signal"].(float64); !ok {
		t.Fatalf("trace missing last_fitness_signal: %+v", trace)
	}
}

func TestPole2BalancingScapeSupportsVectorPushControls(t *testing.T) {
	scape := Pole2BalancingScape{}
	vectorControl := scriptedStepAgent{
		id: "vector-control",
		fn: func(_ []float64) []float64 {
			// Force + damping flag + double-pole flag.
			// Negative damping flag disables damping; negative double-pole flag emulates single-pole mode.
			return []float64{0.25, -1.0, -1.0}
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), vectorControl)
	if err != nil {
		t.Fatalf("evaluate vector-control policy: %v", err)
	}
	vectorSteps, ok := trace["vector_control_steps"].(int)
	if !ok || vectorSteps <= 0 {
		t.Fatalf("expected positive vector_control_steps, got %+v", trace)
	}
	dampingOffSteps, ok := trace["damping_off_steps"].(int)
	if !ok || dampingOffSteps <= 0 {
		t.Fatalf("expected positive damping_off_steps, got %+v", trace)
	}
	singlePoleSteps, ok := trace["single_pole_steps"].(int)
	if !ok || singlePoleSteps <= 0 {
		t.Fatalf("expected positive single_pole_steps, got %+v", trace)
	}
}

func TestPole2BalancingCountsTerminalStepFitness(t *testing.T) {
	cfg := pole2ModeConfig{
		mode:       "terminal-step",
		maxSteps:   8,
		goalSteps:  100,
		angleLimit: 0.01,
		initAngle1: 0.02,
		damping:    true,
		doublePole: true,
	}

	_, trace, err := evaluatePole2Balancing(
		context.Background(),
		cfg,
		"step-agent",
		"derived",
		func(_ context.Context, _ pole2State, _ pole2WorkflowSignal) (pole2Control, error) {
			return pole2Control{force: 0, damping: true, doublePole: true}, nil
		},
	)
	if err != nil {
		t.Fatalf("evaluate terminal-step config: %v", err)
	}

	if steps, ok := trace["steps_survived"].(int); !ok || steps != 1 {
		t.Fatalf("expected single executed step before termination, got %+v", trace)
	}
	want := pole2StepFitness(1, pole2State{
		cartPosition: trace["cart_position"].(float64),
		cartVelocity: trace["cart_velocity"].(float64),
		angle1:       trace["angle1"].(float64),
		velocity1:    trace["velocity1"].(float64),
		angle2:       trace["angle2"].(float64),
		velocity2:    trace["velocity2"].(float64),
	}, true)
	if got, ok := trace["fitness_acc"].(float64); !ok || math.Abs(got-want) > 1e-9 {
		t.Fatalf("expected terminal step fitness_acc=%f, got %+v", want, trace)
	}
	if got, ok := trace["avg_step_fitness"].(float64); !ok || math.Abs(got-want) > 1e-9 {
		t.Fatalf("expected avg_step_fitness to include terminal step, want=%f got=%+v", want, trace)
	}
	if got, ok := trace["last_fitness_signal"].(float64); !ok || math.Abs(got-want) > 1e-9 {
		t.Fatalf("expected last_fitness_signal to reflect terminal step, want=%f got=%+v", want, trace)
	}
	if reason, ok := trace["termination_reason"].(string); !ok || reason != "angle1_limit" {
		t.Fatalf("expected angle1_limit termination, got %+v", trace)
	}
}
