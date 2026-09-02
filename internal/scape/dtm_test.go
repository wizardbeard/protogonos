package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestDTMScapeRewardsJunctionTurningPolicy(t *testing.T) {
	scape := DTMScape{}
	forward := scriptedStepAgent{
		id: "forward",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	junctionTurn := scriptedStepAgent{
		id: "junction-turn",
		fn: func(in []float64) []float64 {
			if len(in) >= 3 && in[0] > 0.5 && in[2] > 0.5 {
				return []float64{1}
			}
			return []float64{0}
		},
	}

	forwardFitness, _, err := scape.Evaluate(context.Background(), forward)
	if err != nil {
		t.Fatalf("evaluate forward: %v", err)
	}
	turnFitness, _, err := scape.Evaluate(context.Background(), junctionTurn)
	if err != nil {
		t.Fatalf("evaluate junction turn: %v", err)
	}
	if turnFitness <= forwardFitness {
		t.Fatalf("expected junction-turn policy to outperform forward policy, got turn=%f forward=%f", turnFitness, forwardFitness)
	}
}

func TestDTMScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.DTMRangeLeftSensorName,
			protoio.DTMRangeFrontSensorName,
			protoio.DTMRangeRightSensorName,
			protoio.DTMRewardSensorName,
		},
		ActuatorIDs: []string{protoio.DTMMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "rl", Activation: "identity"},
			{ID: "rf", Activation: "identity"},
			{ID: "rr", Activation: "identity"},
			{ID: "r", Activation: "identity"},
			{ID: "m", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "rl", To: "m", Weight: 1, Enabled: true},
			{From: "rr", To: "m", Weight: 1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.DTMRangeLeftSensorName:  protoio.NewScalarInputSensor(0),
		protoio.DTMRangeFrontSensorName: protoio.NewScalarInputSensor(0),
		protoio.DTMRangeRightSensorName: protoio.NewScalarInputSensor(0),
		protoio.DTMRewardSensorName:     protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.DTMMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"dtm-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"rl", "rf", "rr", "r"},
		[]string{"m"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := DTMScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["terminal_runs"].(int); !ok {
		t.Fatalf("trace missing terminal_runs: %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "all" {
		t.Fatalf("expected sensor_surface=all, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 4 {
		t.Fatalf("expected sensor_width=4, got %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface != protoio.DTMMoveActuatorName {
		t.Fatalf("expected control_surface=%s, got %+v", protoio.DTMMoveActuatorName, trace)
	}
}

func TestDTMScapeEvaluateWithExtendedIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.DTMRangeLeftSensorName,
			protoio.DTMRangeFrontSensorName,
			protoio.DTMRangeRightSensorName,
			protoio.DTMRewardSensorName,
			protoio.DTMRunProgressSensorName,
			protoio.DTMStepProgressSensorName,
			protoio.DTMSwitchedSensorName,
		},
		ActuatorIDs: []string{protoio.DTMMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "rl", Activation: "identity"},
			{ID: "rf", Activation: "identity"},
			{ID: "rr", Activation: "identity"},
			{ID: "r", Activation: "identity"},
			{ID: "rp", Activation: "identity"},
			{ID: "sp", Activation: "identity"},
			{ID: "sw", Activation: "identity"},
			{ID: "m", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "rl", To: "m", Weight: 0.8, Enabled: true},
			{From: "rr", To: "m", Weight: 0.8, Enabled: true},
			{From: "rp", To: "m", Weight: 0.2, Enabled: true},
			{From: "sp", To: "m", Weight: 0.1, Enabled: true},
			{From: "sw", To: "m", Weight: -0.1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.DTMRangeLeftSensorName:    protoio.NewScalarInputSensor(0),
		protoio.DTMRangeFrontSensorName:   protoio.NewScalarInputSensor(0),
		protoio.DTMRangeRightSensorName:   protoio.NewScalarInputSensor(0),
		protoio.DTMRewardSensorName:       protoio.NewScalarInputSensor(0),
		protoio.DTMRunProgressSensorName:  protoio.NewScalarInputSensor(0),
		protoio.DTMStepProgressSensorName: protoio.NewScalarInputSensor(0),
		protoio.DTMSwitchedSensorName:     protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.DTMMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"dtm-agent-extended-io",
		genome,
		sensors,
		actuators,
		[]string{"rl", "rf", "rr", "r", "rp", "sp", "sw"},
		[]string{"m"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := DTMScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["mean_run_progress"].(float64); !ok {
		t.Fatalf("trace missing mean_run_progress: %+v", trace)
	}
	if _, ok := trace["mean_step_progress"].(float64); !ok {
		t.Fatalf("trace missing mean_step_progress: %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "all" {
		t.Fatalf("expected sensor_surface=all for extended dtm IO, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 7 {
		t.Fatalf("expected sensor_width=7 for extended dtm IO, got %+v", trace)
	}
}

func TestDTMScapeEvaluateWithRangeOnlyIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.DTMRangeLeftSensorName,
			protoio.DTMRangeFrontSensorName,
			protoio.DTMRangeRightSensorName,
		},
		ActuatorIDs: []string{protoio.DTMMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "rl", Activation: "identity"},
			{ID: "rf", Activation: "identity"},
			{ID: "rr", Activation: "identity"},
			{ID: "m", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "rl", To: "m", Weight: 1, Enabled: true},
			{From: "rr", To: "m", Weight: 1, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.DTMRangeLeftSensorName:  protoio.NewScalarInputSensor(0),
		protoio.DTMRangeFrontSensorName: protoio.NewScalarInputSensor(0),
		protoio.DTMRangeRightSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.DTMMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"dtm-agent-range-only",
		genome,
		sensors,
		actuators,
		[]string{"rl", "rf", "rr"},
		[]string{"m"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := DTMScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "range_sense" {
		t.Fatalf("expected sensor_surface=range_sense, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 3 {
		t.Fatalf("expected sensor_width=3, got %+v", trace)
	}
}

func TestDTMScapeEvaluateWithTickSensorsAndNoActuatorSnapshot(t *testing.T) {
	agent := scriptedTickAgent{
		id: "dtm-tick-no-snapshot",
		sensors: map[string]protoio.Sensor{
			protoio.DTMRangeLeftSensorName:  protoio.NewScalarInputSensor(0),
			protoio.DTMRangeFrontSensorName: protoio.NewScalarInputSensor(0),
			protoio.DTMRangeRightSensorName: protoio.NewScalarInputSensor(0),
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			left, err := sensors[protoio.DTMRangeLeftSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			right, err := sensors[protoio.DTMRangeRightSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			leftValue := 0.0
			if len(left) > 0 {
				leftValue = left[0]
			}
			rightValue := 0.0
			if len(right) > 0 {
				rightValue = right[0]
			}
			if leftValue > 0.5 && rightValue > 0.5 {
				return []float64{1}, nil
			}
			return []float64{0}, nil
		},
	}

	scape := DTMScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent without snapshot actuator: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "range_sense" {
		t.Fatalf("expected sensor_surface=range_sense, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 3 {
		t.Fatalf("expected sensor_width=3, got %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface != protoio.DTMMoveActuatorName {
		t.Fatalf("expected control_surface=%s, got %+v", protoio.DTMMoveActuatorName, trace)
	}
}

func TestDTMScapeEvaluateWithRewardOnlyIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs:   []string{protoio.DTMRewardSensorName},
		ActuatorIDs: []string{protoio.DTMMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "r", Activation: "identity"},
			{ID: "m", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "r", To: "m", Weight: 0.7, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.DTMRewardSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.DTMMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"dtm-agent-reward-only",
		genome,
		sensors,
		actuators,
		[]string{"r"},
		[]string{"m"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := DTMScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "reward" {
		t.Fatalf("expected sensor_surface=reward, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 1 {
		t.Fatalf("expected sensor_width=1, got %+v", trace)
	}
}

func TestDTMScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := DTMScape{}
	junctionTurn := scriptedStepAgent{
		id: "junction-turn",
		fn: func(in []float64) []float64 {
			if len(in) >= 3 && in[0] > 0.5 && in[2] > 0.5 {
				return []float64{1}
			}
			return []float64{0}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), junctionTurn, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	validationRuns, ok := validationTrace["total_runs"].(int)
	if !ok || validationRuns <= 0 {
		t.Fatalf("expected positive validation total_runs, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), junctionTurn, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	testRuns, ok := testTrace["total_runs"].(int)
	if !ok || testRuns != validationRuns {
		t.Fatalf("expected matching total_runs between validation/test windows, got validation=%d test=%+v", validationRuns, testTrace["total_runs"])
	}
}

func TestDTMScapeTraceIncludesRunDiagnostics(t *testing.T) {
	scape := DTMScape{}
	junctionTurn := scriptedStepAgent{
		id: "junction-turn",
		fn: func(in []float64) []float64 {
			if len(in) >= 3 && in[0] > 0.5 && in[2] > 0.5 {
				return []float64{1}
			}
			return []float64{0}
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), junctionTurn)
	if err != nil {
		t.Fatalf("evaluate junction turn: %v", err)
	}
	if _, ok := trace["terminal_reward_total"].(float64); !ok {
		t.Fatalf("trace missing terminal_reward_total: %+v", trace)
	}
	if _, ok := trace["avg_steps_per_run"].(float64); !ok {
		t.Fatalf("trace missing avg_steps_per_run: %+v", trace)
	}
	if _, ok := trace["fitness_delta"].(float64); !ok {
		t.Fatalf("trace missing fitness_delta: %+v", trace)
	}
	if _, ok := trace["mean_switched_signal"].(float64); !ok {
		t.Fatalf("trace missing mean_switched_signal: %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "step_input" {
		t.Fatalf("expected sensor_surface=step_input, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 7 {
		t.Fatalf("expected sensor_width=7, got %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface != "step_output" {
		t.Fatalf("expected control_surface=step_output, got %+v", trace)
	}
	if width, ok := trace["feature_width"].(int); !ok || width != 7 {
		t.Fatalf("expected feature_width=7, got %+v", trace)
	}
	if _, ok := trace["switch_triggered_at"].(int); !ok {
		t.Fatalf("trace missing switch_triggered_at: %+v", trace)
	}
	leftRuns, lok := trace["left_terminal_runs"].(int)
	rightRuns, rok := trace["right_terminal_runs"].(int)
	terminalRuns, tok := trace["terminal_runs"].(int)
	if !lok || !rok || !tok {
		t.Fatalf("trace missing terminal run side diagnostics: %+v", trace)
	}
	if leftRuns+rightRuns != terminalRuns {
		t.Fatalf("expected side terminal runs to sum to terminal_runs, got left=%d right=%d total=%d", leftRuns, rightRuns, terminalRuns)
	}
	timeoutRuns, ok := trace["timeout_runs"].(int)
	if !ok {
		t.Fatalf("trace missing timeout_runs: %+v", trace)
	}
	if timeoutRuns != 0 {
		t.Fatalf("expected zero timeout_runs for reference-style dtm run flow, got %+v", trace)
	}
	maxRunStepIndex, ok := trace["max_run_step_index"].(int)
	if !ok || maxRunStepIndex <= 0 {
		t.Fatalf("expected positive max_run_step_index, got %+v", trace)
	}
	if maxRunStepIndex > 3 {
		t.Fatalf("expected bounded dtm run-step depth <= 3, got %+v", trace)
	}
	if stepIndex, ok := trace["last_step_index"].(int); !ok || stepIndex != 0 {
		t.Fatalf("expected reset last_step_index=0 after terminal episode completion, got %+v", trace)
	}
	totalRuns, ok := trace["total_runs"].(int)
	if !ok || totalRuns <= 0 {
		t.Fatalf("trace missing positive total_runs: %+v", trace)
	}
	if runIndex, ok := trace["last_run_index"].(int); !ok || runIndex != totalRuns-1 {
		t.Fatalf("expected last_run_index to report the final completed run, got %+v", trace)
	}
}

func TestDTMScapeSingleRunDoesNotTriggerRewardSwitch(t *testing.T) {
	cfg := dtmModeConfig{
		mode:           "unit",
		totalRuns:      1,
		maxStepsPerRun: 0,
		switchFloor:    20,
		switchSpread:   20,
	}

	fitness, trace, err := evaluateDTM(
		context.Background(),
		"dtm-single-run",
		cfg,
		"step_input",
		7,
		"step_output",
		func(_ context.Context, sense dtmSenseInput) (float64, error) {
			if len(sense.vector) >= 3 && sense.vector[0] > 0.5 && sense.vector[2] > 0.5 {
				return 1, nil
			}
			return 0, nil
		},
	)
	if err != nil {
		t.Fatalf("evaluate single-run dtm: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness from single terminal run, got %f", fitness)
	}
	if switched, ok := trace["switched"].(bool); !ok || switched {
		t.Fatalf("expected single-run episode to skip reward switch, got %+v", trace)
	}
	if triggered, ok := trace["switch_triggered_at"].(int); !ok || triggered != -1 {
		t.Fatalf("expected switch_triggered_at=-1 for single-run episode, got %+v", trace)
	}
	if totalRuns, ok := trace["total_runs"].(int); !ok || totalRuns != 1 {
		t.Fatalf("expected total_runs=1, got %+v", trace)
	}
	if runIndex, ok := trace["last_run_index"].(int); !ok || runIndex != 0 {
		t.Fatalf("expected single completed run to report last_run_index=0, got %+v", trace)
	}
}

func TestDTMSimulatorSenseMoveStateAndReset(t *testing.T) {
	sim, err := NewDTMSimulator("gt")
	if err != nil {
		t.Fatalf("new dtm simulator: %v", err)
	}

	all, err := sim.Sense(context.Background(), "all")
	if err != nil {
		t.Fatalf("sense all: %v", err)
	}
	if len(all) != 4 {
		t.Fatalf("expected all sense width 4, got %d", len(all))
	}
	ranges, err := sim.Sense(context.Background(), "range_sense")
	if err != nil {
		t.Fatalf("sense range_sense: %v", err)
	}
	if len(ranges) != 3 {
		t.Fatalf("expected range_sense width 3, got %d", len(ranges))
	}
	reward, err := sim.Sense(context.Background(), "reward")
	if err != nil {
		t.Fatalf("sense reward: %v", err)
	}
	if len(reward) != 1 || reward[0] != 0 {
		t.Fatalf("expected start reward [0], got %+v", reward)
	}

	fitness, end, err := sim.Move(context.Background(), []float64{0})
	if err != nil {
		t.Fatalf("move forward: %v", err)
	}
	if end || fitness != 0 {
		t.Fatalf("expected non-terminal zero-fitness move, got fitness=%f end=%t", fitness, end)
	}
	state := sim.State()
	if state.PositionX != 0 || state.PositionY != 1 || state.Direction != 90 || state.StepIndex != 1 {
		t.Fatalf("unexpected state after forward move: %+v", state)
	}
	if state.LastMove != 0 || state.LastMoveAction != "forward" || state.StepsExecuted != 1 || state.LastFitness != 0 {
		t.Fatalf("unexpected move diagnostics after forward move: %+v", state)
	}

	sim.Reset()
	state = sim.State()
	if state.PositionX != 0 || state.PositionY != 0 || state.Direction != 90 || state.RunIndex != 0 || state.StepIndex != 0 || state.Halted {
		t.Fatalf("unexpected state after reset: %+v", state)
	}
	if state.StepsExecuted != 0 || state.LastMoveAction != "" || state.TerminationReason != "" || state.LastFitness != 0 {
		t.Fatalf("expected reset to clear diagnostics, got %+v", state)
	}
}

func TestDTMSimulatorTerminalRunAccounting(t *testing.T) {
	sim, err := NewDTMSimulator("gt")
	if err != nil {
		t.Fatalf("new dtm simulator: %v", err)
	}
	sim.episode.position = dtmCoord{x: 1, y: 1}
	sim.episode.runIndex = sim.episode.totalRuns - 1
	sim.episode.fitnessAcc = 50

	fitness, end, err := sim.Move(context.Background(), []float64{0})
	if err != nil {
		t.Fatalf("terminal move: %v", err)
	}
	if !end {
		t.Fatalf("expected terminal move to end")
	}
	if fitness != 51 {
		t.Fatalf("expected terminal fitness 51, got %f", fitness)
	}
	state := sim.State()
	if !state.Halted || state.RunIndex != state.TotalRuns || state.StepIndex != 0 {
		t.Fatalf("unexpected terminal state: %+v", state)
	}
	if state.TerminationReason != "terminal" || state.TerminalRuns != 1 || state.LastReward != 1 || state.LastFitness != fitness {
		t.Fatalf("unexpected terminal diagnostics, fitness=%f state=%+v", fitness, state)
	}
	if state.RightTerminalRuns != 1 || state.LeftTerminalRuns != 0 || state.StepsExecuted != 1 {
		t.Fatalf("unexpected terminal side counters, got %+v", state)
	}
	if _, err := sim.Sense(context.Background(), "all"); err == nil {
		t.Fatalf("expected halted simulator sense to fail")
	}
}

func TestDTMSimulatorSwitchEventState(t *testing.T) {
	sim, err := NewDTMSimulator("gt")
	if err != nil {
		t.Fatalf("new dtm simulator: %v", err)
	}
	sim.episode.runIndex = sim.episode.switchEvent
	if sim.State().Switched {
		t.Fatalf("expected simulator to start unswitched")
	}

	_, err = sim.Sense(context.Background(), "all")
	if err != nil {
		t.Fatalf("sense at switch event: %v", err)
	}
	if !sim.State().Switched {
		t.Fatalf("expected simulator sense to record switched state")
	}
	sim.episode.position = dtmCoord{x: 0, y: 0}
	sim.episode.direction = 90

	_, end, err := sim.Move(context.Background(), []float64{0})
	if err != nil {
		t.Fatalf("move at switch event: %v", err)
	}
	if end {
		t.Fatalf("expected switch-event move to remain non-terminal")
	}
	state := sim.State()
	if !state.Switched {
		t.Fatalf("expected simulator to record switched state: %+v", state)
	}
	if sim.episode.sectors[dtmCoord{x: 1, y: 1}].reward != 0.2 {
		t.Fatalf("expected right reward sector to be swapped")
	}
	if sim.episode.sectors[dtmCoord{x: -1, y: 1}].reward != 1 {
		t.Fatalf("expected left reward sector to be swapped")
	}
}

func TestDTMProcessCommandWrapper(t *testing.T) {
	process := NewDTMProcess()
	ctx := context.Background()

	if response := process.Call(ctx, DTMSenseMessage{Parameter: "all"}); response.Err == nil {
		t.Fatal("expected sense before start to fail")
	}
	start := process.Call(ctx, DTMStartMessage{Mode: "validation"})
	if start.Err != nil || !start.OK {
		t.Fatalf("start response=%+v", start)
	}
	if start.State.Mode != "validation" || start.State.Halted {
		t.Fatalf("unexpected start state=%+v", start.State)
	}

	sense := process.Call(ctx, DTMSenseMessage{Parameter: "range_sense"})
	if sense.Err != nil || !sense.OK {
		t.Fatalf("sense response=%+v", sense)
	}
	if len(sense.Percept) != 3 {
		t.Fatalf("expected range_sense width 3, got response=%+v", sense)
	}

	move := process.Call(ctx, DTMMoveMessage{Output: []float64{0}})
	if move.Err != nil || !move.OK {
		t.Fatalf("move response=%+v", move)
	}
	if move.End || move.Fitness != 0 {
		t.Fatalf("expected non-terminal move response, got %+v", move)
	}
	if move.State.PositionX != 0 || move.State.PositionY != 1 || move.State.StepIndex != 1 {
		t.Fatalf("unexpected move state=%+v", move.State)
	}
	if move.State.LastMoveAction != "forward" || move.State.StepsExecuted != 1 {
		t.Fatalf("expected process move diagnostics, got %+v", move.State)
	}

	state := process.Call(ctx, DTMStateMessage{})
	if state.Err != nil || !state.OK || state.State.StepIndex != 1 {
		t.Fatalf("state response=%+v", state)
	}

	stop := process.Call(ctx, DTMStopMessage{Reason: "normal"})
	if stop.Err != nil || !stop.OK || !stop.End {
		t.Fatalf("stop response=%+v", stop)
	}
	if stop.StopReason != "normal" || !stop.State.Halted {
		t.Fatalf("unexpected stop response=%+v", stop)
	}
	if stop.State.TerminationReason != "normal" {
		t.Fatalf("expected stop reason in state, got %+v", stop.State)
	}
	if response := process.Call(ctx, DTMSenseMessage{Parameter: "all"}); response.Err == nil {
		t.Fatal("expected sense after stop to fail")
	}

	restart := process.Call(ctx, DTMRestartMessage{})
	if restart.Err != nil || !restart.OK {
		t.Fatalf("restart response=%+v", restart)
	}
	if restart.State.Halted || restart.State.RunIndex != 0 || restart.State.StepIndex != 0 || restart.State.Mode != "validation" {
		t.Fatalf("unexpected restart response=%+v", restart)
	}
	if restart.State.StepsExecuted != 0 || restart.State.TerminationReason != "" || restart.State.LastMoveAction != "" {
		t.Fatalf("expected restart to clear process diagnostics, got %+v", restart.State)
	}
}
