package scape

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

type scriptedStepAgent struct {
	id string
	fn func(input []float64) []float64
}

func (a scriptedStepAgent) ID() string { return a.id }

func (a scriptedStepAgent) RunStep(_ context.Context, input []float64) ([]float64, error) {
	return a.fn(input), nil
}

func flatlandGreedyForager(input []float64) []float64 {
	if len(input) == 0 {
		return []float64{0}
	}
	if input[0] > 0 {
		return []float64{1}
	}
	if input[0] < 0 {
		return []float64{-1}
	}
	return []float64{0}
}

func TestFlatlandPublicProcessCommandWrapper(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicTickMessage{}); response.Err == nil {
		t.Fatal("expected tick before start to fail")
	}
	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})

	enter := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{
		ID:   "agent-1",
		Mode: "validation",
	}})
	if enter.Err != nil || !enter.OK {
		t.Fatalf("enter response=%+v", enter)
	}

	sense := process.Call(ctx, FlatlandPublicSenseMessage{AgentID: "agent-1"})
	if sense.Err != nil || !sense.OK {
		t.Fatalf("sense response=%+v", sense)
	}
	if len(sense.Percept) != flatlandBaseFeatureWidth+flatlandScannerWidth {
		t.Fatalf("unexpected percept width=%d response=%+v", len(sense.Percept), sense)
	}
	if surface, _ := sense.Trace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected step_input sense trace, got %+v", sense.Trace)
	}

	act := process.Call(ctx, FlatlandPublicActMessage{AgentID: "agent-1", Output: []float64{1}})
	if act.Err != nil || !act.OK {
		t.Fatalf("act response=%+v", act)
	}
	if act.Fitness <= 0 {
		t.Fatalf("expected positive act fitness, got %+v", act)
	}
	if math.Abs(float64(act.Fitness)-0.001) > 1e-12 {
		t.Fatalf("expected reference-style immediate act fitness=0.001, got %+v", act)
	}
	if referenceFitness, _ := act.Trace["reference_fitness"].(float64); math.Abs(referenceFitness-0.001) > 1e-12 {
		t.Fatalf("expected reference_fitness trace=0.001, got %+v", act.Trace)
	}
	if shapedFitness, _ := act.Trace["shaped_fitness"].(float64); shapedFitness <= float64(act.Fitness) {
		t.Fatalf("expected shaped_fitness diagnostic above immediate feedback, got %+v", act.Trace)
	}
	if width, _ := act.Trace["last_control_width"].(int); width != 1 {
		t.Fatalf("expected single-channel control trace, got %+v", act.Trace)
	}

	agents := process.Call(ctx, FlatlandPublicGetAllMessage{})
	if agents.Err != nil || !agents.OK || len(agents.Agents) != 1 {
		t.Fatalf("get_all response=%+v", agents)
	}
	if len(agents.Avatars) != 1 || agents.Avatars[0].ID != "agent-1" {
		t.Fatalf("expected typed avatar snapshot for agent-1, response=%+v", agents)
	}

	update := process.Call(ctx, FlatlandPublicUpdateAgentsMessage{Agents: []FlatlandPublicAgent{
		{ID: "agent-1"},
		{ID: "agent-2", Mode: "benchmark"},
	}})
	if update.Err != nil || !update.OK {
		t.Fatalf("update response=%+v", update)
	}
	if update.Update.Created != 1 || update.Update.Preserved != 1 || update.Update.ActiveAfter != 2 {
		t.Fatalf("unexpected update summary: %+v", update.Update)
	}
	if len(update.Avatars) != 2 {
		t.Fatalf("expected two post-update avatars, response=%+v", update)
	}
	tick := process.Call(ctx, FlatlandPublicTickMessage{})
	if tick.Err != nil || !tick.OK {
		t.Fatalf("tick response=%+v", tick)
	}
	if active, _ := tick.Trace["active_agents"].(int); active != 2 {
		t.Fatalf("expected two active agents, trace=%+v", tick.Trace)
	}
	if len(tick.Avatars) != 2 {
		t.Fatalf("expected process tick to return two typed avatar snapshots, response=%+v", tick)
	}
	if avatars, ok := tick.Trace["avatars"].([]FlatlandPublicAvatarSnapshot); !ok || len(avatars) != 2 {
		t.Fatalf("expected typed avatars in process tick trace, trace=%+v", tick.Trace)
	}

	leave := process.Call(ctx, FlatlandPublicLeaveMessage{AgentID: "agent-1"})
	if leave.Err != nil || !leave.OK {
		t.Fatalf("leave response=%+v", leave)
	}
	if response := process.Call(ctx, FlatlandPublicSyncMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("sync response=%+v", response)
	}
	stop := process.Call(ctx, FlatlandPublicStopMessage{Reason: "shutdown"})
	if stop.Err != nil || !stop.OK || stop.StopReason != "shutdown" {
		t.Fatalf("stop response=%+v", stop)
	}
}

func TestFlatlandProcessIOAdapters(t *testing.T) {
	sensorIDs := []string{
		protoio.DistanceScannerSensorAliasName,
		protoio.ColorScannerSensorAliasName,
		protoio.EnergyScannerSensorAliasName,
	}
	actuatorIDs := []string{protoio.TwoWheelsActuatorAliasName}

	sensors, actuators, err := NewFlatlandProcessIO("gt", sensorIDs, actuatorIDs)
	if err != nil {
		t.Fatalf("new flatland process io: %v", err)
	}
	t.Cleanup(func() {
		_ = FlatlandScape{}.StopWithReason(context.Background(), "normal")
	})

	for _, sensorID := range sensorIDs {
		reader, ok := sensors[sensorID].(protoio.SensorProcessReader)
		if !ok {
			t.Fatalf("sensor %s does not expose process reader", sensorID)
		}
		values, err := reader.ReadForSensorProcess(context.Background(), protoio.SensorProcessCall{
			Scape:      "flatland",
			SensorName: sensorID,
			VL:         flatlandScannerDensity,
			OpMode:     "gt",
		})
		if err != nil {
			t.Fatalf("read process sensor %s: %v", sensorID, err)
		}
		if len(values) != flatlandScannerDensity {
			t.Fatalf("expected scanner width %d from %s, got %+v", flatlandScannerDensity, sensorID, values)
		}
	}

	writer, ok := actuators[protoio.TwoWheelsActuatorAliasName].(protoio.ActuatorProcessWriter)
	if !ok {
		t.Fatal("two_wheels actuator does not expose process writer")
	}
	sync, err := writer.WriteForActuatorProcess(context.Background(), protoio.ActuatorProcessCall{
		Scape:        "flatland",
		ActuatorName: protoio.TwoWheelsActuatorAliasName,
		OpMode:       "gt",
		Output:       []float64{1, 1},
	})
	if err != nil {
		t.Fatalf("write process actuator: %v", err)
	}
	if sync.EndFlag != 0 || len(sync.Fitness) != 1 {
		t.Fatalf("expected non-terminal single-channel flatland sync, got %+v", sync)
	}
	if math.Abs(sync.Fitness[0]-0.001) > 1e-12 {
		t.Fatalf("expected reference-style immediate flatland sync fitness=0.001, got %+v", sync)
	}
}

func TestFlatlandProcessIOAcceptsPublicCommandActuators(t *testing.T) {
	sensors, actuators, err := NewFlatlandProcessIO(
		"gt",
		[]string{protoio.DistanceScannerSensorAliasName},
		[]string{
			protoio.FlatlandSpeakActuatorAliasName,
			protoio.FlatlandGestaltActuatorName,
			protoio.FlatlandSpearActuatorName,
			protoio.FlatlandShootActuatorName,
			protoio.FlatlandCreateOffspringActuatorName,
		},
	)
	if err != nil {
		t.Fatalf("new flatland command process io: %v", err)
	}
	if len(sensors) != 1 {
		t.Fatalf("expected one process sensor, got=%v", sensors)
	}
	for _, actuatorName := range []string{
		protoio.FlatlandSpeakActuatorAliasName,
		protoio.FlatlandGestaltActuatorName,
		protoio.FlatlandSpearActuatorName,
		protoio.FlatlandShootActuatorName,
		protoio.FlatlandCreateOffspringActuatorName,
	} {
		if _, ok := actuators[actuatorName].(protoio.ActuatorProcessWriter); !ok {
			t.Fatalf("expected flatland command actuator %s to expose process writer", actuatorName)
		}
	}
}

func TestFlatlandTwoWheelsControlUsesReferenceSpeedAndTurn(t *testing.T) {
	control, err := flatlandControlFromActuatorOutput(protoio.FlatlandTwoWheelsActuatorName, []float64{1, -1})
	if err != nil {
		t.Fatalf("two-wheels control: %v", err)
	}
	if control.width != 2 || !control.twoWheels {
		t.Fatalf("expected two-wheel control metadata, got %+v", control)
	}
	if control.move != 0 {
		t.Fatalf("expected zero speed from opposite wheels, got %+v", control)
	}
	if control.turn != 1 {
		t.Fatalf("expected positive turn from right-left wheel delta, got %+v", control)
	}
}

func TestFlatlandPublicProcessStateActuatorCommands(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "speaker"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter response=%+v", response)
	}
	before := process.Call(ctx, FlatlandPublicSenseMessage{AgentID: "speaker"})
	if before.Err != nil || !before.OK {
		t.Fatalf("sense response=%+v", before)
	}
	startAge, _ := before.Trace["age"].(int)
	startEnergy, _ := before.Trace["energy"].(float64)

	speak := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "speaker",
		ActuatorName: protoio.FlatlandSpeakActuatorName,
		Output:       []float64{0.75},
	})
	if speak.Err != nil || !speak.OK {
		t.Fatalf("speak response=%+v", speak)
	}
	if sound, _ := speak.Trace["sound"].(float64); sound != 0.75 {
		t.Fatalf("expected sound=0.75, trace=%+v", speak.Trace)
	}
	if age, _ := speak.Trace["age"].(int); age != startAge {
		t.Fatalf("expected speak to leave age unchanged, before=%d trace=%+v", startAge, speak.Trace)
	}
	if energy, _ := speak.Trace["energy"].(float64); energy != startEnergy {
		t.Fatalf("expected speak to leave energy unchanged, before=%f trace=%+v", startEnergy, speak.Trace)
	}

	gestalt := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "speaker",
		ActuatorName: protoio.FlatlandGestaltActuatorName,
		Output:       []float64{0.1, 0.2, 0.3},
	})
	if gestalt.Err != nil || !gestalt.OK {
		t.Fatalf("gestalt response=%+v", gestalt)
	}
	gotGestalt := mustTraceFloat64Slice(t, gestalt.Trace, "gestalt")
	if !reflect.DeepEqual(gotGestalt, []float64{0.1, 0.2, 0.3}) {
		t.Fatalf("unexpected gestalt trace=%+v", gestalt.Trace)
	}
	if fitness := float64(gestalt.Fitness); math.Abs(fitness-0.001) > 1e-12 {
		t.Fatalf("expected reference-style state-actuator fitness=0.001, got %+v", gestalt)
	}
}

func TestFlatlandPublicProcessSpearCommandUsesReferenceEnergyGate(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "hunter"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter response=%+v", response)
	}

	low := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "hunter",
		ActuatorName: protoio.FlatlandSpearActuatorName,
		Output:       []float64{1},
	})
	if low.Err != nil || !low.OK {
		t.Fatalf("low-energy spear response=%+v", low)
	}
	if spear, _ := low.Trace["spear"].(bool); spear {
		t.Fatalf("expected low-energy spear to remain disabled, trace=%+v", low.Trace)
	}
	if energy, _ := low.Trace["energy"].(float64); math.Abs(energy-(flatlandInitialEnergy-1)) > 1e-12 {
		t.Fatalf("expected low-energy spear cost=1, trace=%+v", low.Trace)
	}
	if end, _ := low.Trace["end"].(bool); end {
		t.Fatalf("expected low-energy spear not to terminate at default energy, trace=%+v", low.Trace)
	}

	process.runtime.mu.Lock()
	state := process.runtime.agents["hunter"]
	state.episode.energy = 150
	state.spear = false
	process.runtime.mu.Unlock()

	high := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "hunter",
		ActuatorName: protoio.FlatlandSpearActuatorName,
		Output:       []float64{1},
	})
	if high.Err != nil || !high.OK {
		t.Fatalf("high-energy spear response=%+v", high)
	}
	if spear, _ := high.Trace["spear"].(bool); !spear {
		t.Fatalf("expected high-energy spear to enable spear flag, trace=%+v", high.Trace)
	}
	if energy, _ := high.Trace["energy"].(float64); math.Abs(energy-140) > 1e-12 {
		t.Fatalf("expected high-energy spear cost=10, trace=%+v", high.Trace)
	}
}

func TestFlatlandPublicProcessSpearCommandDestroysForwardPrey(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "hunter"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter response=%+v", response)
	}

	process.runtime.mu.Lock()
	state := process.runtime.agents["hunter"]
	state.episode.energy = 150
	state.episode.position = 0
	state.episode.heading = 1
	state.episode.food = nil
	state.episode.poison = nil
	state.episode.prey = []flatlandResource{{position: flatlandSpearReach, potency: flatlandPreyEnergyMax}}
	state.episode.predators = nil
	state.spear = false
	process.runtime.mu.Unlock()

	response := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "hunter",
		ActuatorName: protoio.FlatlandSpearActuatorAliasName,
		Output:       []float64{1},
	})
	if response.Err != nil || !response.OK {
		t.Fatalf("spear response=%+v", response)
	}
	if hit, _ := response.Trace["spear_prey_hit"].(bool); !hit {
		t.Fatalf("expected spear_prey_hit=true, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["spear_kills"].(int); kills != 1 {
		t.Fatalf("expected spear_kills=1, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["spear_prey_kills"].(int); kills != 1 {
		t.Fatalf("expected spear_prey_kills=1, trace=%+v", response.Trace)
	}
	if energy, _ := response.Trace["energy"].(float64); math.Abs(energy-flatlandEnergyCap) > 1e-12 {
		t.Fatalf("expected spear prey credit to saturate energy at cap, trace=%+v", response.Trace)
	}
	if state.episode.prey[0].cooldown != flatlandPreyRespawn {
		t.Fatalf("expected speared prey cooldown=%d, got=%d", flatlandPreyRespawn, state.episode.prey[0].cooldown)
	}
}

func TestFlatlandPublicProcessSpearCommandDestroysForwardPredator(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "hunter"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter response=%+v", response)
	}

	process.runtime.mu.Lock()
	state := process.runtime.agents["hunter"]
	state.episode.energy = 150
	state.episode.position = 0
	state.episode.heading = 1
	state.episode.food = nil
	state.episode.poison = nil
	state.episode.prey = nil
	state.episode.predators = []flatlandResource{{position: flatlandSpearReach, potency: flatlandPredatorDamageMax}}
	state.spear = false
	process.runtime.mu.Unlock()

	response := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "hunter",
		ActuatorName: protoio.FlatlandSpearActuatorName,
		Output:       []float64{1},
	})
	if response.Err != nil || !response.OK {
		t.Fatalf("spear response=%+v", response)
	}
	if hit, _ := response.Trace["spear_predator_hit"].(bool); !hit {
		t.Fatalf("expected spear_predator_hit=true, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["spear_kills"].(int); kills != 1 {
		t.Fatalf("expected spear_kills=1, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["spear_predator_kills"].(int); kills != 1 {
		t.Fatalf("expected spear_predator_kills=1, trace=%+v", response.Trace)
	}
	if state.episode.predators[0].cooldown != flatlandPredatorRespawn {
		t.Fatalf("expected speared predator cooldown=%d, got=%d", flatlandPredatorRespawn, state.episode.predators[0].cooldown)
	}
}

func TestFlatlandPublicProcessShootCommandUsesReferenceEnergyGate(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "shooter"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter response=%+v", response)
	}

	noOp := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "shooter",
		ActuatorName: protoio.FlatlandShootActuatorAliasName,
		Output:       []float64{0},
	})
	if noOp.Err != nil || !noOp.OK {
		t.Fatalf("no-op shoot response=%+v", noOp)
	}
	if energy, _ := noOp.Trace["energy"].(float64); math.Abs(energy-flatlandInitialEnergy) > 1e-12 {
		t.Fatalf("expected non-positive shoot output to leave energy unchanged, trace=%+v", noOp.Trace)
	}

	low := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "shooter",
		ActuatorName: protoio.FlatlandShootActuatorName,
		Output:       []float64{1},
	})
	if low.Err != nil || !low.OK {
		t.Fatalf("low-energy shoot response=%+v", low)
	}
	if energy, _ := low.Trace["energy"].(float64); math.Abs(energy-(flatlandInitialEnergy-1)) > 1e-12 {
		t.Fatalf("expected low-energy shoot cost=1, trace=%+v", low.Trace)
	}
	if spear, _ := low.Trace["spear"].(bool); spear {
		t.Fatalf("expected shoot not to enable spear, trace=%+v", low.Trace)
	}

	process.runtime.mu.Lock()
	state := process.runtime.agents["shooter"]
	state.episode.energy = 150
	state.spear = true
	process.runtime.mu.Unlock()

	high := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "shooter",
		ActuatorName: protoio.FlatlandShootActuatorAliasName,
		Output:       []float64{1},
	})
	if high.Err != nil || !high.OK {
		t.Fatalf("high-energy shoot response=%+v", high)
	}
	if energy, _ := high.Trace["energy"].(float64); math.Abs(energy-130) > 1e-12 {
		t.Fatalf("expected high-energy shoot cost=20, trace=%+v", high.Trace)
	}
	if spear, _ := high.Trace["spear"].(bool); !spear {
		t.Fatalf("expected shoot to preserve spear flag, trace=%+v", high.Trace)
	}
}

func TestFlatlandPublicProcessShootCommandHitsForwardPrey(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "shooter"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter response=%+v", response)
	}

	process.runtime.mu.Lock()
	state := process.runtime.agents["shooter"]
	state.episode.energy = 150
	state.episode.position = 0
	state.episode.heading = 1
	state.episode.food = nil
	state.episode.poison = nil
	state.episode.prey = []flatlandResource{{position: flatlandShootReach, potency: flatlandPreyEnergyMax}}
	state.episode.predators = nil
	state.spear = true
	process.runtime.mu.Unlock()

	response := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "shooter",
		ActuatorName: protoio.FlatlandShootActuatorName,
		Output:       []float64{1},
	})
	if response.Err != nil || !response.OK {
		t.Fatalf("shoot response=%+v", response)
	}
	if fired, _ := response.Trace["shot_fired"].(bool); !fired {
		t.Fatalf("expected shot_fired=true, trace=%+v", response.Trace)
	}
	if hit, _ := response.Trace["shoot_prey_hit"].(bool); !hit {
		t.Fatalf("expected shoot_prey_hit=true, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["shoot_kills"].(int); kills != 1 {
		t.Fatalf("expected shoot_kills=1, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["shoot_prey_kills"].(int); kills != 1 {
		t.Fatalf("expected shoot_prey_kills=1, trace=%+v", response.Trace)
	}
	if spear, _ := response.Trace["spear"].(bool); !spear {
		t.Fatalf("expected shoot to preserve spear flag, trace=%+v", response.Trace)
	}
	if state.episode.prey[0].cooldown != flatlandPreyRespawn {
		t.Fatalf("expected shot prey cooldown=%d, got=%d", flatlandPreyRespawn, state.episode.prey[0].cooldown)
	}
}

func TestFlatlandPublicProcessShootCommandHitsForwardPredator(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "shooter"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter response=%+v", response)
	}

	process.runtime.mu.Lock()
	state := process.runtime.agents["shooter"]
	state.episode.energy = 150
	state.episode.position = 0
	state.episode.heading = 1
	state.episode.food = nil
	state.episode.poison = nil
	state.episode.prey = nil
	state.episode.predators = []flatlandResource{{position: flatlandShootReach, potency: flatlandPredatorDamageMax}}
	process.runtime.mu.Unlock()

	response := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "shooter",
		ActuatorName: protoio.FlatlandShootActuatorAliasName,
		Output:       []float64{1},
	})
	if response.Err != nil || !response.OK {
		t.Fatalf("shoot response=%+v", response)
	}
	if fired, _ := response.Trace["shot_fired"].(bool); !fired {
		t.Fatalf("expected shot_fired=true, trace=%+v", response.Trace)
	}
	if hit, _ := response.Trace["shoot_predator_hit"].(bool); !hit {
		t.Fatalf("expected shoot_predator_hit=true, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["shoot_kills"].(int); kills != 1 {
		t.Fatalf("expected shoot_kills=1, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["shoot_predator_kills"].(int); kills != 1 {
		t.Fatalf("expected shoot_predator_kills=1, trace=%+v", response.Trace)
	}
	if state.episode.predators[0].cooldown != flatlandPredatorRespawn {
		t.Fatalf("expected shot predator cooldown=%d, got=%d", flatlandPredatorRespawn, state.episode.predators[0].cooldown)
	}
}

func TestFlatlandPublicProcessMoveTracksPublicAgentCollision(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	for _, id := range []string{"mover", "blocker"} {
		if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: id}}); response.Err != nil || !response.OK {
			t.Fatalf("enter %s response=%+v", id, response)
		}
	}

	process.runtime.mu.Lock()
	mover := process.runtime.agents["mover"]
	blocker := process.runtime.agents["blocker"]
	mover.episode.position = 0
	mover.episode.heading = 1
	mover.episode.food = nil
	mover.episode.poison = nil
	mover.episode.prey = nil
	mover.episode.predators = nil
	blocker.episode.position = 1
	blocker.episode.heading = -1
	process.runtime.mu.Unlock()

	response := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "mover",
		ActuatorName: protoio.FlatlandMoveActuatorName,
		Output:       []float64{1},
	})
	if response.Err != nil || !response.OK {
		t.Fatalf("move response=%+v", response)
	}
	if collision, _ := response.Trace["public_agent_collision"].(bool); !collision {
		t.Fatalf("expected public_agent_collision=true, trace=%+v", response.Trace)
	}
	if collisions, _ := response.Trace["public_agent_collisions"].(int); collisions != 1 {
		t.Fatalf("expected mover public_agent_collisions=1, trace=%+v", response.Trace)
	}
	if mover.episode.publicAgentCollisions != 1 || blocker.episode.publicAgentCollisions != 1 {
		t.Fatalf("expected collision count on both agents, mover=%d blocker=%d", mover.episode.publicAgentCollisions, blocker.episode.publicAgentCollisions)
	}
	if mover.terminated || blocker.terminated {
		t.Fatalf("expected same-position public collision to remain non-terminal")
	}
}

func TestFlatlandPublicProcessSpearCanTerminateForwardPublicAgent(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	for _, id := range []string{"hunter", "target"} {
		if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: id}}); response.Err != nil || !response.OK {
			t.Fatalf("enter %s response=%+v", id, response)
		}
	}

	process.runtime.mu.Lock()
	hunter := process.runtime.agents["hunter"]
	target := process.runtime.agents["target"]
	hunter.episode.energy = 150
	hunter.episode.position = 0
	hunter.episode.heading = 1
	hunter.episode.food = nil
	hunter.episode.poison = nil
	hunter.episode.prey = nil
	hunter.episode.predators = nil
	target.episode.position = flatlandSpearReach
	process.runtime.mu.Unlock()

	response := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "hunter",
		ActuatorName: protoio.FlatlandSpearActuatorName,
		Output:       []float64{1},
	})
	if response.Err != nil || !response.OK {
		t.Fatalf("spear response=%+v", response)
	}
	if targetID, _ := response.Trace["public_agent_target_id"].(string); targetID != "target" {
		t.Fatalf("expected public target id=target, trace=%+v", response.Trace)
	}
	if terminated, _ := response.Trace["public_agent_target_terminated"].(bool); !terminated {
		t.Fatalf("expected public target termination, trace=%+v", response.Trace)
	}
	if kills, _ := response.Trace["public_agent_kills"].(int); kills != 1 {
		t.Fatalf("expected public_agent_kills=1, trace=%+v", response.Trace)
	}
	if !target.terminated {
		t.Fatalf("expected target state to be terminated")
	}
	if deaths := target.episode.publicAgentDeaths; deaths != 1 {
		t.Fatalf("expected target public_agent_deaths=1, got=%d", deaths)
	}
}

func TestFlatlandPublicProcessCreateOffspringCommandReportsGrantState(t *testing.T) {
	process := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := process.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start response=%+v", response)
	}
	t.Cleanup(func() {
		_ = process.Call(context.Background(), FlatlandPublicStopMessage{Reason: "normal"}).Err
	})
	if response := process.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "parent", NeuronCount: 3}}); response.Err != nil || !response.OK {
		t.Fatalf("enter response=%+v", response)
	}

	noOp := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "parent",
		ActuatorName: protoio.FlatlandCreateOffspringActuatorAliasName,
		Output:       []float64{0},
	})
	if noOp.Err != nil || !noOp.OK {
		t.Fatalf("no-op offspring response=%+v", noOp)
	}
	if requested, _ := noOp.Trace["offspring_requested"].(bool); requested {
		t.Fatalf("expected non-positive offspring output not to request clone, trace=%+v", noOp.Trace)
	}
	if energy, _ := noOp.Trace["energy"].(float64); math.Abs(energy-flatlandInitialEnergy) > 1e-12 {
		t.Fatalf("expected no-op offspring output to leave energy unchanged, trace=%+v", noOp.Trace)
	}

	process.runtime.mu.Lock()
	state := process.runtime.agents["parent"]
	state.episode.energy = 1200
	process.runtime.mu.Unlock()

	denied := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "parent",
		ActuatorName: protoio.FlatlandCreateOffspringActuatorName,
		Output:       []float64{1},
	})
	if denied.Err != nil || !denied.OK {
		t.Fatalf("denied offspring response=%+v", denied)
	}
	if requested, _ := denied.Trace["offspring_requested"].(bool); !requested {
		t.Fatalf("expected offspring request to be recorded, trace=%+v", denied.Trace)
	}
	if granted, _ := denied.Trace["offspring_granted"].(bool); granted {
		t.Fatalf("expected offspring request to be denied below grant threshold, trace=%+v", denied.Trace)
	}
	if cost, _ := denied.Trace["offspring_cost"].(float64); cost != 50 {
		t.Fatalf("expected denied offspring cost=50, trace=%+v", denied.Trace)
	}
	if energy, _ := denied.Trace["energy"].(float64); math.Abs(energy-1150) > 1e-12 {
		t.Fatalf("expected denied offspring energy=1150, trace=%+v", denied.Trace)
	}

	process.runtime.mu.Lock()
	state = process.runtime.agents["parent"]
	state.episode.energy = 1400
	process.runtime.mu.Unlock()

	granted := process.Call(ctx, FlatlandPublicActMessage{
		AgentID:      "parent",
		ActuatorName: protoio.FlatlandCreateOffspringActuatorAliasName,
		Output:       []float64{1},
	})
	if granted.Err != nil || !granted.OK {
		t.Fatalf("granted offspring response=%+v", granted)
	}
	if grantedFlag, _ := granted.Trace["offspring_granted"].(bool); !grantedFlag {
		t.Fatalf("expected offspring request to be granted, trace=%+v", granted.Trace)
	}
	if parentID, _ := granted.Trace["offspring_parent_id"].(string); parentID != "parent" {
		t.Fatalf("expected offspring parent id to be recorded, trace=%+v", granted.Trace)
	}
	if cost, _ := granted.Trace["offspring_cost"].(float64); cost != 1300 {
		t.Fatalf("expected granted offspring cost=1300, trace=%+v", granted.Trace)
	}
	if energy, _ := granted.Trace["energy"].(float64); math.Abs(energy-100) > 1e-12 {
		t.Fatalf("expected granted offspring energy=100, trace=%+v", granted.Trace)
	}
}

func TestFlatlandEpisodeTwoWheelsRotatesBeforeMoving(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:            "test",
		maxAge:          16,
		forageGoal:      10,
		foodPositions:   []int{7},
		poisonPositions: []int{18},
		wallPositions:   []int{8},
	})
	episode.position = 3
	episode.heading = -1

	moveStep, _, _, wallCollision, reason := episode.stepControl(flatlandControl{
		move:      1,
		turn:      1,
		twoWheels: true,
		width:     2,
	})
	if reason != "" {
		t.Fatalf("expected non-terminal wheel step, reason=%s", reason)
	}
	if wallCollision {
		t.Fatalf("expected no wall collision")
	}
	if moveStep != 1 {
		t.Fatalf("expected turn-right then forward move step=1, got %d", moveStep)
	}
	if episode.position != 4 || episode.heading != 1 {
		t.Fatalf("expected rotated forward position=4 heading=1, got position=%d heading=%d", episode.position, episode.heading)
	}
}

func TestFlatlandPublicProcessInstancesAreIsolated(t *testing.T) {
	first := NewFlatlandPublicProcess()
	second := NewFlatlandPublicProcess()
	ctx := context.Background()

	if response := first.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start first response=%+v", response)
	}
	if response := second.Call(ctx, FlatlandPublicStartMessage{}); response.Err != nil || !response.OK {
		t.Fatalf("start second response=%+v", response)
	}
	if response := first.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "first-agent"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter first response=%+v", response)
	}
	if response := second.Call(ctx, FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: "second-agent"}}); response.Err != nil || !response.OK {
		t.Fatalf("enter second response=%+v", response)
	}

	firstAgents := first.Call(ctx, FlatlandPublicGetAllMessage{})
	if firstAgents.Err != nil || !firstAgents.OK || len(firstAgents.Agents) != 1 || len(firstAgents.Avatars) != 1 {
		t.Fatalf("get first agents response=%+v", firstAgents)
	}
	if id, _ := firstAgents.Agents[0]["id"].(string); id != "first-agent" {
		t.Fatalf("expected first process to retain only first-agent, got %+v", firstAgents)
	}

	secondAgents := second.Call(ctx, FlatlandPublicGetAllMessage{})
	if secondAgents.Err != nil || !secondAgents.OK || len(secondAgents.Agents) != 1 || len(secondAgents.Avatars) != 1 {
		t.Fatalf("get second agents response=%+v", secondAgents)
	}
	if id, _ := secondAgents.Agents[0]["id"].(string); id != "second-agent" {
		t.Fatalf("expected second process to retain only second-agent, got %+v", secondAgents)
	}
}

func TestFlatlandScapePublicLifecycleAndTick(t *testing.T) {
	scape := FlatlandScape{}
	if _, err := scape.TickPublic(context.Background()); err == nil {
		t.Fatal("expected public tick to fail before start")
	}
	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "a"}); err == nil {
		t.Fatal("expected enter to fail before start")
	}

	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})

	if err := scape.EnterPublicAgent(FlatlandPublicAgent{
		ID:   "forager",
		Mode: "benchmark",
		Decide: func(input []float64) []float64 {
			return flatlandGreedyForager(input)
		},
	}); err != nil {
		t.Fatalf("enter forager: %v", err)
	}
	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "idle"}); err != nil {
		t.Fatalf("enter idle: %v", err)
	}
	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "idle"}); err == nil {
		t.Fatal("expected duplicate public enter to fail")
	}

	var trace Trace
	for i := 0; i < 5; i++ {
		var err error
		trace, err = scape.TickPublic(context.Background())
		if err != nil {
			t.Fatalf("tick %d: %v", i, err)
		}
	}
	if tick, ok := trace["tick"].(int); !ok || tick != 5 {
		t.Fatalf("expected tick=5 trace marker, trace=%+v", trace)
	}
	if active, ok := trace["active_agents"].(int); !ok || active != 2 {
		t.Fatalf("expected active_agents=2, trace=%+v", trace)
	}
	if _, ok := trace["avg_energy"].(float64); !ok {
		t.Fatalf("expected avg_energy in public trace, trace=%+v", trace)
	}
	agents, ok := trace["agents"].([]Trace)
	if !ok || len(agents) != 2 {
		t.Fatalf("expected two public agent traces, trace=%+v", trace)
	}
	if id, _ := agents[0]["id"].(string); id != "forager" && id != "idle" {
		t.Fatalf("unexpected public agent trace ordering/content: %+v", agents)
	}

	if err := scape.LeavePublicAgent("forager"); err != nil {
		t.Fatalf("leave forager: %v", err)
	}
	trace, err := scape.TickPublic(context.Background())
	if err != nil {
		t.Fatalf("tick after leave: %v", err)
	}
	if active, ok := trace["active_agents"].(int); !ok || active != 1 {
		t.Fatalf("expected active_agents=1 after leave, trace=%+v", trace)
	}
	if err := scape.LeavePublicAgent("forager"); err == nil {
		t.Fatal("expected missing public agent leave to fail")
	}
}

func TestFlatlandScapePublicAvatarSnapshotsExposeReferenceFields(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})
	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "avatar", Mode: "benchmark", NeuronCount: 4}); err != nil {
		t.Fatalf("enter avatar: %v", err)
	}

	flatlandPublicWorld.mu.Lock()
	state := flatlandPublicWorld.agents["avatar"]
	state.sound = 0.25
	state.gestalt = []float64{0.1, 0.2}
	state.spear = true
	state.offspringRequested = true
	state.offspringParentID = "avatar"
	state.episode.position = 7
	state.episode.heading = -1
	state.episode.age = 3
	state.episode.energy = 1.5
	state.episode.foodCollected = 2
	state.episode.publicAgentKills = 1
	state.episode.publicAgentCollisions = 1
	state.episode.shootKills = 1
	flatlandPublicWorld.mu.Unlock()

	snapshots, err := scape.PublicAvatarSnapshots()
	if err != nil {
		t.Fatalf("public avatar snapshots: %v", err)
	}
	if len(snapshots) != 1 {
		t.Fatalf("expected one public avatar snapshot, got=%d", len(snapshots))
	}
	snapshot := snapshots[0]
	if snapshot.ID != "avatar" || snapshot.Type != "prey" || snapshot.Specie != "benchmark" {
		t.Fatalf("unexpected avatar identity fields: %+v", snapshot)
	}
	if snapshot.Position != 7 || snapshot.Heading != -1 || snapshot.Age != 3 || snapshot.NeuronCount != 4 {
		t.Fatalf("unexpected avatar runtime fields: %+v", snapshot)
	}
	if math.Abs(snapshot.Energy-1.5) > 1e-12 || snapshot.EnergyNorm <= 0 {
		t.Fatalf("unexpected avatar energy fields: %+v", snapshot)
	}
	if snapshot.Kills != 4 {
		t.Fatalf("expected reference kill total=4, got %+v", snapshot)
	}
	if snapshot.Sound != 0.25 || !reflect.DeepEqual(snapshot.Gestalt, []float64{0.1, 0.2}) || !snapshot.Spear {
		t.Fatalf("unexpected avatar communication/weapon fields: %+v", snapshot)
	}
	if !snapshot.OffspringRequested || snapshot.OffspringParentID != "avatar" {
		t.Fatalf("unexpected offspring fields: %+v", snapshot)
	}
	if snapshot.PublicAgentKills != 1 || snapshot.PublicAgentCollisions != 1 || snapshot.ShootKills != 1 {
		t.Fatalf("unexpected interaction counters: %+v", snapshot)
	}

	snapshots[0].Gestalt[0] = 99
	again, err := scape.PublicAvatarSnapshots()
	if err != nil {
		t.Fatalf("public avatar snapshots again: %v", err)
	}
	if again[0].Gestalt[0] == 99 {
		t.Fatalf("expected snapshot gestalt to be copy-safe, got %+v", again[0])
	}

	flatlandPublicWorld.mu.Lock()
	flatlandPublicWorld.agents["avatar"].terminated = true
	flatlandPublicWorld.mu.Unlock()
	destroyed, err := scape.PublicAvatarSnapshots()
	if err != nil {
		t.Fatalf("public avatar snapshots destroyed: %v", err)
	}
	if destroyed[0].State != "destroyed" || !destroyed[0].Terminated {
		t.Fatalf("expected destroyed snapshot state, got %+v", destroyed[0])
	}
}

func TestFlatlandScapePublicUpdateAndListAgents(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})

	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "seed"}); err != nil {
		t.Fatalf("enter seed: %v", err)
	}

	agents, err := scape.PublicAgents()
	if err != nil {
		t.Fatalf("public agents: %v", err)
	}
	if len(agents) != 1 {
		t.Fatalf("expected one initial public agent, got=%d", len(agents))
	}
	if id, _ := agents[0]["id"].(string); id != "seed" {
		t.Fatalf("expected seed public agent, agents=%+v", agents)
	}

	if err := scape.UpdatePublicAgents([]FlatlandPublicAgent{
		{ID: "seed"},
		{ID: "bench", Mode: "benchmark"},
	}); err != nil {
		t.Fatalf("update public agents: %v", err)
	}
	summary, err := scape.LastPublicUpdateSummary()
	if err != nil {
		t.Fatalf("last public update summary: %v", err)
	}
	if summary.Previous != 1 || summary.Requested != 2 || summary.Preserved != 1 || summary.Created != 1 || summary.ActiveAfter != 2 {
		t.Fatalf("unexpected add update summary: %+v", summary)
	}

	trace, err := scape.TickPublic(context.Background())
	if err != nil {
		t.Fatalf("tick after update: %v", err)
	}
	if active, ok := trace["active_agents"].(int); !ok || active != 2 {
		t.Fatalf("expected active_agents=2 after update, trace=%+v", trace)
	}

	agents, err = scape.PublicAgents()
	if err != nil {
		t.Fatalf("public agents after update: %v", err)
	}
	if len(agents) != 2 {
		t.Fatalf("expected two public agents after update, got=%d", len(agents))
	}
	if mode, _ := agents[1]["mode"].(string); mode == "" {
		t.Fatalf("expected public agent mode annotation, agents=%+v", agents)
	}

	if err := scape.UpdatePublicAgents([]FlatlandPublicAgent{{ID: "bench", Mode: "benchmark"}}); err != nil {
		t.Fatalf("update public agents remove seed: %v", err)
	}
	summary, err = scape.LastPublicUpdateSummary()
	if err != nil {
		t.Fatalf("last public update summary after remove: %v", err)
	}
	if summary.Removed != 1 || !reflect.DeepEqual(summary.RemovedIDs, []string{"seed"}) || summary.Preserved != 1 || summary.ActiveAfter != 1 {
		t.Fatalf("unexpected removal update summary: %+v", summary)
	}
	summary.RemovedIDs[0] = "mutated"
	copied, err := scape.LastPublicUpdateSummary()
	if err != nil {
		t.Fatalf("copied public update summary: %v", err)
	}
	if copied.RemovedIDs[0] != "seed" {
		t.Fatalf("expected summary removed ids to be copy-safe, got %+v", copied)
	}
	trace, err = scape.TickPublic(context.Background())
	if err != nil {
		t.Fatalf("tick after removal update: %v", err)
	}
	if active, ok := trace["active_agents"].(int); !ok || active != 1 {
		t.Fatalf("expected active_agents=1 after removal update, trace=%+v", trace)
	}

	if err := scape.UpdatePublicAgents([]FlatlandPublicAgent{
		{ID: "dup"},
		{ID: "dup"},
	}); err == nil {
		t.Fatal("expected duplicate update ids to fail")
	}
}

func TestFlatlandScapePublicUpdateRevivesTerminatedAgent(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})

	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "seed"}); err != nil {
		t.Fatalf("enter seed: %v", err)
	}

	flatlandPublicWorld.mu.Lock()
	state := flatlandPublicWorld.agents["seed"]
	if state == nil {
		flatlandPublicWorld.mu.Unlock()
		t.Fatal("expected public agent state")
	}
	state.terminated = true
	oldEpisode := state.episode
	flatlandPublicWorld.mu.Unlock()

	if err := scape.UpdatePublicAgents([]FlatlandPublicAgent{{ID: "seed"}}); err != nil {
		t.Fatalf("update public agents: %v", err)
	}
	summary, err := scape.LastPublicUpdateSummary()
	if err != nil {
		t.Fatalf("last public update summary: %v", err)
	}
	if summary.DeadBefore != 1 || summary.Revived != 1 || summary.ActiveAfter != 1 || summary.TerminatedAfter != 0 {
		t.Fatalf("unexpected revive update summary: %+v", summary)
	}

	flatlandPublicWorld.mu.RLock()
	updated := flatlandPublicWorld.agents["seed"]
	flatlandPublicWorld.mu.RUnlock()
	if updated == nil {
		t.Fatal("expected updated public agent state")
	}
	if updated.terminated {
		t.Fatalf("expected update to revive terminated agent, state=%+v", updated)
	}
	if updated.episode == oldEpisode {
		t.Fatal("expected update to rebuild episode for terminated agent")
	}

	trace, err := scape.TickPublic(context.Background())
	if err != nil {
		t.Fatalf("tick after revive: %v", err)
	}
	if terminated, _ := trace["terminated_agents"].(int); terminated != 0 {
		t.Fatalf("expected no terminated agents after revive, trace=%+v", trace)
	}
}

func TestFlatlandScapePublicUpdateCanClearCustomDecider(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})

	custom := func(_ []float64) []float64 { return []float64{1} }
	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "seed", Decide: custom}); err != nil {
		t.Fatalf("enter seed: %v", err)
	}

	flatlandPublicWorld.mu.RLock()
	initial := flatlandPublicWorld.agents["seed"]
	flatlandPublicWorld.mu.RUnlock()
	if initial == nil || initial.decide == nil {
		t.Fatalf("expected initial custom decider, state=%+v", initial)
	}

	if err := scape.UpdatePublicAgents([]FlatlandPublicAgent{{ID: "seed"}}); err != nil {
		t.Fatalf("update public agents: %v", err)
	}

	flatlandPublicWorld.mu.RLock()
	updated := flatlandPublicWorld.agents["seed"]
	flatlandPublicWorld.mu.RUnlock()
	if updated == nil {
		t.Fatal("expected updated public agent state")
	}
	if updated.decide != nil {
		t.Fatalf("expected nil update decider to restore default policy, state=%+v", updated)
	}
}

func TestFlatlandScapePublicTraceIncludesTerminatedAgentAggregates(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})

	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "seed"}); err != nil {
		t.Fatalf("enter public agent: %v", err)
	}

	flatlandPublicWorld.mu.Lock()
	state := flatlandPublicWorld.agents["seed"]
	if state == nil || state.episode == nil {
		flatlandPublicWorld.mu.Unlock()
		t.Fatal("expected public agent episode state")
	}
	state.terminated = true
	state.episode.energy = 7.5
	state.episode.foodCollected = 3
	state.episode.preyCollected = 2
	state.episode.predatorHits = 1
	flatlandPublicWorld.mu.Unlock()

	trace, err := scape.TickPublic(context.Background())
	if err != nil {
		t.Fatalf("tick public: %v", err)
	}
	if active, _ := trace["active_agents"].(int); active != 0 {
		t.Fatalf("expected active_agents=0 when all public agents are terminated, trace=%+v", trace)
	}
	if terminated, _ := trace["terminated_agents"].(int); terminated != 1 {
		t.Fatalf("expected terminated_agents=1, trace=%+v", trace)
	}
	if avgEnergy, _ := trace["avg_energy"].(float64); avgEnergy != 7.5 {
		t.Fatalf("expected avg_energy to include terminated agent energy, trace=%+v", trace)
	}
	if totalFood, _ := trace["total_food_collected"].(int); totalFood != 3 {
		t.Fatalf("expected total_food_collected to include terminated agent totals, trace=%+v", trace)
	}
	if totalPrey, _ := trace["total_prey_collected"].(int); totalPrey != 2 {
		t.Fatalf("expected total_prey_collected to include terminated agent totals, trace=%+v", trace)
	}
	if totalHits, _ := trace["total_predator_hits"].(int); totalHits != 1 {
		t.Fatalf("expected total_predator_hits to include terminated agent totals, trace=%+v", trace)
	}
	if avatars, ok := trace["avatars"].([]FlatlandPublicAvatarSnapshot); !ok || len(avatars) != 1 || !avatars[0].Terminated {
		t.Fatalf("expected typed terminated avatar snapshot in public tick trace, trace=%+v", trace)
	}
}

func TestFlatlandScapePublicTraceAggregatesMatchActiveAgentTrace(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})

	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "seed"}); err != nil {
		t.Fatalf("enter public agent: %v", err)
	}

	trace, err := scape.TickPublic(context.Background())
	if err != nil {
		t.Fatalf("tick public: %v", err)
	}
	agents, ok := trace["agents"].([]Trace)
	if !ok || len(agents) != 1 {
		t.Fatalf("expected one public agent trace, trace=%+v", trace)
	}
	agent := agents[0]
	if avgEnergy, _ := trace["avg_energy"].(float64); avgEnergy != agent["energy"] {
		t.Fatalf("expected avg_energy to match active agent trace, trace=%+v agent=%+v", trace, agent)
	}
	if totalFood, _ := trace["total_food_collected"].(int); totalFood != agent["food_collected"] {
		t.Fatalf("expected total_food_collected to match active agent trace, trace=%+v agent=%+v", trace, agent)
	}
	if totalPrey, _ := trace["total_prey_collected"].(int); totalPrey != agent["prey_collected"] {
		t.Fatalf("expected total_prey_collected to match active agent trace, trace=%+v agent=%+v", trace, agent)
	}
	if totalHits, _ := trace["total_predator_hits"].(int); totalHits != agent["predator_hits"] {
		t.Fatalf("expected total_predator_hits to match active agent trace, trace=%+v agent=%+v", trace, agent)
	}
	if totalPublicCollisions, _ := trace["total_public_agent_collisions"].(int); totalPublicCollisions != agent["public_agent_collisions"] {
		t.Fatalf("expected total_public_agent_collisions to match active agent trace, trace=%+v agent=%+v", trace, agent)
	}
	if avatars, ok := trace["avatars"].([]FlatlandPublicAvatarSnapshot); !ok || len(avatars) != 1 || avatars[0].ID != "seed" {
		t.Fatalf("expected typed avatar snapshot in public tick trace, trace=%+v", trace)
	}
}

func TestFlatlandScapePublicTickAggregatesInteractionTotals(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})
	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "seed"}); err != nil {
		t.Fatalf("enter public agent: %v", err)
	}

	flatlandPublicWorld.mu.Lock()
	state := flatlandPublicWorld.agents["seed"]
	state.episode.publicAgentCollisions = 2
	state.episode.publicAgentKills = 1
	state.episode.publicAgentDeaths = 1
	state.episode.spearKills = 3
	state.episode.shootKills = 4
	flatlandPublicWorld.mu.Unlock()

	trace, err := scape.TickPublic(context.Background())
	if err != nil {
		t.Fatalf("tick public: %v", err)
	}
	if total, _ := trace["total_public_agent_collisions"].(int); total != 2 {
		t.Fatalf("expected public collision aggregate=2, trace=%+v", trace)
	}
	if total, _ := trace["total_public_agent_kills"].(int); total != 1 {
		t.Fatalf("expected public kill aggregate=1, trace=%+v", trace)
	}
	if total, _ := trace["total_public_agent_deaths"].(int); total != 1 {
		t.Fatalf("expected public death aggregate=1, trace=%+v", trace)
	}
	if total, _ := trace["total_spear_kills"].(int); total != 3 {
		t.Fatalf("expected spear kill aggregate=3, trace=%+v", trace)
	}
	if total, _ := trace["total_shoot_kills"].(int); total != 4 {
		t.Fatalf("expected shoot kill aggregate=4, trace=%+v", trace)
	}
}

func TestFlatlandScapeRunPublicTicksUntilCancel(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		_ = scape.Stop(context.Background())
	})
	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "runner"}); err != nil {
		t.Fatalf("enter runner: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	if err := scape.RunPublic(ctx, 2*time.Millisecond); err != nil {
		t.Fatalf("run public: %v", err)
	}

	trace, err := scape.TickPublic(context.Background())
	if err != nil {
		t.Fatalf("tick after run public: %v", err)
	}
	if tick, ok := trace["tick"].(int); !ok || tick <= 1 {
		t.Fatalf("expected run public to advance world ticks before manual tick, trace=%+v", trace)
	}
}

func TestFlatlandScapeRunPublicRejectsNonPositiveInterval(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.RunPublic(context.Background(), 0); err == nil {
		t.Fatal("expected run public with zero interval to fail")
	}
}

func TestFlatlandScapeRunPublicStopsCleanlyWhenWorldStops(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := scape.EnterPublicAgent(FlatlandPublicAgent{ID: "runner"}); err != nil {
		t.Fatalf("enter runner: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		done <- scape.RunPublic(context.Background(), 2*time.Millisecond)
	}()

	time.Sleep(10 * time.Millisecond)
	if err := scape.Stop(context.Background()); err != nil {
		t.Fatalf("stop during run public: %v", err)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("expected run public to stop cleanly after world stop, got %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("run public did not exit after world stop")
	}
}

func TestFlatlandScapePublicStopReasons(t *testing.T) {
	scape := FlatlandScape{}
	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	if err := scape.Shutdown(context.Background()); err != nil {
		t.Fatalf("shutdown: %v", err)
	}
	if reason := scape.LastPublicStopReason(); reason != "shutdown" {
		t.Fatalf("expected shutdown last stop reason, got=%q", reason)
	}

	if err := scape.Start(context.Background()); err != nil {
		t.Fatalf("start after shutdown: %v", err)
	}
	if reason := scape.LastPublicStopReason(); reason != "" {
		t.Fatalf("expected start to clear prior stop reason, got=%q", reason)
	}
	if err := scape.Stop(context.Background()); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if reason := scape.LastPublicStopReason(); reason != "normal" {
		t.Fatalf("expected normal last stop reason, got=%q", reason)
	}

	if err := scape.StopWithReason(context.Background(), "bad_reason"); err == nil {
		t.Fatal("expected unsupported stop reason to fail")
	}
}

func TestFlatlandScapeForagingCollectsResources(t *testing.T) {
	scape := FlatlandScape{}
	stationary := scriptedStepAgent{
		id: "stationary",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	forager := scriptedStepAgent{
		id: "forager",
		fn: flatlandGreedyForager,
	}

	stationaryFitness, stationaryTrace, err := scape.Evaluate(context.Background(), stationary)
	if err != nil {
		t.Fatalf("evaluate stationary: %v", err)
	}
	foragerFitness, foragerTrace, err := scape.Evaluate(context.Background(), forager)
	if err != nil {
		t.Fatalf("evaluate forager: %v", err)
	}
	if foragerFitness <= 0 || stationaryFitness <= 0 {
		t.Fatalf("expected positive fitness signals, got forager=%f stationary=%f", foragerFitness, stationaryFitness)
	}
	stationaryFood, ok := stationaryTrace["food_collected"].(int)
	if !ok {
		t.Fatalf("stationary trace missing food_collected: %+v", stationaryTrace)
	}
	foragerFood, ok := foragerTrace["food_collected"].(int)
	if !ok {
		t.Fatalf("forager trace missing food_collected: %+v", foragerTrace)
	}
	if foragerFood <= stationaryFood {
		t.Fatalf(
			"expected forager to collect more food than stationary, got forager=%d stationary=%d forager_trace=%+v stationary_trace=%+v",
			foragerFood,
			stationaryFood,
			foragerTrace,
			stationaryTrace,
		)
	}
}

func TestFlatlandScapeStepInputSurfaceIncludesScannerAndExtendedChannels(t *testing.T) {
	scape := FlatlandScape{}
	var lastInput []float64
	agent := scriptedStepAgent{
		id: "flatland-step-input-surface",
		fn: func(input []float64) []float64 {
			lastInput = append([]float64(nil), input...)
			return flatlandGreedyForager(input)
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if len(lastInput) != flatlandBaseFeatureWidth+flatlandScannerWidth {
		t.Fatalf("expected step input width=%d, got=%d input=%v", flatlandBaseFeatureWidth+flatlandScannerWidth, len(lastInput), lastInput)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "step_input" {
		t.Fatalf("expected step_input sensor surface, trace=%+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != flatlandBaseFeatureWidth+flatlandScannerWidth {
		t.Fatalf("expected trace sensor_width=%d, trace=%+v", flatlandBaseFeatureWidth+flatlandScannerWidth, trace)
	}
	if width, ok := trace["step_input_width"].(int); !ok || width != flatlandBaseFeatureWidth+flatlandScannerWidth {
		t.Fatalf("expected trace step_input_width=%d, trace=%+v", flatlandBaseFeatureWidth+flatlandScannerWidth, trace)
	}
	if bins, ok := trace["scanner_runtime_active_bins"].([]int); !ok || !reflect.DeepEqual(bins, []int{0, 1, 2, 3, 4}) {
		t.Fatalf("expected full runtime scanner bins, trace=%+v", trace)
	}
	if density, ok := trace["scanner_density_runtime"].(int); !ok || density != 5 {
		t.Fatalf("expected runtime scanner density=5, trace=%+v", trace)
	}
	if width, ok := trace["scanner_feature_width_runtime"].(int); !ok || width != 15 {
		t.Fatalf("expected runtime scanner feature width=15, trace=%+v", trace)
	}

	expectEqual := func(name string, got, want float64) {
		t.Helper()
		if math.Abs(got-want) > 1e-9 {
			t.Fatalf("expected %s=%f got=%f", name, want, got)
		}
	}

	expectEqual("distance", lastInput[0], mustTraceFloat64(t, trace, "last_food_distance"))
	expectEqual("prey", lastInput[2], mustTraceFloat64(t, trace, "last_prey_signal"))
	expectEqual("predator", lastInput[3], mustTraceFloat64(t, trace, "last_predator_signal"))
	expectEqual("poison", lastInput[4], mustTraceFloat64(t, trace, "last_poison_signal"))
	expectEqual("wall", lastInput[5], mustTraceFloat64(t, trace, "last_wall_signal"))
	expectEqual("food_proximity", lastInput[6], mustTraceFloat64(t, trace, "last_food_proximity"))
	expectEqual("prey_proximity", lastInput[7], mustTraceFloat64(t, trace, "last_prey_proximity"))
	expectEqual("predator_proximity", lastInput[8], mustTraceFloat64(t, trace, "last_predator_proximity"))
	expectEqual("poison_proximity", lastInput[9], mustTraceFloat64(t, trace, "last_poison_proximity"))
	expectEqual("wall_proximity", lastInput[10], mustTraceFloat64(t, trace, "last_wall_proximity"))
	expectEqual("resource_balance", lastInput[11], mustTraceFloat64(t, trace, "last_resource_balance"))

	distanceBins := mustTraceFloat64Slice(t, trace, "last_distance_scan_bins")
	colorBins := mustTraceFloat64Slice(t, trace, "last_color_scan_bins")
	energyBins := mustTraceFloat64Slice(t, trace, "last_energy_scan_bins")
	for i := 0; i < flatlandScannerDensity; i++ {
		expectEqual("distance_scan_bin", lastInput[flatlandBaseFeatureWidth+i], distanceBins[i])
		expectEqual("color_scan_bin", lastInput[flatlandBaseFeatureWidth+flatlandScannerDensity+i], colorBins[i])
		expectEqual("energy_scan_bin", lastInput[flatlandBaseFeatureWidth+2*flatlandScannerDensity+i], energyBins[i])
	}
}

func TestFlatlandScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FlatlandDistanceSensorName,
			protoio.FlatlandEnergySensorName,
		},
		ActuatorIDs: []string{protoio.FlatlandMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "distance", Activation: "identity"},
			{ID: "energy", Activation: "identity"},
			{ID: "move", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "distance", To: "move", Weight: 1, Enabled: true},
			{From: "energy", To: "move", Weight: 0.2, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FlatlandDistanceSensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandEnergySensorName:   protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FlatlandMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"flatland-agent-io",
		genome,
		sensors,
		actuators,
		[]string{"distance", "energy"},
		[]string{"move"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["energy"].(float64); !ok {
		t.Fatalf("trace missing energy: %+v", trace)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "classic" {
		t.Fatalf("expected classic sensor surface, trace=%+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 2 {
		t.Fatalf("expected classic sensor width=2, trace=%+v", trace)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FlatlandMoveActuatorName {
		t.Fatalf("expected flatland_move control surface, trace=%+v", trace)
	}
	if width, ok := trace["last_control_width"].(int); !ok || width != 1 {
		t.Fatalf("expected single-channel control width, trace=%+v", trace)
	}
}

func TestFlatlandScapeEvaluateWithExtendedIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FlatlandPreySensorName,
			protoio.FlatlandPredatorSensorName,
			protoio.FlatlandPoisonSensorName,
			protoio.FlatlandWallSensorName,
			protoio.FlatlandFoodProximitySensorName,
			protoio.FlatlandPreyProximitySensorName,
			protoio.FlatlandPredatorProximitySensorName,
			protoio.FlatlandPoisonProximitySensorName,
			protoio.FlatlandWallProximitySensorName,
			protoio.FlatlandResourceBalanceSensorName,
		},
		ActuatorIDs: []string{protoio.FlatlandMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "prey", Activation: "identity"},
			{ID: "predator", Activation: "identity"},
			{ID: "poison", Activation: "identity"},
			{ID: "wall", Activation: "identity"},
			{ID: "food_prox", Activation: "identity"},
			{ID: "prey_prox", Activation: "identity"},
			{ID: "predator_prox", Activation: "identity"},
			{ID: "poison_prox", Activation: "identity"},
			{ID: "wall_prox", Activation: "identity"},
			{ID: "balance", Activation: "identity"},
			{ID: "move", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "prey", To: "move", Weight: 0.5, Enabled: true},
			{From: "predator", To: "move", Weight: -0.6, Enabled: true},
			{From: "poison", To: "move", Weight: -0.8, Enabled: true},
			{From: "wall", To: "move", Weight: -0.6, Enabled: true},
			{From: "food_prox", To: "move", Weight: 0.9, Enabled: true},
			{From: "prey_prox", To: "move", Weight: 0.7, Enabled: true},
			{From: "predator_prox", To: "move", Weight: -0.8, Enabled: true},
			{From: "poison_prox", To: "move", Weight: -0.7, Enabled: true},
			{From: "wall_prox", To: "move", Weight: -0.5, Enabled: true},
			{From: "balance", To: "move", Weight: 0.4, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FlatlandPreySensorName:              protoio.NewScalarInputSensor(0),
		protoio.FlatlandPredatorSensorName:          protoio.NewScalarInputSensor(0),
		protoio.FlatlandPoisonSensorName:            protoio.NewScalarInputSensor(0),
		protoio.FlatlandWallSensorName:              protoio.NewScalarInputSensor(0),
		protoio.FlatlandFoodProximitySensorName:     protoio.NewScalarInputSensor(0),
		protoio.FlatlandPreyProximitySensorName:     protoio.NewScalarInputSensor(0),
		protoio.FlatlandPredatorProximitySensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandPoisonProximitySensorName:   protoio.NewScalarInputSensor(0),
		protoio.FlatlandWallProximitySensorName:     protoio.NewScalarInputSensor(0),
		protoio.FlatlandResourceBalanceSensorName:   protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FlatlandMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"flatland-agent-io-extended",
		genome,
		sensors,
		actuators,
		[]string{"prey", "predator", "poison", "wall", "food_prox", "prey_prox", "predator_prox", "poison_prox", "wall_prox", "balance"},
		[]string{"move"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "extended" {
		t.Fatalf("expected extended sensor surface, trace=%+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 10 {
		t.Fatalf("expected extended sensor width=10, trace=%+v", trace)
	}
	if width, ok := trace["feature_width"].(int); !ok || width != flatlandBaseFeatureWidth {
		t.Fatalf("expected extended feature width marker, trace=%+v", trace)
	}
	if width, ok := trace["scanner_feature_width"].(int); !ok || width != 15 {
		t.Fatalf("expected scanner feature width marker, trace=%+v", trace)
	}
}

func TestFlatlandScapeEvaluateWithScannerIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FlatlandDistanceScan0SensorName,
			protoio.FlatlandDistanceScan1SensorName,
			protoio.FlatlandDistanceScan2SensorName,
			protoio.FlatlandDistanceScan3SensorName,
			protoio.FlatlandDistanceScan4SensorName,
		},
		ActuatorIDs: []string{protoio.FlatlandMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "d0", Activation: "identity"},
			{ID: "d1", Activation: "identity"},
			{ID: "d2", Activation: "identity"},
			{ID: "d3", Activation: "identity"},
			{ID: "d4", Activation: "identity"},
			{ID: "move", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "d0", To: "move", Weight: -0.8, Enabled: true},
			{From: "d1", To: "move", Weight: -0.4, Enabled: true},
			{From: "d2", To: "move", Weight: 0.0, Enabled: true},
			{From: "d3", To: "move", Weight: 0.4, Enabled: true},
			{From: "d4", To: "move", Weight: 0.8, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FlatlandDistanceScan0SensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandDistanceScan1SensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandDistanceScan2SensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandDistanceScan3SensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandDistanceScan4SensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FlatlandMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"flatland-agent-io-scanner",
		genome,
		sensors,
		actuators,
		[]string{"d0", "d1", "d2", "d3", "d4"},
		[]string{"move"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "scanner" {
		t.Fatalf("expected scanner sensor surface, trace=%+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 15 {
		t.Fatalf("expected scanner sensor width=15, trace=%+v", trace)
	}
	if density, ok := trace["scanner_density"].(int); !ok || density != 5 {
		t.Fatalf("expected scanner density marker, trace=%+v", trace)
	}
	if _, ok := trace["last_distance_scan_mean"].(float64); !ok {
		t.Fatalf("expected last_distance_scan_mean trace marker, trace=%+v", trace)
	}
	distanceBins, ok := trace["last_distance_scan_bins"].([]float64)
	if !ok || len(distanceBins) != 5 {
		t.Fatalf("expected distance scanner bins len=5, trace=%+v", trace)
	}
}

func TestFlatlandScapeEvaluateWithAlignedPartialScannerIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FlatlandDistanceScan1SensorName,
			protoio.FlatlandDistanceScan2SensorName,
			protoio.FlatlandDistanceScan3SensorName,
			protoio.FlatlandColorScan1SensorName,
			protoio.FlatlandColorScan2SensorName,
			protoio.FlatlandColorScan3SensorName,
			protoio.FlatlandEnergyScan1SensorName,
			protoio.FlatlandEnergyScan2SensorName,
			protoio.FlatlandEnergyScan3SensorName,
		},
		ActuatorIDs: []string{protoio.FlatlandMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "d1", Activation: "identity"},
			{ID: "d2", Activation: "identity"},
			{ID: "d3", Activation: "identity"},
			{ID: "c1", Activation: "identity"},
			{ID: "c2", Activation: "identity"},
			{ID: "c3", Activation: "identity"},
			{ID: "e1", Activation: "identity"},
			{ID: "e2", Activation: "identity"},
			{ID: "e3", Activation: "identity"},
			{ID: "move", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "d1", To: "move", Weight: -0.8, Enabled: true},
			{From: "d2", To: "move", Weight: 0.2, Enabled: true},
			{From: "d3", To: "move", Weight: 0.6, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FlatlandDistanceScan1SensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandDistanceScan2SensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandDistanceScan3SensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandColorScan1SensorName:    protoio.NewScalarInputSensor(0),
		protoio.FlatlandColorScan2SensorName:    protoio.NewScalarInputSensor(0),
		protoio.FlatlandColorScan3SensorName:    protoio.NewScalarInputSensor(0),
		protoio.FlatlandEnergyScan1SensorName:   protoio.NewScalarInputSensor(0),
		protoio.FlatlandEnergyScan2SensorName:   protoio.NewScalarInputSensor(0),
		protoio.FlatlandEnergyScan3SensorName:   protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FlatlandMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"flatland-agent-io-partial-scanner",
		genome,
		sensors,
		actuators,
		[]string{"d1", "d2", "d3", "c1", "c2", "c3", "e1", "e2", "e3"},
		[]string{"move"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FlatlandMoveActuatorName {
		t.Fatalf("expected tick-agent control surface via flatland_move, trace=%+v", trace)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "scanner" {
		t.Fatalf("expected scanner sensor surface, trace=%+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 9 {
		t.Fatalf("expected scanner sensor width=9, trace=%+v", trace)
	}
	if bins, ok := trace["scanner_runtime_active_bins"].([]int); !ok || !reflect.DeepEqual(bins, []int{1, 2, 3}) {
		t.Fatalf("expected runtime scanner bins [1 2 3], trace=%+v", trace)
	}
	if density, ok := trace["scanner_density_runtime"].(int); !ok || density != 3 {
		t.Fatalf("expected runtime scanner density=3, trace=%+v", trace)
	}
	if width, ok := trace["scanner_feature_width_runtime"].(int); !ok || width != 9 {
		t.Fatalf("expected runtime scanner feature width=9, trace=%+v", trace)
	}
	if active, ok := trace["scanner_density_active"].(int); !ok || active != 3 {
		t.Fatalf("expected active scanner density=3, trace=%+v", trace)
	}
	if density, ok := trace["scanner_density"].(int); !ok || density != 5 {
		t.Fatalf("expected fixed scanner density=5, trace=%+v", trace)
	}
}

func TestFlatlandScapeEvaluateWithTwoWheelsActuator(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FlatlandDistanceSensorName,
			protoio.FlatlandEnergySensorName,
		},
		ActuatorIDs: []string{protoio.FlatlandTwoWheelsActuatorName},
		Neurons: []model.Neuron{
			{ID: "distance", Activation: "identity"},
			{ID: "energy", Activation: "identity"},
			{ID: "left", Activation: "tanh"},
			{ID: "right", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "distance", To: "left", Weight: -0.8, Enabled: true},
			{From: "energy", To: "left", Weight: 0.25, Enabled: true},
			{From: "distance", To: "right", Weight: 0.8, Enabled: true},
			{From: "energy", To: "right", Weight: 0.25, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FlatlandDistanceSensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandEnergySensorName:   protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FlatlandTwoWheelsActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"flatland-agent-two-wheels",
		genome,
		sensors,
		actuators,
		[]string{"distance", "energy"},
		[]string{"left", "right"},
		nil,
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FlatlandTwoWheelsActuatorName {
		t.Fatalf("expected flatland_two_wheels control surface, trace=%+v", trace)
	}
	if width, ok := trace["last_control_width"].(int); !ok || width != 2 {
		t.Fatalf("expected two-channel control width, trace=%+v", trace)
	}
}

func TestFlatlandScapeEvaluateWithActorProcessIO(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.FlatlandDistanceSensorName,
			protoio.FlatlandEnergySensorName,
		},
		ActuatorIDs: []string{protoio.FlatlandTwoWheelsActuatorName},
		Neurons: []model.Neuron{
			{ID: "distance", Activation: "identity"},
			{ID: "energy", Activation: "identity"},
			{ID: "left", Activation: "tanh"},
			{ID: "right", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "distance", To: "left", Weight: -0.8, Enabled: true},
			{From: "energy", To: "left", Weight: 0.25, Enabled: true},
			{From: "distance", To: "right", Weight: 0.8, Enabled: true},
			{From: "energy", To: "right", Weight: 0.25, Enabled: true},
		},
	}

	sensors, actuators, err := NewFlatlandProcessIO("gt", genome.SensorIDs, genome.ActuatorIDs)
	if err != nil {
		t.Fatalf("new flatland process io: %v", err)
	}
	t.Cleanup(func() {
		_ = FlatlandScape{}.StopWithReason(context.Background(), "normal")
	})
	cortex, err := agent.NewCortex(
		"flatland-agent-actor-process-io",
		genome,
		sensors,
		actuators,
		[]string{"distance", "energy"},
		[]string{"left", "right"},
		nil,
		agent.WithIOProcessContext("flatland", "gt"),
		agent.WithIOActors(),
	)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}
	t.Cleanup(cortex.Terminate)

	fitness, trace, err := FlatlandScape{}.EvaluateMode(context.Background(), cortex, "gt")
	if err != nil {
		t.Fatalf("evaluate actor process io: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive actor-process flatland fitness, got %f trace=%+v", fitness, trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface != "classic" {
		t.Fatalf("expected classic flatland sensor surface, got %+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 2 {
		t.Fatalf("expected flatland sensor_width=2, got %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface != protoio.FlatlandTwoWheelsActuatorName {
		t.Fatalf("expected control_surface=%s, got %+v", protoio.FlatlandTwoWheelsActuatorName, trace)
	}
}

func TestFlatlandScapeEvaluateWithWriteOnlyMoveActuator(t *testing.T) {
	agent := scriptedTickAgent{
		id: "flatland-tick-write-only",
		sensors: map[string]protoio.Sensor{
			protoio.FlatlandDistanceSensorName: protoio.NewScalarInputSensor(0),
			protoio.FlatlandEnergySensorName:   protoio.NewScalarInputSensor(0),
		},
		actuators: map[string]protoio.Actuator{
			protoio.FlatlandMoveActuatorName: &writeOnlyActuator{name: protoio.FlatlandMoveActuatorName},
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			distance, err := sensors[protoio.FlatlandDistanceSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			energy, err := sensors[protoio.FlatlandEnergySensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			dist := 0.0
			if len(distance) > 0 {
				dist = distance[0]
			}
			en := 0.0
			if len(energy) > 0 {
				en = energy[0]
			}
			return []float64{dist - 0.2*en}, nil
		},
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent with write-only actuator: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FlatlandMoveActuatorName {
		t.Fatalf("expected flatland_move control surface, trace=%+v", trace)
	}
	if surface, _ := trace["sensor_surface"].(string); surface != "classic" {
		t.Fatalf("expected classic sensor surface, trace=%+v", trace)
	}
	if width, ok := trace["sensor_width"].(int); !ok || width != 2 {
		t.Fatalf("expected sensor_width=2, trace=%+v", trace)
	}
}

func TestFlatlandScapeEvaluateWithInferredMoveControlSurface(t *testing.T) {
	agent := scriptedTickAgent{
		id: "flatland-tick-inferred-move",
		sensors: map[string]protoio.Sensor{
			protoio.FlatlandDistanceSensorName: protoio.NewScalarInputSensor(0),
			protoio.FlatlandEnergySensorName:   protoio.NewScalarInputSensor(0),
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			distance, err := sensors[protoio.FlatlandDistanceSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			energy, err := sensors[protoio.FlatlandEnergySensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			dist := 0.0
			if len(distance) > 0 {
				dist = distance[0]
			}
			en := 0.0
			if len(energy) > 0 {
				en = energy[0]
			}
			return []float64{dist - 0.2*en}, nil
		},
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent with inferred move surface: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FlatlandMoveActuatorName {
		t.Fatalf("expected inferred flatland_move control surface, trace=%+v", trace)
	}
	if width, ok := trace["last_control_width"].(int); !ok || width != 1 {
		t.Fatalf("expected scalar control width=1, trace=%+v", trace)
	}
}

func TestFlatlandScapeEvaluateWithInferredTwoWheelsControlSurface(t *testing.T) {
	agent := scriptedTickAgent{
		id: "flatland-tick-inferred-two-wheels",
		sensors: map[string]protoio.Sensor{
			protoio.FlatlandDistanceSensorName: protoio.NewScalarInputSensor(0),
			protoio.FlatlandEnergySensorName:   protoio.NewScalarInputSensor(0),
		},
		fn: func(ctx context.Context, sensors map[string]protoio.Sensor) ([]float64, error) {
			distance, err := sensors[protoio.FlatlandDistanceSensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			energy, err := sensors[protoio.FlatlandEnergySensorName].Read(ctx)
			if err != nil {
				return nil, err
			}
			dist := 0.0
			if len(distance) > 0 {
				dist = distance[0]
			}
			en := 0.0
			if len(energy) > 0 {
				en = energy[0]
			}
			return []float64{-0.8*dist + 0.25*en, 0.8*dist + 0.25*en}, nil
		},
	}

	scape := FlatlandScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate tick agent with inferred two-wheel surface: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if surface, _ := trace["control_surface"].(string); surface != protoio.FlatlandTwoWheelsActuatorName {
		t.Fatalf("expected inferred flatland_two_wheels control surface, trace=%+v", trace)
	}
	if width, ok := trace["last_control_width"].(int); !ok || width != 2 {
		t.Fatalf("expected two-wheel control width=2, trace=%+v", trace)
	}
}

func TestFlatlandScapeStepAgentSupportsWideControlVector(t *testing.T) {
	scape := FlatlandScape{}
	agent := scriptedStepAgent{
		id: "flatland-wide-control",
		fn: func(_ []float64) []float64 {
			return []float64{0.8, -0.2, 0.6, -0.4}
		},
	}

	fitness, trace, err := scape.Evaluate(context.Background(), agent)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if width, ok := trace["last_control_width"].(int); !ok || width != 4 {
		t.Fatalf("expected control width=4 for wide control vector, trace=%+v", trace)
	}
}

func TestFlatlandScapeTraceCapturesMetabolicsAndCollisions(t *testing.T) {
	scape := FlatlandScape{}
	forager := scriptedStepAgent{
		id: "forager",
		fn: flatlandGreedyForager,
	}

	_, trace, err := scape.Evaluate(context.Background(), forager)
	if err != nil {
		t.Fatalf("evaluate forager: %v", err)
	}
	if _, ok := trace["age"].(int); !ok {
		t.Fatalf("trace missing age: %+v", trace)
	}
	if _, ok := trace["food_collected"].(int); !ok {
		t.Fatalf("trace missing food_collected: %+v", trace)
	}
	if _, ok := trace["poison_hits"].(int); !ok {
		t.Fatalf("trace missing poison_hits: %+v", trace)
	}
	if _, ok := trace["wall_collisions"].(int); !ok {
		t.Fatalf("trace missing wall_collisions: %+v", trace)
	}
	if _, ok := trace["resource_respawns"].(int); !ok {
		t.Fatalf("trace missing resource_respawns: %+v", trace)
	}
	if reason, ok := trace["terminal_reason"].(string); !ok || reason == "" {
		t.Fatalf("trace missing terminal_reason: %+v", trace)
	}
	if _, ok := trace["last_poison_signal"].(float64); !ok {
		t.Fatalf("trace missing last_poison_signal: %+v", trace)
	}
	if _, ok := trace["last_prey_signal"].(float64); !ok {
		t.Fatalf("trace missing last_prey_signal: %+v", trace)
	}
	if _, ok := trace["last_predator_signal"].(float64); !ok {
		t.Fatalf("trace missing last_predator_signal: %+v", trace)
	}
	if _, ok := trace["last_wall_signal"].(float64); !ok {
		t.Fatalf("trace missing last_wall_signal: %+v", trace)
	}
	if _, ok := trace["last_food_proximity"].(float64); !ok {
		t.Fatalf("trace missing last_food_proximity: %+v", trace)
	}
	if _, ok := trace["last_prey_proximity"].(float64); !ok {
		t.Fatalf("trace missing last_prey_proximity: %+v", trace)
	}
	if _, ok := trace["last_predator_proximity"].(float64); !ok {
		t.Fatalf("trace missing last_predator_proximity: %+v", trace)
	}
	if _, ok := trace["last_poison_proximity"].(float64); !ok {
		t.Fatalf("trace missing last_poison_proximity: %+v", trace)
	}
	if _, ok := trace["last_wall_proximity"].(float64); !ok {
		t.Fatalf("trace missing last_wall_proximity: %+v", trace)
	}
	if _, ok := trace["last_resource_balance"].(float64); !ok {
		t.Fatalf("trace missing last_resource_balance: %+v", trace)
	}
	if _, ok := trace["last_distance_scan_mean"].(float64); !ok {
		t.Fatalf("trace missing last_distance_scan_mean: %+v", trace)
	}
	if _, ok := trace["last_color_scan_mean"].(float64); !ok {
		t.Fatalf("trace missing last_color_scan_mean: %+v", trace)
	}
	if _, ok := trace["last_energy_scan_mean"].(float64); !ok {
		t.Fatalf("trace missing last_energy_scan_mean: %+v", trace)
	}
	if bins, ok := trace["last_distance_scan_bins"].([]float64); !ok || len(bins) != 5 {
		t.Fatalf("trace missing last_distance_scan_bins len=5: %+v", trace)
	}
	if bins, ok := trace["last_color_scan_bins"].([]float64); !ok || len(bins) != 5 {
		t.Fatalf("trace missing last_color_scan_bins len=5: %+v", trace)
	}
	if bins, ok := trace["last_energy_scan_bins"].([]float64); !ok || len(bins) != 5 {
		t.Fatalf("trace missing last_energy_scan_bins len=5: %+v", trace)
	}
	if _, ok := trace["scanner_spread"].(float64); !ok {
		t.Fatalf("trace missing scanner_spread: %+v", trace)
	}
	if _, ok := trace["scanner_offset"].(float64); !ok {
		t.Fatalf("trace missing scanner_offset: %+v", trace)
	}
	if _, ok := trace["scanner_heading"].(int); !ok {
		t.Fatalf("trace missing scanner_heading: %+v", trace)
	}
	if _, ok := trace["initial_heading"].(int); !ok {
		t.Fatalf("trace missing initial_heading: %+v", trace)
	}
	if profile, ok := trace["scanner_profile"].(string); !ok || profile == "" {
		t.Fatalf("trace missing scanner_profile: %+v", trace)
	}
	if weights, ok := trace["scanner_profile_weights"].([]float64); !ok || len(weights) != flatlandScannerDensity {
		t.Fatalf("trace missing scanner_profile_weights len=%d: %+v", flatlandScannerDensity, trace)
	}
	if effective, ok := trace["scanner_density_effective"].(int); !ok || effective <= 0 || effective > flatlandScannerDensity {
		t.Fatalf("trace missing scanner_density_effective in range 1..%d: %+v", flatlandScannerDensity, trace)
	}
	effective, ok := trace["scanner_density_effective"].(int)
	if !ok || effective <= 0 || effective > flatlandScannerDensity {
		t.Fatalf("trace missing scanner_density_effective in range 1..%d: %+v", flatlandScannerDensity, trace)
	}
	if width, ok := trace["scanner_feature_width_effective"].(int); !ok || width != effective*3 {
		t.Fatalf("trace missing scanner_feature_width_effective aligned to effective density: %+v", trace)
	}
	if bins, ok := trace["scanner_profile_active_bins"].([]int); !ok || len(bins) != effective {
		t.Fatalf("trace missing scanner_profile_active_bins aligned to effective density: %+v", trace)
	}
	if runtimeDensity, ok := trace["scanner_density_runtime"].(int); !ok || runtimeDensity <= 0 || runtimeDensity > effective {
		t.Fatalf("trace missing scanner_density_runtime bounded by effective density: %+v", trace)
	}
	if runtimeWidth, ok := trace["scanner_feature_width_runtime"].(int); !ok || runtimeWidth != mustTraceFlatlandInt(t, trace, "scanner_density_runtime")*3 {
		t.Fatalf("trace missing scanner_feature_width_runtime aligned to runtime density: %+v", trace)
	}
	if _, ok := trace["layout_variant"].(int); !ok {
		t.Fatalf("trace missing layout_variant: %+v", trace)
	}
	if _, ok := trace["layout_shift"].(int); !ok {
		t.Fatalf("trace missing layout_shift: %+v", trace)
	}
	if _, ok := trace["last_control_width"].(int); !ok {
		t.Fatalf("trace missing last_control_width: %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface == "" {
		t.Fatalf("trace missing control_surface: %+v", trace)
	}
	if surface, ok := trace["sensor_surface"].(string); !ok || surface == "" {
		t.Fatalf("trace missing sensor_surface: %+v", trace)
	}
	if _, ok := trace["sensor_width"].(int); !ok {
		t.Fatalf("trace missing sensor_width: %+v", trace)
	}
}

func TestFlatlandScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := FlatlandScape{}
	forager := scriptedStepAgent{
		id: "forager",
		fn: flatlandGreedyForager,
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), forager, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}
	if profile, _ := validationTrace["scanner_profile"].(string); profile != flatlandScannerProfileForward {
		t.Fatalf("expected validation scanner profile %q, trace=%+v", flatlandScannerProfileForward, validationTrace)
	}
	if effective, _ := validationTrace["scanner_density_effective"].(int); effective != 5 {
		t.Fatalf("expected validation effective scanner density=5, trace=%+v", validationTrace)
	}
	if active, _ := validationTrace["scanner_density_active"].(int); active != 5 {
		t.Fatalf("expected validation active scanner density=5, trace=%+v", validationTrace)
	}
	if runtime, _ := validationTrace["scanner_density_runtime"].(int); runtime != 5 {
		t.Fatalf("expected validation runtime scanner density=5, trace=%+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), forager, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
	if profile, _ := testTrace["scanner_profile"].(string); profile != flatlandScannerProfileBalanced {
		t.Fatalf("expected test scanner profile %q, trace=%+v", flatlandScannerProfileBalanced, testTrace)
	}
	if effective, _ := testTrace["scanner_density_effective"].(int); effective != 5 {
		t.Fatalf("expected test effective scanner density=5, trace=%+v", testTrace)
	}
	if active, _ := testTrace["scanner_density_active"].(int); active != 5 {
		t.Fatalf("expected test active scanner density=5, trace=%+v", testTrace)
	}
	if runtime, _ := testTrace["scanner_density_runtime"].(int); runtime != 5 {
		t.Fatalf("expected test runtime scanner density=5, trace=%+v", testTrace)
	}

	_, benchmarkTrace, err := scape.EvaluateMode(context.Background(), forager, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark mode: %v", err)
	}
	if mode, _ := benchmarkTrace["mode"].(string); mode != "benchmark" {
		t.Fatalf("expected benchmark mode trace marker, got %+v", benchmarkTrace)
	}
	if profile, _ := benchmarkTrace["scanner_profile"].(string); profile != flatlandScannerProfileCore {
		t.Fatalf("expected benchmark scanner profile %q, trace=%+v", flatlandScannerProfileCore, benchmarkTrace)
	}
	if effective, _ := benchmarkTrace["scanner_density_effective"].(int); effective != 3 {
		t.Fatalf("expected benchmark effective scanner density=3, trace=%+v", benchmarkTrace)
	}
	if active, _ := benchmarkTrace["scanner_density_active"].(int); active != 3 {
		t.Fatalf("expected benchmark active scanner density=3, trace=%+v", benchmarkTrace)
	}
	if width, _ := benchmarkTrace["scanner_feature_width_effective"].(int); width != 9 {
		t.Fatalf("expected benchmark effective scanner feature width=9, trace=%+v", benchmarkTrace)
	}
	if bins, ok := benchmarkTrace["scanner_runtime_active_bins"].([]int); !ok || !reflect.DeepEqual(bins, []int{1, 2, 3}) {
		t.Fatalf("expected benchmark runtime scanner bins [1 2 3], trace=%+v", benchmarkTrace)
	}
	if runtime, _ := benchmarkTrace["scanner_density_runtime"].(int); runtime != 3 {
		t.Fatalf("expected benchmark runtime scanner density=3, trace=%+v", benchmarkTrace)
	}
	if _, ok := benchmarkTrace["layout_variant"].(int); !ok {
		t.Fatalf("expected benchmark layout_variant trace marker, got %+v", benchmarkTrace)
	}
	if _, ok := benchmarkTrace["layout_shift"].(int); !ok {
		t.Fatalf("expected benchmark layout_shift trace marker, got %+v", benchmarkTrace)
	}
}

func TestFlatlandScapeBenchmarkModeUsesDeterministicLayoutVariant(t *testing.T) {
	scape := FlatlandScape{}
	agentID := "flatland-benchmark-agent"
	forager := scriptedStepAgent{
		id: agentID,
		fn: flatlandGreedyForager,
	}

	cfg, err := flatlandConfigForMode("benchmark")
	if err != nil {
		t.Fatalf("flatland benchmark config: %v", err)
	}
	wantVariant, wantShift, _ := flatlandLayoutVariant(cfg, agentID)
	wantHeading := flatlandLayoutHeading(wantVariant)

	_, traceA, err := scape.EvaluateMode(context.Background(), forager, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark mode first run: %v", err)
	}
	_, traceB, err := scape.EvaluateMode(context.Background(), forager, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark mode second run: %v", err)
	}

	variantA, ok := traceA["layout_variant"].(int)
	if !ok {
		t.Fatalf("missing layout_variant on first run: %+v", traceA)
	}
	shiftA, ok := traceA["layout_shift"].(int)
	if !ok {
		t.Fatalf("missing layout_shift on first run: %+v", traceA)
	}
	initialHeadingA, ok := traceA["initial_heading"].(int)
	if !ok {
		t.Fatalf("missing initial_heading on first run: %+v", traceA)
	}
	if variantA != wantVariant || shiftA != wantShift || initialHeadingA != wantHeading {
		t.Fatalf(
			"unexpected deterministic layout metadata first run: got variant=%d shift=%d heading=%d want variant=%d shift=%d heading=%d",
			variantA,
			shiftA,
			initialHeadingA,
			wantVariant,
			wantShift,
			wantHeading,
		)
	}

	variantB, _ := traceB["layout_variant"].(int)
	shiftB, _ := traceB["layout_shift"].(int)
	initialHeadingB, _ := traceB["initial_heading"].(int)
	if variantA != variantB || shiftA != shiftB || initialHeadingA != initialHeadingB {
		t.Fatalf(
			"expected deterministic benchmark layout metadata across runs, first=%+v second=%+v",
			traceA,
			traceB,
		)
	}
}

func TestFlatlandLayoutVariantRespondsToAgentIDOnlyInBenchmarkMode(t *testing.T) {
	gtCfg, err := flatlandConfigForMode("gt")
	if err != nil {
		t.Fatalf("flatland gt config: %v", err)
	}
	if variant, shift, _ := flatlandLayoutVariant(gtCfg, "agent-0"); variant != 0 || shift != 0 {
		t.Fatalf("expected gt mode to keep fixed layout, got variant=%d shift=%d", variant, shift)
	}

	benchmarkCfg, err := flatlandConfigForMode("benchmark")
	if err != nil {
		t.Fatalf("flatland benchmark config: %v", err)
	}
	baseline, _, _ := flatlandLayoutVariant(benchmarkCfg, "agent-0")
	foundDifferent := false
	for i := 1; i < 64; i++ {
		id := fmt.Sprintf("agent-%d", i)
		candidate, _, _ := flatlandLayoutVariant(benchmarkCfg, id)
		if candidate != baseline {
			foundDifferent = true
			break
		}
	}
	if !foundDifferent {
		t.Fatalf("expected benchmark layout variants to respond to agent id, baseline=%d", baseline)
	}
}

func TestFlatlandEpisodeWallCollisionPenalizesAndTracks(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:            "test",
		maxAge:          16,
		forageGoal:      10,
		foodPositions:   []int{7},
		poisonPositions: []int{18},
		wallPositions:   []int{1},
	})

	startEnergy := episode.energy
	moveStep, hitFood, hitPoison, wallCollision, _ := episode.step(1)
	if moveStep != 1 {
		t.Fatalf("expected move step to record attempted right move, got %d", moveStep)
	}
	if hitFood || hitPoison {
		t.Fatalf("expected wall collision step without resource collision, got food=%t poison=%t", hitFood, hitPoison)
	}
	if !wallCollision {
		t.Fatalf("expected wall collision signal")
	}
	if episode.wallCollisions != 1 {
		t.Fatalf("expected wall collision count=1, got %d", episode.wallCollisions)
	}
	if episode.energy >= startEnergy {
		t.Fatalf("expected wall collision to reduce energy, before=%f after=%f", startEnergy, episode.energy)
	}
}

func TestFlatlandScapeBenchmarkTraceIncludesSocialDynamics(t *testing.T) {
	scape := FlatlandScape{}
	forager := scriptedStepAgent{
		id: "flatland-social-benchmark",
		fn: flatlandGreedyForager,
	}

	_, trace, err := scape.EvaluateMode(context.Background(), forager, "benchmark")
	if err != nil {
		t.Fatalf("evaluate benchmark: %v", err)
	}

	if enabled, ok := trace["social_dynamics"].(bool); !ok || !enabled {
		t.Fatalf("expected social_dynamics=true, trace=%+v", trace)
	}
	prey, ok := trace["prey_collected"].(int)
	if !ok {
		t.Fatalf("expected prey_collected in trace, trace=%+v", trace)
	}
	predatorHits, ok := trace["predator_hits"].(int)
	if !ok {
		t.Fatalf("expected predator_hits in trace, trace=%+v", trace)
	}
	socialCollisions, ok := trace["social_collisions"].(int)
	if !ok {
		t.Fatalf("expected social_collisions in trace, trace=%+v", trace)
	}
	if socialCollisions != prey+predatorHits {
		t.Fatalf("expected social_collisions=%d (prey=%d + predator_hits=%d), trace=%+v", prey+predatorHits, prey, predatorHits, trace)
	}
	if _, ok := trace["active_prey"].(int); !ok {
		t.Fatalf("expected active_prey in trace, trace=%+v", trace)
	}
	if _, ok := trace["active_predators"].(int); !ok {
		t.Fatalf("expected active_predators in trace, trace=%+v", trace)
	}
	if _, ok := trace["prey_hunted"].(int); !ok {
		t.Fatalf("expected prey_hunted in trace, trace=%+v", trace)
	}
	if _, ok := trace["predator_feeds"].(int); !ok {
		t.Fatalf("expected predator_feeds in trace, trace=%+v", trace)
	}
	if _, ok := trace["predator_pressure_events"].(int); !ok {
		t.Fatalf("expected predator_pressure_events in trace, trace=%+v", trace)
	}
}

func TestFlatlandEpisodePredatorCollisionPenalizesAndTracks(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:              "test",
		maxAge:            16,
		forageGoal:        8,
		foodPositions:     []int{},
		poisonPositions:   []int{},
		wallPositions:     []int{},
		predatorPositions: []int{1},
	})

	startEnergy := episode.energy
	episode.step(1)
	if episode.predatorHits != 1 {
		t.Fatalf("expected predator hit count=1, got %d", episode.predatorHits)
	}
	if episode.energy >= startEnergy {
		t.Fatalf("expected predator collision to reduce energy, before=%f after=%f", startEnergy, episode.energy)
	}
	if len(episode.predators) != 1 {
		t.Fatalf("expected single predator resource, got=%d", len(episode.predators))
	}
	if episode.predators[0].cooldown != flatlandPredatorRespawn {
		t.Fatalf("expected predator respawn cooldown=%d, got=%d", flatlandPredatorRespawn, episode.predators[0].cooldown)
	}
}

func TestFlatlandEpisodePreyCollisionRewardsAndTracks(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:            "test",
		maxAge:          16,
		forageGoal:      8,
		foodPositions:   []int{},
		poisonPositions: []int{},
		wallPositions:   []int{},
		preyPositions:   []int{1},
	})

	startEnergy := episode.energy
	episode.step(1)
	if episode.preyCollected != 1 {
		t.Fatalf("expected prey_collected=1, got %d", episode.preyCollected)
	}
	if episode.energy <= startEnergy {
		t.Fatalf("expected prey collision to increase energy, before=%f after=%f", startEnergy, episode.energy)
	}
	if len(episode.prey) != 1 {
		t.Fatalf("expected single prey resource, got=%d", len(episode.prey))
	}
	if episode.prey[0].cooldown != flatlandPreyRespawn {
		t.Fatalf("expected prey respawn cooldown=%d, got=%d", flatlandPreyRespawn, episode.prey[0].cooldown)
	}
}

func TestFlatlandEpisodePredatorHuntsNearbyPrey(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:              "test",
		maxAge:            16,
		forageGoal:        8,
		foodPositions:     []int{},
		poisonPositions:   []int{},
		wallPositions:     []int{},
		preyPositions:     []int{10},
		predatorPositions: []int{9},
		socialDynamics:    true,
	})
	episode.position = 30

	episode.step(0)
	if episode.preyHunted != 1 {
		t.Fatalf("expected prey_hunted=1, got %d", episode.preyHunted)
	}
	if episode.predatorFeeds != 1 {
		t.Fatalf("expected predator_feeds=1, got %d", episode.predatorFeeds)
	}
	if len(episode.prey) != 1 || episode.prey[0].cooldown != flatlandPreyRespawn {
		t.Fatalf("expected hunted prey to enter respawn cooldown=%d, prey=%+v", flatlandPreyRespawn, episode.prey)
	}
	if len(episode.predators) != 1 || episode.predators[0].potency <= flatlandPredatorDamageMin {
		t.Fatalf("expected predator potency boost after feed, predators=%+v", episode.predators)
	}
}

func TestFlatlandEpisodePredatorPressurePenalizesNearMiss(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:              "test",
		maxAge:            16,
		forageGoal:        8,
		foodPositions:     []int{},
		poisonPositions:   []int{},
		wallPositions:     []int{},
		preyPositions:     []int{},
		predatorPositions: []int{2},
		socialDynamics:    true,
	})
	episode.position = 0

	startEnergy := episode.energy
	baselineNoPressure := startEnergy - (flatlandBaseMetabolic + flatlandIdleMetabolic)
	episode.step(0)
	if episode.predatorHits != 0 {
		t.Fatalf("expected near-miss pressure scenario without direct hit, predator_hits=%d", episode.predatorHits)
	}
	if episode.predatorPressureEvents == 0 {
		t.Fatalf("expected at least one predator pressure event, got=%d", episode.predatorPressureEvents)
	}
	if episode.energy >= baselineNoPressure {
		t.Fatalf("expected pressure to reduce energy below baseline=%f, got=%f", baselineNoPressure, episode.energy)
	}
}

func TestFlatlandEpisodeRespawnsFoodAwayFromConsumedCell(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:            "test",
		maxAge:          64,
		forageGoal:      10,
		foodPositions:   []int{1},
		poisonPositions: []int{},
		wallPositions:   []int{2},
	})

	if len(episode.food) == 0 {
		t.Fatal("expected at least one food resource")
	}
	consumedPosition := episode.food[0].position
	_, hitFood, _, _, _ := episode.step(1)
	if !hitFood {
		t.Fatalf("expected food collision on first move, episode=%+v", episode)
	}

	for i := 0; i < flatlandFoodRespawn; i++ {
		episode.advanceRespawns()
	}
	if episode.food[0].cooldown != 0 {
		t.Fatalf("expected respawned food cooldown=0, got %d", episode.food[0].cooldown)
	}
	if episode.food[0].position == consumedPosition {
		t.Fatalf("expected respawned food to relocate from %d, got %d", consumedPosition, episode.food[0].position)
	}
	if episode.resourceRespawns == 0 {
		t.Fatalf("expected respawn counter to increase, got %d", episode.resourceRespawns)
	}
}

func TestFlatlandEpisodeScannerProbeOffsetsRespectHeadingAndOffset(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:            "test",
		maxAge:          64,
		forageGoal:      6,
		foodPositions:   []int{6, 12, 20, 28, 36, 44},
		poisonPositions: []int{15, 31},
		wallPositions:   []int{9, 17, 25, 33, 41},
		scannerSpread:   0.2,
		scannerOffset:   0,
	})

	episode.heading = 1
	forward := episode.scannerProbeOffsets()
	episode.heading = -1
	reverse := episode.scannerProbeOffsets()
	for i := range forward {
		if forward[i] != -reverse[i] {
			t.Fatalf("expected mirrored offsets at index %d, forward=%v reverse=%v", i, forward, reverse)
		}
	}

	episode.heading = 1
	episode.scannerOffset = 0.6
	shifted := episode.scannerProbeOffsets()
	if shifted[flatlandScannerDensity/2] <= forward[flatlandScannerDensity/2] {
		t.Fatalf("expected positive scanner offset to shift center probe forward, baseline=%v shifted=%v", forward, shifted)
	}
}

func TestFlatlandEpisodeScannerProfileCoreMasksEdgeBins(t *testing.T) {
	probes := []flatlandResource{
		{position: 43, potency: flatlandFoodEnergyMax},
		{position: 46, potency: flatlandFoodEnergyMax},
		{position: 0, potency: flatlandFoodEnergyMax},
		{position: 2, potency: flatlandFoodEnergyMax},
		{position: 5, potency: flatlandFoodEnergyMax},
	}

	balanced := newFlatlandEpisode(flatlandModeConfig{
		mode:            "test",
		maxAge:          64,
		forageGoal:      6,
		foodPositions:   []int{6, 12, 20, 28, 36, 44},
		poisonPositions: []int{15, 31},
		wallPositions:   []int{9, 17, 25, 33, 41},
		scannerSpread:   0.2,
		scannerOffset:   0,
		scannerProfile:  flatlandScannerProfileBalanced,
	})
	balanced.position = 0
	balanced.heading = 1
	balanced.food = append([]flatlandResource(nil), probes...)
	balanced.poison = nil
	balanced.walls = map[int]struct{}{}

	core := newFlatlandEpisode(flatlandModeConfig{
		mode:            "test",
		maxAge:          64,
		forageGoal:      6,
		foodPositions:   []int{6, 12, 20, 28, 36, 44},
		poisonPositions: []int{15, 31},
		wallPositions:   []int{9, 17, 25, 33, 41},
		scannerSpread:   0.2,
		scannerOffset:   0,
		scannerProfile:  flatlandScannerProfileCore,
	})
	core.position = 0
	core.heading = 1
	core.food = append([]flatlandResource(nil), probes...)
	core.poison = nil
	core.walls = map[int]struct{}{}

	balancedDistance, _, _ := balanced.senseScannerVectors()
	coreDistance, coreColor, coreEnergy := core.senseScannerVectors()
	if balancedDistance[0] <= 0 || balancedDistance[flatlandScannerDensity-1] <= 0 {
		t.Fatalf("expected balanced profile to preserve edge scan energy, got=%v", balancedDistance)
	}
	if coreDistance[0] != 0 || coreDistance[flatlandScannerDensity-1] != 0 {
		t.Fatalf("expected core profile to mask edge distance bins, got=%v", coreDistance)
	}
	if coreColor[0] != 0 || coreColor[flatlandScannerDensity-1] != 0 {
		t.Fatalf("expected core profile to mask edge color bins, got=%v", coreColor)
	}
	if coreEnergy[0] != 0 || coreEnergy[flatlandScannerDensity-1] != 0 {
		t.Fatalf("expected core profile to mask edge energy bins, got=%v", coreEnergy)
	}
	if coreDistance[flatlandScannerDensity/2] <= 0 {
		t.Fatalf("expected core profile center bin to remain active, got=%v", coreDistance)
	}
}

func TestFlatlandEpisodeNearestEntityFromClassifiesSocialAndResourceTypes(t *testing.T) {
	episode := newFlatlandEpisode(flatlandModeConfig{
		mode:            "test",
		maxAge:          64,
		forageGoal:      6,
		foodPositions:   []int{},
		poisonPositions: []int{},
		wallPositions:   []int{},
	})
	episode.food = []flatlandResource{{position: 3, potency: flatlandFoodEnergyMax}}
	episode.prey = []flatlandResource{{position: 7, potency: flatlandPreyEnergyMax}}
	episode.poison = []flatlandResource{{position: 11, potency: flatlandPoisonDamageMax}}
	episode.predators = []flatlandResource{{position: 15, potency: flatlandPredatorDamageMax}}
	episode.walls = map[int]struct{}{19: {}}

	cases := []struct {
		name        string
		origin      int
		wantKind    flatlandScannerEntityKind
		wantPotency float64
	}{
		{name: "plant", origin: 3, wantKind: flatlandScannerEntityPlant, wantPotency: flatlandFoodEnergyMax},
		{name: "prey", origin: 7, wantKind: flatlandScannerEntityPrey, wantPotency: flatlandPreyEnergyMax},
		{name: "poison", origin: 11, wantKind: flatlandScannerEntityPoison, wantPotency: flatlandPoisonDamageMax},
		{name: "predator", origin: 15, wantKind: flatlandScannerEntityPredator, wantPotency: flatlandPredatorDamageMax},
		{name: "wall", origin: 19, wantKind: flatlandScannerEntityWall, wantPotency: flatlandWallPenalty},
	}
	for _, tc := range cases {
		kind, distance, potency, ok := episode.nearestEntityFrom(tc.origin)
		if !ok {
			t.Fatalf("%s: expected nearest entity, got none", tc.name)
		}
		if distance != 0 {
			t.Fatalf("%s: expected distance=0, got=%d", tc.name, distance)
		}
		if kind != tc.wantKind {
			t.Fatalf("%s: expected kind=%v got=%v", tc.name, tc.wantKind, kind)
		}
		if math.Abs(potency-tc.wantPotency) > 1e-9 {
			t.Fatalf("%s: expected potency=%f got=%f", tc.name, tc.wantPotency, potency)
		}
	}
}

func TestFlatlandEpisodeScannerColorSemanticsMatchEntityClasses(t *testing.T) {
	newScannerEpisode := func() *flatlandEpisode {
		episode := newFlatlandEpisode(flatlandModeConfig{
			mode:            "test",
			maxAge:          64,
			forageGoal:      6,
			foodPositions:   []int{},
			poisonPositions: []int{},
			wallPositions:   []int{},
			scannerSpread:   0.2,
			scannerOffset:   0,
			scannerProfile:  flatlandScannerProfileBalanced,
		})
		episode.position = 0
		episode.heading = 1
		episode.food = nil
		episode.prey = nil
		episode.poison = nil
		episode.predators = nil
		episode.walls = map[int]struct{}{}
		return episode
	}

	assertCenter := func(name string, setup func(*flatlandEpisode), wantColor float64, energySign int) {
		t.Helper()
		episode := newScannerEpisode()
		setup(episode)
		distance, color, energy := episode.senseScannerVectors()
		center := flatlandScannerDensity / 2
		if distance[center] <= 0 {
			t.Fatalf("%s: expected center distance signal > 0, got=%v", name, distance)
		}
		if math.Abs(color[center]-wantColor) > 1e-9 {
			t.Fatalf("%s: expected center color=%f got=%f bins=%v", name, wantColor, color[center], color)
		}
		switch energySign {
		case 1:
			if energy[center] <= 0 {
				t.Fatalf("%s: expected positive center energy signal, got=%v", name, energy)
			}
		case -1:
			if energy[center] >= 0 {
				t.Fatalf("%s: expected negative center energy signal, got=%v", name, energy)
			}
		}
	}

	assertCenter("plant", func(e *flatlandEpisode) {
		e.food = []flatlandResource{{position: 0, potency: flatlandFoodEnergyMax}}
	}, flatlandScannerColorPlant, 1)
	assertCenter("prey", func(e *flatlandEpisode) {
		e.prey = []flatlandResource{{position: 0, potency: flatlandPreyEnergyMax}}
	}, flatlandScannerColorPrey, 1)
	assertCenter("poison", func(e *flatlandEpisode) {
		e.poison = []flatlandResource{{position: 0, potency: flatlandPoisonDamageMax}}
	}, flatlandScannerColorPoison, -1)
	assertCenter("predator", func(e *flatlandEpisode) {
		e.predators = []flatlandResource{{position: 0, potency: flatlandPredatorDamageMax}}
	}, flatlandScannerColorPredator, -1)
	assertCenter("wall", func(e *flatlandEpisode) {
		e.walls = map[int]struct{}{0: {}}
	}, flatlandScannerColorWall, -1)
}

func mustTraceFloat64(t *testing.T, trace Trace, key string) float64 {
	t.Helper()
	value, ok := trace[key].(float64)
	if !ok {
		t.Fatalf("expected trace[%q] float64, got trace=%+v", key, trace)
	}
	return value
}

func mustTraceFloat64Slice(t *testing.T, trace Trace, key string) []float64 {
	t.Helper()
	value, ok := trace[key].([]float64)
	if !ok {
		t.Fatalf("expected trace[%q] []float64, got trace=%+v", key, trace)
	}
	return value
}

func mustTraceFlatlandInt(t *testing.T, trace Trace, key string) int {
	t.Helper()
	value, ok := trace[key].(int)
	if !ok {
		t.Fatalf("expected trace[%q] int, got trace=%+v", key, trace)
	}
	return value
}
