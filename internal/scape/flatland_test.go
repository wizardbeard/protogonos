package scape

import (
	"context"
	"testing"

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
			protoio.FlatlandPoisonSensorName,
			protoio.FlatlandWallSensorName,
			protoio.FlatlandFoodProximitySensorName,
			protoio.FlatlandPoisonProximitySensorName,
			protoio.FlatlandWallProximitySensorName,
			protoio.FlatlandResourceBalanceSensorName,
		},
		ActuatorIDs: []string{protoio.FlatlandMoveActuatorName},
		Neurons: []model.Neuron{
			{ID: "poison", Activation: "identity"},
			{ID: "wall", Activation: "identity"},
			{ID: "food_prox", Activation: "identity"},
			{ID: "poison_prox", Activation: "identity"},
			{ID: "wall_prox", Activation: "identity"},
			{ID: "balance", Activation: "identity"},
			{ID: "move", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "poison", To: "move", Weight: -0.8, Enabled: true},
			{From: "wall", To: "move", Weight: -0.6, Enabled: true},
			{From: "food_prox", To: "move", Weight: 0.9, Enabled: true},
			{From: "poison_prox", To: "move", Weight: -0.7, Enabled: true},
			{From: "wall_prox", To: "move", Weight: -0.5, Enabled: true},
			{From: "balance", To: "move", Weight: 0.4, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FlatlandPoisonSensorName:          protoio.NewScalarInputSensor(0),
		protoio.FlatlandWallSensorName:            protoio.NewScalarInputSensor(0),
		protoio.FlatlandFoodProximitySensorName:   protoio.NewScalarInputSensor(0),
		protoio.FlatlandPoisonProximitySensorName: protoio.NewScalarInputSensor(0),
		protoio.FlatlandWallProximitySensorName:   protoio.NewScalarInputSensor(0),
		protoio.FlatlandResourceBalanceSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FlatlandMoveActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"flatland-agent-io-extended",
		genome,
		sensors,
		actuators,
		[]string{"poison", "wall", "food_prox", "poison_prox", "wall_prox", "balance"},
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
	if width, ok := trace["feature_width"].(int); !ok || width != 8 {
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
	if density, ok := trace["scanner_density"].(int); !ok || density != 5 {
		t.Fatalf("expected scanner density marker, trace=%+v", trace)
	}
	if _, ok := trace["last_distance_scan_mean"].(float64); !ok {
		t.Fatalf("expected last_distance_scan_mean trace marker, trace=%+v", trace)
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
	if _, ok := trace["last_wall_signal"].(float64); !ok {
		t.Fatalf("trace missing last_wall_signal: %+v", trace)
	}
	if _, ok := trace["last_food_proximity"].(float64); !ok {
		t.Fatalf("trace missing last_food_proximity: %+v", trace)
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
	if _, ok := trace["last_control_width"].(int); !ok {
		t.Fatalf("trace missing last_control_width: %+v", trace)
	}
	if surface, ok := trace["control_surface"].(string); !ok || surface == "" {
		t.Fatalf("trace missing control_surface: %+v", trace)
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

	_, testTrace, err := scape.EvaluateMode(context.Background(), forager, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
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
