package scape

import (
	"context"
	"fmt"
	"math"
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
	if width, ok := trace["step_input_width"].(int); !ok || width != flatlandBaseFeatureWidth+flatlandScannerWidth {
		t.Fatalf("expected trace step_input_width=%d, trace=%+v", flatlandBaseFeatureWidth+flatlandScannerWidth, trace)
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
