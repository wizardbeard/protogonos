package scape

import (
	"context"
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

	_, testTrace, err := scape.EvaluateMode(context.Background(), follow, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
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
			{ID: "trade", Activation: "tanh"},
		},
		Synapses: []model.Synapse{
			{From: "signal", To: "trade", Weight: 1.1, Enabled: true},
			{From: "mom", To: "trade", Weight: 0.8, Enabled: true},
			{From: "vol", To: "trade", Weight: -0.5, Enabled: true},
			{From: "nav", To: "trade", Weight: 0.25, Enabled: true},
			{From: "dd", To: "trade", Weight: -0.6, Enabled: true},
			{From: "pos", To: "trade", Weight: 0.15, Enabled: true},
			{From: "price", To: "trade", Weight: 0.2, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.FXPriceSensorName:      protoio.NewScalarInputSensor(0),
		protoio.FXSignalSensorName:     protoio.NewScalarInputSensor(0),
		protoio.FXMomentumSensorName:   protoio.NewScalarInputSensor(0),
		protoio.FXVolatilitySensorName: protoio.NewScalarInputSensor(0),
		protoio.FXNAVSensorName:        protoio.NewScalarInputSensor(0),
		protoio.FXDrawdownSensorName:   protoio.NewScalarInputSensor(0),
		protoio.FXPositionSensorName:   protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.FXTradeActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"fx-agent-io-extended",
		genome,
		sensors,
		actuators,
		[]string{"price", "signal", "mom", "vol", "nav", "dd", "pos"},
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
}

func TestFXScapeStepPerceptIncludesMarketInternals(t *testing.T) {
	scape := FXScape{}
	maxPerceptWidth := 0
	follow := scriptedStepAgent{
		id: "follow",
		fn: func(input []float64) []float64 {
			if len(input) > maxPerceptWidth {
				maxPerceptWidth = len(input)
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
