package scape

import (
	"context"
	"testing"

	"protogonos/internal/agent"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
)

func TestLLVMPhaseOrderingScapeRewardsPhaseAwarePolicy(t *testing.T) {
	scape := LLVMPhaseOrderingScape{}
	flat := scriptedStepAgent{
		id: "flat",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	phaseAware := scriptedStepAgent{
		id: "phase-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			passNorm := in[1]
			return []float64{1 - 2*passNorm}
		},
	}

	flatFitness, _, err := scape.Evaluate(context.Background(), flat)
	if err != nil {
		t.Fatalf("evaluate flat: %v", err)
	}
	phaseAwareFitness, _, err := scape.Evaluate(context.Background(), phaseAware)
	if err != nil {
		t.Fatalf("evaluate phase-aware: %v", err)
	}
	if phaseAwareFitness <= flatFitness {
		t.Fatalf("expected phase-aware policy to outperform flat policy, got aware=%f flat=%f", phaseAwareFitness, flatFitness)
	}
}

func TestLLVMPhaseOrderingScapeEvaluateWithIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.LLVMComplexitySensorName,
			protoio.LLVMPassIndexSensorName,
		},
		ActuatorIDs: []string{protoio.LLVMPhaseActuatorName},
		Neurons: []model.Neuron{
			{ID: "c", Activation: "identity"},
			{ID: "p", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 1},
		},
		Synapses: []model.Synapse{
			{From: "p", To: "o", Weight: -2, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.LLVMComplexitySensorName: protoio.NewScalarInputSensor(0),
		protoio.LLVMPassIndexSensorName:  protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.LLVMPhaseActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("llvm-agent-io", genome, sensors, actuators, []string{"c", "p"}, []string{"o"}, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := LLVMPhaseOrderingScape{}
	fitness, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if fitness <= 0 {
		t.Fatalf("expected positive fitness, got %f", fitness)
	}
	if _, ok := trace["phases"].(int); !ok {
		t.Fatalf("trace missing phases: %+v", trace)
	}
}

func TestLLVMPhaseOrderingScapeEvaluateModeAnnotatesMode(t *testing.T) {
	scape := LLVMPhaseOrderingScape{}
	phaseAware := scriptedStepAgent{
		id: "phase-aware",
		fn: func(in []float64) []float64 {
			if len(in) < 2 {
				return []float64{0}
			}
			return []float64{1 - 2*in[1]}
		},
	}

	_, validationTrace, err := scape.EvaluateMode(context.Background(), phaseAware, "validation")
	if err != nil {
		t.Fatalf("evaluate validation mode: %v", err)
	}
	if mode, _ := validationTrace["mode"].(string); mode != "validation" {
		t.Fatalf("expected validation mode trace marker, got %+v", validationTrace)
	}

	_, testTrace, err := scape.EvaluateMode(context.Background(), phaseAware, "test")
	if err != nil {
		t.Fatalf("evaluate test mode: %v", err)
	}
	if mode, _ := testTrace["mode"].(string); mode != "test" {
		t.Fatalf("expected test mode trace marker, got %+v", testTrace)
	}
}

func TestLLVMPhaseOrderingScapeSupportsVectorOptimizationSurface(t *testing.T) {
	scape := LLVMPhaseOrderingScape{}
	vectorPolicy := scriptedStepAgent{
		id: "vector-policy",
		fn: func(in []float64) []float64 {
			out := make([]float64, len(llvmOptimizationList))
			passNorm := 0.0
			if len(in) > 1 {
				passNorm = in[1]
			}
			switch {
			case passNorm < 0.25:
				out[16] = 1 // gvn
			case passNorm < 0.50:
				out[31] = 1 // loop-simplify
			case passNorm < 0.80:
				out[47] = 1 // simplifycfg
			default:
				out[0] = 1 // done
			}
			return out
		},
	}

	_, trace, err := scape.Evaluate(context.Background(), vectorPolicy)
	if err != nil {
		t.Fatalf("evaluate vector policy: %v", err)
	}
	if surface, ok := trace["optimization_surface"].(int); !ok || surface != len(llvmOptimizationList) {
		t.Fatalf("expected optimization surface width=%d, got %+v", len(llvmOptimizationList), trace)
	}
	if vectors, ok := trace["vector_decisions"].(int); !ok || vectors <= 0 {
		t.Fatalf("expected positive vector_decisions, got %+v", trace)
	}
	if width, ok := trace["percept_width"].(int); !ok || width != 31 {
		t.Fatalf("expected percept_width=31, got %+v", trace)
	}
	history, ok := trace["selected_optimizations"].([]string)
	if !ok || len(history) == 0 {
		t.Fatalf("expected selected_optimizations history, got %+v", trace)
	}
}
