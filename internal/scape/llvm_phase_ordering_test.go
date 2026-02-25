package scape

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"protogonos/internal/agent"
	"protogonos/internal/genotype"
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

func TestLLVMPhaseOrderingScapeEvaluateWithExtendedIOComponents(t *testing.T) {
	genome := model.Genome{
		SensorIDs: []string{
			protoio.LLVMComplexitySensorName,
			protoio.LLVMPassIndexSensorName,
			protoio.LLVMAlignmentSensorName,
			protoio.LLVMDiversitySensorName,
			protoio.LLVMRuntimeGainSensorName,
		},
		ActuatorIDs: []string{protoio.LLVMPhaseActuatorName},
		Neurons: []model.Neuron{
			{ID: "c", Activation: "identity"},
			{ID: "p", Activation: "identity"},
			{ID: "a", Activation: "identity"},
			{ID: "d", Activation: "identity"},
			{ID: "r", Activation: "identity"},
			{ID: "o", Activation: "identity", Bias: 0.2},
		},
		Synapses: []model.Synapse{
			{From: "c", To: "o", Weight: -0.6, Enabled: true},
			{From: "p", To: "o", Weight: -1.2, Enabled: true},
			{From: "a", To: "o", Weight: 0.5, Enabled: true},
			{From: "d", To: "o", Weight: 0.4, Enabled: true},
			{From: "r", To: "o", Weight: 0.3, Enabled: true},
		},
	}

	sensors := map[string]protoio.Sensor{
		protoio.LLVMComplexitySensorName:  protoio.NewScalarInputSensor(0),
		protoio.LLVMPassIndexSensorName:   protoio.NewScalarInputSensor(0),
		protoio.LLVMAlignmentSensorName:   protoio.NewScalarInputSensor(0),
		protoio.LLVMDiversitySensorName:   protoio.NewScalarInputSensor(0),
		protoio.LLVMRuntimeGainSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.LLVMPhaseActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex(
		"llvm-agent-extended-io",
		genome,
		sensors,
		actuators,
		[]string{"c", "p", "a", "d", "r"},
		[]string{"o"},
		nil,
	)
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
	if _, ok := trace["mean_diversity"].(float64); !ok {
		t.Fatalf("trace missing mean_diversity: %+v", trace)
	}
	if _, ok := trace["mean_runtime_gain"].(float64); !ok {
		t.Fatalf("trace missing mean_runtime_gain: %+v", trace)
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
			out := make([]float64, len(defaultLLVMOptimizations))
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
	if surface, ok := trace["optimization_surface"].(int); !ok || surface != len(defaultLLVMOptimizations) {
		t.Fatalf("expected optimization surface width=%d, got %+v", len(defaultLLVMOptimizations), trace)
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

func TestLLVMPhaseOrderingScapeEvaluateWithSeedVectorCortex(t *testing.T) {
	seed, err := genotype.ConstructSeedPopulation("llvm-phase-ordering", 1, 71)
	if err != nil {
		t.Fatalf("construct llvm seed population: %v", err)
	}
	genome := seed.Genomes[0]

	sensors := map[string]protoio.Sensor{
		protoio.LLVMComplexitySensorName:  protoio.NewScalarInputSensor(0),
		protoio.LLVMPassIndexSensorName:   protoio.NewScalarInputSensor(0),
		protoio.LLVMAlignmentSensorName:   protoio.NewScalarInputSensor(0),
		protoio.LLVMDiversitySensorName:   protoio.NewScalarInputSensor(0),
		protoio.LLVMRuntimeGainSensorName: protoio.NewScalarInputSensor(0),
	}
	actuators := map[string]protoio.Actuator{
		protoio.LLVMPhaseActuatorName: protoio.NewScalarOutputActuator(),
	}

	cortex, err := agent.NewCortex("llvm-seed-vector", genome, sensors, actuators, seed.InputNeuronIDs, seed.OutputNeuronIDs, nil)
	if err != nil {
		t.Fatalf("new cortex: %v", err)
	}

	scape := LLVMPhaseOrderingScape{}
	_, trace, err := scape.Evaluate(context.Background(), cortex)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if vectors, ok := trace["vector_decisions"].(int); !ok || vectors <= 0 {
		t.Fatalf("expected vector decisions from seed cortex, got %+v", trace)
	}
	if surface, ok := trace["optimization_surface"].(int); !ok || surface != len(defaultLLVMOptimizations) {
		t.Fatalf("expected optimization surface=%d, got %+v", len(defaultLLVMOptimizations), trace)
	}
	if alignment, ok := trace["last_alignment"].(float64); !ok || alignment < 0 || alignment > 1 {
		t.Fatalf("expected last_alignment in [0,1], got %+v", trace)
	}
}

func TestLLVMPhaseOrderingScapeLoadWorkflowJSON(t *testing.T) {
	ResetLLVMWorkflowSource()
	t.Cleanup(ResetLLVMWorkflowSource)

	path := filepath.Join(t.TempDir(), "llvm_workflow.json")
	data := `{
  "name": "llvm.custom.v1",
  "optimizations": ["done", "instcombine", "licm", "gvn"],
  "modes": {
    "gt": {
      "program": "custom-prog",
      "max_phases": 12,
      "initial_complexity": 1.1,
      "target_complexity": 0.4,
      "base_runtime": 0.9
    }
  }
}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write workflow json: %v", err)
	}
	if err := LoadLLVMWorkflowJSON(path); err != nil {
		t.Fatalf("load workflow: %v", err)
	}

	scape := LLVMPhaseOrderingScape{}
	flat := scriptedStepAgent{
		id: "flat",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	_, trace, err := scape.Evaluate(context.Background(), flat)
	if err != nil {
		t.Fatalf("evaluate with loaded workflow: %v", err)
	}
	if name, ok := trace["workflow_name"].(string); !ok || name != "llvm.custom.v1" {
		t.Fatalf("expected custom workflow name in trace, got %+v", trace)
	}
	if surface, ok := trace["optimization_surface"].(int); !ok || surface != 4 {
		t.Fatalf("expected custom optimization surface=4, got %+v", trace)
	}
}

func TestLLVMPhaseOrderingScapeContextWorkflowOverride(t *testing.T) {
	ResetLLVMWorkflowSource()
	t.Cleanup(ResetLLVMWorkflowSource)

	path := filepath.Join(t.TempDir(), "llvm_ctx_workflow.json")
	data := `{
  "name": "llvm.ctx.v1",
  "optimizations": ["done", "adce", "instcombine"],
  "modes": {
    "gt": {
      "program": "ctx-prog",
      "max_phases": 8,
      "initial_complexity": 1.0,
      "target_complexity": 0.5,
      "base_runtime": 1.0
    }
  }
}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write workflow json: %v", err)
	}

	ctx, err := WithDataSources(context.Background(), DataSources{
		LLVM: LLVMDataSource{WorkflowJSONPath: path},
	})
	if err != nil {
		t.Fatalf("with data sources: %v", err)
	}

	scape := LLVMPhaseOrderingScape{}
	flat := scriptedStepAgent{
		id: "flat",
		fn: func(_ []float64) []float64 { return []float64{0} },
	}
	_, scopedTrace, err := scape.Evaluate(ctx, flat)
	if err != nil {
		t.Fatalf("evaluate scoped workflow: %v", err)
	}
	_, defaultTrace, err := scape.Evaluate(context.Background(), flat)
	if err != nil {
		t.Fatalf("evaluate default workflow: %v", err)
	}
	if name, _ := scopedTrace["workflow_name"].(string); !strings.Contains(name, "llvm.ctx.v1") {
		t.Fatalf("expected scoped workflow name, got %+v", scopedTrace)
	}
	if name, _ := defaultTrace["workflow_name"].(string); name != "llvm.synthetic.v1" {
		t.Fatalf("expected default workflow name, got %+v", defaultTrace)
	}
}
