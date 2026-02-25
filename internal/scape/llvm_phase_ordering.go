package scape

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

// LLVMPhaseOrderingScape is a deterministic surrogate for the reference
// phase-ordering workflow, preserving a phase-indexed optimize loop.
type LLVMPhaseOrderingScape struct{}

type llvmModeProfile struct {
	Program           string  `json:"program"`
	MaxPhases         int     `json:"max_phases"`
	InitialComplexity float64 `json:"initial_complexity"`
	TargetComplexity  float64 `json:"target_complexity"`
	BaseRuntime       float64 `json:"base_runtime"`
}

type llvmWorkflow struct {
	name          string
	optimizations []string
	modes         map[string]llvmModeProfile
}

type llvmWorkflowFile struct {
	Name          string                     `json:"name"`
	Optimizations []string                   `json:"optimizations"`
	Modes         map[string]llvmModeProfile `json:"modes"`
}

var (
	llvmWorkflowSourceMu sync.RWMutex
	llvmWorkflowSource   = defaultLLVMWorkflow()
)

func (LLVMPhaseOrderingScape) Name() string {
	return "llvm-phase-ordering"
}

// ResetLLVMWorkflowSource restores the deterministic built-in LLVM workflow.
func ResetLLVMWorkflowSource() {
	llvmWorkflowSourceMu.Lock()
	defer llvmWorkflowSourceMu.Unlock()
	llvmWorkflowSource = defaultLLVMWorkflow()
}

// LoadLLVMWorkflowJSON loads LLVM optimization surface/runtime profiles from JSON.
func LoadLLVMWorkflowJSON(path string) error {
	workflow, err := loadLLVMWorkflowJSON(path)
	if err != nil {
		return err
	}

	llvmWorkflowSourceMu.Lock()
	defer llvmWorkflowSourceMu.Unlock()
	llvmWorkflowSource = workflow
	return nil
}

func (LLVMPhaseOrderingScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return LLVMPhaseOrderingScape{}.EvaluateMode(ctx, agent, "gt")
}

func (LLVMPhaseOrderingScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	workflow := currentLLVMWorkflow(ctx)
	cfg, err := llvmPhaseOrderingConfigForMode(mode, workflow)
	if err != nil {
		return 0, nil, err
	}

	if ticker, ok := agent.(TickAgent); ok {
		fitness, trace, err := evaluateLLVMPhaseOrderingWithTick(ctx, ticker, cfg)
		if err == nil {
			return fitness, trace, nil
		}
	}

	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}
	return evaluateLLVMPhaseOrderingWithStep(ctx, runner, cfg)
}

func evaluateLLVMPhaseOrderingWithStep(ctx context.Context, runner StepAgent, cfg llvmPhaseOrderingConfig) (Fitness, Trace, error) {
	return evaluateLLVMPhaseOrdering(
		ctx,
		cfg,
		func(ctx context.Context, in []float64) ([]float64, error) {
			out, err := runner.RunStep(ctx, in)
			if err != nil {
				return nil, err
			}
			if len(out) == 0 {
				return nil, fmt.Errorf("llvm-phase-ordering requires at least one output")
			}
			return append([]float64(nil), out...), nil
		},
	)
}

func evaluateLLVMPhaseOrderingWithTick(ctx context.Context, ticker TickAgent, cfg llvmPhaseOrderingConfig) (Fitness, Trace, error) {
	complexitySetter, passSetter, phaseOutput, err := llvmPhaseOrderingIO(ticker)
	if err != nil {
		return 0, nil, err
	}

	return evaluateLLVMPhaseOrdering(
		ctx,
		cfg,
		func(ctx context.Context, in []float64) ([]float64, error) {
			complexitySetter.Set(in[0])
			passSetter.Set(in[1])

			out, err := ticker.Tick(ctx)
			if err != nil {
				return nil, err
			}

			last := phaseOutput.Last()
			if len(last) > 0 {
				return append([]float64(nil), last...), nil
			}
			if len(out) > 0 {
				return append([]float64(nil), out...), nil
			}
			return []float64{0}, nil
		},
	)
}

func evaluateLLVMPhaseOrdering(
	ctx context.Context,
	cfg llvmPhaseOrderingConfig,
	choosePhase func(context.Context, []float64) ([]float64, error),
) (Fitness, Trace, error) {
	complexity := cfg.initialComplexity
	bestComplexity := complexity
	alignmentAcc := 0.0
	phasesUsed := 0
	done := false
	terminationReason := "max_phases"
	optimizationHistory := make([]string, 0, cfg.maxPhases)
	scalarDecisions := 0
	vectorDecisions := 0
	uniqueOpts := make(map[string]struct{}, cfg.maxPhases)
	runtimeBaseline := cfg.baseRuntime * (0.7 + 1.1*cfg.initialComplexity)

	for phase := 1; phase <= cfg.maxPhases; phase++ {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		passNorm := 0.0
		if cfg.maxPhases > 1 {
			passNorm = float64(phase-1) / float64(cfg.maxPhases-1)
		}
		percept := llvmPerceptVector(cfg.program, phase, cfg.maxPhases, complexity, passNorm, optimizationHistory)
		output, err := choosePhase(ctx, percept)
		if err != nil {
			return 0, nil, err
		}
		if len(output) == 0 {
			return 0, nil, fmt.Errorf("llvm-phase-ordering requires at least one output")
		}

		decision := decodeLLVMDecision(output, passNorm, cfg.optimizations)
		if decision.mode == "scalar" {
			scalarDecisions++
		} else {
			vectorDecisions++
		}
		alignmentAcc += decision.alignment
		phasesUsed++

		if decision.done {
			done = true
			terminationReason = "done_action"
			break
		}

		optimizationHistory = append(optimizationHistory, decision.optimization)
		uniqueOpts[decision.optimization] = struct{}{}
		gain := llvmOptimizationGain(cfg, decision, phase, complexity, optimizationHistory)
		complexity = clampLLVM(complexity-gain, 0.03, 2.5)
		if complexity < bestComplexity {
			bestComplexity = complexity
		}
		if complexity <= cfg.targetComplexity {
			done = true
			terminationReason = "target_complexity"
			break
		}
	}

	if phasesUsed == 0 {
		return 0, Trace{"phases": 0, "final_complexity": complexity}, nil
	}

	alignmentAvg := alignmentAcc / float64(phasesUsed)
	diversity := float64(len(uniqueOpts)) / float64(maxIntLLVM(1, len(optimizationHistory)))
	runtimeEstimate := llvmRuntimeEstimate(cfg, complexity, phasesUsed, diversity, done)
	runtimeScore := 1.0 / (1.0 + runtimeEstimate)
	fitness := 0.56*runtimeScore + 0.24*alignmentAvg + 0.20*diversity
	if !done && phasesUsed >= cfg.maxPhases {
		fitness -= 0.03
	}
	fitness = clampLLVM(fitness, 0, 1.5)
	improvement := 0.0
	if runtimeBaseline > 0 {
		improvement = (runtimeBaseline - runtimeEstimate) / runtimeBaseline
	}

	return Fitness(fitness), Trace{
		"fitness":                fitness,
		"phases":                 phasesUsed,
		"max_phases":             cfg.maxPhases,
		"done":                   done,
		"alignment":              alignmentAvg,
		"final_complexity":       complexity,
		"best_complexity":        bestComplexity,
		"estimated_runtime":      runtimeEstimate,
		"runtime_improvement":    improvement,
		"mode":                   cfg.mode,
		"program":                cfg.program,
		"termination_reason":     terminationReason,
		"workflow_name":          cfg.workflowName,
		"optimization_surface":   len(cfg.optimizations),
		"percept_width":          llvmPerceptWidth,
		"percept_layout":         "legacy2+extended29",
		"selected_optimizations": optimizationHistory,
		"unique_optimizations":   len(uniqueOpts),
		"scalar_decisions":       scalarDecisions,
		"vector_decisions":       vectorDecisions,
	}, nil
}

type llvmPhaseOrderingConfig struct {
	mode              string
	program           string
	maxPhases         int
	initialComplexity float64
	targetComplexity  float64
	baseRuntime       float64
	workflowName      string
	optimizations     []string
}

func llvmPhaseOrderingConfigForMode(mode string, workflow llvmWorkflow) (llvmPhaseOrderingConfig, error) {
	mode = strings.TrimSpace(strings.ToLower(mode))
	if mode == "" {
		mode = "gt"
	}
	profile, ok := workflow.profileForMode(mode)
	if !ok {
		return llvmPhaseOrderingConfig{}, fmt.Errorf("unsupported llvm-phase-ordering mode: %s", mode)
	}
	return llvmPhaseOrderingConfig{
		mode:              mode,
		program:           profile.Program,
		maxPhases:         profile.MaxPhases,
		initialComplexity: profile.InitialComplexity,
		targetComplexity:  profile.TargetComplexity,
		baseRuntime:       profile.BaseRuntime,
		workflowName:      workflow.name,
		optimizations:     append([]string(nil), workflow.optimizations...),
	}, nil
}

type llvmDecision struct {
	mode              string
	scalarAction      float64
	optimizationIndex int
	optimization      string
	done              bool
	alignment         float64
}

const llvmPerceptWidth = 31

var defaultLLVMOptimizations = []string{
	"done",
	"adce",
	"always-inline",
	"argpromotion",
	"bb-vectorize",
	"break-crit-edges",
	"codegenprepare",
	"constmerge",
	"constprop",
	"dce",
	"deadargelim",
	"die",
	"dse",
	"functionattrs",
	"globaldce",
	"globalopt",
	"gvn",
	"indvars",
	"inline",
	"instcombine",
	"internalize",
	"ipconstprop",
	"ipsccp",
	"jump-threading",
	"lcssa",
	"licm",
	"loop-deletion",
	"loop-extract",
	"loop-extract-single",
	"loop-reduce",
	"loop-rotate",
	"loop-simplify",
	"loop-unroll",
	"loop-unswitch",
	"loweratomic",
	"lowerinvoke",
	"lowerswitch",
	"mem2reg",
	"memcpyopt",
	"mergefunc",
	"mergereturn",
	"partial-inliner",
	"prune-eh",
	"reassociate",
	"reg2mem",
	"scalarrepl",
	"sccp",
	"simplifycfg",
	"sink",
	"strip",
	"strip-dead-debug-info",
	"strip-dead-prototypes",
	"strip-debug-declare",
	"strip-nondebug",
	"tailcallelim",
}

func defaultLLVMWorkflow() llvmWorkflow {
	return llvmWorkflow{
		name:          "llvm.synthetic.v1",
		optimizations: append([]string(nil), defaultLLVMOptimizations...),
		modes: map[string]llvmModeProfile{
			"gt": {
				Program:           "bzip2",
				MaxPhases:         50,
				InitialComplexity: 1.25,
				TargetComplexity:  0.34,
				BaseRuntime:       1.0,
			},
			"validation": {
				Program:           "gcc",
				MaxPhases:         40,
				InitialComplexity: 1.40,
				TargetComplexity:  0.40,
				BaseRuntime:       1.08,
			},
			"test": {
				Program:           "parser",
				MaxPhases:         40,
				InitialComplexity: 1.50,
				TargetComplexity:  0.45,
				BaseRuntime:       1.12,
			},
			"benchmark": {
				Program:           "bzip2",
				MaxPhases:         50,
				InitialComplexity: 1.30,
				TargetComplexity:  0.36,
				BaseRuntime:       1.00,
			},
		},
	}
}

func currentLLVMWorkflow(ctx context.Context) llvmWorkflow {
	if workflow, ok := llvmWorkflowFromContext(ctx); ok {
		return workflow
	}
	llvmWorkflowSourceMu.RLock()
	defer llvmWorkflowSourceMu.RUnlock()
	return llvmWorkflowSource
}

func loadLLVMWorkflowJSON(path string) (llvmWorkflow, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return llvmWorkflow{}, fmt.Errorf("llvm workflow json path is required")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return llvmWorkflow{}, fmt.Errorf("read llvm workflow json %s: %w", path, err)
	}
	var file llvmWorkflowFile
	if err := json.Unmarshal(data, &file); err != nil {
		return llvmWorkflow{}, fmt.Errorf("decode llvm workflow json %s: %w", path, err)
	}

	workflow := defaultLLVMWorkflow()
	if strings.TrimSpace(file.Name) != "" {
		workflow.name = strings.TrimSpace(file.Name)
	} else {
		workflow.name = fmt.Sprintf("llvm.json.%s", filepath.Base(path))
	}
	if len(file.Optimizations) > 0 {
		workflow.optimizations = normalizeLLVMOptimizations(file.Optimizations)
	}
	for mode, profile := range file.Modes {
		normalizedMode := strings.TrimSpace(strings.ToLower(mode))
		if normalizedMode == "" {
			continue
		}
		if profile.Program == "" {
			profile.Program = workflow.modes["gt"].Program
		}
		if profile.MaxPhases <= 0 {
			profile.MaxPhases = workflow.modes["gt"].MaxPhases
		}
		if profile.InitialComplexity <= 0 {
			profile.InitialComplexity = workflow.modes["gt"].InitialComplexity
		}
		if profile.TargetComplexity <= 0 {
			profile.TargetComplexity = workflow.modes["gt"].TargetComplexity
		}
		if profile.BaseRuntime <= 0 {
			profile.BaseRuntime = workflow.modes["gt"].BaseRuntime
		}
		workflow.modes[normalizedMode] = profile
	}
	return workflow, nil
}

func normalizeLLVMOptimizations(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in)+1)
	for _, item := range in {
		name := strings.TrimSpace(strings.ToLower(item))
		if name == "" {
			continue
		}
		if _, exists := seen[name]; exists {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	if len(out) == 0 {
		return append([]string(nil), defaultLLVMOptimizations...)
	}
	if out[0] != "done" {
		out = append([]string{"done"}, out...)
	}
	return out
}

func (w llvmWorkflow) profileForMode(mode string) (llvmModeProfile, bool) {
	if profile, ok := w.modes[mode]; ok {
		return profile, true
	}
	return llvmModeProfile{}, false
}

func decodeLLVMDecision(output []float64, passNorm float64, optimizations []string) llvmDecision {
	if len(optimizations) == 0 {
		optimizations = defaultLLVMOptimizations
	}
	if len(output) == 1 {
		action := clampLLVM(output[0], -1, 1)
		target := llvmTargetPhase(passNorm)
		alignment := 1.0 - 0.5*math.Abs(action-target)
		optIndex := scalarActionToOptimizationIndex(action, optimizations)
		done := action < -0.995 && passNorm >= 0.80
		return llvmDecision{
			mode:              "scalar",
			scalarAction:      action,
			optimizationIndex: optIndex,
			optimization:      llvmOptimizationName(optIndex, optimizations),
			done:              done,
			alignment:         clampLLVM(alignment, 0, 1),
		}
	}

	optIndex := argmax(output)
	if optIndex >= len(optimizations) {
		optIndex = len(optimizations) - 1
	}
	targetIndex := int(math.Round((1.0 - passNorm) * float64(len(optimizations)-1)))
	targetIndex = clampIntLLVM(targetIndex, 0, len(optimizations)-1)
	distance := math.Abs(float64(optIndex - targetIndex))
	alignment := 1.0 - distance/float64(maxIntLLVM(1, len(optimizations)-1))
	return llvmDecision{
		mode:              "vector",
		scalarAction:      0,
		optimizationIndex: optIndex,
		optimization:      llvmOptimizationName(optIndex, optimizations),
		done:              optIndex == 0,
		alignment:         clampLLVM(alignment, 0, 1),
	}
}

func llvmPerceptVector(program string, phaseIndex, maxPhases int, complexity, passNorm float64, history []string) []float64 {
	percept := make([]float64, llvmPerceptWidth)
	// Keep the first two dimensions backward-compatible with existing scalar policies.
	percept[0] = clampLLVM(complexity, 0, 2.5)
	percept[1] = clampLLVM(passNorm, 0, 1)

	inversePhase := 1.0 / float64(maxIntLLVM(1, phaseIndex))
	for i := 2; i < llvmPerceptWidth; i++ {
		base := math.Sin(float64(i)*0.37 + float64(phaseIndex)*0.21)
		trend := math.Cos(float64(i)*0.11 + complexity*1.7)
		historyBias := llvmHistorySignal(history, i)
		programBias := llvmProgramFeatureBias(program, i)
		value := 0.5 + 0.18*base + 0.17*trend + 0.12*historyBias + 0.20*programBias + 0.33*inversePhase
		percept[i] = clampLLVM(value, 0, 1)
	}
	return percept
}

func llvmOptimizationGain(cfg llvmPhaseOrderingConfig, decision llvmDecision, phase int, complexity float64, history []string) float64 {
	if decision.mode == "scalar" {
		progress := float64(phase-1) / float64(maxIntLLVM(1, cfg.maxPhases-1))
		effortPenalty := 0.012 * math.Abs(decision.scalarAction)
		gain := (0.028 + 0.032*decision.alignment) * (1.0 - 0.35*progress)
		gain = (gain - effortPenalty) * (0.65 + 0.35*complexity)
		return clampLLVM(gain, -0.03, 0.11)
	}

	affinity := llvmProgramOptimizationAffinity(cfg.program, decision.optimization)
	progress := float64(phase) / float64(maxIntLLVM(1, cfg.maxPhases))
	gain := (0.015 + 0.055*affinity) * (1.0 - 0.45*progress)
	gain = gain * (0.55 + 0.45*complexity)

	if len(history) >= 2 && history[len(history)-2] == decision.optimization {
		gain -= 0.012
	}
	if len(history) >= 3 && history[len(history)-3] == decision.optimization {
		gain -= 0.008
	}
	if strings.Contains(decision.optimization, "loop") && progress < 0.40 {
		gain += 0.004
	}
	if strings.Contains(decision.optimization, "strip") && progress > 0.60 {
		gain += 0.003
	}
	return clampLLVM(gain, -0.025, 0.12)
}

func llvmRuntimeEstimate(cfg llvmPhaseOrderingConfig, complexity float64, phasesUsed int, diversity float64, done bool) float64 {
	runtime := cfg.baseRuntime * (0.70 + 1.10*complexity)
	runtime += 0.015 * float64(phasesUsed)
	runtime -= 0.08 * diversity
	if !done && phasesUsed >= cfg.maxPhases {
		runtime += 0.12
	}
	if runtime < 0.05 {
		return 0.05
	}
	return runtime
}

func scalarActionToOptimizationIndex(action float64, optimizations []string) int {
	if len(optimizations) == 0 {
		return 0
	}
	scale := (clampLLVM(action, -1, 1) + 1) / 2
	index := int(math.Round(scale * float64(len(optimizations)-1)))
	return clampIntLLVM(index, 0, len(optimizations)-1)
}

func llvmOptimizationName(index int, optimizations []string) string {
	if len(optimizations) == 0 {
		return "done"
	}
	index = clampIntLLVM(index, 0, len(optimizations)-1)
	return optimizations[index]
}

func llvmProgramOptimizationAffinity(program, optimization string) float64 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(program))
	_, _ = h.Write([]byte("|"))
	_, _ = h.Write([]byte(optimization))
	return float64(h.Sum32()%1000) / 1000.0
}

func llvmProgramFeatureBias(program string, feature int) float64 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(program))
	_, _ = h.Write([]byte(":"))
	_, _ = h.Write([]byte(fmt.Sprintf("%d", feature)))
	raw := float64(h.Sum32()%1000) / 1000.0
	return raw - 0.5
}

func llvmHistorySignal(history []string, feature int) float64 {
	if len(history) == 0 {
		return 0
	}
	count := maxIntLLVM(1, minIntLLVM(3, len(history)))
	acc := 0.0
	for i := 0; i < count; i++ {
		opt := history[len(history)-1-i]
		h := fnv.New32a()
		_, _ = h.Write([]byte(opt))
		_, _ = h.Write([]byte("#"))
		_, _ = h.Write([]byte(fmt.Sprintf("%d", feature+i)))
		acc += float64(h.Sum32()%200)/100.0 - 1.0
	}
	return acc / float64(count)
}

func llvmTargetPhase(passNorm float64) float64 {
	return clampLLVM(1.0-2.0*passNorm, -1, 1)
}

func llvmPhaseOrderingIO(agent TickAgent) (protoio.ScalarSensorSetter, protoio.ScalarSensorSetter, protoio.SnapshotActuator, error) {
	typed, ok := agent.(interface {
		RegisteredSensor(id string) (protoio.Sensor, bool)
		RegisteredActuator(id string) (protoio.Actuator, bool)
	})
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s does not expose IO registry access", agent.ID())
	}

	complexity, ok := typed.RegisteredSensor(protoio.LLVMComplexitySensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.LLVMComplexitySensorName)
	}
	complexitySetter, ok := complexity.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.LLVMComplexitySensorName)
	}

	passIndex, ok := typed.RegisteredSensor(protoio.LLVMPassIndexSensorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing sensor %s", agent.ID(), protoio.LLVMPassIndexSensorName)
	}
	passSetter, ok := passIndex.(protoio.ScalarSensorSetter)
	if !ok {
		return nil, nil, nil, fmt.Errorf("sensor %s does not support scalar set", protoio.LLVMPassIndexSensorName)
	}

	actuator, ok := typed.RegisteredActuator(protoio.LLVMPhaseActuatorName)
	if !ok {
		return nil, nil, nil, fmt.Errorf("agent %s missing actuator %s", agent.ID(), protoio.LLVMPhaseActuatorName)
	}
	output, ok := actuator.(protoio.SnapshotActuator)
	if !ok {
		return nil, nil, nil, fmt.Errorf("actuator %s does not support output snapshot", protoio.LLVMPhaseActuatorName)
	}

	return complexitySetter, passSetter, output, nil
}

func argmax(values []float64) int {
	if len(values) == 0 {
		return 0
	}
	maxIndex := 0
	maxVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
			maxIndex = i
		}
	}
	return maxIndex
}

func maxIntLLVM(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minIntLLVM(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clampIntLLVM(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampLLVM(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
