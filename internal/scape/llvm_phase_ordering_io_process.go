package scape

import (
	"context"
	"fmt"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

type llvmPhaseOrderingProcessIOState struct {
	mu      sync.Mutex
	process *LLVMPhaseOrderingProcess
	cache   []float64
}

type llvmPhaseOrderingProcessIOSensor struct {
	name  string
	index int
	state *llvmPhaseOrderingProcessIOState
}

func (s *llvmPhaseOrderingProcessIOSensor) Name() string {
	return protoio.ScalarInputSensorName
}

func (s *llvmPhaseOrderingProcessIOSensor) Read(ctx context.Context) ([]float64, error) {
	return s.ReadForSensorProcess(ctx, protoio.SensorProcessCall{})
}

func (s *llvmPhaseOrderingProcessIOSensor) Set(float64) {}

func (s *llvmPhaseOrderingProcessIOSensor) ReadForSensorProcess(ctx context.Context, call protoio.SensorProcessCall) ([]float64, error) {
	if s == nil || s.state == nil || s.state.process == nil {
		return nil, fmt.Errorf("llvm-phase-ordering process sensor is not initialized")
	}
	s.state.mu.Lock()
	defer s.state.mu.Unlock()

	percept, err := s.state.cachedPercept(ctx)
	if err != nil {
		return nil, err
	}
	if s.index < 0 || s.index >= len(percept) {
		return nil, fmt.Errorf("llvm-phase-ordering percept index out of range: sensor=%s index=%d width=%d", s.name, s.index, len(percept))
	}
	return []float64{percept[s.index]}, nil
}

func (s *llvmPhaseOrderingProcessIOState) cachedPercept(ctx context.Context) ([]float64, error) {
	if s.cache != nil {
		return append([]float64(nil), s.cache...), nil
	}
	response := s.process.Call(ctx, LLVMPhaseOrderingSenseMessage{Parameter: "all"})
	if response.Err != nil {
		return nil, response.Err
	}
	if !response.OK {
		return nil, fmt.Errorf("llvm-phase-ordering sense failed")
	}
	s.cache = append([]float64(nil), response.Percept...)
	return append([]float64(nil), s.cache...), nil
}

type llvmPhaseOrderingProcessIOActuator struct {
	state  *llvmPhaseOrderingProcessIOState
	opMode string
	last   []float64
}

func (a *llvmPhaseOrderingProcessIOActuator) Name() string {
	return protoio.ScalarOutputActuatorName
}

func (a *llvmPhaseOrderingProcessIOActuator) Write(ctx context.Context, values []float64) error {
	_, err := a.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{Output: values})
	return err
}

func (a *llvmPhaseOrderingProcessIOActuator) Last() []float64 {
	a.state.mu.Lock()
	defer a.state.mu.Unlock()
	return append([]float64(nil), a.last...)
}

func (a *llvmPhaseOrderingProcessIOActuator) WriteForActuatorProcess(ctx context.Context, call protoio.ActuatorProcessCall) (protoio.ActuatorSyncMessage, error) {
	if a == nil || a.state == nil || a.state.process == nil {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("llvm-phase-ordering process actuator is not initialized")
	}
	a.state.mu.Lock()
	defer a.state.mu.Unlock()

	output := append([]float64(nil), call.Output...)
	a.last = append([]float64(nil), output...)
	response := a.state.process.Call(ctx, LLVMPhaseOrderingOptimizeMessage{Output: output})
	a.state.cache = nil
	if response.Err != nil {
		return protoio.ActuatorSyncMessage{}, response.Err
	}
	if !response.OK {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("llvm-phase-ordering optimize failed")
	}
	opMode := a.opMode
	if strings.TrimSpace(call.OpMode) != "" {
		opMode = call.OpMode
	}
	return protoio.ActuatorSyncMessage{
		Fitness: referenceFitnessVector(opMode, response.Fitness),
		EndFlag: boolEndFlag(response.End),
	}, nil
}

func NewLLVMPhaseOrderingProcessIO(mode string, sensorIDs, actuatorIDs []string) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	process := NewLLVMPhaseOrderingProcess()
	start := process.Call(context.Background(), LLVMPhaseOrderingStartMessage{Mode: mode})
	if start.Err != nil {
		return nil, nil, start.Err
	}
	if !start.OK {
		return nil, nil, fmt.Errorf("llvm-phase-ordering process did not start")
	}
	state := &llvmPhaseOrderingProcessIOState{process: process}
	sensors := make(map[string]protoio.Sensor, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		index, err := llvmPhaseOrderingProcessSensorIndex(sensorID)
		if err != nil {
			return nil, nil, err
		}
		sensors[sensorID] = &llvmPhaseOrderingProcessIOSensor{name: sensorID, index: index, state: state}
	}
	actuators := make(map[string]protoio.Actuator, len(actuatorIDs))
	for _, actuatorID := range actuatorIDs {
		if protoio.CanonicalActuatorName(actuatorID) != protoio.LLVMPhaseActuatorName {
			return nil, nil, fmt.Errorf("unsupported llvm-phase-ordering process actuator: %s", actuatorID)
		}
		actuators[actuatorID] = &llvmPhaseOrderingProcessIOActuator{state: state, opMode: mode}
	}
	return sensors, actuators, nil
}

func llvmPhaseOrderingProcessSensorIndex(sensorID string) (int, error) {
	switch protoio.CanonicalSensorName(sensorID) {
	case protoio.LLVMComplexitySensorName:
		return 0, nil
	case protoio.LLVMPassIndexSensorName:
		return 1, nil
	case protoio.LLVMAlignmentSensorName:
		return 2, nil
	case protoio.LLVMDiversitySensorName:
		return 3, nil
	case protoio.LLVMRuntimeGainSensorName:
		return 4, nil
	default:
		return 0, fmt.Errorf("unsupported llvm-phase-ordering process sensor: %s", sensorID)
	}
}

func UsesReferenceLLVMPhaseOrderingIO(scapeName, ioExecution string) bool {
	switch strings.TrimSpace(strings.ToLower(ioExecution)) {
	case "process", "processes", "actor", "actors":
		switch strings.TrimSpace(strings.ToLower(scapeName)) {
		case "llvm-phase-ordering", "llvm_phase_ordering", "llvm_phase_ordering_sim", "scape_llvmphaseordering", "scape_llvm_phase_ordering":
			return true
		default:
			return false
		}
	default:
		return false
	}
}
