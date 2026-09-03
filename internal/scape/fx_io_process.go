package scape

import (
	"context"
	"fmt"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

type fxProcessIOState struct {
	mu      sync.Mutex
	process *FXProcess
	cache   []float64
}

type fxProcessIOSensor struct {
	name  string
	index int
	state *fxProcessIOState
}

func (s *fxProcessIOSensor) Name() string {
	return protoio.ScalarInputSensorName
}

func (s *fxProcessIOSensor) Read(ctx context.Context) ([]float64, error) {
	return s.ReadForSensorProcess(ctx, protoio.SensorProcessCall{})
}

func (s *fxProcessIOSensor) Set(float64) {}

func (s *fxProcessIOSensor) ReadForSensorProcess(ctx context.Context, call protoio.SensorProcessCall) ([]float64, error) {
	if s == nil || s.state == nil || s.state.process == nil {
		return nil, fmt.Errorf("fx process sensor is not initialized")
	}
	s.state.mu.Lock()
	defer s.state.mu.Unlock()

	percept, err := s.state.cachedPercept(ctx)
	if err != nil {
		return nil, err
	}
	if s.index < 0 || s.index >= len(percept) {
		return nil, fmt.Errorf("fx percept index out of range: sensor=%s index=%d width=%d", s.name, s.index, len(percept))
	}
	return []float64{percept[s.index]}, nil
}

func (s *fxProcessIOState) cachedPercept(ctx context.Context) ([]float64, error) {
	if s.cache != nil {
		return append([]float64(nil), s.cache...), nil
	}
	response := s.process.Call(ctx, FXSenseMessage{})
	if response.Err != nil {
		return nil, response.Err
	}
	if !response.OK {
		return nil, fmt.Errorf("fx sense failed")
	}
	s.cache = append([]float64(nil), response.Percept...)
	return append([]float64(nil), s.cache...), nil
}

type fxProcessIOActuator struct {
	state  *fxProcessIOState
	opMode string
	last   []float64
}

func (a *fxProcessIOActuator) Name() string {
	return protoio.ScalarOutputActuatorName
}

func (a *fxProcessIOActuator) Write(ctx context.Context, values []float64) error {
	_, err := a.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{Output: values})
	return err
}

func (a *fxProcessIOActuator) Last() []float64 {
	a.state.mu.Lock()
	defer a.state.mu.Unlock()
	return append([]float64(nil), a.last...)
}

func (a *fxProcessIOActuator) WriteForActuatorProcess(ctx context.Context, call protoio.ActuatorProcessCall) (protoio.ActuatorSyncMessage, error) {
	if a == nil || a.state == nil || a.state.process == nil {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("fx process actuator is not initialized")
	}
	a.state.mu.Lock()
	defer a.state.mu.Unlock()

	output := append([]float64(nil), call.Output...)
	a.last = append([]float64(nil), output...)
	action := 0.0
	if len(output) > 0 {
		action = output[0]
	}
	response := a.state.process.Call(ctx, FXTradeMessage{Action: action})
	a.state.cache = nil
	if response.Err != nil {
		return protoio.ActuatorSyncMessage{}, response.Err
	}
	if !response.OK {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("fx trade failed")
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

func NewFXProcessIO(mode string, sensorIDs, actuatorIDs []string) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	process := NewFXProcess()
	start := process.Call(context.Background(), FXStartMessage{Mode: mode})
	if start.Err != nil {
		return nil, nil, start.Err
	}
	if !start.OK {
		return nil, nil, fmt.Errorf("fx process did not start")
	}
	state := &fxProcessIOState{process: process}
	sensors := make(map[string]protoio.Sensor, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		index, err := fxProcessSensorIndex(sensorID)
		if err != nil {
			return nil, nil, err
		}
		sensors[sensorID] = &fxProcessIOSensor{name: sensorID, index: index, state: state}
	}
	actuators := make(map[string]protoio.Actuator, len(actuatorIDs))
	for _, actuatorID := range actuatorIDs {
		if protoio.CanonicalActuatorName(actuatorID) != protoio.FXTradeActuatorName {
			return nil, nil, fmt.Errorf("unsupported fx process actuator: %s", actuatorID)
		}
		actuators[actuatorID] = &fxProcessIOActuator{state: state, opMode: mode}
	}
	return sensors, actuators, nil
}

func fxProcessSensorIndex(sensorID string) (int, error) {
	switch protoio.CanonicalSensorName(sensorID) {
	case protoio.FXPriceSensorName:
		return 0, nil
	case protoio.FXSignalSensorName:
		return 1, nil
	case protoio.FXMomentumSensorName:
		return 2, nil
	case protoio.FXVolatilitySensorName:
		return 4, nil
	case protoio.FXDrawdownSensorName:
		return 6, nil
	case protoio.FXNAVSensorName:
		return 7, nil
	case protoio.FXPositionSensorName:
		return 8, nil
	case protoio.FXEntrySensorName:
		return 10, nil
	case protoio.FXPercentChangeSensorName:
		return 11, nil
	case protoio.FXPrevPercentChangeSensorName:
		return 12, nil
	case protoio.FXProfitSensorName:
		return 13, nil
	default:
		return 0, fmt.Errorf("unsupported fx process sensor: %s", sensorID)
	}
}

func UsesReferenceFXIO(scapeName, ioExecution string) bool {
	switch strings.TrimSpace(strings.ToLower(ioExecution)) {
	case "process", "processes", "actor", "actors":
		switch strings.TrimSpace(strings.ToLower(scapeName)) {
		case "fx", "fx_sim", "scape_fx_sim", "forex_trader":
			return true
		default:
			return false
		}
	default:
		return false
	}
}
