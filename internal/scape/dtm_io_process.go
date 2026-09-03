package scape

import (
	"context"
	"fmt"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

type dtmProcessIOSensor struct {
	name    string
	index   int
	process *DTMProcess
	mu      *sync.Mutex
}

func (s *dtmProcessIOSensor) Name() string {
	return protoio.ScalarInputSensorName
}

func (s *dtmProcessIOSensor) Read(ctx context.Context) ([]float64, error) {
	return s.ReadForSensorProcess(ctx, protoio.SensorProcessCall{})
}

func (s *dtmProcessIOSensor) Set(float64) {}

func (s *dtmProcessIOSensor) ReadForSensorProcess(ctx context.Context, call protoio.SensorProcessCall) ([]float64, error) {
	if s == nil || s.process == nil || s.mu == nil {
		return nil, fmt.Errorf("dtm process sensor is not initialized")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	response := s.process.Call(ctx, DTMSenseMessage{Parameter: "all"})
	if response.Err != nil {
		return nil, response.Err
	}
	if !response.OK {
		return nil, fmt.Errorf("dtm sense failed")
	}
	percept := append([]float64(nil), response.Percept...)
	percept = append(percept, response.State.RunProgress, response.State.StepProgress, boolFloat(response.State.Switched))
	if s.index < 0 || s.index >= len(percept) {
		return nil, fmt.Errorf("dtm percept index out of range: sensor=%s index=%d width=%d", s.name, s.index, len(percept))
	}
	return []float64{percept[s.index]}, nil
}

type dtmProcessIOActuator struct {
	process *DTMProcess
	mu      *sync.Mutex
	opMode  string
	last    []float64
}

func (a *dtmProcessIOActuator) Name() string {
	return protoio.ScalarOutputActuatorName
}

func (a *dtmProcessIOActuator) Write(ctx context.Context, values []float64) error {
	_, err := a.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{Output: values})
	return err
}

func (a *dtmProcessIOActuator) Last() []float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]float64(nil), a.last...)
}

func (a *dtmProcessIOActuator) WriteForActuatorProcess(ctx context.Context, call protoio.ActuatorProcessCall) (protoio.ActuatorSyncMessage, error) {
	if a == nil || a.process == nil || a.mu == nil {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("dtm process actuator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	output := append([]float64(nil), call.Output...)
	a.last = append([]float64(nil), output...)
	response := a.process.Call(ctx, DTMMoveMessage{Output: output})
	if response.Err != nil {
		return protoio.ActuatorSyncMessage{}, response.Err
	}
	if !response.OK {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("dtm move failed")
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

func NewDTMProcessIO(mode string, sensorIDs, actuatorIDs []string) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	process := NewDTMProcess()
	start := process.Call(context.Background(), DTMStartMessage{Mode: mode})
	if start.Err != nil {
		return nil, nil, start.Err
	}
	if !start.OK {
		return nil, nil, fmt.Errorf("dtm process did not start")
	}
	mu := &sync.Mutex{}
	sensors := make(map[string]protoio.Sensor, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		index, err := dtmProcessSensorIndex(sensorID)
		if err != nil {
			return nil, nil, err
		}
		sensors[sensorID] = &dtmProcessIOSensor{name: sensorID, index: index, process: process, mu: mu}
	}
	actuators := make(map[string]protoio.Actuator, len(actuatorIDs))
	for _, actuatorID := range actuatorIDs {
		if protoio.CanonicalActuatorName(actuatorID) != protoio.DTMMoveActuatorName {
			return nil, nil, fmt.Errorf("unsupported dtm process actuator: %s", actuatorID)
		}
		actuators[actuatorID] = &dtmProcessIOActuator{process: process, mu: mu, opMode: mode}
	}
	return sensors, actuators, nil
}

func dtmProcessSensorIndex(sensorID string) (int, error) {
	switch protoio.CanonicalSensorName(sensorID) {
	case protoio.DTMRangeLeftSensorName:
		return 0, nil
	case protoio.DTMRangeFrontSensorName:
		return 1, nil
	case protoio.DTMRangeRightSensorName:
		return 2, nil
	case protoio.DTMRewardSensorName:
		return 3, nil
	case protoio.DTMRunProgressSensorName:
		return 4, nil
	case protoio.DTMStepProgressSensorName:
		return 5, nil
	case protoio.DTMSwitchedSensorName:
		return 6, nil
	default:
		return 0, fmt.Errorf("unsupported dtm process sensor: %s", sensorID)
	}
}

func UsesReferenceDTMIO(scapeName, ioExecution string) bool {
	switch strings.TrimSpace(strings.ToLower(ioExecution)) {
	case "process", "processes", "actor", "actors":
		return strings.TrimSpace(strings.ToLower(scapeName)) == "dtm"
	default:
		return false
	}
}

func referenceFitnessVector(opMode string, fitness Fitness) []float64 {
	if strings.TrimSpace(strings.ToLower(opMode)) == "test" {
		return []float64{float64(fitness), 0, 0}
	}
	return []float64{float64(fitness)}
}

func boolFloat(v bool) float64 {
	if v {
		return 1
	}
	return 0
}
