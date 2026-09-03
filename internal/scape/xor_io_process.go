package scape

import (
	"context"
	"fmt"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

type xorProcessIOSensor struct {
	name    string
	index   int
	process *XORProcess
	mu      *sync.Mutex
}

func (s *xorProcessIOSensor) Name() string {
	return protoio.ScalarInputSensorName
}

func (s *xorProcessIOSensor) Read(ctx context.Context) ([]float64, error) {
	return s.ReadForSensorProcess(ctx, protoio.SensorProcessCall{})
}

func (s *xorProcessIOSensor) Set(float64) {}

func (s *xorProcessIOSensor) ReadForSensorProcess(ctx context.Context, call protoio.SensorProcessCall) ([]float64, error) {
	if s == nil || s.process == nil || s.mu == nil {
		return nil, fmt.Errorf("xor process sensor is not initialized")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	response := s.process.Call(ctx, XORSenseMessage{})
	if response.Err != nil {
		return nil, response.Err
	}
	if !response.OK {
		return nil, fmt.Errorf("xor sense failed")
	}
	if s.index < 0 || s.index >= len(response.Percept) {
		return nil, fmt.Errorf("xor percept index out of range: index=%d width=%d", s.index, len(response.Percept))
	}
	return []float64{response.Percept[s.index]}, nil
}

type xorProcessIOActuator struct {
	process *XORProcess
	mu      *sync.Mutex
	last    []float64
}

func (a *xorProcessIOActuator) Name() string {
	return protoio.ScalarOutputActuatorName
}

func (a *xorProcessIOActuator) Write(ctx context.Context, values []float64) error {
	_, err := a.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{Output: values})
	return err
}

func (a *xorProcessIOActuator) Last() []float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]float64(nil), a.last...)
}

func (a *xorProcessIOActuator) WriteForActuatorProcess(ctx context.Context, call protoio.ActuatorProcessCall) (protoio.ActuatorSyncMessage, error) {
	if a == nil || a.process == nil || a.mu == nil {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("xor process actuator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	output := append([]float64(nil), call.Output...)
	a.last = append([]float64(nil), output...)
	response := a.process.Call(ctx, XORPredictMessage{Output: output})
	if response.Err != nil {
		return protoio.ActuatorSyncMessage{}, response.Err
	}
	if !response.OK {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("xor action failed")
	}
	return protoio.ActuatorSyncMessage{
		Fitness: []float64{float64(response.Fitness)},
		EndFlag: boolEndFlag(response.End),
	}, nil
}

func NewXORProcessIO(mode string, sensorIDs, actuatorIDs []string) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	process := NewXORProcess()
	start := process.Call(context.Background(), XORStartMessage{Mode: mode})
	if start.Err != nil {
		return nil, nil, start.Err
	}
	if !start.OK {
		return nil, nil, fmt.Errorf("xor process did not start")
	}
	mu := &sync.Mutex{}
	sensors := make(map[string]protoio.Sensor, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		switch protoio.CanonicalSensorName(sensorID) {
		case protoio.XORInputLeftSensorName:
			sensors[sensorID] = &xorProcessIOSensor{name: sensorID, index: 0, process: process, mu: mu}
		case protoio.XORInputRightSensorName:
			sensors[sensorID] = &xorProcessIOSensor{name: sensorID, index: 1, process: process, mu: mu}
		default:
			return nil, nil, fmt.Errorf("unsupported xor process sensor: %s", sensorID)
		}
	}
	actuators := make(map[string]protoio.Actuator, len(actuatorIDs))
	for _, actuatorID := range actuatorIDs {
		if protoio.CanonicalActuatorName(actuatorID) != protoio.XOROutputActuatorName {
			return nil, nil, fmt.Errorf("unsupported xor process actuator: %s", actuatorID)
		}
		actuators[actuatorID] = &xorProcessIOActuator{process: process, mu: mu}
	}
	return sensors, actuators, nil
}

func boolEndFlag(v bool) int {
	if v {
		return 1
	}
	return 0
}

func UsesReferenceXORIO(scapeName, ioExecution string) bool {
	switch strings.TrimSpace(strings.ToLower(ioExecution)) {
	case "process", "processes", "actor", "actors":
		return strings.TrimSpace(strings.ToLower(scapeName)) == "xor"
	default:
		return false
	}
}
