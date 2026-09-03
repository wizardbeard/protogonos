package scape

import (
	"context"
	"fmt"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

type pole2ProcessIOSensor struct {
	name    string
	index   int
	process *Pole2Process
	mu      *sync.Mutex
}

func (s *pole2ProcessIOSensor) Name() string {
	return protoio.ScalarInputSensorName
}

func (s *pole2ProcessIOSensor) Read(ctx context.Context) ([]float64, error) {
	return s.ReadForSensorProcess(ctx, protoio.SensorProcessCall{})
}

func (s *pole2ProcessIOSensor) Set(float64) {}

func (s *pole2ProcessIOSensor) ReadForSensorProcess(ctx context.Context, call protoio.SensorProcessCall) ([]float64, error) {
	if s == nil || s.process == nil || s.mu == nil {
		return nil, fmt.Errorf("pole2 process sensor is not initialized")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	response := s.process.Call(ctx, Pole2SenseMessage{Parameter: "6"})
	if response.Err != nil {
		return nil, response.Err
	}
	if !response.OK {
		return nil, fmt.Errorf("pole2 sense failed")
	}
	percept := append([]float64(nil), response.Percept...)
	percept = append(percept, response.State.RunProgress, response.State.StepProgress, response.State.LastStepFitness)
	if s.index < 0 || s.index >= len(percept) {
		return nil, fmt.Errorf("pole2 percept index out of range: sensor=%s index=%d width=%d", s.name, s.index, len(percept))
	}
	return []float64{percept[s.index]}, nil
}

type pole2ProcessIOActuator struct {
	process *Pole2Process
	mu      *sync.Mutex
	opMode  string
	last    []float64
}

func (a *pole2ProcessIOActuator) Name() string {
	return protoio.ScalarOutputActuatorName
}

func (a *pole2ProcessIOActuator) Write(ctx context.Context, values []float64) error {
	_, err := a.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{Output: values})
	return err
}

func (a *pole2ProcessIOActuator) Last() []float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]float64(nil), a.last...)
}

func (a *pole2ProcessIOActuator) WriteForActuatorProcess(ctx context.Context, call protoio.ActuatorProcessCall) (protoio.ActuatorSyncMessage, error) {
	if a == nil || a.process == nil || a.mu == nil {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("pole2 process actuator is not initialized")
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	output := append([]float64(nil), call.Output...)
	a.last = append([]float64(nil), output...)
	response := a.process.Call(ctx, Pole2PushMessage{Output: output})
	if response.Err != nil {
		return protoio.ActuatorSyncMessage{}, response.Err
	}
	if !response.OK {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("pole2 push failed")
	}
	opMode := a.opMode
	if strings.TrimSpace(call.OpMode) != "" {
		opMode = call.OpMode
	}
	return protoio.ActuatorSyncMessage{
		Fitness:     referenceFitnessVector(opMode, response.Fitness),
		EndFlag:     boolEndFlag(response.End),
		GoalReached: response.State.GoalReached,
	}, nil
}

func NewPole2ProcessIO(mode string, sensorIDs, actuatorIDs []string) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	process := NewPole2Process()
	start := process.Call(context.Background(), Pole2StartMessage{Mode: mode})
	if start.Err != nil {
		return nil, nil, start.Err
	}
	if !start.OK {
		return nil, nil, fmt.Errorf("pole2 process did not start")
	}
	mu := &sync.Mutex{}
	sensors := make(map[string]protoio.Sensor, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		index, err := pole2ProcessSensorIndex(sensorID)
		if err != nil {
			return nil, nil, err
		}
		sensors[sensorID] = &pole2ProcessIOSensor{name: sensorID, index: index, process: process, mu: mu}
	}
	actuators := make(map[string]protoio.Actuator, len(actuatorIDs))
	for _, actuatorID := range actuatorIDs {
		if protoio.CanonicalActuatorName(actuatorID) != protoio.Pole2PushActuatorName {
			return nil, nil, fmt.Errorf("unsupported pole2 process actuator: %s", actuatorID)
		}
		actuators[actuatorID] = &pole2ProcessIOActuator{process: process, mu: mu, opMode: mode}
	}
	return sensors, actuators, nil
}

func pole2ProcessSensorIndex(sensorID string) (int, error) {
	switch protoio.CanonicalSensorName(sensorID) {
	case protoio.Pole2CartPositionSensorName:
		return 0, nil
	case protoio.Pole2CartVelocitySensorName:
		return 1, nil
	case protoio.Pole2Angle1SensorName:
		return 2, nil
	case protoio.Pole2Velocity1SensorName:
		return 3, nil
	case protoio.Pole2Angle2SensorName:
		return 4, nil
	case protoio.Pole2Velocity2SensorName:
		return 5, nil
	case protoio.Pole2RunProgressSensorName:
		return 6, nil
	case protoio.Pole2StepProgressSensorName:
		return 7, nil
	case protoio.Pole2FitnessSignalSensorName:
		return 8, nil
	default:
		return 0, fmt.Errorf("unsupported pole2 process sensor: %s", sensorID)
	}
}

func UsesReferencePole2IO(scapeName, ioExecution string) bool {
	switch strings.TrimSpace(strings.ToLower(ioExecution)) {
	case "process", "processes", "actor", "actors":
		switch strings.TrimSpace(strings.ToLower(scapeName)) {
		case "pole2-balancing", "pb_sim", "pole_balancing":
			return true
		default:
			return false
		}
	default:
		return false
	}
}
