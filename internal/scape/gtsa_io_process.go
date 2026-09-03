package scape

import (
	"context"
	"fmt"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

type gtsaProcessIOState struct {
	mu      sync.Mutex
	process *GTSAProcess
	cache   *gtsaProcessPercept
}

type gtsaProcessPercept struct {
	current    float64
	delta      float64
	windowMean float64
	progress   float64
}

type gtsaProcessIOSensor struct {
	name  string
	index int
	state *gtsaProcessIOState
}

func (s *gtsaProcessIOSensor) Name() string {
	return protoio.ScalarInputSensorName
}

func (s *gtsaProcessIOSensor) Read(ctx context.Context) ([]float64, error) {
	return s.ReadForSensorProcess(ctx, protoio.SensorProcessCall{})
}

func (s *gtsaProcessIOSensor) Set(float64) {}

func (s *gtsaProcessIOSensor) ReadForSensorProcess(ctx context.Context, call protoio.SensorProcessCall) ([]float64, error) {
	if s == nil || s.state == nil || s.state.process == nil {
		return nil, fmt.Errorf("gtsa process sensor is not initialized")
	}
	s.state.mu.Lock()
	defer s.state.mu.Unlock()

	percept, err := s.state.cachedPercept(ctx)
	if err != nil {
		return nil, err
	}
	switch s.index {
	case 0:
		return []float64{percept.current}, nil
	case 1:
		return []float64{percept.delta}, nil
	case 2:
		return []float64{percept.windowMean}, nil
	case 3:
		return []float64{percept.progress}, nil
	default:
		return nil, fmt.Errorf("gtsa percept index out of range: sensor=%s index=%d", s.name, s.index)
	}
}

func (s *gtsaProcessIOState) cachedPercept(ctx context.Context) (gtsaProcessPercept, error) {
	if s.cache != nil {
		return *s.cache, nil
	}
	response := s.process.Call(ctx, GTSASensePerceptMessage{})
	if response.Err != nil {
		return gtsaProcessPercept{}, response.Err
	}
	if !response.OK {
		return gtsaProcessPercept{}, fmt.Errorf("gtsa sense failed")
	}
	percept := gtsaProcessPercept{
		current:    response.Current,
		delta:      response.Delta,
		windowMean: response.WindowMean,
		progress:   response.Progress,
	}
	s.cache = &percept
	return percept, nil
}

type gtsaProcessIOActuator struct {
	state  *gtsaProcessIOState
	opMode string
	last   []float64
}

func (a *gtsaProcessIOActuator) Name() string {
	return protoio.ScalarOutputActuatorName
}

func (a *gtsaProcessIOActuator) Write(ctx context.Context, values []float64) error {
	_, err := a.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{Output: values})
	return err
}

func (a *gtsaProcessIOActuator) Last() []float64 {
	a.state.mu.Lock()
	defer a.state.mu.Unlock()
	return append([]float64(nil), a.last...)
}

func (a *gtsaProcessIOActuator) WriteForActuatorProcess(ctx context.Context, call protoio.ActuatorProcessCall) (protoio.ActuatorSyncMessage, error) {
	if a == nil || a.state == nil || a.state.process == nil {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("gtsa process actuator is not initialized")
	}
	a.state.mu.Lock()
	defer a.state.mu.Unlock()

	output := append([]float64(nil), call.Output...)
	a.last = append([]float64(nil), output...)
	prediction := 0.0
	if len(output) > 0 {
		prediction = output[0]
	}
	response := a.state.process.Call(ctx, GTSAPredictValueMessage{Prediction: prediction})
	a.state.cache = nil
	if response.Err != nil {
		return protoio.ActuatorSyncMessage{}, response.Err
	}
	if !response.OK {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("gtsa predict failed")
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

func NewGTSAProcessIO(mode string, sensorIDs, actuatorIDs []string) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	process := NewGTSAProcess()
	start := process.Call(context.Background(), GTSAStartMessage{Mode: mode})
	if start.Err != nil {
		return nil, nil, start.Err
	}
	if !start.OK {
		return nil, nil, fmt.Errorf("gtsa process did not start")
	}
	state := &gtsaProcessIOState{process: process}
	sensors := make(map[string]protoio.Sensor, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		index, err := gtsaProcessSensorIndex(sensorID)
		if err != nil {
			return nil, nil, err
		}
		sensors[sensorID] = &gtsaProcessIOSensor{name: sensorID, index: index, state: state}
	}
	actuators := make(map[string]protoio.Actuator, len(actuatorIDs))
	for _, actuatorID := range actuatorIDs {
		if protoio.CanonicalActuatorName(actuatorID) != protoio.GTSAPredictActuatorName {
			return nil, nil, fmt.Errorf("unsupported gtsa process actuator: %s", actuatorID)
		}
		actuators[actuatorID] = &gtsaProcessIOActuator{state: state, opMode: mode}
	}
	return sensors, actuators, nil
}

func gtsaProcessSensorIndex(sensorID string) (int, error) {
	switch protoio.CanonicalSensorName(sensorID) {
	case protoio.GTSAInputSensorName:
		return 0, nil
	case protoio.GTSADeltaSensorName:
		return 1, nil
	case protoio.GTSAWindowMeanSensorName:
		return 2, nil
	case protoio.GTSAProgressSensorName:
		return 3, nil
	default:
		return 0, fmt.Errorf("unsupported gtsa process sensor: %s", sensorID)
	}
}

func UsesReferenceGTSAIO(scapeName, ioExecution string) bool {
	switch strings.TrimSpace(strings.ToLower(ioExecution)) {
	case "process", "processes", "actor", "actors":
		switch strings.TrimSpace(strings.ToLower(scapeName)) {
		case "gtsa", "gtsa_sim", "scape_gtsa", "general_predictor":
			return true
		default:
			return false
		}
	default:
		return false
	}
}
