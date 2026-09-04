package scape

import (
	"context"
	"fmt"
	"strings"
	"sync"

	protoio "protogonos/internal/io"
)

type epitopesProcessIOState struct {
	mu      sync.Mutex
	process *EpitopesProcess
	cache   *epitopesProcessPercept
}

type epitopesProcessPercept struct {
	signal   float64
	memory   float64
	target   float64
	progress float64
	margin   float64
}

type epitopesProcessIOSensor struct {
	name  string
	index int
	state *epitopesProcessIOState
}

func (s *epitopesProcessIOSensor) Name() string {
	return protoio.ScalarInputSensorName
}

func (s *epitopesProcessIOSensor) Read(ctx context.Context) ([]float64, error) {
	return s.ReadForSensorProcess(ctx, protoio.SensorProcessCall{})
}

func (s *epitopesProcessIOSensor) Set(float64) {}

func (s *epitopesProcessIOSensor) ReadForSensorProcess(ctx context.Context, call protoio.SensorProcessCall) ([]float64, error) {
	if s == nil || s.state == nil || s.state.process == nil {
		return nil, fmt.Errorf("epitopes process sensor is not initialized")
	}
	s.state.mu.Lock()
	defer s.state.mu.Unlock()

	percept, err := s.state.cachedPercept(ctx)
	if err != nil {
		return nil, err
	}
	switch s.index {
	case 0:
		return []float64{percept.signal}, nil
	case 1:
		return []float64{percept.memory}, nil
	case 2:
		return []float64{percept.target}, nil
	case 3:
		return []float64{percept.progress}, nil
	case 4:
		return []float64{percept.margin}, nil
	default:
		return nil, fmt.Errorf("epitopes percept index out of range: sensor=%s index=%d", s.name, s.index)
	}
}

func (s *epitopesProcessIOState) cachedPercept(ctx context.Context) (epitopesProcessPercept, error) {
	if s.cache != nil {
		return *s.cache, nil
	}
	response := s.process.Call(ctx, EpitopesSenseMessage{})
	if response.Err != nil {
		return epitopesProcessPercept{}, response.Err
	}
	if !response.OK {
		return epitopesProcessPercept{}, fmt.Errorf("epitopes sense failed")
	}
	if len(response.Percept) < 2 {
		return epitopesProcessPercept{}, fmt.Errorf("epitopes percept too short: %d", len(response.Percept))
	}
	signal := response.Percept[0]
	memory := response.Percept[1]
	margin := signal + 0.7*memory
	target := -1.0
	if margin >= 0 {
		target = 1
	}
	progress := 0.0
	if response.State.IndexCurrent > 0 {
		progress = epitopesSessionProgress(response.State.IndexCurrent, response.State.StartIndex, response.State.EndIndex)
	}
	percept := epitopesProcessPercept{
		signal:   signal,
		memory:   memory,
		target:   target,
		progress: progress,
		margin:   margin,
	}
	s.cache = &percept
	return percept, nil
}

type epitopesProcessIOActuator struct {
	state  *epitopesProcessIOState
	opMode string
	last   []float64
}

func (a *epitopesProcessIOActuator) Name() string {
	return protoio.ScalarOutputActuatorName
}

func (a *epitopesProcessIOActuator) Write(ctx context.Context, values []float64) error {
	_, err := a.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{Output: values})
	return err
}

func (a *epitopesProcessIOActuator) Last() []float64 {
	a.state.mu.Lock()
	defer a.state.mu.Unlock()
	return append([]float64(nil), a.last...)
}

func (a *epitopesProcessIOActuator) WriteForActuatorProcess(ctx context.Context, call protoio.ActuatorProcessCall) (protoio.ActuatorSyncMessage, error) {
	if a == nil || a.state == nil || a.state.process == nil {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("epitopes process actuator is not initialized")
	}
	a.state.mu.Lock()
	defer a.state.mu.Unlock()

	output := append([]float64(nil), call.Output...)
	a.last = append([]float64(nil), output...)
	response := a.state.process.Call(ctx, EpitopesClassifyMessage{Output: output})
	a.state.cache = nil
	if response.Err != nil {
		return protoio.ActuatorSyncMessage{}, response.Err
	}
	if !response.OK {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("epitopes classify failed")
	}
	opMode := a.opMode
	if strings.TrimSpace(call.OpMode) != "" {
		opMode = call.OpMode
	}
	return protoio.ActuatorSyncMessage{
		Fitness: referenceFitnessVector(opMode, Fitness(response.Reward)),
		EndFlag: boolEndFlag(response.End),
	}, nil
}

func NewEpitopesProcessIO(mode string, sensorIDs, actuatorIDs []string) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	process := NewEpitopesProcess()
	opMode, params := epitopesProcessStartConfig(mode)
	start := process.Call(context.Background(), EpitopesStartMessage{OpMode: opMode, Params: params})
	if start.Err != nil {
		return nil, nil, start.Err
	}
	if !start.OK {
		return nil, nil, fmt.Errorf("epitopes process did not start")
	}
	state := &epitopesProcessIOState{process: process}
	sensors := make(map[string]protoio.Sensor, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		index, err := epitopesProcessSensorIndex(sensorID)
		if err != nil {
			return nil, nil, err
		}
		sensors[sensorID] = &epitopesProcessIOSensor{name: sensorID, index: index, state: state}
	}
	actuators := make(map[string]protoio.Actuator, len(actuatorIDs))
	for _, actuatorID := range actuatorIDs {
		if protoio.CanonicalActuatorName(actuatorID) != protoio.EpitopesResponseActuatorName {
			return nil, nil, fmt.Errorf("unsupported epitopes process actuator: %s", actuatorID)
		}
		actuators[actuatorID] = &epitopesProcessIOActuator{state: state, opMode: mode}
	}
	return sensors, actuators, nil
}

func epitopesProcessSensorIndex(sensorID string) (int, error) {
	switch protoio.CanonicalSensorName(sensorID) {
	case protoio.EpitopesSignalSensorName:
		return 0, nil
	case protoio.EpitopesMemorySensorName:
		return 1, nil
	case protoio.EpitopesTargetSensorName:
		return 2, nil
	case protoio.EpitopesProgressSensorName:
		return 3, nil
	case protoio.EpitopesMarginSensorName:
		return 4, nil
	default:
		return 0, fmt.Errorf("unsupported epitopes process sensor: %s", sensorID)
	}
}

func epitopesProcessMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "gt":
		return "gt"
	case "benchmark", "validation", "test":
		return "benchmark"
	default:
		return mode
	}
}

func epitopesProcessStartConfig(mode string) (string, EpitopesSimParameters) {
	normalized := strings.ToLower(strings.TrimSpace(mode))
	if normalized != "validation" && normalized != "test" {
		return epitopesProcessMode(mode), EpitopesSimParameters{}
	}

	source := currentEpitopesSource(context.Background())
	params := EpitopesSimParameters{}
	switch normalized {
	case "validation":
		params.StartBenchmarkIndex = source.windows.validationStart
		params.EndBenchmarkIndex = source.windows.validationEnd
	case "test":
		params.StartBenchmarkIndex = source.windows.testStart
		params.EndBenchmarkIndex = source.windows.testEnd
	}
	return "benchmark", params
}

func UsesReferenceEpitopesIO(scapeName, ioExecution string) bool {
	switch strings.TrimSpace(strings.ToLower(ioExecution)) {
	case "process", "processes", "actor", "actors":
		switch strings.TrimSpace(strings.ToLower(scapeName)) {
		case "epitopes", "epitopes_sim", "scape_epitopes_sim":
			return true
		default:
			return false
		}
	default:
		return false
	}
}
