package scape

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	protoio "protogonos/internal/io"
)

var flatlandProcessIOCounter uint64

type flatlandProcessIOState struct {
	mu      sync.Mutex
	process *FlatlandPublicProcess
	agentID string
	opMode  string
	cache   []float64
	last    []float64
}

type flatlandProcessIOSensor struct {
	name   string
	index  int
	family string
	state  *flatlandProcessIOState
}

func (s *flatlandProcessIOSensor) Name() string {
	return protoio.ScalarInputSensorName
}

func (s *flatlandProcessIOSensor) Read(ctx context.Context) ([]float64, error) {
	return s.ReadForSensorProcess(ctx, protoio.SensorProcessCall{})
}

func (s *flatlandProcessIOSensor) Set(float64) {}

func (s *flatlandProcessIOSensor) ReadForSensorProcess(ctx context.Context, call protoio.SensorProcessCall) ([]float64, error) {
	if s == nil || s.state == nil || s.state.process == nil {
		return nil, fmt.Errorf("flatland process sensor is not initialized")
	}
	s.state.mu.Lock()
	defer s.state.mu.Unlock()

	percept, err := s.state.cachedPercept(ctx)
	if err != nil {
		return nil, err
	}
	if s.family != "" {
		return flatlandProcessScannerFamily(percept, s.family)
	}
	if s.index < 0 || s.index >= len(percept) {
		return nil, fmt.Errorf("flatland percept index out of range: sensor=%s index=%d width=%d", s.name, s.index, len(percept))
	}
	return []float64{percept[s.index]}, nil
}

func (s *flatlandProcessIOState) cachedPercept(ctx context.Context) ([]float64, error) {
	if s.cache != nil {
		return append([]float64(nil), s.cache...), nil
	}
	response := s.process.Call(ctx, FlatlandPublicSenseMessage{AgentID: s.agentID})
	if response.Err != nil {
		return nil, response.Err
	}
	if !response.OK {
		return nil, fmt.Errorf("flatland sense failed")
	}
	s.cache = append([]float64(nil), response.Percept...)
	return append([]float64(nil), s.cache...), nil
}

type flatlandProcessIOActuator struct {
	state *flatlandProcessIOState
	name  string
}

func (a *flatlandProcessIOActuator) Name() string {
	return protoio.ScalarOutputActuatorName
}

func (a *flatlandProcessIOActuator) Write(ctx context.Context, values []float64) error {
	_, err := a.WriteForActuatorProcess(ctx, protoio.ActuatorProcessCall{Output: values})
	return err
}

func (a *flatlandProcessIOActuator) Last() []float64 {
	a.state.mu.Lock()
	defer a.state.mu.Unlock()
	return append([]float64(nil), a.state.last...)
}

func (a *flatlandProcessIOActuator) WriteForActuatorProcess(ctx context.Context, call protoio.ActuatorProcessCall) (protoio.ActuatorSyncMessage, error) {
	if a == nil || a.state == nil || a.state.process == nil {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("flatland process actuator is not initialized")
	}
	a.state.mu.Lock()
	defer a.state.mu.Unlock()

	output := append([]float64(nil), call.Output...)
	a.state.last = append([]float64(nil), output...)
	response := a.state.process.Call(ctx, FlatlandPublicActMessage{AgentID: a.state.agentID, Output: output})
	a.state.cache = nil
	if response.Err != nil {
		return protoio.ActuatorSyncMessage{}, response.Err
	}
	if !response.OK {
		return protoio.ActuatorSyncMessage{}, fmt.Errorf("flatland act failed")
	}
	opMode := a.state.opMode
	if strings.TrimSpace(call.OpMode) != "" {
		opMode = call.OpMode
	}
	return protoio.ActuatorSyncMessage{
		Fitness: referenceFitnessVector(opMode, response.Fitness),
		EndFlag: boolEndFlag(response.End),
	}, nil
}

func NewFlatlandProcessIO(mode string, sensorIDs, actuatorIDs []string) (map[string]protoio.Sensor, map[string]protoio.Actuator, error) {
	process := NewFlatlandPublicProcess()
	start := process.Call(context.Background(), FlatlandPublicStartMessage{})
	if start.Err != nil {
		return nil, nil, start.Err
	}
	if !start.OK {
		return nil, nil, fmt.Errorf("flatland process did not start")
	}
	agentID := fmt.Sprintf("flatland-process-agent-%d", atomic.AddUint64(&flatlandProcessIOCounter, 1))
	opMode := strings.TrimSpace(mode)
	enter := process.Call(context.Background(), FlatlandPublicEnterMessage{Agent: FlatlandPublicAgent{ID: agentID, Mode: opMode}})
	if enter.Err != nil {
		return nil, nil, enter.Err
	}
	if !enter.OK {
		return nil, nil, fmt.Errorf("flatland process agent did not enter")
	}
	state := &flatlandProcessIOState{process: process, agentID: agentID, opMode: mode}

	sensors := make(map[string]protoio.Sensor, len(sensorIDs))
	for _, sensorID := range sensorIDs {
		index, family, err := flatlandProcessSensorIndex(sensorID)
		if err != nil {
			return nil, nil, err
		}
		sensors[sensorID] = &flatlandProcessIOSensor{name: sensorID, index: index, family: family, state: state}
	}
	actuators := make(map[string]protoio.Actuator, len(actuatorIDs))
	for _, actuatorID := range actuatorIDs {
		canonical := protoio.CanonicalActuatorName(actuatorID)
		if canonical != protoio.FlatlandMoveActuatorName && canonical != protoio.FlatlandTwoWheelsActuatorName {
			return nil, nil, fmt.Errorf("unsupported flatland process actuator: %s", actuatorID)
		}
		actuators[actuatorID] = &flatlandProcessIOActuator{state: state, name: actuatorID}
	}
	return sensors, actuators, nil
}

func flatlandProcessSensorIndex(sensorID string) (int, string, error) {
	switch strings.ToLower(strings.TrimSpace(sensorID)) {
	case strings.ToLower(protoio.DistanceScannerSensorAliasName):
		return 0, "distance", nil
	case strings.ToLower(protoio.ColorScannerSensorAliasName):
		return 0, "color", nil
	case strings.ToLower(protoio.EnergyScannerSensorAliasName), strings.ToLower(protoio.EnergyScannerSensorCorrectedName):
		return 0, "energy", nil
	}

	switch protoio.CanonicalSensorName(sensorID) {
	case protoio.FlatlandDistanceSensorName:
		return 0, "", nil
	case protoio.FlatlandEnergySensorName:
		return 1, "", nil
	case protoio.FlatlandPreySensorName:
		return 2, "", nil
	case protoio.FlatlandPredatorSensorName:
		return 3, "", nil
	case protoio.FlatlandPoisonSensorName:
		return 4, "", nil
	case protoio.FlatlandWallSensorName:
		return 5, "", nil
	case protoio.FlatlandFoodProximitySensorName:
		return 6, "", nil
	case protoio.FlatlandPreyProximitySensorName:
		return 7, "", nil
	case protoio.FlatlandPredatorProximitySensorName:
		return 8, "", nil
	case protoio.FlatlandPoisonProximitySensorName:
		return 9, "", nil
	case protoio.FlatlandWallProximitySensorName:
		return 10, "", nil
	case protoio.FlatlandResourceBalanceSensorName:
		return 11, "", nil
	default:
		if index, ok := flatlandProcessScannerBinIndex(sensorID, flatlandDistanceScannerSensors, flatlandBaseFeatureWidth); ok {
			return index, "", nil
		}
		if index, ok := flatlandProcessScannerBinIndex(sensorID, flatlandColorScannerSensors, flatlandBaseFeatureWidth+flatlandScannerDensity); ok {
			return index, "", nil
		}
		if index, ok := flatlandProcessScannerBinIndex(sensorID, flatlandEnergyScannerSensors, flatlandBaseFeatureWidth+2*flatlandScannerDensity); ok {
			return index, "", nil
		}
		return 0, "", fmt.Errorf("unsupported flatland process sensor: %s", sensorID)
	}
}

func flatlandProcessScannerBinIndex(sensorID string, sensors [flatlandScannerDensity]string, offset int) (int, bool) {
	canonical := protoio.CanonicalSensorName(sensorID)
	for i, name := range sensors {
		if canonical == name {
			return offset + i, true
		}
	}
	return 0, false
}

func flatlandProcessScannerFamily(percept []float64, family string) ([]float64, error) {
	offset := flatlandBaseFeatureWidth
	switch family {
	case "distance":
		offset = flatlandBaseFeatureWidth
	case "color":
		offset = flatlandBaseFeatureWidth + flatlandScannerDensity
	case "energy":
		offset = flatlandBaseFeatureWidth + 2*flatlandScannerDensity
	default:
		return nil, fmt.Errorf("unsupported flatland scanner family: %s", family)
	}
	end := offset + flatlandScannerDensity
	if offset < 0 || end > len(percept) {
		return nil, fmt.Errorf("flatland scanner family %s out of range: offset=%d width=%d", family, offset, len(percept))
	}
	return append([]float64(nil), percept[offset:end]...), nil
}

func UsesReferenceFlatlandIO(scapeName, ioExecution string) bool {
	switch strings.TrimSpace(strings.ToLower(ioExecution)) {
	case "process", "processes", "actor", "actors":
		switch strings.TrimSpace(strings.ToLower(scapeName)) {
		case "flatland", "scape_flatland", "prey", "predator":
			return true
		default:
			return false
		}
	default:
		return false
	}
}
