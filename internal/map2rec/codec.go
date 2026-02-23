package map2rec

import (
	"encoding/json"
	"errors"
	"fmt"
)

const (
	SupportedSchemaVersion = 1
	SupportedCodecVersion  = 1
)

var ErrRecordVersionMismatch = errors.New("record version mismatch")

type RecordEnvelope struct {
	SchemaVersion int             `json:"schema_version"`
	CodecVersion  int             `json:"codec_version"`
	Kind          string          `json:"kind"`
	Payload       json.RawMessage `json:"payload"`
}

func DefaultRecord(kind string) (any, error) {
	switch kind {
	case "constraint":
		return defaultConstraintRecord(), nil
	case "pmp":
		return defaultPMPRecord(), nil
	case "experiment":
		return defaultExperimentRecord(), nil
	case "circuit":
		return defaultCircuitRecord(), nil
	case "layer":
		return defaultLayerRecord(), nil
	case "layer2":
		return defaultLayer2Record(), nil
	case "layer_spec":
		return defaultLayerSpecRecord(), nil
	case "neurode":
		return defaultNeurodeRecord(), nil
	case "sensor":
		return defaultSensorRecord(), nil
	case "actuator":
		return defaultActuatorRecord(), nil
	case "neuron":
		return defaultNeuronRecord(), nil
	case "agent":
		return defaultAgentRecord(), nil
	case "cortex":
		return defaultCortexRecord(), nil
	case "substrate":
		return defaultSubstrateRecord(), nil
	case "polis":
		return defaultPolisRecord(), nil
	case "scape":
		return defaultScapeRecord(), nil
	case "sector":
		return defaultSectorRecord(), nil
	case "avatar":
		return defaultAvatarRecord(), nil
	case "object":
		return defaultObjectRecord(), nil
	case "circle":
		return defaultCircleRecord(), nil
	case "square":
		return defaultSquareRecord(), nil
	case "line":
		return defaultLineRecord(), nil
	case "e":
		return defaultERecord(), nil
	case "a":
		return defaultARecord(), nil
	case "specie":
		return defaultSpecieRecord(), nil
	case "population":
		return defaultPopulationRecord(), nil
	case "trace":
		return defaultTraceRecord(), nil
	case "stat":
		return defaultStatRecord(), nil
	case "topology_summary":
		return defaultTopologySummaryRecord(), nil
	case "signature":
		return defaultSignatureRecord(), nil
	case "champion":
		return defaultChampionRecord(), nil
	default:
		return nil, ErrUnsupportedKind
	}
}

func EncodeRecord(kind string, record any) ([]byte, error) {
	if _, err := DefaultRecord(kind); err != nil {
		return nil, err
	}

	payload, err := json.Marshal(record)
	if err != nil {
		return nil, fmt.Errorf("marshal %s payload: %w", kind, err)
	}
	env := RecordEnvelope{
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Kind:          kind,
		Payload:       payload,
	}
	data, err := json.Marshal(env)
	if err != nil {
		return nil, fmt.Errorf("marshal %s envelope: %w", kind, err)
	}
	return data, nil
}

func DecodeRecord(data []byte) (string, any, error) {
	var env RecordEnvelope
	if err := json.Unmarshal(data, &env); err != nil {
		return "", nil, err
	}
	if env.SchemaVersion != SupportedSchemaVersion || env.CodecVersion != SupportedCodecVersion {
		return "", nil, fmt.Errorf("%w: schema=%d codec=%d", ErrRecordVersionMismatch, env.SchemaVersion, env.CodecVersion)
	}

	record, err := decodeRecordPayload(env.Kind, env.Payload)
	if err != nil {
		return "", nil, err
	}
	return env.Kind, record, nil
}

func decodeRecordPayload(kind string, payload []byte) (any, error) {
	switch kind {
	case "constraint":
		var rec ConstraintRecord
		return rec, json.Unmarshal(payload, &rec)
	case "pmp":
		var rec PMPRecord
		return rec, json.Unmarshal(payload, &rec)
	case "experiment":
		var rec ExperimentRecord
		return rec, json.Unmarshal(payload, &rec)
	case "circuit":
		var rec CircuitRecord
		return rec, json.Unmarshal(payload, &rec)
	case "layer":
		var rec LayerRecord
		return rec, json.Unmarshal(payload, &rec)
	case "layer2":
		var rec Layer2Record
		return rec, json.Unmarshal(payload, &rec)
	case "layer_spec":
		var rec LayerSpecRecord
		return rec, json.Unmarshal(payload, &rec)
	case "neurode":
		var rec NeurodeRecord
		return rec, json.Unmarshal(payload, &rec)
	case "sensor":
		var rec SensorRecord
		return rec, json.Unmarshal(payload, &rec)
	case "actuator":
		var rec ActuatorRecord
		return rec, json.Unmarshal(payload, &rec)
	case "neuron":
		var rec NeuronRecord
		return rec, json.Unmarshal(payload, &rec)
	case "agent":
		var rec AgentRecord
		return rec, json.Unmarshal(payload, &rec)
	case "cortex":
		var rec CortexRecord
		return rec, json.Unmarshal(payload, &rec)
	case "substrate":
		var rec SubstrateRecord
		return rec, json.Unmarshal(payload, &rec)
	case "polis":
		var rec PolisRecord
		return rec, json.Unmarshal(payload, &rec)
	case "scape":
		var rec ScapeRecord
		return rec, json.Unmarshal(payload, &rec)
	case "sector":
		var rec SectorRecord
		return rec, json.Unmarshal(payload, &rec)
	case "avatar":
		var rec AvatarRecord
		return rec, json.Unmarshal(payload, &rec)
	case "object":
		var rec ObjectRecord
		return rec, json.Unmarshal(payload, &rec)
	case "circle":
		var rec CircleRecord
		return rec, json.Unmarshal(payload, &rec)
	case "square":
		var rec SquareRecord
		return rec, json.Unmarshal(payload, &rec)
	case "line":
		var rec LineRecord
		return rec, json.Unmarshal(payload, &rec)
	case "e":
		var rec ERecord
		return rec, json.Unmarshal(payload, &rec)
	case "a":
		var rec ARecord
		return rec, json.Unmarshal(payload, &rec)
	case "specie":
		var rec SpecieRecord
		return rec, json.Unmarshal(payload, &rec)
	case "population":
		var rec PopulationRecord
		return rec, json.Unmarshal(payload, &rec)
	case "trace":
		var rec TraceRecord
		return rec, json.Unmarshal(payload, &rec)
	case "stat":
		var rec StatRecord
		return rec, json.Unmarshal(payload, &rec)
	case "topology_summary":
		var rec TopologySummaryRecord
		return rec, json.Unmarshal(payload, &rec)
	case "signature":
		var rec SignatureRecord
		return rec, json.Unmarshal(payload, &rec)
	case "champion":
		var rec ChampionRecord
		return rec, json.Unmarshal(payload, &rec)
	default:
		return nil, ErrUnsupportedKind
	}
}
