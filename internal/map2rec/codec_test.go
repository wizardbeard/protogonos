package map2rec

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"testing"
)

func TestDefaultRecordSupportsAllRecordKinds(t *testing.T) {
	for _, kind := range allRecordKinds() {
		record, err := DefaultRecord(kind)
		if err != nil {
			t.Fatalf("default record for %s: %v", kind, err)
		}
		if record == nil {
			t.Fatalf("default record for %s is nil", kind)
		}
	}
}

func TestDefaultRecordUnsupportedKind(t *testing.T) {
	if _, err := DefaultRecord("unknown"); !errors.Is(err, ErrUnsupportedKind) {
		t.Fatalf("expected ErrUnsupportedKind, got %v", err)
	}
}

func TestEncodeDecodeRecordRoundTripAllKinds(t *testing.T) {
	for _, kind := range allRecordKinds() {
		record, err := DefaultRecord(kind)
		if err != nil {
			t.Fatalf("default record for %s: %v", kind, err)
		}
		if kind == "pmp" {
			// JSON does not support +/-Inf.
			pmp := record.(PMPRecord)
			pmp.FitnessGoal = 0
			record = pmp
		}

		data, err := EncodeRecord(kind, record)
		if err != nil {
			t.Fatalf("encode %s: %v", kind, err)
		}
		gotKind, gotRecord, err := DecodeRecord(data)
		if err != nil {
			t.Fatalf("decode %s: %v", kind, err)
		}
		if gotKind != kind {
			t.Fatalf("kind mismatch: got=%s want=%s", gotKind, kind)
		}
		gotJSON := canonicalJSON(t, gotRecord)
		wantJSON := canonicalJSON(t, record)
		if !bytes.Equal(gotJSON, wantJSON) {
			t.Fatalf("record mismatch for %s:\n got=%s\nwant=%s", kind, string(gotJSON), string(wantJSON))
		}
	}
}

func TestEncodeRecordUnsupportedKind(t *testing.T) {
	if _, err := EncodeRecord("unknown", struct{}{}); !errors.Is(err, ErrUnsupportedKind) {
		t.Fatalf("expected ErrUnsupportedKind, got %v", err)
	}
}

func TestEncodeRecordWithInfinitePMPFitnessGoalFails(t *testing.T) {
	record := defaultPMPRecord()
	if !math.IsInf(record.FitnessGoal, 1) {
		t.Fatalf("expected default fitness goal to be +Inf")
	}
	if _, err := EncodeRecord("pmp", record); err == nil {
		t.Fatal("expected encode failure for +Inf fitness goal")
	}
}

func TestDecodeRecordVersionMismatch(t *testing.T) {
	payload, err := json.Marshal(defaultTraceRecord())
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	data, err := json.Marshal(RecordEnvelope{
		SchemaVersion: SupportedSchemaVersion + 1,
		CodecVersion:  SupportedCodecVersion,
		Kind:          "trace",
		Payload:       payload,
	})
	if err != nil {
		t.Fatalf("marshal envelope: %v", err)
	}

	if _, _, err := DecodeRecord(data); !errors.Is(err, ErrRecordVersionMismatch) {
		t.Fatalf("expected ErrRecordVersionMismatch, got %v", err)
	}
}

func TestDecodeRecordUnsupportedKind(t *testing.T) {
	payload, err := json.Marshal(map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	data, err := json.Marshal(RecordEnvelope{
		SchemaVersion: SupportedSchemaVersion,
		CodecVersion:  SupportedCodecVersion,
		Kind:          "unknown",
		Payload:       payload,
	})
	if err != nil {
		t.Fatalf("marshal envelope: %v", err)
	}

	if _, _, err := DecodeRecord(data); !errors.Is(err, ErrUnsupportedKind) {
		t.Fatalf("expected ErrUnsupportedKind, got %v", err)
	}
}

func allRecordKinds() []string {
	return []string{
		"constraint",
		"pmp",
		"experiment",
		"circuit",
		"layer",
		"layer2",
		"layer_spec",
		"neurode",
		"sensor",
		"actuator",
		"neuron",
		"agent",
		"cortex",
		"substrate",
		"polis",
		"scape",
		"sector",
		"avatar",
		"object",
		"circle",
		"square",
		"line",
		"e",
		"a",
		"specie",
		"population",
		"trace",
		"stat",
		"topology_summary",
		"signature",
		"champion",
	}
}

func canonicalJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal canonical value: %v", err)
	}
	var generic any
	if err := json.Unmarshal(raw, &generic); err != nil {
		t.Fatalf("unmarshal canonical value: %v", err)
	}
	out, err := json.Marshal(generic)
	if err != nil {
		t.Fatalf("marshal canonical generic: %v", err)
	}
	return out
}
