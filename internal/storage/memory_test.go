package storage

import (
	"context"
	"testing"

	"protogonos/internal/model"
)

func TestMemoryStoreLineageRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}

	input := []model.LineageRecord{{
		VersionedRecord: model.VersionedRecord{SchemaVersion: CurrentSchemaVersion, CodecVersion: CurrentCodecVersion},
		GenomeID:        "g1",
		Generation:      1,
		Operation:       "mutate",
	}}
	if err := store.SaveLineage(ctx, "run-1", input); err != nil {
		t.Fatalf("save lineage: %v", err)
	}

	output, ok, err := store.GetLineage(ctx, "run-1")
	if err != nil {
		t.Fatalf("get lineage: %v", err)
	}
	if !ok {
		t.Fatal("expected persisted lineage")
	}
	if len(output) != 1 || output[0].GenomeID != "g1" {
		t.Fatalf("unexpected lineage: %+v", output)
	}
}
