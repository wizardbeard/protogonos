package genotype

import (
	"context"
	"fmt"
	"strings"

	"protogonos/internal/model"
	"protogonos/internal/storage"
)

const (
	RecordTableGenome     = "genome"
	RecordTablePopulation = "population"
	RecordTableScape      = "scape"
)

type RecordKey struct {
	Table string
	ID    string
}

// Read is a genotype-level typed store wrapper analogous to genotype:read/1.
func Read(ctx context.Context, store storage.Store, key RecordKey) (any, bool, error) {
	if store == nil {
		return nil, false, fmt.Errorf("store is required")
	}
	if key.ID == "" {
		return nil, false, fmt.Errorf("record id is required")
	}
	switch normalizeRecordTable(key.Table) {
	case RecordTableGenome:
		return store.GetGenome(ctx, key.ID)
	case RecordTablePopulation:
		return store.GetPopulation(ctx, key.ID)
	case RecordTableScape:
		return store.GetScapeSummary(ctx, key.ID)
	default:
		return nil, false, fmt.Errorf("unsupported record table: %s", key.Table)
	}
}

// DirtyRead mirrors genotype:dirty_read/1 semantics; storage backends control
// transaction behavior so this is an alias to Read.
func DirtyRead(ctx context.Context, store storage.Store, key RecordKey) (any, bool, error) {
	return Read(ctx, store, key)
}

// Write is a genotype-level typed store wrapper analogous to genotype:write/1.
func Write(ctx context.Context, store storage.Store, record any) error {
	if store == nil {
		return fmt.Errorf("store is required")
	}
	switch rec := record.(type) {
	case model.Genome:
		return store.SaveGenome(ctx, rec)
	case *model.Genome:
		if rec == nil {
			return fmt.Errorf("record is required")
		}
		return store.SaveGenome(ctx, *rec)
	case model.Population:
		return store.SavePopulation(ctx, rec)
	case *model.Population:
		if rec == nil {
			return fmt.Errorf("record is required")
		}
		return store.SavePopulation(ctx, *rec)
	case model.ScapeSummary:
		return store.SaveScapeSummary(ctx, rec)
	case *model.ScapeSummary:
		if rec == nil {
			return fmt.Errorf("record is required")
		}
		return store.SaveScapeSummary(ctx, *rec)
	default:
		return fmt.Errorf("unsupported record type: %T", record)
	}
}

// DirtyWrite mirrors genotype:dirty_write/1 semantics; storage backends control
// transaction behavior so this is an alias to Write.
func DirtyWrite(ctx context.Context, store storage.Store, record any) error {
	return Write(ctx, store, record)
}

// Delete is a genotype-level typed store wrapper analogous to genotype:delete/1.
func Delete(ctx context.Context, store storage.Store, key RecordKey) error {
	if store == nil {
		return fmt.Errorf("store is required")
	}
	if key.ID == "" {
		return fmt.Errorf("record id is required")
	}
	switch normalizeRecordTable(key.Table) {
	case RecordTableGenome:
		return store.DeleteGenome(ctx, key.ID)
	case RecordTablePopulation:
		return store.DeletePopulation(ctx, key.ID)
	case RecordTableScape:
		return fmt.Errorf("delete is not supported for record table: %s", key.Table)
	default:
		return fmt.Errorf("unsupported record table: %s", key.Table)
	}
}

// DirtyDelete mirrors genotype:dirty_delete/1 semantics; storage backends
// control transaction behavior so this is an alias to Delete.
func DirtyDelete(ctx context.Context, store storage.Store, key RecordKey) error {
	return Delete(ctx, store, key)
}

func normalizeRecordTable(table string) string {
	return strings.TrimSpace(strings.ToLower(table))
}
