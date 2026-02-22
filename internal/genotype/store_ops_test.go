package genotype

import (
	"context"
	"testing"

	"protogonos/internal/model"
	"protogonos/internal/storage"
)

func TestReadWriteDeleteGenomeWrappers(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}

	genome := model.Genome{
		VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1},
		ID:              "g-wrap-1",
	}
	if err := Write(ctx, store, genome); err != nil {
		t.Fatalf("write genome: %v", err)
	}

	got, ok, err := Read(ctx, store, RecordKey{Table: RecordTableGenome, ID: genome.ID})
	if err != nil {
		t.Fatalf("read genome: %v", err)
	}
	if !ok {
		t.Fatal("expected stored genome")
	}
	g, ok := got.(model.Genome)
	if !ok {
		t.Fatalf("expected model.Genome result type, got %T", got)
	}
	if g.ID != genome.ID {
		t.Fatalf("unexpected genome read result: %+v", g)
	}

	if err := Delete(ctx, store, RecordKey{Table: RecordTableGenome, ID: genome.ID}); err != nil {
		t.Fatalf("delete genome: %v", err)
	}
	_, ok, err = Read(ctx, store, RecordKey{Table: RecordTableGenome, ID: genome.ID})
	if err != nil {
		t.Fatalf("read after delete: %v", err)
	}
	if ok {
		t.Fatal("expected deleted genome to be absent")
	}
}

func TestDirtyAliasesPopulationWrappers(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}

	population := &model.Population{
		VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1},
		ID:              "pop-wrap-1",
		AgentIDs:        []string{"g1", "g2"},
		Generation:      3,
	}
	if err := DirtyWrite(ctx, store, population); err != nil {
		t.Fatalf("dirty write population: %v", err)
	}

	got, ok, err := DirtyRead(ctx, store, RecordKey{Table: RecordTablePopulation, ID: population.ID})
	if err != nil {
		t.Fatalf("dirty read population: %v", err)
	}
	if !ok {
		t.Fatal("expected stored population")
	}
	pop, ok := got.(model.Population)
	if !ok {
		t.Fatalf("expected model.Population result type, got %T", got)
	}
	if pop.Generation != 3 || len(pop.AgentIDs) != 2 {
		t.Fatalf("unexpected population read result: %+v", pop)
	}

	if err := DirtyDelete(ctx, store, RecordKey{Table: RecordTablePopulation, ID: population.ID}); err != nil {
		t.Fatalf("dirty delete population: %v", err)
	}
	_, ok, err = DirtyRead(ctx, store, RecordKey{Table: RecordTablePopulation, ID: population.ID})
	if err != nil {
		t.Fatalf("dirty read after delete: %v", err)
	}
	if ok {
		t.Fatal("expected deleted population to be absent")
	}
}

func TestReadWriteScapeSummaryWrappers(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}

	summary := model.ScapeSummary{
		VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1},
		Name:            "xor",
		Description:     "xor scape",
		BestFitness:     1.0,
	}
	if err := Write(ctx, store, summary); err != nil {
		t.Fatalf("write scape summary: %v", err)
	}

	got, ok, err := Read(ctx, store, RecordKey{Table: RecordTableScape, ID: "xor"})
	if err != nil {
		t.Fatalf("read scape summary: %v", err)
	}
	if !ok {
		t.Fatal("expected stored scape summary")
	}
	scape, ok := got.(model.ScapeSummary)
	if !ok {
		t.Fatalf("expected model.ScapeSummary result type, got %T", got)
	}
	if scape.Name != "xor" || scape.BestFitness != 1.0 {
		t.Fatalf("unexpected scape summary result: %+v", scape)
	}
}

func TestStoreWrappersValidateInputs(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}

	if _, _, err := Read(ctx, nil, RecordKey{Table: RecordTableGenome, ID: "x"}); err == nil {
		t.Fatal("expected read to fail with nil store")
	}
	if _, _, err := Read(ctx, store, RecordKey{Table: "unknown", ID: "x"}); err == nil {
		t.Fatal("expected read to fail for unsupported table")
	}
	if err := Write(ctx, store, 123); err == nil {
		t.Fatal("expected write to fail for unsupported record type")
	}
	if err := Delete(ctx, store, RecordKey{Table: "unknown", ID: "x"}); err == nil {
		t.Fatal("expected delete to fail for unsupported table")
	}
	if err := Delete(ctx, store, RecordKey{Table: RecordTableScape, ID: "xor"}); err == nil {
		t.Fatal("expected delete to fail for scape table without delete support")
	}
}
