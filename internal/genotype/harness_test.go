package genotype

import (
	"context"
	"math/rand"
	"testing"

	"protogonos/internal/model"
	"protogonos/internal/storage"
)

func TestCreateTestReplacesExistingGenome(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init memory store: %v", err)
	}
	if err := store.SaveGenome(ctx, model.Genome{
		VersionedRecord: model.VersionedRecord{
			SchemaVersion: storage.CurrentSchemaVersion,
			CodecVersion:  storage.CurrentCodecVersion,
		},
		ID: "test",
		Neurons: []model.Neuron{
			{ID: "legacy"},
		},
	}); err != nil {
		t.Fatalf("seed legacy test genome: %v", err)
	}

	created, err := CreateTest(ctx, store, DefaultConstructConstraint(), rand.New(rand.NewSource(21)))
	if err != nil {
		t.Fatalf("create test genome: %v", err)
	}
	if created.Genome.ID != "test" {
		t.Fatalf("unexpected created id: %s", created.Genome.ID)
	}

	loaded, ok, err := store.GetGenome(ctx, "test")
	if err != nil {
		t.Fatalf("get created test genome: %v", err)
	}
	if !ok {
		t.Fatal("expected test genome to be persisted")
	}
	if len(loaded.Neurons) == 0 {
		t.Fatal("expected non-empty reconstructed test genome")
	}
	if len(loaded.Neurons) == 1 && loaded.Neurons[0].ID == "legacy" {
		t.Fatalf("expected legacy genome to be replaced, got=%+v", loaded.Neurons)
	}
}

func TestRunTestConstructsCloneThenDeletesBoth(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init memory store: %v", err)
	}

	if err := RunTest(ctx, store, DefaultConstructConstraint(), rand.New(rand.NewSource(33))); err != nil {
		t.Fatalf("run test harness: %v", err)
	}

	if _, ok, err := store.GetGenome(ctx, "test"); err != nil {
		t.Fatalf("get test genome: %v", err)
	} else if ok {
		t.Fatal("expected test genome to be deleted by harness")
	}

	if _, ok, err := store.GetGenome(ctx, "test_clone"); err != nil {
		t.Fatalf("get test_clone genome: %v", err)
	} else if ok {
		t.Fatal("expected test_clone genome to be deleted by harness")
	}
}

func TestCreateTestValidatesStore(t *testing.T) {
	if _, err := CreateTest(context.Background(), nil, DefaultConstructConstraint(), rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("expected error for nil store")
	}
}

func TestRunTestValidatesStore(t *testing.T) {
	if err := RunTest(context.Background(), nil, DefaultConstructConstraint(), rand.New(rand.NewSource(1))); err == nil {
		t.Fatal("expected error for nil store")
	}
}
