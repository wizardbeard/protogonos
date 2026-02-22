package genotype

import (
	"context"
	"testing"

	"protogonos/internal/model"
	"protogonos/internal/storage"
)

func TestCloneGenomeDeepCopy(t *testing.T) {
	in := model.Genome{
		ID:          "g1",
		Neurons:     []model.Neuron{{ID: "n1", Activation: "identity", PlasticityBiasParams: []float64{0.2, -0.3}}},
		Synapses:    []model.Synapse{{ID: "s1", From: "n1", To: "n1", Weight: 1, Enabled: true, PlasticityParams: []float64{0.3, -0.1}}},
		SensorIDs:   []string{"s"},
		ActuatorIDs: []string{"a"},
		ActuatorTunables: map[string]float64{
			"a": 0.25,
		},
		Substrate: &model.SubstrateConfig{
			CPPName:    "set_weight",
			CPPIDs:     []string{"substrate:cpp:d3:0"},
			CEPIDs:     []string{"substrate:cep:d3:0"},
			Parameters: map[string]float64{"scale": 1},
		},
		Plasticity: &model.PlasticityConfig{Rule: "hebbian", Rate: 0.1},
	}

	out := CloneGenome(in)
	out.Neurons[0].Activation = "relu"
	out.Neurons[0].PlasticityBiasParams[0] = 7
	out.Synapses[0].PlasticityParams[0] = 9
	out.ActuatorTunables["a"] = 0.75
	out.Substrate.Parameters["scale"] = 2
	out.Substrate.CPPIDs[0] = "substrate:cpp:d3:99"
	out.Substrate.CEPIDs[0] = "substrate:cep:d3:99"

	if in.Neurons[0].Activation != "identity" {
		t.Fatal("expected original neuron slice to remain unchanged")
	}
	if in.Neurons[0].PlasticityBiasParams[0] != 0.2 {
		t.Fatal("expected original neuron plasticity bias parameters to remain unchanged")
	}
	if in.Substrate.Parameters["scale"] != 1 {
		t.Fatal("expected original substrate map to remain unchanged")
	}
	if in.Substrate.CPPIDs[0] != "substrate:cpp:d3:0" || in.Substrate.CEPIDs[0] != "substrate:cep:d3:0" {
		t.Fatal("expected original substrate endpoint id slices to remain unchanged")
	}
	if in.ActuatorTunables["a"] != 0.25 {
		t.Fatal("expected original actuator tunable map to remain unchanged")
	}
	if in.Synapses[0].PlasticityParams[0] != 0.3 {
		t.Fatal("expected original synapse plasticity parameters to remain unchanged")
	}
}

func TestSavePopulationSnapshot(t *testing.T) {
	store := storage.NewMemoryStore()
	if err := store.Init(context.Background()); err != nil {
		t.Fatalf("init store: %v", err)
	}
	genomes := []model.Genome{
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "g1"},
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "g2"},
	}

	if err := SavePopulationSnapshot(context.Background(), store, "pop1", 3, genomes); err != nil {
		t.Fatalf("save snapshot: %v", err)
	}

	pop, ok, err := store.GetPopulation(context.Background(), "pop1")
	if err != nil || !ok {
		t.Fatalf("get population err=%v ok=%t", err, ok)
	}
	if pop.Generation != 3 || len(pop.AgentIDs) != 2 {
		t.Fatalf("unexpected population snapshot: %+v", pop)
	}
}

func TestSavePopulationSnapshotDeduplicatesAgentIDs(t *testing.T) {
	store := storage.NewMemoryStore()
	if err := store.Init(context.Background()); err != nil {
		t.Fatalf("init store: %v", err)
	}
	genomes := []model.Genome{
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "g1"},
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "g1"},
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "g2"},
	}

	if err := SavePopulationSnapshot(context.Background(), store, "pop1", 1, genomes); err != nil {
		t.Fatalf("save snapshot: %v", err)
	}

	pop, ok, err := store.GetPopulation(context.Background(), "pop1")
	if err != nil || !ok {
		t.Fatalf("get population err=%v ok=%t", err, ok)
	}
	if len(pop.AgentIDs) != 2 || pop.AgentIDs[0] != "g1" || pop.AgentIDs[1] != "g2" {
		t.Fatalf("unexpected population agents: %+v", pop.AgentIDs)
	}
}

func TestSavePopulationSnapshotReconcilesMembership(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}
	if err := store.SaveGenome(ctx, model.Genome{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "old-1"}); err != nil {
		t.Fatalf("seed old genome: %v", err)
	}
	if err := store.SaveGenome(ctx, model.Genome{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "keep"}); err != nil {
		t.Fatalf("seed keep genome: %v", err)
	}
	if err := store.SavePopulation(ctx, model.Population{
		VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1},
		ID:              "pop1",
		Generation:      0,
		AgentIDs:        []string{"old-1", "keep"},
	}); err != nil {
		t.Fatalf("seed population: %v", err)
	}

	genomes := []model.Genome{
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "keep"},
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "new-2"},
	}
	if err := SavePopulationSnapshot(ctx, store, "pop1", 2, genomes); err != nil {
		t.Fatalf("save snapshot: %v", err)
	}

	pop, ok, err := store.GetPopulation(ctx, "pop1")
	if err != nil || !ok {
		t.Fatalf("get population err=%v ok=%t", err, ok)
	}
	if pop.Generation != 2 {
		t.Fatalf("unexpected generation: %d", pop.Generation)
	}
	if len(pop.AgentIDs) != 2 || pop.AgentIDs[0] != "keep" || pop.AgentIDs[1] != "new-2" {
		t.Fatalf("unexpected population agents: %+v", pop.AgentIDs)
	}
	if _, ok, err := store.GetGenome(ctx, "old-1"); err != nil {
		t.Fatalf("get old genome: %v", err)
	} else if ok {
		t.Fatal("expected stale genome to be removed from store")
	}
	if _, ok, err := store.GetGenome(ctx, "keep"); err != nil {
		t.Fatalf("get kept genome: %v", err)
	} else if !ok {
		t.Fatal("expected retained genome to remain in store")
	}
	if _, ok, err := store.GetGenome(ctx, "new-2"); err != nil {
		t.Fatalf("get new genome: %v", err)
	} else if !ok {
		t.Fatal("expected new genome to be saved")
	}
}

func TestLoadPopulationSnapshot(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}
	genomes := []model.Genome{
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "g1"},
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "g2"},
	}
	if err := SavePopulationSnapshot(ctx, store, "pop-load", 4, genomes); err != nil {
		t.Fatalf("seed snapshot: %v", err)
	}

	pop, loaded, err := LoadPopulationSnapshot(ctx, store, "pop-load")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if pop.Generation != 4 || len(loaded) != 2 {
		t.Fatalf("unexpected loaded snapshot pop=%+v genomes=%d", pop, len(loaded))
	}
	if loaded[0].ID != "g1" || loaded[1].ID != "g2" {
		t.Fatalf("unexpected loaded genome ordering: %+v", []string{loaded[0].ID, loaded[1].ID})
	}
}

func TestDeletePopulationSnapshot(t *testing.T) {
	ctx := context.Background()
	store := storage.NewMemoryStore()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("init store: %v", err)
	}
	genomes := []model.Genome{
		{VersionedRecord: model.VersionedRecord{SchemaVersion: 1, CodecVersion: 1}, ID: "g1"},
	}
	if err := SavePopulationSnapshot(ctx, store, "pop-del", 1, genomes); err != nil {
		t.Fatalf("seed snapshot: %v", err)
	}
	if err := DeletePopulationSnapshot(ctx, store, "pop-del"); err != nil {
		t.Fatalf("delete snapshot: %v", err)
	}
	_, ok, err := store.GetPopulation(ctx, "pop-del")
	if err != nil {
		t.Fatalf("get after delete: %v", err)
	}
	if ok {
		t.Fatal("expected deleted population to be missing")
	}
}
