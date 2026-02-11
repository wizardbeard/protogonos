package genotype

import (
	"context"
	"fmt"

	"protogonos/internal/model"
	"protogonos/internal/storage"
)

func SavePopulationSnapshot(ctx context.Context, store storage.Store, populationID string, generation int, genomes []model.Genome) error {
	if store == nil {
		return fmt.Errorf("store is required")
	}
	if populationID == "" {
		return fmt.Errorf("population id is required")
	}

	agentIDs := make([]string, 0, len(genomes))
	for _, g := range genomes {
		if err := store.SaveGenome(ctx, g); err != nil {
			return err
		}
		agentIDs = append(agentIDs, g.ID)
	}

	return store.SavePopulation(ctx, model.Population{
		VersionedRecord: model.VersionedRecord{
			SchemaVersion: storage.CurrentSchemaVersion,
			CodecVersion:  storage.CurrentCodecVersion,
		},
		ID:         populationID,
		AgentIDs:   agentIDs,
		Generation: generation,
	})
}
