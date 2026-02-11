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
	seen := make(map[string]struct{}, len(genomes))
	for _, g := range genomes {
		if err := store.SaveGenome(ctx, g); err != nil {
			return err
		}
		if g.ID == "" {
			continue
		}
		if _, ok := seen[g.ID]; ok {
			continue
		}
		seen[g.ID] = struct{}{}
		agentIDs = append(agentIDs, g.ID)
	}

	if err := reconcilePopulationMembership(ctx, store, populationID, seen); err != nil {
		return err
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

func reconcilePopulationMembership(ctx context.Context, store storage.Store, populationID string, keep map[string]struct{}) error {
	population, ok, err := store.GetPopulation(ctx, populationID)
	if err != nil {
		return err
	}
	if !ok {
		return nil
	}

	for _, agentID := range population.AgentIDs {
		if _, stillPresent := keep[agentID]; stillPresent {
			continue
		}
		if err := DeleteAgentFromPopulation(ctx, store, populationID, agentID); err != nil {
			return err
		}
	}
	return nil
}
