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

func LoadPopulationSnapshot(ctx context.Context, store storage.Store, populationID string) (model.Population, []model.Genome, error) {
	if store == nil {
		return model.Population{}, nil, fmt.Errorf("store is required")
	}
	if populationID == "" {
		return model.Population{}, nil, fmt.Errorf("population id is required")
	}

	pop, ok, err := store.GetPopulation(ctx, populationID)
	if err != nil {
		return model.Population{}, nil, err
	}
	if !ok {
		return model.Population{}, nil, fmt.Errorf("population not found: %s", populationID)
	}

	genomes := make([]model.Genome, 0, len(pop.AgentIDs))
	for _, agentID := range pop.AgentIDs {
		g, ok, err := store.GetGenome(ctx, agentID)
		if err != nil {
			return model.Population{}, nil, err
		}
		if !ok {
			return model.Population{}, nil, fmt.Errorf("genome not found for population %s agent %s", populationID, agentID)
		}
		genomes = append(genomes, g)
	}
	return pop, genomes, nil
}

func DeletePopulationSnapshot(ctx context.Context, store storage.Store, populationID string) error {
	if store == nil {
		return fmt.Errorf("store is required")
	}
	if populationID == "" {
		return fmt.Errorf("population id is required")
	}
	_, ok, err := store.GetPopulation(ctx, populationID)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("population not found: %s", populationID)
	}
	return store.DeletePopulation(ctx, populationID)
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
