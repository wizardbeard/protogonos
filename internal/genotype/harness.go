package genotype

import (
	"context"
	"fmt"
	"math/rand"

	"protogonos/internal/storage"
)

// CreateTest mirrors genotype:create_test/0 intent by ensuring a fresh test
// genome with ID "test" is (re)constructed and persisted.
func CreateTest(
	ctx context.Context,
	store storage.Store,
	constraint ConstructConstraint,
	rng *rand.Rand,
) (ConstructedAgent, error) {
	if store == nil {
		return ConstructedAgent{}, fmt.Errorf("store is required")
	}
	if constraint.Morphology == "" {
		constraint = DefaultConstructConstraint()
	}

	_, exists, err := store.GetGenome(ctx, "test")
	if err != nil {
		return ConstructedAgent{}, err
	}
	if exists {
		if err := DeleteAgent(ctx, store, "test"); err != nil {
			return ConstructedAgent{}, err
		}
	}

	agent, err := ConstructAgent("test", "test", constraint, rng)
	if err != nil {
		return ConstructedAgent{}, err
	}
	if err := store.SaveGenome(ctx, agent.Genome); err != nil {
		return ConstructedAgent{}, err
	}
	return agent, nil
}

// RunTest mirrors genotype:test/0 intent by constructing a test agent,
// cloning it, and then deleting both records.
func RunTest(
	ctx context.Context,
	store storage.Store,
	constraint ConstructConstraint,
	rng *rand.Rand,
) error {
	if store == nil {
		return fmt.Errorf("store is required")
	}
	if constraint.Morphology == "" {
		constraint = DefaultConstructConstraint()
	}
	base, err := ConstructAgent("test", "test", constraint, rng)
	if err != nil {
		return err
	}
	if err := store.SaveGenome(ctx, base.Genome); err != nil {
		return err
	}

	clone := CloneAgentWithRemappedIDs(base.Genome, "test_clone", append(append([]string(nil), base.InputNeuronIDs...), base.OutputNeuronIDs...))
	if err := store.SaveGenome(ctx, clone); err != nil {
		return err
	}

	if err := DeleteAgent(ctx, store, "test"); err != nil {
		return err
	}
	return DeleteAgent(ctx, store, "test_clone")
}
