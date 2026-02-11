package storage

import (
	"context"
	"sync"

	"protogonos/internal/model"
)

type MemoryStore struct {
	mu          sync.RWMutex
	initialized bool
	genomes     map[string]model.Genome
	populations map[string]model.Population
	lineage     map[string][]model.LineageRecord
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{}
}

func (s *MemoryStore) Init(_ context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.initialized = true
	s.genomes = make(map[string]model.Genome)
	s.populations = make(map[string]model.Population)
	s.lineage = make(map[string][]model.LineageRecord)
	return nil
}

func (s *MemoryStore) SaveGenome(_ context.Context, genome model.Genome) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.genomes[genome.ID] = genome
	return nil
}

func (s *MemoryStore) GetGenome(_ context.Context, id string) (model.Genome, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	genome, ok := s.genomes[id]
	return genome, ok, nil
}

func (s *MemoryStore) SavePopulation(_ context.Context, population model.Population) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.populations[population.ID] = population
	return nil
}

func (s *MemoryStore) GetPopulation(_ context.Context, id string) (model.Population, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	population, ok := s.populations[id]
	return population, ok, nil
}

func (s *MemoryStore) SaveLineage(_ context.Context, runID string, lineage []model.LineageRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	copied := make([]model.LineageRecord, len(lineage))
	copy(copied, lineage)
	s.lineage[runID] = copied
	return nil
}

func (s *MemoryStore) GetLineage(_ context.Context, runID string) ([]model.LineageRecord, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	lineage, ok := s.lineage[runID]
	if !ok {
		return nil, false, nil
	}
	copied := make([]model.LineageRecord, len(lineage))
	copy(copied, lineage)
	return copied, true, nil
}
