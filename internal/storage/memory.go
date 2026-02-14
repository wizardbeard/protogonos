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
	scapes      map[string]model.ScapeSummary
	history     map[string][]float64
	diagnostics map[string][]model.GenerationDiagnostics
	speciesHist map[string][]model.SpeciesGeneration
	topGenomes  map[string][]model.TopGenomeRecord
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
	s.scapes = make(map[string]model.ScapeSummary)
	s.history = make(map[string][]float64)
	s.diagnostics = make(map[string][]model.GenerationDiagnostics)
	s.speciesHist = make(map[string][]model.SpeciesGeneration)
	s.topGenomes = make(map[string][]model.TopGenomeRecord)
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

func (s *MemoryStore) DeletePopulation(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.populations, id)
	return nil
}

func (s *MemoryStore) SaveScapeSummary(_ context.Context, summary model.ScapeSummary) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.scapes[summary.Name] = summary
	return nil
}

func (s *MemoryStore) GetScapeSummary(_ context.Context, name string) (model.ScapeSummary, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	summary, ok := s.scapes[name]
	return summary, ok, nil
}

func (s *MemoryStore) SaveFitnessHistory(_ context.Context, runID string, history []float64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	copied := append([]float64(nil), history...)
	s.history[runID] = copied
	return nil
}

func (s *MemoryStore) GetFitnessHistory(_ context.Context, runID string) ([]float64, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	history, ok := s.history[runID]
	if !ok {
		return nil, false, nil
	}
	copied := append([]float64(nil), history...)
	return copied, true, nil
}

func (s *MemoryStore) SaveGenerationDiagnostics(_ context.Context, runID string, diagnostics []model.GenerationDiagnostics) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	copied := make([]model.GenerationDiagnostics, len(diagnostics))
	copy(copied, diagnostics)
	s.diagnostics[runID] = copied
	return nil
}

func (s *MemoryStore) GetGenerationDiagnostics(_ context.Context, runID string) ([]model.GenerationDiagnostics, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	diagnostics, ok := s.diagnostics[runID]
	if !ok {
		return nil, false, nil
	}
	copied := make([]model.GenerationDiagnostics, len(diagnostics))
	copy(copied, diagnostics)
	return copied, true, nil
}

func (s *MemoryStore) SaveTopGenomes(_ context.Context, runID string, top []model.TopGenomeRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	copied := make([]model.TopGenomeRecord, len(top))
	copy(copied, top)
	s.topGenomes[runID] = copied
	return nil
}

func (s *MemoryStore) SaveSpeciesHistory(_ context.Context, runID string, history []model.SpeciesGeneration) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	copied := make([]model.SpeciesGeneration, 0, len(history))
	for _, generation := range history {
		species := make([]model.SpeciesMetrics, len(generation.Species))
		copy(species, generation.Species)
		copied = append(copied, model.SpeciesGeneration{
			Generation:     generation.Generation,
			Species:        species,
			NewSpecies:     append([]string(nil), generation.NewSpecies...),
			ExtinctSpecies: append([]string(nil), generation.ExtinctSpecies...),
		})
	}
	s.speciesHist[runID] = copied
	return nil
}

func (s *MemoryStore) GetSpeciesHistory(_ context.Context, runID string) ([]model.SpeciesGeneration, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	history, ok := s.speciesHist[runID]
	if !ok {
		return nil, false, nil
	}
	copied := make([]model.SpeciesGeneration, 0, len(history))
	for _, generation := range history {
		species := make([]model.SpeciesMetrics, len(generation.Species))
		copy(species, generation.Species)
		copied = append(copied, model.SpeciesGeneration{
			Generation:     generation.Generation,
			Species:        species,
			NewSpecies:     append([]string(nil), generation.NewSpecies...),
			ExtinctSpecies: append([]string(nil), generation.ExtinctSpecies...),
		})
	}
	return copied, true, nil
}

func (s *MemoryStore) GetTopGenomes(_ context.Context, runID string) ([]model.TopGenomeRecord, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	top, ok := s.topGenomes[runID]
	if !ok {
		return nil, false, nil
	}
	copied := make([]model.TopGenomeRecord, len(top))
	copy(copied, top)
	return copied, true, nil
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
