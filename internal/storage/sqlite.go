//go:build sqlite

package storage

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sync"

	"protogonos/internal/model"

	_ "modernc.org/sqlite"
)

type SQLiteStore struct {
	path string

	mu sync.RWMutex
	db *sql.DB
}

func NewSQLiteStore(path string) *SQLiteStore {
	return &SQLiteStore{path: path}
}

func (s *SQLiteStore) Init(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.path == "" {
		return errors.New("sqlite path is required")
	}
	if s.db != nil {
		return nil
	}

	db, err := sql.Open("sqlite", s.path)
	if err != nil {
		return err
	}

	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return err
	}

	if err := createTables(ctx, db); err != nil {
		_ = db.Close()
		return err
	}

	s.db = db
	return nil
}

func (s *SQLiteStore) SaveGenome(ctx context.Context, genome model.Genome) error {
	db, err := s.getDB()
	if err != nil {
		return err
	}

	payload, err := EncodeGenome(genome)
	if err != nil {
		return err
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO genomes (id, schema_version, codec_version, payload)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			schema_version = excluded.schema_version,
			codec_version = excluded.codec_version,
			payload = excluded.payload
	`, genome.ID, genome.SchemaVersion, genome.CodecVersion, payload)
	return err
}

func (s *SQLiteStore) GetGenome(ctx context.Context, id string) (model.Genome, bool, error) {
	db, err := s.getDB()
	if err != nil {
		return model.Genome{}, false, err
	}

	var payload []byte
	err = db.QueryRowContext(ctx, `SELECT payload FROM genomes WHERE id = ?`, id).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return model.Genome{}, false, nil
		}
		return model.Genome{}, false, err
	}

	genome, err := DecodeGenome(payload)
	if err != nil {
		return model.Genome{}, false, fmt.Errorf("decode genome %s: %w", id, err)
	}
	return genome, true, nil
}

func (s *SQLiteStore) SavePopulation(ctx context.Context, population model.Population) error {
	db, err := s.getDB()
	if err != nil {
		return err
	}

	payload, err := EncodePopulation(population)
	if err != nil {
		return err
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO populations (id, schema_version, codec_version, payload)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			schema_version = excluded.schema_version,
			codec_version = excluded.codec_version,
			payload = excluded.payload
	`, population.ID, population.SchemaVersion, population.CodecVersion, payload)
	return err
}

func (s *SQLiteStore) GetPopulation(ctx context.Context, id string) (model.Population, bool, error) {
	db, err := s.getDB()
	if err != nil {
		return model.Population{}, false, err
	}

	var payload []byte
	err = db.QueryRowContext(ctx, `SELECT payload FROM populations WHERE id = ?`, id).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return model.Population{}, false, nil
		}
		return model.Population{}, false, err
	}

	population, err := DecodePopulation(payload)
	if err != nil {
		return model.Population{}, false, fmt.Errorf("decode population %s: %w", id, err)
	}
	return population, true, nil
}

func (s *SQLiteStore) SaveScapeSummary(ctx context.Context, summary model.ScapeSummary) error {
	db, err := s.getDB()
	if err != nil {
		return err
	}

	payload, err := EncodeScapeSummary(summary)
	if err != nil {
		return err
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO scape_summaries (name, schema_version, codec_version, payload)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET
			schema_version = excluded.schema_version,
			codec_version = excluded.codec_version,
			payload = excluded.payload
	`, summary.Name, summary.SchemaVersion, summary.CodecVersion, payload)
	return err
}

func (s *SQLiteStore) GetScapeSummary(ctx context.Context, name string) (model.ScapeSummary, bool, error) {
	db, err := s.getDB()
	if err != nil {
		return model.ScapeSummary{}, false, err
	}

	var payload []byte
	err = db.QueryRowContext(ctx, `SELECT payload FROM scape_summaries WHERE name = ?`, name).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return model.ScapeSummary{}, false, nil
		}
		return model.ScapeSummary{}, false, err
	}

	summary, err := DecodeScapeSummary(payload)
	if err != nil {
		return model.ScapeSummary{}, false, fmt.Errorf("decode scape summary %s: %w", name, err)
	}
	return summary, true, nil
}

func (s *SQLiteStore) SaveFitnessHistory(ctx context.Context, runID string, history []float64) error {
	db, err := s.getDB()
	if err != nil {
		return err
	}

	payload, err := EncodeFitnessHistory(history)
	if err != nil {
		return err
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO fitness_history (run_id, payload)
		VALUES (?, ?)
		ON CONFLICT(run_id) DO UPDATE SET
			payload = excluded.payload
	`, runID, payload)
	return err
}

func (s *SQLiteStore) GetFitnessHistory(ctx context.Context, runID string) ([]float64, bool, error) {
	db, err := s.getDB()
	if err != nil {
		return nil, false, err
	}

	var payload []byte
	err = db.QueryRowContext(ctx, `SELECT payload FROM fitness_history WHERE run_id = ?`, runID).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, false, nil
		}
		return nil, false, err
	}

	history, err := DecodeFitnessHistory(payload)
	if err != nil {
		return nil, false, fmt.Errorf("decode fitness history %s: %w", runID, err)
	}
	return history, true, nil
}

func (s *SQLiteStore) SaveGenerationDiagnostics(ctx context.Context, runID string, diagnostics []model.GenerationDiagnostics) error {
	db, err := s.getDB()
	if err != nil {
		return err
	}

	payload, err := EncodeGenerationDiagnostics(diagnostics)
	if err != nil {
		return err
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO generation_diagnostics (run_id, payload)
		VALUES (?, ?)
		ON CONFLICT(run_id) DO UPDATE SET
			payload = excluded.payload
	`, runID, payload)
	return err
}

func (s *SQLiteStore) GetGenerationDiagnostics(ctx context.Context, runID string) ([]model.GenerationDiagnostics, bool, error) {
	db, err := s.getDB()
	if err != nil {
		return nil, false, err
	}

	var payload []byte
	err = db.QueryRowContext(ctx, `SELECT payload FROM generation_diagnostics WHERE run_id = ?`, runID).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, false, nil
		}
		return nil, false, err
	}

	diagnostics, err := DecodeGenerationDiagnostics(payload)
	if err != nil {
		return nil, false, fmt.Errorf("decode generation diagnostics %s: %w", runID, err)
	}
	return diagnostics, true, nil
}

func (s *SQLiteStore) SaveTopGenomes(ctx context.Context, runID string, top []model.TopGenomeRecord) error {
	db, err := s.getDB()
	if err != nil {
		return err
	}

	payload, err := EncodeTopGenomes(top)
	if err != nil {
		return err
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO top_genomes (run_id, payload)
		VALUES (?, ?)
		ON CONFLICT(run_id) DO UPDATE SET
			payload = excluded.payload
	`, runID, payload)
	return err
}

func (s *SQLiteStore) SaveSpeciesHistory(ctx context.Context, runID string, history []model.SpeciesGeneration) error {
	db, err := s.getDB()
	if err != nil {
		return err
	}

	payload, err := EncodeSpeciesHistory(history)
	if err != nil {
		return err
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO species_history (run_id, payload)
		VALUES (?, ?)
		ON CONFLICT(run_id) DO UPDATE SET
			payload = excluded.payload
	`, runID, payload)
	return err
}

func (s *SQLiteStore) GetSpeciesHistory(ctx context.Context, runID string) ([]model.SpeciesGeneration, bool, error) {
	db, err := s.getDB()
	if err != nil {
		return nil, false, err
	}

	var payload []byte
	err = db.QueryRowContext(ctx, `SELECT payload FROM species_history WHERE run_id = ?`, runID).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, false, nil
		}
		return nil, false, err
	}

	history, err := DecodeSpeciesHistory(payload)
	if err != nil {
		return nil, false, fmt.Errorf("decode species history %s: %w", runID, err)
	}
	return history, true, nil
}

func (s *SQLiteStore) GetTopGenomes(ctx context.Context, runID string) ([]model.TopGenomeRecord, bool, error) {
	db, err := s.getDB()
	if err != nil {
		return nil, false, err
	}

	var payload []byte
	err = db.QueryRowContext(ctx, `SELECT payload FROM top_genomes WHERE run_id = ?`, runID).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, false, nil
		}
		return nil, false, err
	}

	top, err := DecodeTopGenomes(payload)
	if err != nil {
		return nil, false, fmt.Errorf("decode top genomes %s: %w", runID, err)
	}
	return top, true, nil
}

func (s *SQLiteStore) SaveLineage(ctx context.Context, runID string, lineage []model.LineageRecord) error {
	db, err := s.getDB()
	if err != nil {
		return err
	}

	payload, err := EncodeLineage(lineage)
	if err != nil {
		return err
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO lineage (run_id, payload)
		VALUES (?, ?)
		ON CONFLICT(run_id) DO UPDATE SET
			payload = excluded.payload
	`, runID, payload)
	return err
}

func (s *SQLiteStore) GetLineage(ctx context.Context, runID string) ([]model.LineageRecord, bool, error) {
	db, err := s.getDB()
	if err != nil {
		return nil, false, err
	}

	var payload []byte
	err = db.QueryRowContext(ctx, `SELECT payload FROM lineage WHERE run_id = ?`, runID).Scan(&payload)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, false, nil
		}
		return nil, false, err
	}

	lineage, err := DecodeLineage(payload)
	if err != nil {
		return nil, false, fmt.Errorf("decode lineage %s: %w", runID, err)
	}
	return lineage, true, nil
}

func (s *SQLiteStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.db == nil {
		return nil
	}
	err := s.db.Close()
	s.db = nil
	return err
}

func (s *SQLiteStore) getDB() (*sql.DB, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.db == nil {
		return nil, errors.New("store is not initialized")
	}
	return s.db, nil
}

func createTables(ctx context.Context, db *sql.DB) error {
	_, err := db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS genomes (
			id TEXT PRIMARY KEY,
			schema_version INTEGER NOT NULL,
			codec_version INTEGER NOT NULL,
			payload BLOB NOT NULL
		);
		CREATE TABLE IF NOT EXISTS populations (
			id TEXT PRIMARY KEY,
			schema_version INTEGER NOT NULL,
			codec_version INTEGER NOT NULL,
			payload BLOB NOT NULL
		);
		CREATE TABLE IF NOT EXISTS scape_summaries (
			name TEXT PRIMARY KEY,
			schema_version INTEGER NOT NULL,
			codec_version INTEGER NOT NULL,
			payload BLOB NOT NULL
		);
		CREATE TABLE IF NOT EXISTS fitness_history (
			run_id TEXT PRIMARY KEY,
			payload BLOB NOT NULL
		);
		CREATE TABLE IF NOT EXISTS generation_diagnostics (
			run_id TEXT PRIMARY KEY,
			payload BLOB NOT NULL
		);
		CREATE TABLE IF NOT EXISTS top_genomes (
			run_id TEXT PRIMARY KEY,
			payload BLOB NOT NULL
		);
		CREATE TABLE IF NOT EXISTS species_history (
			run_id TEXT PRIMARY KEY,
			payload BLOB NOT NULL
		);
		CREATE TABLE IF NOT EXISTS lineage (
			run_id TEXT PRIMARY KEY,
			payload BLOB NOT NULL
		);
	`)
	return err
}
