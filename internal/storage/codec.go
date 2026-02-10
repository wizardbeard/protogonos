package storage

import (
	"encoding/json"
	"errors"

	"protogonos/internal/model"
)

const (
	CurrentSchemaVersion = 1
	CurrentCodecVersion  = 1
)

var ErrVersionMismatch = errors.New("record version mismatch")

func EncodeGenome(g model.Genome) ([]byte, error) {
	return json.Marshal(g)
}

func DecodeGenome(data []byte) (model.Genome, error) {
	var genome model.Genome
	if err := json.Unmarshal(data, &genome); err != nil {
		return model.Genome{}, err
	}
	if err := checkVersion(genome.VersionedRecord); err != nil {
		return model.Genome{}, err
	}
	return genome, nil
}

func EncodeAgent(a model.Agent) ([]byte, error) {
	return json.Marshal(a)
}

func DecodeAgent(data []byte) (model.Agent, error) {
	var agent model.Agent
	if err := json.Unmarshal(data, &agent); err != nil {
		return model.Agent{}, err
	}
	if err := checkVersion(agent.VersionedRecord); err != nil {
		return model.Agent{}, err
	}
	return agent, nil
}

func EncodePopulation(p model.Population) ([]byte, error) {
	return json.Marshal(p)
}

func DecodePopulation(data []byte) (model.Population, error) {
	var population model.Population
	if err := json.Unmarshal(data, &population); err != nil {
		return model.Population{}, err
	}
	if err := checkVersion(population.VersionedRecord); err != nil {
		return model.Population{}, err
	}
	return population, nil
}

func EncodeScapeSummary(s model.ScapeSummary) ([]byte, error) {
	return json.Marshal(s)
}

func DecodeScapeSummary(data []byte) (model.ScapeSummary, error) {
	var summary model.ScapeSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		return model.ScapeSummary{}, err
	}
	if err := checkVersion(summary.VersionedRecord); err != nil {
		return model.ScapeSummary{}, err
	}
	return summary, nil
}

func checkVersion(v model.VersionedRecord) error {
	if v.SchemaVersion != CurrentSchemaVersion || v.CodecVersion != CurrentCodecVersion {
		return ErrVersionMismatch
	}
	return nil
}
