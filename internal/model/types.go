package model

// VersionedRecord captures schema and codec evolution for persistent data.
type VersionedRecord struct {
	SchemaVersion int `json:"schema_version"`
	CodecVersion  int `json:"codec_version"`
}

type Genome struct {
	VersionedRecord
	ID          string    `json:"id"`
	Neurons     []Neuron  `json:"neurons"`
	Synapses    []Synapse `json:"synapses"`
	SensorIDs   []string  `json:"sensor_ids"`
	ActuatorIDs []string  `json:"actuator_ids"`
}

type Neuron struct {
	ID         string  `json:"id"`
	Activation string  `json:"activation"`
	Bias       float64 `json:"bias"`
}

type Synapse struct {
	ID        string  `json:"id"`
	From      string  `json:"from"`
	To        string  `json:"to"`
	Weight    float64 `json:"weight"`
	Enabled   bool    `json:"enabled"`
	Recurrent bool    `json:"recurrent"`
}

type Agent struct {
	VersionedRecord
	ID       string `json:"id"`
	GenomeID string `json:"genome_id"`
}

type Population struct {
	VersionedRecord
	ID         string   `json:"id"`
	AgentIDs   []string `json:"agent_ids"`
	Generation int      `json:"generation"`
}

type ScapeSummary struct {
	VersionedRecord
	Name        string  `json:"name"`
	Description string  `json:"description"`
	BestFitness float64 `json:"best_fitness"`
}
