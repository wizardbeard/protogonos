package model

// VersionedRecord captures schema and codec evolution for persistent data.
type VersionedRecord struct {
	SchemaVersion int `json:"schema_version"`
	CodecVersion  int `json:"codec_version"`
}

type Genome struct {
	VersionedRecord
	ID                  string               `json:"id"`
	Neurons             []Neuron             `json:"neurons"`
	Synapses            []Synapse            `json:"synapses"`
	SensorIDs           []string             `json:"sensor_ids"`
	ActuatorIDs         []string             `json:"actuator_ids"`
	ActuatorTunables    map[string]float64   `json:"actuator_tunables,omitempty"`
	ActuatorGenerations map[string]int       `json:"actuator_generations,omitempty"`
	SensorNeuronLinks   []SensorNeuronLink   `json:"sensor_neuron_links,omitempty"`
	NeuronActuatorLinks []NeuronActuatorLink `json:"neuron_actuator_links,omitempty"`
	SensorLinks         int                  `json:"sensor_links,omitempty"`
	ActuatorLinks       int                  `json:"actuator_links,omitempty"`
	Substrate           *SubstrateConfig     `json:"substrate,omitempty"`
	Plasticity          *PlasticityConfig    `json:"plasticity,omitempty"`
	Strategy            *StrategyConfig      `json:"strategy,omitempty"`
}

type SensorNeuronLink struct {
	SensorID string `json:"sensor_id"`
	NeuronID string `json:"neuron_id"`
}

type NeuronActuatorLink struct {
	NeuronID   string `json:"neuron_id"`
	ActuatorID string `json:"actuator_id"`
}

type StrategyConfig struct {
	TuningSelection  string  `json:"tuning_selection"`
	AnnealingFactor  float64 `json:"annealing_factor"`
	TopologicalMode  string  `json:"topological_mode"`
	TopologicalParam float64 `json:"topological_param,omitempty"`
	HeredityType     string  `json:"heredity_type"`
}

type SubstrateConfig struct {
	CPPName     string             `json:"cpp_name"`
	CEPName     string             `json:"cep_name"`
	Dimensions  []int              `json:"dimensions"`
	Parameters  map[string]float64 `json:"parameters"`
	WeightCount int                `json:"weight_count"`
}

type PlasticityConfig struct {
	Rule            string  `json:"rule"`
	Rate            float64 `json:"rate"`
	SaturationLimit float64 `json:"saturation_limit"`
	CoeffA          float64 `json:"coeff_a,omitempty"`
	CoeffB          float64 `json:"coeff_b,omitempty"`
	CoeffC          float64 `json:"coeff_c,omitempty"`
	CoeffD          float64 `json:"coeff_d,omitempty"`
}

type Neuron struct {
	ID             string  `json:"id"`
	Generation     int     `json:"generation,omitempty"`
	Activation     string  `json:"activation"`
	Aggregator     string  `json:"aggregator,omitempty"`
	PlasticityRule string  `json:"plasticity_rule,omitempty"`
	PlasticityRate float64 `json:"plasticity_rate,omitempty"`
	PlasticityA    float64 `json:"plasticity_a,omitempty"`
	PlasticityB    float64 `json:"plasticity_b,omitempty"`
	PlasticityC    float64 `json:"plasticity_c,omitempty"`
	PlasticityD    float64 `json:"plasticity_d,omitempty"`
	Bias           float64 `json:"bias"`
}

type Synapse struct {
	ID               string    `json:"id"`
	From             string    `json:"from"`
	To               string    `json:"to"`
	Weight           float64   `json:"weight"`
	Enabled          bool      `json:"enabled"`
	Recurrent        bool      `json:"recurrent"`
	PlasticityParams []float64 `json:"plasticity_params,omitempty"`
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

type LineageSummary struct {
	TotalNeurons           int            `json:"total_neurons"`
	TotalSynapses          int            `json:"total_synapses"`
	TotalRecurrentSynapses int            `json:"total_recurrent_synapses"`
	TotalSensors           int            `json:"total_sensors"`
	TotalActuators         int            `json:"total_actuators"`
	ActivationDistribution map[string]int `json:"activation_distribution"`
	AggregatorDistribution map[string]int `json:"aggregator_distribution"`
}

type LineageRecord struct {
	VersionedRecord
	GenomeID    string         `json:"genome_id"`
	ParentID    string         `json:"parent_id"`
	Generation  int            `json:"generation"`
	Operation   string         `json:"operation"`
	Fingerprint string         `json:"fingerprint,omitempty"`
	Summary     LineageSummary `json:"summary,omitempty"`
}

type GenerationDiagnostics struct {
	Generation            int     `json:"generation"`
	BestFitness           float64 `json:"best_fitness"`
	MeanFitness           float64 `json:"mean_fitness"`
	MinFitness            float64 `json:"min_fitness"`
	SpeciesCount          int     `json:"species_count"`
	FingerprintDiversity  int     `json:"fingerprint_diversity"`
	SpeciationThreshold   float64 `json:"speciation_threshold"`
	TargetSpeciesCount    int     `json:"target_species_count"`
	MeanSpeciesSize       float64 `json:"mean_species_size"`
	LargestSpeciesSize    int     `json:"largest_species_size"`
	TuningInvocations     int     `json:"tuning_invocations"`
	TuningAttempts        int     `json:"tuning_attempts"`
	TuningEvaluations     int     `json:"tuning_evaluations"`
	TuningAccepted        int     `json:"tuning_accepted"`
	TuningRejected        int     `json:"tuning_rejected"`
	TuningGoalHits        int     `json:"tuning_goal_hits"`
	TuningAcceptRate      float64 `json:"tuning_accept_rate"`
	TuningEvalsPerAttempt float64 `json:"tuning_evals_per_attempt"`
}

type SpeciesGeneration struct {
	Generation     int              `json:"generation"`
	Species        []SpeciesMetrics `json:"species"`
	NewSpecies     []string         `json:"new_species,omitempty"`
	ExtinctSpecies []string         `json:"extinct_species,omitempty"`
}

type SpeciesMetrics struct {
	Key         string  `json:"key"`
	Size        int     `json:"size"`
	MeanFitness float64 `json:"mean_fitness"`
	BestFitness float64 `json:"best_fitness"`
}

type TopGenomeRecord struct {
	Rank    int     `json:"rank"`
	Fitness float64 `json:"fitness"`
	Genome  Genome  `json:"genome"`
}

type ScapeSummary struct {
	VersionedRecord
	Name        string  `json:"name"`
	Description string  `json:"description"`
	BestFitness float64 `json:"best_fitness"`
}
