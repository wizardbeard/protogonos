package map2rec

import "math"

type WeightedOperator struct {
	Name   string
	Weight float64
}

type MutationCountPolicy struct {
	Name  string
	Param float64
}

type DurationSpec struct {
	Name  string
	Param float64
}

type ConstraintRecord struct {
	Morphology                  string
	ConnectionArchitecture      string
	NeuralAFs                   []string
	NeuralPFNs                  []string
	SubstratePlasticities       []string
	SubstrateLinkforms          []string
	NeuralAggrFs                []string
	TuningSelectionFs           []string
	TuningDurationF             DurationSpec
	AnnealingParameters         []float64
	PerturbationRanges          []float64
	AgentEncodingTypes          []string
	HeredityTypes               []string
	MutationOperators           []WeightedOperator
	TotTopologicalMutationsFs   []MutationCountPolicy
	PopulationEvoAlgF           string
	PopulationFitnessProcessorF string
	PopulationSelectionF        string
	SpecieDistinguishers        []string
	HOFDistinguishers           []string
	Objectives                  []string
}

type PMPRecord struct {
	OpMode             string
	PopulationID       string
	SurvivalPercentage float64
	SpecieSizeLimit    int
	InitSpecieSize     int
	PolisID            string
	GenerationLimit    int
	EvaluationsLimit   int
	FitnessGoal        float64
	BenchmarkerPID     string
	CommitteePID       string
}

type SensorRecord struct {
	ID           any
	Name         string
	Type         string
	CortexID     any
	Scape        any
	VL           int
	FanoutIDs    []any
	Generation   int
	Format       any
	Parameters   any
	GTParameters any
	PhysRep      any
	VisRep       any
	PreF         string
	PostF        string
}

type ActuatorRecord struct {
	ID           any
	Name         string
	Type         string
	CortexID     any
	Scape        any
	VL           int
	FaninIDs     []any
	Generation   int
	Format       any
	Parameters   any
	GTParameters any
	PhysRep      any
	VisRep       any
	PreF         string
	PostF        string
}

type NeuronRecord struct {
	ID                 any
	Generation         int
	CortexID           any
	PreProcessor       string
	SignalIntegrator   string
	ActivationFunction string
	PostProcessor      string
	PlasticityFunction any
	AggregatorFunction string
	InputIDPs          []any
	InputIDPsMod       []any
	OutputIDs          []any
	RecurrentOutputIDs []any
}

type AgentRecord struct {
	ID                 any
	EncodingType       string
	Generation         int
	PopulationID       any
	SpecieID           any
	CortexID           any
	Fingerprint        any
	Constraint         any
	EvoHist            []any
	Fitness            float64
	InnovationFactor   any
	Pattern            []any
	TuningSelectionF   string
	AnnealingParameter any
	TuningDurationF    any
	PerturbationRange  any
	MutationOperators  []any
	TotTopologicalMutF any
	HeredityType       string
	SubstrateID        any
	OffspringIDs       []any
	ParentIDs          []any
	ChampionFlag       []any
	Evolvability       float64
	Brittleness        float64
	Robustness         float64
	EvolutionaryCap    float64
	BehavioralTrace    any
	FS                 float64
	MainFitness        float64
}

type CortexRecord struct {
	ID          any
	AgentID     any
	NeuronIDs   []any
	SensorIDs   []any
	ActuatorIDs []any
}

type SpecieRecord struct {
	ID                any
	PopulationID      any
	Fingerprint       any
	Constraint        any
	AllAgentIDs       []any
	AgentIDs          []any
	DeadPool          []any
	ChampionIDs       []any
	Fitness           any
	InnovationFactor  any
	Stats             []any
	SeedAgentIDs      []any
	HOFDistinguishers []any
	SpecieDistinguish []any
	HallOfFame        []any
}

type PopulationRecord struct {
	ID               any
	PolisID          any
	SpecieIDs        []any
	Morphologies     []any
	InnovationFactor any
	EvoAlgF          string
	FitnessPostprocF string
	SelectionF       string
	Trace            any
	SeedAgentIDs     []any
	SeedSpecieIDs    []any
}

func defaultConstraintRecord() ConstraintRecord {
	return ConstraintRecord{
		Morphology:             "xor_mimic",
		ConnectionArchitecture: "recurrent",
		NeuralAFs:              []string{"tanh", "cos", "gaussian"},
		NeuralPFNs:             []string{"none"},
		SubstratePlasticities:  []string{"none"},
		SubstrateLinkforms:     []string{"l2l_feedforward"},
		NeuralAggrFs:           []string{"dot_product"},
		TuningSelectionFs:      []string{"dynamic_random"},
		TuningDurationF:        DurationSpec{Name: "wsize_proportional", Param: 0.5},
		AnnealingParameters:    []float64{0.5},
		PerturbationRanges:     []float64{1},
		AgentEncodingTypes:     []string{"neural"},
		HeredityTypes:          []string{"darwinian"},
		MutationOperators: []WeightedOperator{
			{Name: "add_bias", Weight: 10},
			{Name: "add_outlink", Weight: 40},
			{Name: "add_inlink", Weight: 40},
			{Name: "add_neuron", Weight: 40},
			{Name: "outsplice", Weight: 40},
			{Name: "add_sensorlink", Weight: 1},
			{Name: "add_sensor", Weight: 1},
			{Name: "add_actuator", Weight: 1},
			{Name: "add_cpp", Weight: 1},
			{Name: "add_cep", Weight: 1},
		},
		TotTopologicalMutationsFs:   []MutationCountPolicy{{Name: "ncount_exponential", Param: 0.5}},
		PopulationEvoAlgF:           "generational",
		PopulationFitnessProcessorF: "size_proportional",
		PopulationSelectionF:        "hof_competition",
		SpecieDistinguishers:        []string{"tot_n"},
		HOFDistinguishers:           []string{"tot_n"},
		Objectives:                  []string{"main_fitness", "inverse_tot_n"},
	}
}

func defaultPMPRecord() PMPRecord {
	return PMPRecord{
		OpMode:             "gt",
		PopulationID:       "test",
		SurvivalPercentage: 0.5,
		SpecieSizeLimit:    10,
		InitSpecieSize:     20,
		PolisID:            "mathema",
		GenerationLimit:    100,
		EvaluationsLimit:   100000,
		FitnessGoal:        math.Inf(1),
	}
}

func defaultSensorRecord() SensorRecord {
	return SensorRecord{
		FanoutIDs: []any{},
	}
}

func defaultActuatorRecord() ActuatorRecord {
	return ActuatorRecord{
		FaninIDs: []any{},
	}
}

func defaultNeuronRecord() NeuronRecord {
	return NeuronRecord{
		InputIDPs:          []any{},
		InputIDPsMod:       []any{},
		OutputIDs:          []any{},
		RecurrentOutputIDs: []any{},
	}
}

func defaultAgentRecord() AgentRecord {
	return AgentRecord{
		EvoHist:           []any{},
		Pattern:           []any{},
		MutationOperators: []any{},
		OffspringIDs:      []any{},
		ParentIDs:         []any{},
		ChampionFlag:      []any{false},
		FS:                1,
	}
}

func defaultCortexRecord() CortexRecord {
	return CortexRecord{
		NeuronIDs:   []any{},
		SensorIDs:   []any{},
		ActuatorIDs: []any{},
	}
}

func defaultSpecieRecord() SpecieRecord {
	return SpecieRecord{
		AllAgentIDs:       []any{},
		AgentIDs:          []any{},
		DeadPool:          []any{},
		ChampionIDs:       []any{},
		Stats:             []any{},
		SeedAgentIDs:      []any{},
		HOFDistinguishers: []any{"tot_n"},
		SpecieDistinguish: []any{"tot_n"},
		HallOfFame:        []any{},
	}
}

func defaultPopulationRecord() PopulationRecord {
	return PopulationRecord{
		SpecieIDs:     []any{},
		Morphologies:  []any{},
		SeedAgentIDs:  []any{},
		SeedSpecieIDs: []any{},
	}
}
