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
