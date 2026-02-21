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

type ExperimentRecord struct {
	ID              any
	BackupFlag      bool
	PMParameters    any
	InitConstraints any
	ProgressFlag    any
	TraceAcc        []any
	RunIndex        int
	TotalRuns       int
	Notes           any
	Started         any
	Completed       any
	Interruptions   []any
}

type CircuitRecord struct {
	ID             any
	Input          any
	OutputVectorL  any
	InputVectorL   any
	Training       any
	Output         any
	Parameters     any
	Dynamics       any
	Layers         any
	Type           any
	Noise          any
	NoiseType      any
	LPDecay        float64
	LPMin          float64
	LPMax          float64
	Memory         []any
	MemorySize     any
	Validation     any
	Testing        any
	ReceptiveField any
	Step           int
	BlockSize      int
	ErrAcc         float64
	BackpropTuning any
	TrainingLength int
}

type LayerSpecRecord struct {
	Type           any
	AF             any
	IVL            int
	Dynamics       any
	ReceptiveField any
	Step           int
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

type SubstrateRecord struct {
	ID         any
	AgentID    any
	Densities  any
	Linkform   any
	Plasticity any
	CPPIDs     []any
	CEPIDs     []any
}

type PolisRecord struct {
	ID            any
	ScapeIDs      []any
	PopulationIDs []any
	SpecieIDs     []any
	DXIDs         []any
	Parameters    []any
}

type ScapeRecord struct {
	ID             any
	Type           any
	Physics        any
	Metabolics     any
	Sector2Avatars any
	Avatars        []any
	Plants         []any
	Walls          []any
	Pillars        []any
	Laws           []any
	Anomolies      []any
	Artifacts      []any
	Objects        []any
	Elements       []any
	Atoms          []any
	Scheduler      int
}

type SectorRecord struct {
	ID             any
	Type           any
	ScapePID       any
	SectorSize     any
	Physics        any
	Metabolics     any
	Sector2Avatars any
	Avatars        []any
	Plants         []any
	Walls          []any
	Pillars        []any
	Laws           []any
	Anomolies      []any
	Artifacts      []any
	Objects        []any
	Elements       []any
	Atoms          []any
}

type AvatarRecord struct {
	ID         any
	Sector     any
	Morphology any
	Type       any
	Specie     any
	Energy     float64
	Health     float64
	Food       float64
	Age        float64
	Kills      float64
	Loc        any
	Direction  any
	R          any
	Mass       any
	Objects    any
	Vis        []any
	State      any
	Stats      any
	Actuators  any
	Sensors    any
	Sound      any
	Gestalt    any
	Spear      any
}

type ObjectRecord struct {
	ID         any
	Sector     any
	Type       any
	Color      any
	Loc        any
	Pivot      any
	Elements   []any
	Parameters []any
}

type CircleRecord struct {
	ID     any
	Sector any
	Color  any
	Loc    any
	Pivot  any
	R      any
}

type SquareRecord struct {
	ID     any
	Sector any
	Color  any
	Loc    any
	Pivot  any
	R      any
}

type LineRecord struct {
	ID     any
	Sector any
	Color  any
	Loc    any
	Pivot  any
	Coords any
}

type ERecord struct {
	ID     any
	Sector any
	VID    any
	Type   any
	Loc    any
	Pivot  any
}

type ARecord struct {
	ID         any
	Sector     any
	VID        any
	Type       any
	Loc        any
	Pivot      any
	Mass       any
	Properties any
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

type TraceRecord struct {
	Stats            []any
	TotalEvaluations int
	StepSize         int
}

type StatRecord struct {
	Morphology        any
	SpecieID          any
	AvgNeurons        float64
	StdNeurons        float64
	AvgFitness        float64
	StdFitness        float64
	MaxFitness        float64
	MinFitness        float64
	ValidationFitness float64
	TestFitness       float64
	AvgDiversity      float64
	Evaluations       int
	TimeStamp         any
}

type TopologySummaryRecord struct {
	Type           any
	TotalNeurons   int
	TotalNILs      int
	TotalNOLs      int
	TotalNROs      int
	AFDistribution any
}

type SignatureRecord struct {
	GeneralizedPattern   any
	GeneralizedEvoHist   any
	GeneralizedSensors   any
	GeneralizedActuators any
	TopologySummary      any
}

type ChampionRecord struct {
	HOFFingerprint        any
	ID                    any
	Fitness               float64
	ValidationFitness     float64
	TestFitness           float64
	MainFitness           float64
	TotalNeurons          int
	Evolvability          float64
	Robustness            float64
	Brittleness           float64
	Generation            int
	BehavioralDifferences any
	FS                    float64
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

func defaultExperimentRecord() ExperimentRecord {
	return ExperimentRecord{
		BackupFlag:    true,
		ProgressFlag:  "in_progress",
		TraceAcc:      []any{},
		RunIndex:      1,
		TotalRuns:     10,
		Interruptions: []any{},
	}
}

func defaultCircuitRecord() CircuitRecord {
	return CircuitRecord{
		Type:           "standard",
		NoiseType:      "zero_mask",
		LPDecay:        0.999999,
		LPMin:          0.0000001,
		LPMax:          0.1,
		Memory:         []any{},
		MemorySize:     []any{0, 100000},
		ReceptiveField: "full",
		Step:           0,
		BlockSize:      100,
		ErrAcc:         0,
		BackpropTuning: "off",
		TrainingLength: 1000,
	}
}

func defaultLayerSpecRecord() LayerSpecRecord {
	return LayerSpecRecord{}
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

func defaultSubstrateRecord() SubstrateRecord {
	return SubstrateRecord{
		Plasticity: "none",
		CPPIDs:     []any{},
		CEPIDs:     []any{},
	}
}

func defaultPolisRecord() PolisRecord {
	return PolisRecord{
		ScapeIDs:      []any{},
		PopulationIDs: []any{},
		SpecieIDs:     []any{},
		DXIDs:         []any{},
		Parameters:    []any{},
	}
}

func defaultScapeRecord() ScapeRecord {
	return ScapeRecord{
		Avatars:   []any{},
		Plants:    []any{},
		Walls:     []any{},
		Pillars:   []any{},
		Laws:      []any{},
		Anomolies: []any{},
		Artifacts: []any{},
		Objects:   []any{},
		Elements:  []any{},
		Atoms:     []any{},
		Scheduler: 0,
	}
}

func defaultSectorRecord() SectorRecord {
	return SectorRecord{
		Avatars:   []any{},
		Plants:    []any{},
		Walls:     []any{},
		Pillars:   []any{},
		Laws:      []any{},
		Anomolies: []any{},
		Artifacts: []any{},
		Objects:   []any{},
		Elements:  []any{},
		Atoms:     []any{},
	}
}

func defaultAvatarRecord() AvatarRecord {
	return AvatarRecord{
		Energy: 0,
		Health: 0,
		Food:   0,
		Age:    0,
		Kills:  0,
		Vis:    []any{},
	}
}

func defaultObjectRecord() ObjectRecord {
	return ObjectRecord{
		Elements:   []any{},
		Parameters: []any{},
	}
}

func defaultCircleRecord() CircleRecord {
	return CircleRecord{}
}

func defaultSquareRecord() SquareRecord {
	return SquareRecord{}
}

func defaultLineRecord() LineRecord {
	return LineRecord{}
}

func defaultERecord() ERecord {
	return ERecord{}
}

func defaultARecord() ARecord {
	return ARecord{}
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

func defaultTraceRecord() TraceRecord {
	return TraceRecord{
		Stats:            []any{},
		TotalEvaluations: 0,
		StepSize:         500,
	}
}

func defaultStatRecord() StatRecord {
	return StatRecord{}
}

func defaultTopologySummaryRecord() TopologySummaryRecord {
	return TopologySummaryRecord{}
}

func defaultSignatureRecord() SignatureRecord {
	return SignatureRecord{}
}

func defaultChampionRecord() ChampionRecord {
	return ChampionRecord{
		FS: 0,
	}
}
