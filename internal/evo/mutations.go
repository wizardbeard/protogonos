package evo

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"

	"protogonos/internal/genotype"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/nn"
	"protogonos/internal/substrate"
	"protogonos/internal/tuning"
)

var (
	ErrNoSynapses = errors.New("genome has no synapses")
	ErrNoNeurons  = errors.New("genome has no neurons")
)

// ContextualOperator can declare whether it is applicable to a genome under a
// specific scape context. PopulationMonitor uses this to avoid selecting
// incompatible operators.
type ContextualOperator interface {
	Operator
	Applicable(genome model.Genome, scapeName string) bool
}

var (
	ErrSynapseExists   = errors.New("synapse already exists")
	ErrSynapseNotFound = errors.New("synapse not found")
	ErrNeuronExists    = errors.New("neuron already exists")
	ErrNeuronNotFound  = errors.New("neuron not found")
	ErrInvalidEndpoint = errors.New("invalid synapse endpoint")
)

// PerturbWeightAt mutates one synapse weight by a fixed delta.
type PerturbWeightAt struct {
	Index int
	Delta float64
}

func (o PerturbWeightAt) Name() string {
	return "perturb_weight_at"
}

func (o PerturbWeightAt) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	if o.Index < 0 || o.Index >= len(genome.Synapses) {
		return model.Genome{}, fmt.Errorf("synapse index out of range: %d", o.Index)
	}

	mutated := cloneGenome(genome)
	mutated.Synapses[o.Index].Weight += o.Delta
	return mutated, nil
}

// PerturbRandomWeight mutates a random synapse using uniform delta in [-MaxDelta, MaxDelta].
type PerturbRandomWeight struct {
	Rand     *rand.Rand
	MaxDelta float64
}

func (o *PerturbRandomWeight) Name() string {
	return "perturb_random_weight"
}

func (o *PerturbRandomWeight) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Synapses) > 0
}

func (o *PerturbRandomWeight) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if o.MaxDelta <= 0 {
		return model.Genome{}, errors.New("max delta must be > 0")
	}

	idx := o.Rand.Intn(len(genome.Synapses))
	delta := (o.Rand.Float64()*2 - 1) * o.MaxDelta

	mutated := cloneGenome(genome)
	mutated.Synapses[idx].Weight += delta
	return mutated, nil
}

// PerturbWeightsProportional mutates a random subset of synapses using the
// reference-style mutate probability 1/sqrt(total_weights). At least one
// synapse is always perturbed when synapses are present.
type PerturbWeightsProportional struct {
	Rand     *rand.Rand
	MaxDelta float64
}

func (o *PerturbWeightsProportional) Name() string {
	return "perturb_weights_proportional"
}

func (o *PerturbWeightsProportional) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Synapses) > 0
}

func (o *PerturbWeightsProportional) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if o.MaxDelta <= 0 {
		return model.Genome{}, errors.New("max delta must be > 0")
	}

	mutated := cloneGenome(genome)
	mp := 1 / math.Sqrt(float64(len(mutated.Synapses)))
	mutatedCount := 0
	for i := range mutated.Synapses {
		if o.Rand.Float64() >= mp {
			continue
		}
		delta := (o.Rand.Float64()*2 - 1) * o.MaxDelta
		mutated.Synapses[i].Weight += delta
		mutatedCount++
	}
	if mutatedCount == 0 {
		idx := o.Rand.Intn(len(mutated.Synapses))
		delta := (o.Rand.Float64()*2 - 1) * o.MaxDelta
		mutated.Synapses[idx].Weight += delta
	}
	return mutated, nil
}

// MutateWeights mirrors the reference mutate_weights operator name.
type MutateWeights struct {
	Rand     *rand.Rand
	MaxDelta float64
}

func (o *MutateWeights) Name() string {
	return "mutate_weights"
}

func (o *MutateWeights) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Synapses) > 0
}

func (o *MutateWeights) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&PerturbWeightsProportional{Rand: o.Rand, MaxDelta: o.MaxDelta}).Apply(ctx, genome)
}

// PerturbRandomBias mutates a random neuron bias using uniform delta in [-MaxDelta, MaxDelta].
type PerturbRandomBias struct {
	Rand     *rand.Rand
	MaxDelta float64
}

func (o *PerturbRandomBias) Name() string {
	return "perturb_random_bias"
}

func (o *PerturbRandomBias) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *PerturbRandomBias) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if o.MaxDelta <= 0 {
		return model.Genome{}, errors.New("max delta must be > 0")
	}

	idx := o.Rand.Intn(len(genome.Neurons))
	delta := (o.Rand.Float64()*2 - 1) * o.MaxDelta

	mutated := cloneGenome(genome)
	mutated.Neurons[idx].Bias += delta
	return mutated, nil
}

// AddBias mirrors the reference add_bias operator name.
type AddBias struct {
	Rand     *rand.Rand
	MaxDelta float64
}

func (o *AddBias) Name() string {
	return "add_bias"
}

func (o *AddBias) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *AddBias) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&PerturbRandomBias{Rand: o.Rand, MaxDelta: o.MaxDelta}).Apply(ctx, genome)
}

// RemoveRandomBias clears one random neuron bias.
type RemoveRandomBias struct {
	Rand *rand.Rand
}

func (o *RemoveRandomBias) Name() string {
	return "remove_random_bias"
}

func (o *RemoveRandomBias) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *RemoveRandomBias) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	idx := o.Rand.Intn(len(genome.Neurons))
	mutated := cloneGenome(genome)
	mutated.Neurons[idx].Bias = 0
	return mutated, nil
}

// RemoveBias mirrors the reference remove_bias operator name.
type RemoveBias struct {
	Rand *rand.Rand
}

func (o *RemoveBias) Name() string {
	return "remove_bias"
}

func (o *RemoveBias) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *RemoveBias) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&RemoveRandomBias{Rand: o.Rand}).Apply(ctx, genome)
}

// ChangeRandomActivation mutates one neuron's activation function.
type ChangeRandomActivation struct {
	Rand        *rand.Rand
	Activations []string
}

func (o *ChangeRandomActivation) Name() string {
	return "change_random_activation"
}

func (o *ChangeRandomActivation) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *ChangeRandomActivation) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	activations := o.Activations
	if len(activations) == 0 {
		activations = []string{"identity", "relu", "tanh", "sigmoid"}
	}

	idx := o.Rand.Intn(len(genome.Neurons))
	current := genome.Neurons[idx].Activation
	choices := make([]string, 0, len(activations))
	for _, name := range activations {
		if name != "" && name != current {
			choices = append(choices, name)
		}
	}
	if len(choices) == 0 {
		return cloneGenome(genome), nil
	}

	mutated := cloneGenome(genome)
	mutated.Neurons[idx].Activation = choices[o.Rand.Intn(len(choices))]
	return mutated, nil
}

// MutateAF mirrors the reference mutate_af operator name.
type MutateAF struct {
	Rand        *rand.Rand
	Activations []string
}

func (o *MutateAF) Name() string {
	return "mutate_af"
}

func (o *MutateAF) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *MutateAF) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&ChangeRandomActivation{Rand: o.Rand, Activations: o.Activations}).Apply(ctx, genome)
}

// ChangeRandomAggregator mutates one neuron's aggregation function.
type ChangeRandomAggregator struct {
	Rand        *rand.Rand
	Aggregators []string
}

func (o *ChangeRandomAggregator) Name() string {
	return "change_random_aggregator"
}

func (o *ChangeRandomAggregator) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *ChangeRandomAggregator) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	aggregators := o.Aggregators
	if len(aggregators) == 0 {
		aggregators = []string{"dot_product", "mult_product", "diff_product"}
	}

	idx := o.Rand.Intn(len(genome.Neurons))
	current := genome.Neurons[idx].Aggregator
	choices := make([]string, 0, len(aggregators))
	for _, name := range aggregators {
		if name != "" && name != current {
			choices = append(choices, name)
		}
	}
	if len(choices) == 0 {
		return cloneGenome(genome), nil
	}

	mutated := cloneGenome(genome)
	mutated.Neurons[idx].Aggregator = choices[o.Rand.Intn(len(choices))]
	return mutated, nil
}

// MutateAggrF mirrors the reference mutate_aggrf operator name.
type MutateAggrF struct {
	Rand        *rand.Rand
	Aggregators []string
}

func (o *MutateAggrF) Name() string {
	return "mutate_aggrf"
}

func (o *MutateAggrF) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *MutateAggrF) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&ChangeRandomAggregator{Rand: o.Rand, Aggregators: o.Aggregators}).Apply(ctx, genome)
}

// AddRandomSynapse adds a random synapse between existing neurons.
type AddRandomSynapse struct {
	Rand         *rand.Rand
	MaxAbsWeight float64
}

func (o *AddRandomSynapse) Name() string {
	return "add_random_synapse"
}

func (o *AddRandomSynapse) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *AddRandomSynapse) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	if o.MaxAbsWeight <= 0 {
		return model.Genome{}, errors.New("max abs weight must be > 0")
	}

	from := genome.Neurons[o.Rand.Intn(len(genome.Neurons))].ID
	to := genome.Neurons[o.Rand.Intn(len(genome.Neurons))].ID
	id := uniqueSynapseID(genome, o.Rand)
	weight := (o.Rand.Float64()*2 - 1) * o.MaxAbsWeight

	mutated := cloneGenome(genome)
	mutated.Synapses = append(mutated.Synapses, model.Synapse{
		ID:        id,
		From:      from,
		To:        to,
		Weight:    weight,
		Enabled:   true,
		Recurrent: from == to,
	})
	return mutated, nil
}

// AddRandomInlink adds a synapse biased toward input->non-input direction.
type AddRandomInlink struct {
	Rand           *rand.Rand
	MaxAbsWeight   float64
	InputNeuronIDs []string
}

func (o *AddRandomInlink) Name() string {
	return "add_inlink"
}

func (o *AddRandomInlink) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Neurons) <= 1 {
		return false
	}
	inputSet := toIDSet(o.InputNeuronIDs)
	fromCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := inputSet[id]
		return ok
	})
	toCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := inputSet[id]
		return !ok
	})
	return len(fromCandidates) > 0 && len(toCandidates) > 0
}

func (o *AddRandomInlink) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	if o.MaxAbsWeight <= 0 {
		return model.Genome{}, errors.New("max abs weight must be > 0")
	}

	inputSet := toIDSet(o.InputNeuronIDs)
	fromCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := inputSet[id]
		return ok
	})
	toCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := inputSet[id]
		return !ok
	})
	return addDirectedRandomSynapse(genome, o.Rand, o.MaxAbsWeight, fromCandidates, toCandidates)
}

// AddRandomOutlink adds a synapse biased toward non-output->output direction.
type AddRandomOutlink struct {
	Rand            *rand.Rand
	MaxAbsWeight    float64
	OutputNeuronIDs []string
}

func (o *AddRandomOutlink) Name() string {
	return "add_outlink"
}

func (o *AddRandomOutlink) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Neurons) <= 1 {
		return false
	}
	outputSet := toIDSet(o.OutputNeuronIDs)
	fromCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := outputSet[id]
		return !ok
	})
	toCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := outputSet[id]
		return ok
	})
	return len(fromCandidates) > 0 && len(toCandidates) > 0
}

func (o *AddRandomOutlink) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	if o.MaxAbsWeight <= 0 {
		return model.Genome{}, errors.New("max abs weight must be > 0")
	}

	outputSet := toIDSet(o.OutputNeuronIDs)
	fromCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := outputSet[id]
		return !ok
	})
	toCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := outputSet[id]
		return ok
	})
	return addDirectedRandomSynapse(genome, o.Rand, o.MaxAbsWeight, fromCandidates, toCandidates)
}

// RemoveRandomSynapse removes a random synapse.
type RemoveRandomSynapse struct {
	Rand *rand.Rand
}

func (o *RemoveRandomSynapse) Name() string {
	return "remove_random_synapse"
}

func (o *RemoveRandomSynapse) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Synapses) > 0
}

func (o *RemoveRandomSynapse) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}

	idx := o.Rand.Intn(len(genome.Synapses))
	mutated := cloneGenome(genome)
	mutated.Synapses = append(mutated.Synapses[:idx], mutated.Synapses[idx+1:]...)
	return mutated, nil
}

// RemoveRandomInlink removes a synapse biased toward input->non-input direction.
type RemoveRandomInlink struct {
	Rand           *rand.Rand
	InputNeuronIDs []string
}

func (o *RemoveRandomInlink) Name() string {
	return "remove_inlink"
}

func (o *RemoveRandomInlink) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Synapses) == 0 {
		return false
	}
	inputSet := toIDSet(o.InputNeuronIDs)
	for _, syn := range genome.Synapses {
		_, fromInput := inputSet[syn.From]
		_, toInput := inputSet[syn.To]
		if fromInput && !toInput {
			return true
		}
	}
	return false
}

func (o *RemoveRandomInlink) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	inputSet := toIDSet(o.InputNeuronIDs)
	return removeDirectedRandomSynapse(genome, o.Rand, func(s model.Synapse) bool {
		_, fromInput := inputSet[s.From]
		_, toInput := inputSet[s.To]
		return fromInput && !toInput
	})
}

// RemoveRandomOutlink removes a synapse biased toward non-output->output direction.
type RemoveRandomOutlink struct {
	Rand            *rand.Rand
	OutputNeuronIDs []string
}

func (o *RemoveRandomOutlink) Name() string {
	return "remove_outlink"
}

func (o *RemoveRandomOutlink) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Synapses) == 0 {
		return false
	}
	outputSet := toIDSet(o.OutputNeuronIDs)
	for _, syn := range genome.Synapses {
		_, fromOutput := outputSet[syn.From]
		_, toOutput := outputSet[syn.To]
		if !fromOutput && toOutput {
			return true
		}
	}
	return false
}

func (o *RemoveRandomOutlink) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	outputSet := toIDSet(o.OutputNeuronIDs)
	return removeDirectedRandomSynapse(genome, o.Rand, func(s model.Synapse) bool {
		_, fromOutput := outputSet[s.From]
		_, toOutput := outputSet[s.To]
		return !fromOutput && toOutput
	})
}

// CutlinkFromNeuronToNeuron mirrors the reference cutlink operator name for
// neuron-to-neuron links. In the simplified model, this delegates to random
// synapse removal.
type CutlinkFromNeuronToNeuron struct {
	Rand *rand.Rand
}

func (o *CutlinkFromNeuronToNeuron) Name() string {
	return "cutlink_FromNeuronToNeuron"
}

func (o *CutlinkFromNeuronToNeuron) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Synapses) > 0
}

func (o *CutlinkFromNeuronToNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&RemoveRandomSynapse{Rand: o.Rand}).Apply(ctx, genome)
}

// CutlinkFromElementToElement mirrors the generic reference helper mutator
// name used by genome_mutator for directional cutlink paths.
type CutlinkFromElementToElement struct {
	Rand *rand.Rand
}

func (o *CutlinkFromElementToElement) Name() string {
	return "cutlink_FromElementToElement"
}

func (o *CutlinkFromElementToElement) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Synapses) > 0
}

func (o *CutlinkFromElementToElement) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&RemoveRandomSynapse{Rand: o.Rand}).Apply(ctx, genome)
}

// LinkFromElementToElement mirrors the generic reference helper mutator name
// used by genome_mutator for directional add-link paths.
type LinkFromElementToElement struct {
	Rand         *rand.Rand
	MaxAbsWeight float64
}

func (o *LinkFromElementToElement) Name() string {
	return "link_FromElementToElement"
}

func (o *LinkFromElementToElement) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *LinkFromElementToElement) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&AddRandomSynapse{Rand: o.Rand, MaxAbsWeight: o.MaxAbsWeight}).Apply(ctx, genome)
}

// LinkFromNeuronToNeuron mirrors the explicit reference helper name used for
// neuron-to-neuron add-link paths.
type LinkFromNeuronToNeuron struct {
	Rand         *rand.Rand
	MaxAbsWeight float64
}

func (o *LinkFromNeuronToNeuron) Name() string {
	return "link_FromNeuronToNeuron"
}

func (o *LinkFromNeuronToNeuron) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Neurons) > 0
}

func (o *LinkFromNeuronToNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&AddRandomSynapse{Rand: o.Rand, MaxAbsWeight: o.MaxAbsWeight}).Apply(ctx, genome)
}

// LinkFromSensorToNeuron mirrors the explicit reference helper name used for
// sensor-to-neuron add-link paths.
type LinkFromSensorToNeuron struct {
	Rand      *rand.Rand
	ScapeName string
}

func (o *LinkFromSensorToNeuron) Name() string {
	return "link_FromSensorToNeuron"
}

func (o *LinkFromSensorToNeuron) Applicable(genome model.Genome, scapeName string) bool {
	return (&AddRandomSensorLink{Rand: o.Rand, ScapeName: scapeName}).Applicable(genome, scapeName)
}

func (o *LinkFromSensorToNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&AddRandomSensorLink{Rand: o.Rand, ScapeName: o.ScapeName}).Apply(ctx, genome)
}

// LinkFromNeuronToActuator mirrors the explicit reference helper name used for
// neuron-to-actuator add-link paths.
type LinkFromNeuronToActuator struct {
	Rand      *rand.Rand
	ScapeName string
}

func (o *LinkFromNeuronToActuator) Name() string {
	return "link_FromNeuronToActuator"
}

func (o *LinkFromNeuronToActuator) Applicable(genome model.Genome, scapeName string) bool {
	return (&AddRandomActuatorLink{Rand: o.Rand, ScapeName: scapeName}).Applicable(genome, scapeName)
}

func (o *LinkFromNeuronToActuator) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&AddRandomActuatorLink{Rand: o.Rand, ScapeName: o.ScapeName}).Apply(ctx, genome)
}

// AddRandomNeuron inserts a neuron by splitting a random synapse.
type AddRandomNeuron struct {
	Rand        *rand.Rand
	Activations []string
}

func (o *AddRandomNeuron) Name() string {
	return "add_random_neuron"
}

func (o *AddRandomNeuron) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Synapses) > 0
}

func (o *AddRandomNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return addRandomNeuronWithSynapseCandidates(ctx, genome, o.Rand, o.Activations, nil)
}

// AddNeuron mirrors the reference add_neuron operator name.
type AddNeuron struct {
	Rand        *rand.Rand
	Activations []string
}

func (o *AddNeuron) Name() string {
	return "add_neuron"
}

func (o *AddNeuron) Applicable(genome model.Genome, _ string) bool {
	return len(genome.Synapses) > 0
}

func (o *AddNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&AddRandomNeuron{Rand: o.Rand, Activations: o.Activations}).Apply(ctx, genome)
}

// AddRandomOutsplice inserts a neuron by splitting a synapse biased toward
// non-output->output direction.
type AddRandomOutsplice struct {
	Rand            *rand.Rand
	Activations     []string
	OutputNeuronIDs []string
}

func (o *AddRandomOutsplice) Name() string {
	return "outsplice"
}

func (o *AddRandomOutsplice) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Synapses) == 0 {
		return false
	}
	outputSet := toIDSet(o.OutputNeuronIDs)
	for _, syn := range genome.Synapses {
		_, fromOutput := outputSet[syn.From]
		_, toOutput := outputSet[syn.To]
		if !fromOutput && toOutput {
			return true
		}
	}
	return false
}

func (o *AddRandomOutsplice) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	outputSet := toIDSet(o.OutputNeuronIDs)
	return addRandomNeuronWithSynapseCandidates(ctx, genome, o.Rand, o.Activations, func(s model.Synapse) bool {
		_, fromOutput := outputSet[s.From]
		_, toOutput := outputSet[s.To]
		return !fromOutput && toOutput
	})
}

// AddRandomInsplice inserts a neuron by splitting a synapse biased toward
// input->non-input direction.
type AddRandomInsplice struct {
	Rand           *rand.Rand
	Activations    []string
	InputNeuronIDs []string
}

func (o *AddRandomInsplice) Name() string {
	return "insplice"
}

func (o *AddRandomInsplice) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Synapses) == 0 {
		return false
	}
	inputSet := toIDSet(o.InputNeuronIDs)
	for _, syn := range genome.Synapses {
		_, fromInput := inputSet[syn.From]
		_, toInput := inputSet[syn.To]
		if fromInput && !toInput {
			return true
		}
	}
	return false
}

func (o *AddRandomInsplice) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	inputSet := toIDSet(o.InputNeuronIDs)
	return addRandomNeuronWithSynapseCandidates(ctx, genome, o.Rand, o.Activations, func(s model.Synapse) bool {
		_, fromInput := inputSet[s.From]
		_, toInput := inputSet[s.To]
		return fromInput && !toInput
	})
}

func addRandomNeuronWithSynapseCandidates(
	ctx context.Context,
	genome model.Genome,
	rng *rand.Rand,
	activations []string,
	filter func(model.Synapse) bool,
) (model.Genome, error) {
	if rng == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}

	if len(activations) == 0 {
		activations = []string{"identity", "relu", "tanh", "sigmoid"}
	}

	candidates := make([]int, 0, len(genome.Synapses))
	for i, syn := range genome.Synapses {
		if filter == nil || filter(syn) {
			candidates = append(candidates, i)
		}
	}
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoSynapses
	}

	activation := activations[rng.Intn(len(activations))]
	op := AddNeuronAtSynapse{
		SynapseIndex: candidates[rng.Intn(len(candidates))],
		NeuronID:     uniqueNeuronID(genome, rng),
		Activation:   activation,
		Bias:         0,
	}
	return op.Apply(ctx, genome)
}

// RemoveRandomNeuron removes a random neuron, optionally skipping protected IDs.
type RemoveRandomNeuron struct {
	Rand      *rand.Rand
	Protected map[string]struct{}
}

func (o *RemoveRandomNeuron) Name() string {
	return "remove_random_neuron"
}

func (o *RemoveRandomNeuron) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Neurons) == 0 {
		return false
	}
	for _, neuron := range genome.Neurons {
		if _, protected := o.Protected[neuron.ID]; !protected {
			return true
		}
	}
	return false
}

func (o *RemoveRandomNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}

	candidates := make([]string, 0, len(genome.Neurons))
	for _, n := range genome.Neurons {
		if _, protected := o.Protected[n.ID]; protected {
			continue
		}
		candidates = append(candidates, n.ID)
	}
	if len(candidates) == 0 {
		return model.Genome{}, errors.New("no removable neurons")
	}

	target := candidates[o.Rand.Intn(len(candidates))]
	return RemoveNeuron{ID: target}.Apply(ctx, genome)
}

// RemoveNeuronMutation mirrors the reference remove_neuron operator name.
type RemoveNeuronMutation struct {
	Rand      *rand.Rand
	Protected map[string]struct{}
}

func (o *RemoveNeuronMutation) Name() string {
	return "remove_neuron"
}

func (o *RemoveNeuronMutation) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Neurons) == 0 {
		return false
	}
	for _, neuron := range genome.Neurons {
		if _, protected := o.Protected[neuron.ID]; !protected {
			return true
		}
	}
	return false
}

func (o *RemoveNeuronMutation) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&RemoveRandomNeuron{Rand: o.Rand, Protected: o.Protected}).Apply(ctx, genome)
}

// PerturbPlasticityRate mutates the plasticity learning rate when configured.
type PerturbPlasticityRate struct {
	Rand     *rand.Rand
	MaxDelta float64
}

func (o *PerturbPlasticityRate) Name() string {
	return "perturb_plasticity_rate"
}

func (o *PerturbPlasticityRate) Applicable(genome model.Genome, _ string) bool {
	return genome.Plasticity != nil
}

func (o *PerturbPlasticityRate) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if o.MaxDelta <= 0 {
		return model.Genome{}, errors.New("max delta must be > 0")
	}
	if genome.Plasticity == nil {
		return cloneGenome(genome), nil
	}
	mutated := cloneGenome(genome)
	delta := (o.Rand.Float64()*2 - 1) * o.MaxDelta
	mutated.Plasticity.Rate += delta
	if mutated.Plasticity.Rate < 0 {
		mutated.Plasticity.Rate = 0
	}
	return mutated, nil
}

// MutatePlasticityParameters mirrors mutate_plasticity_parameters.
type MutatePlasticityParameters struct {
	Rand     *rand.Rand
	MaxDelta float64
}

func (o *MutatePlasticityParameters) Name() string {
	return "mutate_plasticity_parameters"
}

func (o *MutatePlasticityParameters) Applicable(genome model.Genome, _ string) bool {
	return genome.Plasticity != nil
}

func (o *MutatePlasticityParameters) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&PerturbPlasticityRate{Rand: o.Rand, MaxDelta: o.MaxDelta}).Apply(ctx, genome)
}

// ChangePlasticityRule mutates the configured plasticity rule.
type ChangePlasticityRule struct {
	Rand  *rand.Rand
	Rules []string
}

func (o *ChangePlasticityRule) Name() string {
	return "change_plasticity_rule"
}

func (o *ChangePlasticityRule) Applicable(genome model.Genome, _ string) bool {
	return genome.Plasticity != nil
}

func (o *ChangePlasticityRule) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if genome.Plasticity == nil {
		return cloneGenome(genome), nil
	}
	rules := o.Rules
	if len(rules) == 0 {
		rules = []string{nn.PlasticityNone, nn.PlasticityHebbian, nn.PlasticityOja}
	}

	current := genome.Plasticity.Rule
	choices := make([]string, 0, len(rules))
	for _, rule := range rules {
		if rule == "" || rule == current {
			continue
		}
		choices = append(choices, rule)
	}
	if len(choices) == 0 {
		return cloneGenome(genome), nil
	}

	mutated := cloneGenome(genome)
	mutated.Plasticity.Rule = choices[o.Rand.Intn(len(choices))]
	return mutated, nil
}

// MutatePF mirrors mutate_pf in the reference source.
type MutatePF struct {
	Rand  *rand.Rand
	Rules []string
}

func (o *MutatePF) Name() string {
	return "mutate_pf"
}

func (o *MutatePF) Applicable(genome model.Genome, _ string) bool {
	return genome.Plasticity != nil
}

func (o *MutatePF) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	return (&ChangePlasticityRule{Rand: o.Rand, Rules: o.Rules}).Apply(ctx, genome)
}

// PerturbSubstrateParameter mutates one substrate parameter when configured.
type PerturbSubstrateParameter struct {
	Rand     *rand.Rand
	MaxDelta float64
	Keys     []string
}

func (o *PerturbSubstrateParameter) Name() string {
	return "perturb_substrate_parameter"
}

func (o *PerturbSubstrateParameter) Applicable(genome model.Genome, _ string) bool {
	return genome.Substrate != nil && len(genome.Substrate.Parameters) > 0
}

func (o *PerturbSubstrateParameter) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if o.MaxDelta <= 0 {
		return model.Genome{}, errors.New("max delta must be > 0")
	}
	if genome.Substrate == nil || len(genome.Substrate.Parameters) == 0 {
		return cloneGenome(genome), nil
	}

	keys := append([]string(nil), o.Keys...)
	if len(keys) == 0 {
		for key := range genome.Substrate.Parameters {
			keys = append(keys, key)
		}
	}
	filtered := make([]string, 0, len(keys))
	for _, key := range keys {
		if _, ok := genome.Substrate.Parameters[key]; ok {
			filtered = append(filtered, key)
		}
	}
	if len(filtered) == 0 {
		return cloneGenome(genome), nil
	}

	selected := filtered[o.Rand.Intn(len(filtered))]
	delta := (o.Rand.Float64()*2 - 1) * o.MaxDelta
	mutated := cloneGenome(genome)
	mutated.Substrate.Parameters[selected] += delta
	return mutated, nil
}

// MutateTuningSelection mirrors mutate_tuning_selection.
type MutateTuningSelection struct {
	Rand  *rand.Rand
	Modes []string
}

func (o *MutateTuningSelection) Name() string {
	return "mutate_tuning_selection"
}

func (o *MutateTuningSelection) Applicable(_ model.Genome, _ string) bool {
	return true
}

func (o *MutateTuningSelection) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	modes := append([]string(nil), o.Modes...)
	if len(modes) == 0 {
		modes = []string{
			tuning.CandidateSelectBestSoFar,
			tuning.CandidateSelectOriginal,
			tuning.CandidateSelectDynamicA,
			tuning.CandidateSelectDynamic,
			tuning.CandidateSelectAll,
			tuning.CandidateSelectAllRandom,
			tuning.CandidateSelectCurrent,
			tuning.CandidateSelectCurrentRd,
		}
	}
	for i := range modes {
		modes[i] = tuning.NormalizeCandidateSelectionName(modes[i])
	}
	mutated := cloneGenome(genome)
	ensureStrategyConfig(&mutated)
	current := tuning.NormalizeCandidateSelectionName(mutated.Strategy.TuningSelection)
	choices := filterOutString(modes, current)
	if len(choices) == 0 {
		return mutated, nil
	}
	mutated.Strategy.TuningSelection = choices[o.Rand.Intn(len(choices))]
	return mutated, nil
}

// MutateTuningAnnealing mirrors mutate_tuning_annealing.
type MutateTuningAnnealing struct {
	Rand   *rand.Rand
	Values []float64
}

func (o *MutateTuningAnnealing) Name() string {
	return "mutate_tuning_annealing"
}

func (o *MutateTuningAnnealing) Applicable(_ model.Genome, _ string) bool {
	return true
}

func (o *MutateTuningAnnealing) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	values := append([]float64(nil), o.Values...)
	if len(values) == 0 {
		values = []float64{0.5, 0.65, 0.8, 0.9, 0.95, 1.0}
	}
	mutated := cloneGenome(genome)
	ensureStrategyConfig(&mutated)
	current := mutated.Strategy.AnnealingFactor
	if current == 0 {
		current = 1.0
	}
	choices := make([]float64, 0, len(values))
	for _, value := range values {
		if value <= 0 {
			continue
		}
		if math.Abs(value-current) < 1e-9 {
			continue
		}
		choices = append(choices, value)
	}
	if len(choices) == 0 {
		return mutated, nil
	}
	mutated.Strategy.AnnealingFactor = choices[o.Rand.Intn(len(choices))]
	return mutated, nil
}

// MutateTotTopologicalMutations mirrors mutate_tot_topological_mutations.
type MutateTotTopologicalMutations struct {
	Rand     *rand.Rand
	Policies []string
}

func (o *MutateTotTopologicalMutations) Name() string {
	return "mutate_tot_topological_mutations"
}

func (o *MutateTotTopologicalMutations) Applicable(_ model.Genome, _ string) bool {
	return true
}

func (o *MutateTotTopologicalMutations) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	policies := append([]string(nil), o.Policies...)
	if len(policies) == 0 {
		policies = []string{"const", "ncount_linear", "ncount_exponential"}
	}
	mutated := cloneGenome(genome)
	ensureStrategyConfig(&mutated)
	current := mutated.Strategy.TopologicalMode
	if current == "" {
		current = "const"
	}
	choices := filterOutString(policies, current)
	if len(choices) == 0 {
		return mutated, nil
	}
	mutated.Strategy.TopologicalMode = choices[o.Rand.Intn(len(choices))]
	return mutated, nil
}

// MutateHeredityType mirrors mutate_heredity_type.
type MutateHeredityType struct {
	Rand  *rand.Rand
	Types []string
}

func (o *MutateHeredityType) Name() string {
	return "mutate_heredity_type"
}

func (o *MutateHeredityType) Applicable(_ model.Genome, _ string) bool {
	return true
}

func (o *MutateHeredityType) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	types := append([]string(nil), o.Types...)
	if len(types) == 0 {
		types = []string{"asexual", "crossover", "competition"}
	}
	mutated := cloneGenome(genome)
	ensureStrategyConfig(&mutated)
	current := mutated.Strategy.HeredityType
	if current == "" {
		current = "asexual"
	}
	choices := filterOutString(types, current)
	if len(choices) == 0 {
		return mutated, nil
	}
	mutated.Strategy.HeredityType = choices[o.Rand.Intn(len(choices))]
	return mutated, nil
}

// AddRandomSensor adds one compatible sensor id to genome.SensorIDs.
type AddRandomSensor struct {
	Rand      *rand.Rand
	ScapeName string
}

func (o *AddRandomSensor) Name() string {
	return "add_sensor"
}

func (o *AddRandomSensor) Applicable(genome model.Genome, _ string) bool {
	return len(sensorCandidates(genome, o.ScapeName)) > 0
}

func (o *AddRandomSensor) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	candidates := sensorCandidates(genome, o.ScapeName)
	if len(candidates) == 0 {
		return cloneGenome(genome), nil
	}
	choice := candidates[o.Rand.Intn(len(candidates))]
	mutated := cloneGenome(genome)
	mutated.SensorIDs = append(mutated.SensorIDs, choice)
	return mutated, nil
}

// AddRandomSensorLink mirrors add_sensorlink in the simplified genome model.
type AddRandomSensorLink struct {
	Rand      *rand.Rand
	ScapeName string
}

func (o *AddRandomSensorLink) Name() string {
	return "add_sensorlink"
}

func (o *AddRandomSensorLink) Applicable(genome model.Genome, _ string) bool {
	return len(genome.SensorIDs) > 0
}

func (o *AddRandomSensorLink) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.SensorIDs) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	mutated := cloneGenome(genome)
	mutated.SensorLinks++
	return mutated, nil
}

// AddRandomActuator adds one compatible actuator id to genome.ActuatorIDs.
type AddRandomActuator struct {
	Rand      *rand.Rand
	ScapeName string
}

func (o *AddRandomActuator) Name() string {
	return "add_actuator"
}

func (o *AddRandomActuator) Applicable(genome model.Genome, _ string) bool {
	return len(actuatorCandidates(genome, o.ScapeName)) > 0
}

func (o *AddRandomActuator) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	candidates := actuatorCandidates(genome, o.ScapeName)
	if len(candidates) == 0 {
		return cloneGenome(genome), nil
	}
	choice := candidates[o.Rand.Intn(len(candidates))]
	mutated := cloneGenome(genome)
	mutated.ActuatorIDs = append(mutated.ActuatorIDs, choice)
	return mutated, nil
}

// AddRandomActuatorLink mirrors add_actuatorlink in the simplified genome model.
type AddRandomActuatorLink struct {
	Rand      *rand.Rand
	ScapeName string
}

func (o *AddRandomActuatorLink) Name() string {
	return "add_actuatorlink"
}

func (o *AddRandomActuatorLink) Applicable(genome model.Genome, _ string) bool {
	return len(genome.ActuatorIDs) > 0
}

func (o *AddRandomActuatorLink) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.ActuatorIDs) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	mutated := cloneGenome(genome)
	mutated.ActuatorLinks++
	return mutated, nil
}

// RemoveRandomSensor removes one sensor id from genome.SensorIDs.
type RemoveRandomSensor struct {
	Rand *rand.Rand
}

func (o *RemoveRandomSensor) Name() string {
	return "remove_sensor"
}

func (o *RemoveRandomSensor) Applicable(genome model.Genome, _ string) bool {
	return len(genome.SensorIDs) > 0
}

func (o *RemoveRandomSensor) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.SensorIDs) == 0 {
		return cloneGenome(genome), nil
	}
	selected := genome.SensorIDs[o.Rand.Intn(len(genome.SensorIDs))]
	mutated := cloneGenome(genome)
	filtered := mutated.SensorIDs[:0]
	for _, id := range mutated.SensorIDs {
		if id == selected {
			continue
		}
		filtered = append(filtered, id)
	}
	mutated.SensorIDs = filtered
	mutated.SensorLinks = 0
	return mutated, nil
}

// CutlinkFromSensorToNeuron mirrors the reference cutlink operator name.
// In the simplified genome model, sensor links are represented by membership
// in SensorIDs, so this delegates to RemoveRandomSensor.
type CutlinkFromSensorToNeuron struct {
	Rand *rand.Rand
}

func (o *CutlinkFromSensorToNeuron) Name() string {
	return "cutlink_FromSensorToNeuron"
}

func (o *CutlinkFromSensorToNeuron) Applicable(genome model.Genome, _ string) bool {
	return genome.SensorLinks > 0
}

func (o *CutlinkFromSensorToNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if genome.SensorLinks <= 0 {
		return model.Genome{}, ErrNoSynapses
	}
	mutated := cloneGenome(genome)
	mutated.SensorLinks--
	return mutated, nil
}

// RemoveRandomActuator removes one actuator id from genome.ActuatorIDs.
type RemoveRandomActuator struct {
	Rand *rand.Rand
}

func (o *RemoveRandomActuator) Name() string {
	return "remove_actuator"
}

func (o *RemoveRandomActuator) Applicable(genome model.Genome, _ string) bool {
	return len(genome.ActuatorIDs) > 0
}

func (o *RemoveRandomActuator) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.ActuatorIDs) == 0 {
		return cloneGenome(genome), nil
	}
	selected := genome.ActuatorIDs[o.Rand.Intn(len(genome.ActuatorIDs))]
	mutated := cloneGenome(genome)
	filtered := mutated.ActuatorIDs[:0]
	for _, id := range mutated.ActuatorIDs {
		if id == selected {
			continue
		}
		filtered = append(filtered, id)
	}
	mutated.ActuatorIDs = filtered
	mutated.ActuatorLinks = 0
	return mutated, nil
}

// CutlinkFromNeuronToActuator mirrors the reference cutlink operator name.
// In the simplified genome model, actuator links are represented by membership
// in ActuatorIDs, so this delegates to RemoveRandomActuator.
type CutlinkFromNeuronToActuator struct {
	Rand *rand.Rand
}

func (o *CutlinkFromNeuronToActuator) Name() string {
	return "cutlink_FromNeuronToActuator"
}

func (o *CutlinkFromNeuronToActuator) Applicable(genome model.Genome, _ string) bool {
	return genome.ActuatorLinks > 0
}

func (o *CutlinkFromNeuronToActuator) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if genome.ActuatorLinks <= 0 {
		return model.Genome{}, ErrNoSynapses
	}
	mutated := cloneGenome(genome)
	mutated.ActuatorLinks--
	return mutated, nil
}

// AddRandomCPP mutates substrate CPP selection from the registered CPP set.
type AddRandomCPP struct {
	Rand *rand.Rand
}

func (o *AddRandomCPP) Name() string {
	return "add_cpp"
}

func (o *AddRandomCPP) Applicable(genome model.Genome, _ string) bool {
	return genome.Substrate != nil && len(substrate.ListCPPs()) > 0
}

func (o *AddRandomCPP) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if genome.Substrate == nil {
		return cloneGenome(genome), nil
	}
	candidates := substrate.ListCPPs()
	if len(candidates) == 0 {
		return cloneGenome(genome), nil
	}

	current := ""
	if genome.Substrate != nil {
		current = genome.Substrate.CPPName
	}
	choices := filterOutString(candidates, current)
	if len(choices) == 0 {
		choices = candidates
	}
	selected := choices[o.Rand.Intn(len(choices))]

	mutated := cloneGenome(genome)
	mutated.Substrate.CPPName = selected
	if mutated.Substrate.CEPName == "" {
		mutated.Substrate.CEPName = substrate.DefaultCEPName
	}
	if mutated.Substrate.Parameters == nil {
		mutated.Substrate.Parameters = map[string]float64{}
	}
	return mutated, nil
}

// AddRandomCEP mutates substrate CEP selection from the registered CEP set.
type AddRandomCEP struct {
	Rand *rand.Rand
}

func (o *AddRandomCEP) Name() string {
	return "add_cep"
}

func (o *AddRandomCEP) Applicable(genome model.Genome, _ string) bool {
	return genome.Substrate != nil && len(substrate.ListCEPs()) > 0
}

func (o *AddRandomCEP) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if genome.Substrate == nil {
		return cloneGenome(genome), nil
	}
	candidates := substrate.ListCEPs()
	if len(candidates) == 0 {
		return cloneGenome(genome), nil
	}

	current := ""
	if genome.Substrate != nil {
		current = genome.Substrate.CEPName
	}
	choices := filterOutString(candidates, current)
	if len(choices) == 0 {
		choices = candidates
	}
	selected := choices[o.Rand.Intn(len(choices))]

	mutated := cloneGenome(genome)
	mutated.Substrate.CEPName = selected
	if mutated.Substrate.CPPName == "" {
		mutated.Substrate.CPPName = substrate.DefaultCPPName
	}
	if mutated.Substrate.Parameters == nil {
		mutated.Substrate.Parameters = map[string]float64{}
	}
	return mutated, nil
}

// RemoveRandomCPP clears substrate CPP selection.
type RemoveRandomCPP struct{}

func (o *RemoveRandomCPP) Name() string {
	return "remove_cpp"
}

func (o *RemoveRandomCPP) Applicable(genome model.Genome, _ string) bool {
	return genome.Substrate != nil && genome.Substrate.CPPName != ""
}

func (o *RemoveRandomCPP) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if genome.Substrate == nil {
		return cloneGenome(genome), nil
	}
	mutated := cloneGenome(genome)
	mutated.Substrate.CPPName = ""
	return mutated, nil
}

// RemoveRandomCEP clears substrate CEP selection.
type RemoveRandomCEP struct{}

func (o *RemoveRandomCEP) Name() string {
	return "remove_cep"
}

func (o *RemoveRandomCEP) Applicable(genome model.Genome, _ string) bool {
	return genome.Substrate != nil && genome.Substrate.CEPName != ""
}

func (o *RemoveRandomCEP) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if genome.Substrate == nil {
		return cloneGenome(genome), nil
	}
	mutated := cloneGenome(genome)
	mutated.Substrate.CEPName = ""
	return mutated, nil
}

// AddCircuitNode mutates substrate dimensions by adding one node to a random layer.
type AddCircuitNode struct {
	Rand *rand.Rand
}

func (o *AddCircuitNode) Name() string {
	return "add_circuit_node"
}

func (o *AddCircuitNode) Applicable(genome model.Genome, _ string) bool {
	return genome.Substrate != nil && len(genome.Substrate.Dimensions) > 0
}

func (o *AddCircuitNode) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if genome.Substrate == nil {
		return cloneGenome(genome), nil
	}
	mutated := cloneGenome(genome)
	if len(mutated.Substrate.Dimensions) == 0 {
		return mutated, nil
	}
	idx := o.Rand.Intn(len(mutated.Substrate.Dimensions))
	if mutated.Substrate.Dimensions[idx] < 1 {
		mutated.Substrate.Dimensions[idx] = 1
	}
	mutated.Substrate.Dimensions[idx]++
	return mutated, nil
}

// DeleteCircuitNode mutates substrate dimensions by removing one node from a
// random layer where width > 1.
type DeleteCircuitNode struct {
	Rand *rand.Rand
}

func (o *DeleteCircuitNode) Name() string {
	return "delete_circuit_node"
}

func (o *DeleteCircuitNode) Applicable(genome model.Genome, _ string) bool {
	if genome.Substrate == nil || len(genome.Substrate.Dimensions) == 0 {
		return false
	}
	for _, width := range genome.Substrate.Dimensions {
		if width > 1 {
			return true
		}
	}
	return false
}

func (o *DeleteCircuitNode) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if genome.Substrate == nil || len(genome.Substrate.Dimensions) == 0 {
		return cloneGenome(genome), nil
	}
	candidates := make([]int, 0, len(genome.Substrate.Dimensions))
	for i, width := range genome.Substrate.Dimensions {
		if width > 1 {
			candidates = append(candidates, i)
		}
	}
	if len(candidates) == 0 {
		return cloneGenome(genome), nil
	}
	mutated := cloneGenome(genome)
	idx := candidates[o.Rand.Intn(len(candidates))]
	mutated.Substrate.Dimensions[idx]--
	return mutated, nil
}

// AddCircuitLayer mutates substrate dimensions by inserting a new layer.
type AddCircuitLayer struct {
	Rand *rand.Rand
}

func (o *AddCircuitLayer) Name() string {
	return "add_circuit_layer"
}

func (o *AddCircuitLayer) Applicable(genome model.Genome, _ string) bool {
	return genome.Substrate != nil && len(genome.Substrate.Dimensions) > 0
}

func (o *AddCircuitLayer) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if genome.Substrate == nil {
		return cloneGenome(genome), nil
	}
	mutated := cloneGenome(genome)
	dims := append([]int(nil), mutated.Substrate.Dimensions...)
	if len(dims) == 0 {
		return mutated, nil
	}
	if len(dims) == 1 {
		mutated.Substrate.Dimensions = []int{dims[0], 1}
		return mutated, nil
	}
	insertAt := len(dims) - 1
	updated := make([]int, 0, len(dims)+1)
	updated = append(updated, dims[:insertAt]...)
	updated = append(updated, 1)
	updated = append(updated, dims[insertAt:]...)
	mutated.Substrate.Dimensions = updated
	return mutated, nil
}

// ChangeActivationAt mutates one neuron's activation function label.
type ChangeActivationAt struct {
	Index      int
	Activation string
}

func (o ChangeActivationAt) Name() string {
	return "change_activation_at"
}

func (o ChangeActivationAt) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	if o.Index < 0 || o.Index >= len(genome.Neurons) {
		return model.Genome{}, fmt.Errorf("neuron index out of range: %d", o.Index)
	}
	if o.Activation == "" {
		return model.Genome{}, errors.New("activation is required")
	}

	mutated := cloneGenome(genome)
	mutated.Neurons[o.Index].Activation = o.Activation
	return mutated, nil
}

// AddSynapse inserts a synapse connecting existing neurons.
type AddSynapse struct {
	ID      string
	From    string
	To      string
	Weight  float64
	Enabled bool
}

func (o AddSynapse) Name() string {
	return "add_synapse"
}

func (o AddSynapse) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o.ID == "" {
		return model.Genome{}, errors.New("synapse id is required")
	}
	if !hasNeuron(genome, o.From) || !hasNeuron(genome, o.To) {
		return model.Genome{}, ErrInvalidEndpoint
	}
	if hasSynapse(genome, o.ID) {
		return model.Genome{}, fmt.Errorf("%w: %s", ErrSynapseExists, o.ID)
	}

	mutated := cloneGenome(genome)
	mutated.Synapses = append(mutated.Synapses, model.Synapse{
		ID:        o.ID,
		From:      o.From,
		To:        o.To,
		Weight:    o.Weight,
		Enabled:   o.Enabled,
		Recurrent: o.From == o.To,
	})
	return mutated, nil
}

// RemoveSynapse removes one synapse by id.
type RemoveSynapse struct {
	ID string
}

func (o RemoveSynapse) Name() string {
	return "remove_synapse"
}

func (o RemoveSynapse) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o.ID == "" {
		return model.Genome{}, errors.New("synapse id is required")
	}

	mutated := cloneGenome(genome)
	idx := -1
	for i := range mutated.Synapses {
		if mutated.Synapses[i].ID == o.ID {
			idx = i
			break
		}
	}
	if idx < 0 {
		return model.Genome{}, fmt.Errorf("%w: %s", ErrSynapseNotFound, o.ID)
	}
	mutated.Synapses = append(mutated.Synapses[:idx], mutated.Synapses[idx+1:]...)
	return mutated, nil
}

// AddNeuronAtSynapse splits one synapse with a new hidden neuron.
type AddNeuronAtSynapse struct {
	SynapseIndex int
	NeuronID     string
	Activation   string
	Bias         float64
}

func (o AddNeuronAtSynapse) Name() string {
	return "add_neuron"
}

func (o AddNeuronAtSynapse) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	if o.SynapseIndex < 0 || o.SynapseIndex >= len(genome.Synapses) {
		return model.Genome{}, fmt.Errorf("synapse index out of range: %d", o.SynapseIndex)
	}
	if o.NeuronID == "" {
		return model.Genome{}, errors.New("neuron id is required")
	}
	if o.Activation == "" {
		return model.Genome{}, errors.New("activation is required")
	}
	if hasNeuron(genome, o.NeuronID) {
		return model.Genome{}, fmt.Errorf("%w: %s", ErrNeuronExists, o.NeuronID)
	}

	mutated := cloneGenome(genome)
	target := mutated.Synapses[o.SynapseIndex]
	mutated.Synapses = append(mutated.Synapses[:o.SynapseIndex], mutated.Synapses[o.SynapseIndex+1:]...)

	mutated.Neurons = append(mutated.Neurons, model.Neuron{
		ID:         o.NeuronID,
		Activation: o.Activation,
		Bias:       o.Bias,
	})
	mutated.Synapses = append(mutated.Synapses,
		model.Synapse{
			ID:        target.ID + "a",
			From:      target.From,
			To:        o.NeuronID,
			Weight:    1.0,
			Enabled:   target.Enabled,
			Recurrent: target.From == o.NeuronID,
		},
		model.Synapse{
			ID:        target.ID + "b",
			From:      o.NeuronID,
			To:        target.To,
			Weight:    target.Weight,
			Enabled:   target.Enabled,
			Recurrent: o.NeuronID == target.To,
		},
	)
	return mutated, nil
}

// RemoveNeuron removes a neuron and all incident synapses.
type RemoveNeuron struct {
	ID string
}

func (o RemoveNeuron) Name() string {
	return "remove_neuron"
}

func (o RemoveNeuron) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o.ID == "" {
		return model.Genome{}, errors.New("neuron id is required")
	}

	mutated := cloneGenome(genome)
	neuronIdx := -1
	for i := range mutated.Neurons {
		if mutated.Neurons[i].ID == o.ID {
			neuronIdx = i
			break
		}
	}
	if neuronIdx < 0 {
		return model.Genome{}, fmt.Errorf("%w: %s", ErrNeuronNotFound, o.ID)
	}

	mutated.Neurons = append(mutated.Neurons[:neuronIdx], mutated.Neurons[neuronIdx+1:]...)
	filtered := mutated.Synapses[:0]
	for _, s := range mutated.Synapses {
		if s.From == o.ID || s.To == o.ID {
			continue
		}
		filtered = append(filtered, s)
	}
	mutated.Synapses = filtered
	return mutated, nil
}

func cloneGenome(g model.Genome) model.Genome {
	return genotype.CloneGenome(g)
}

func hasNeuron(g model.Genome, id string) bool {
	for _, n := range g.Neurons {
		if n.ID == id {
			return true
		}
	}
	return false
}

func hasSynapse(g model.Genome, id string) bool {
	for _, s := range g.Synapses {
		if s.ID == id {
			return true
		}
	}
	return false
}

func uniqueSynapseID(g model.Genome, rng *rand.Rand) string {
	for {
		candidate := fmt.Sprintf("srand-%d", rng.Int63())
		if !hasSynapse(g, candidate) {
			return candidate
		}
	}
}

func uniqueNeuronID(g model.Genome, rng *rand.Rand) string {
	for {
		candidate := fmt.Sprintf("nrand-%d", rng.Int63())
		if !hasNeuron(g, candidate) {
			return candidate
		}
	}
}

func toIDSet(ids []string) map[string]struct{} {
	out := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		if id == "" {
			continue
		}
		out[id] = struct{}{}
	}
	return out
}

func ensureStrategyConfig(g *model.Genome) {
	if g == nil {
		return
	}
	if g.Strategy == nil {
		g.Strategy = &model.StrategyConfig{
			TuningSelection: tuning.CandidateSelectBestSoFar,
			AnnealingFactor: 1.0,
			TopologicalMode: "const",
			HeredityType:    "asexual",
		}
		return
	}
	if g.Strategy.TuningSelection == "" {
		g.Strategy.TuningSelection = tuning.CandidateSelectBestSoFar
	}
	if g.Strategy.AnnealingFactor == 0 {
		g.Strategy.AnnealingFactor = 1.0
	}
	if g.Strategy.TopologicalMode == "" {
		g.Strategy.TopologicalMode = "const"
	}
	if g.Strategy.HeredityType == "" {
		g.Strategy.HeredityType = "asexual"
	}
}

func filterNeuronIDs(g model.Genome, keep func(id string) bool) []string {
	out := make([]string, 0, len(g.Neurons))
	for _, n := range g.Neurons {
		if keep == nil || keep(n.ID) {
			out = append(out, n.ID)
		}
	}
	return out
}

func addDirectedRandomSynapse(genome model.Genome, rng *rand.Rand, maxAbsWeight float64, fromCandidates, toCandidates []string) (model.Genome, error) {
	if len(fromCandidates) == 0 || len(toCandidates) == 0 {
		return model.Genome{}, ErrNoSynapses
	}

	from := fromCandidates[rng.Intn(len(fromCandidates))]
	to := toCandidates[rng.Intn(len(toCandidates))]
	id := uniqueSynapseID(genome, rng)
	weight := (rng.Float64()*2 - 1) * maxAbsWeight

	mutated := cloneGenome(genome)
	mutated.Synapses = append(mutated.Synapses, model.Synapse{
		ID:        id,
		From:      from,
		To:        to,
		Weight:    weight,
		Enabled:   true,
		Recurrent: from == to,
	})
	return mutated, nil
}

func removeDirectedRandomSynapse(genome model.Genome, rng *rand.Rand, keep func(s model.Synapse) bool) (model.Genome, error) {
	candidates := make([]int, 0, len(genome.Synapses))
	for i, syn := range genome.Synapses {
		if keep == nil || keep(syn) {
			candidates = append(candidates, i)
		}
	}
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoSynapses
	}
	idx := candidates[rng.Intn(len(candidates))]
	mutated := cloneGenome(genome)
	mutated.Synapses = append(mutated.Synapses[:idx], mutated.Synapses[idx+1:]...)
	return mutated, nil
}

func sensorCandidates(genome model.Genome, scapeName string) []string {
	existing := toIDSet(genome.SensorIDs)
	candidates := make([]string, 0)
	for _, name := range protoio.ListSensors() {
		if _, ok := existing[name]; ok {
			continue
		}
		if _, err := protoio.ResolveSensor(name, scapeName); err != nil {
			continue
		}
		candidates = append(candidates, name)
	}
	return candidates
}

func actuatorCandidates(genome model.Genome, scapeName string) []string {
	existing := toIDSet(genome.ActuatorIDs)
	candidates := make([]string, 0)
	for _, name := range protoio.ListActuators() {
		if _, ok := existing[name]; ok {
			continue
		}
		if _, err := protoio.ResolveActuator(name, scapeName); err != nil {
			continue
		}
		candidates = append(candidates, name)
	}
	return candidates
}

func filterOutString(values []string, drop string) []string {
	out := make([]string, 0, len(values))
	for _, item := range values {
		if item == drop {
			continue
		}
		out = append(out, item)
	}
	return out
}

func ensureSubstrateConfig(genome *model.Genome) {
	if genome.Substrate != nil {
		if genome.Substrate.CPPName == "" {
			genome.Substrate.CPPName = substrate.DefaultCPPName
		}
		if genome.Substrate.CEPName == "" {
			genome.Substrate.CEPName = substrate.DefaultCEPName
		}
		if genome.Substrate.Parameters == nil {
			genome.Substrate.Parameters = map[string]float64{}
		}
		return
	}
	genome.Substrate = &model.SubstrateConfig{
		CPPName:    substrate.DefaultCPPName,
		CEPName:    substrate.DefaultCEPName,
		Dimensions: []int{1, 1},
		Parameters: map[string]float64{},
	}
}
