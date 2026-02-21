package evo

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"

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
	ErrSynapseExists    = errors.New("synapse already exists")
	ErrSynapseNotFound  = errors.New("synapse not found")
	ErrNeuronExists     = errors.New("neuron already exists")
	ErrNeuronNotFound   = errors.New("neuron not found")
	ErrInvalidEndpoint  = errors.New("invalid synapse endpoint")
	ErrNoMutationChoice = errors.New("no mutation choice available")
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
		return model.Genome{}, ErrNoMutationChoice
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
		return model.Genome{}, ErrNoMutationChoice
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
	return len(genome.Synapses) > 0 || len(genome.ActuatorIDs) > 0
}

func (o *MutateWeights) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.Synapses) == 0 && len(genome.ActuatorIDs) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if o.MaxDelta <= 0 {
		return model.Genome{}, errors.New("max delta must be > 0")
	}

	mutated := cloneGenome(genome)
	ensureStrategyConfig(&mutated)
	selectedNeuronSpreads := selectedNeuronSpreadsForMutateWeights(
		mutated,
		o.Rand,
		o.MaxDelta,
		mutated.Strategy.AnnealingFactor,
	)
	if len(selectedNeuronSpreads) == 0 {
		return mutated, nil
	}

	changed := 0
	candidateFallback := make([]int, 0, len(mutated.Synapses))
	currentGeneration := currentGenomeGeneration(mutated)
	for _, target := range selectedNeuronSpreads {
		if target.sourceKind == tuningElementActuator {
			if perturbActuatorTunable(&mutated, target.sourceID, target.spread, o.Rand) {
				changed++
				touchActuatorGeneration(&mutated, target.sourceID, currentGeneration)
			}
			continue
		}
		neuronID := target.id
		incoming := incomingSynapseIndexes(mutated, neuronID)
		if len(incoming) == 0 {
			continue
		}
		candidateFallback = append(candidateFallback, incoming...)
		spread := target.spread
		if spread <= 0 {
			continue
		}
		mp := 1 / math.Sqrt(float64(len(incoming)))
		mutatedLocal := 0
		for _, idx := range incoming {
			if o.Rand.Float64() >= mp {
				continue
			}
			delta := (o.Rand.Float64()*2 - 1) * spread
			mutated.Synapses[idx].Weight += delta
			mutatedLocal++
			changed++
		}
		if mutatedLocal == 0 {
			idx := incoming[o.Rand.Intn(len(incoming))]
			delta := (o.Rand.Float64()*2 - 1) * spread
			mutated.Synapses[idx].Weight += delta
			changed++
		}
		touchNeuronGeneration(mutated.Neurons, neuronID, currentGeneration)
	}

	if changed == 0 {
		if len(mutated.Synapses) == 0 {
			return model.Genome{}, ErrNoMutationChoice
		}
		idx := 0
		if len(candidateFallback) > 0 {
			idx = candidateFallback[o.Rand.Intn(len(candidateFallback))]
		} else {
			idx = o.Rand.Intn(len(mutated.Synapses))
		}
		delta := (o.Rand.Float64()*2 - 1) * o.MaxDelta
		mutated.Synapses[idx].Weight += delta
	}
	return mutated, nil
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
	mutated.Neurons[idx].Generation = currentGenomeGeneration(mutated)
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
	mutated.Neurons[idx].Generation = currentGenomeGeneration(mutated)
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
		return model.Genome{}, ErrNoMutationChoice
	}

	mutated := cloneGenome(genome)
	mutated.Neurons[idx].Activation = choices[o.Rand.Intn(len(choices))]
	mutated.Neurons[idx].Generation = currentGenomeGeneration(mutated)
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
	if len(genome.Neurons) == 0 {
		return false
	}
	activations := append([]string(nil), o.Activations...)
	if len(activations) == 0 {
		activations = []string{"identity", "relu", "tanh", "sigmoid"}
	}
	options := normalizeNonEmptyStrings(activations)
	if len(options) == 0 {
		return false
	}
	for _, neuron := range genome.Neurons {
		for _, option := range options {
			if option != neuron.Activation {
				return true
			}
		}
	}
	return false
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
		return model.Genome{}, ErrNoMutationChoice
	}

	mutated := cloneGenome(genome)
	mutated.Neurons[idx].Aggregator = choices[o.Rand.Intn(len(choices))]
	mutated.Neurons[idx].Generation = currentGenomeGeneration(mutated)
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
	if len(genome.Neurons) == 0 {
		return false
	}
	aggregators := append([]string(nil), o.Aggregators...)
	if len(aggregators) == 0 {
		aggregators = []string{"dot_product", "mult_product", "diff_product"}
	}
	options := normalizeNonEmptyStrings(aggregators)
	if len(options) == 0 {
		return false
	}
	for _, neuron := range genome.Neurons {
		for _, option := range options {
			if option != neuron.Aggregator {
				return true
			}
		}
	}
	return false
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

	type pair struct {
		from string
		to   string
	}
	candidates := make([]pair, 0, len(genome.Neurons)*len(genome.Neurons))
	for _, from := range genome.Neurons {
		for _, to := range genome.Neurons {
			if hasDirectedSynapse(genome, from.ID, to.ID) {
				continue
			}
			candidates = append(candidates, pair{from: from.ID, to: to.ID})
		}
	}
	if len(candidates) == 0 {
		return model.Genome{}, ErrSynapseExists
	}
	selected := candidates[o.Rand.Intn(len(candidates))]
	id := uniqueSynapseID(genome, o.Rand)
	weight := (o.Rand.Float64()*2 - 1) * o.MaxAbsWeight

	mutated := cloneGenome(genome)
	mutated.Synapses = append(mutated.Synapses, model.Synapse{
		ID:        id,
		From:      selected.from,
		To:        selected.to,
		Weight:    weight,
		Enabled:   true,
		Recurrent: selected.from == selected.to,
	})
	return mutated, nil
}

// AddRandomInlink adds a synapse biased toward input->non-input direction.
type AddRandomInlink struct {
	Rand            *rand.Rand
	MaxAbsWeight    float64
	InputNeuronIDs  []string
	FeedForwardOnly bool
}

func (o *AddRandomInlink) Name() string {
	return "add_inlink"
}

func (o *AddRandomInlink) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Neurons) == 0 {
		return false
	}
	inputSet := toIDSet(o.InputNeuronIDs)
	layers := inferFeedforwardLayers(genome, o.InputNeuronIDs, nil)
	fromCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := inputSet[id]
		return ok
	})
	toCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := inputSet[id]
		return !ok
	})
	if o.FeedForwardOnly {
		fromCandidates, toCandidates = filterDirectedFeedforwardCandidates(fromCandidates, toCandidates, layers)
	}
	return len(availableInlinkNeuronPairs(genome, fromCandidates, toCandidates)) > 0 ||
		len(availableSensorToNeuronPairs(genome, toCandidates)) > 0
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
	layers := inferFeedforwardLayers(genome, o.InputNeuronIDs, nil)
	fromCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := inputSet[id]
		return ok
	})
	toCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := inputSet[id]
		return !ok
	})
	if o.FeedForwardOnly {
		fromCandidates, toCandidates = filterDirectedFeedforwardCandidates(fromCandidates, toCandidates, layers)
	}
	neuronPairs := availableInlinkNeuronPairs(genome, fromCandidates, toCandidates)
	sensorPairs := availableSensorToNeuronPairs(genome, toCandidates)
	totalCandidates := len(neuronPairs) + len(sensorPairs)
	if totalCandidates == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	selected := o.Rand.Intn(totalCandidates)
	if selected < len(neuronPairs) {
		pair := neuronPairs[selected]
		weight := (o.Rand.Float64()*2 - 1) * o.MaxAbsWeight
		mutated := cloneGenome(genome)
		mutated.Synapses = append(mutated.Synapses, model.Synapse{
			ID:        uniqueSynapseID(genome, o.Rand),
			From:      pair.from,
			To:        pair.to,
			Weight:    weight,
			Enabled:   true,
			Recurrent: pair.from == pair.to,
		})
		return mutated, nil
	}
	mutated := cloneGenome(genome)
	mutated.SensorNeuronLinks = append(mutated.SensorNeuronLinks, sensorPairs[selected-len(neuronPairs)])
	syncIOLinkCounts(&mutated)
	return mutated, nil
}

// AddRandomOutlink adds a synapse biased toward non-output->output direction.
type AddRandomOutlink struct {
	Rand            *rand.Rand
	MaxAbsWeight    float64
	OutputNeuronIDs []string
	FeedForwardOnly bool
}

func (o *AddRandomOutlink) Name() string {
	return "add_outlink"
}

func (o *AddRandomOutlink) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Neurons) <= 1 {
		return false
	}
	outputSet := toIDSet(o.OutputNeuronIDs)
	layers := inferFeedforwardLayers(genome, nil, o.OutputNeuronIDs)
	fromCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := outputSet[id]
		return !ok
	})
	toCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := outputSet[id]
		return ok
	})
	if o.FeedForwardOnly {
		fromCandidates, toCandidates = filterDirectedFeedforwardCandidates(fromCandidates, toCandidates, layers)
	}
	return hasAvailableDirectedPair(genome, fromCandidates, toCandidates)
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
	layers := inferFeedforwardLayers(genome, nil, o.OutputNeuronIDs)
	fromCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := outputSet[id]
		return !ok
	})
	toCandidates := filterNeuronIDs(genome, func(id string) bool {
		_, ok := outputSet[id]
		return ok
	})
	if o.FeedForwardOnly {
		fromCandidates, toCandidates = filterDirectedFeedforwardCandidates(fromCandidates, toCandidates, layers)
	}
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
	Rand            *rand.Rand
	InputNeuronIDs  []string
	FeedForwardOnly bool
}

func (o *RemoveRandomInlink) Name() string {
	return "remove_inlink"
}

func (o *RemoveRandomInlink) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Synapses) == 0 {
		return false
	}
	inputSet := toIDSet(o.InputNeuronIDs)
	layers := inferFeedforwardLayers(genome, o.InputNeuronIDs, nil)
	for _, syn := range genome.Synapses {
		_, fromInput := inputSet[syn.From]
		_, toInput := inputSet[syn.To]
		if fromInput && !toInput && (!o.FeedForwardOnly || isFeedforwardEdge(layers, syn.From, syn.To)) {
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
		return model.Genome{}, ErrNoMutationChoice
	}
	inputSet := toIDSet(o.InputNeuronIDs)
	layers := inferFeedforwardLayers(genome, o.InputNeuronIDs, nil)
	return removeDirectedRandomSynapse(genome, o.Rand, func(s model.Synapse) bool {
		_, fromInput := inputSet[s.From]
		_, toInput := inputSet[s.To]
		return fromInput && !toInput && (!o.FeedForwardOnly || isFeedforwardEdge(layers, s.From, s.To))
	})
}

// RemoveRandomOutlink removes a synapse biased toward non-output->output direction.
type RemoveRandomOutlink struct {
	Rand            *rand.Rand
	OutputNeuronIDs []string
	FeedForwardOnly bool
}

func (o *RemoveRandomOutlink) Name() string {
	return "remove_outlink"
}

func (o *RemoveRandomOutlink) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Synapses) == 0 {
		return false
	}
	outputSet := toIDSet(o.OutputNeuronIDs)
	layers := inferFeedforwardLayers(genome, nil, o.OutputNeuronIDs)
	for _, syn := range genome.Synapses {
		_, fromOutput := outputSet[syn.From]
		_, toOutput := outputSet[syn.To]
		if !fromOutput && toOutput && (!o.FeedForwardOnly || isFeedforwardEdge(layers, syn.From, syn.To)) {
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
		return model.Genome{}, ErrNoMutationChoice
	}
	outputSet := toIDSet(o.OutputNeuronIDs)
	layers := inferFeedforwardLayers(genome, nil, o.OutputNeuronIDs)
	return removeDirectedRandomSynapse(genome, o.Rand, func(s model.Synapse) bool {
		_, fromOutput := outputSet[s.From]
		_, toOutput := outputSet[s.To]
		return !fromOutput && toOutput && (!o.FeedForwardOnly || isFeedforwardEdge(layers, s.From, s.To))
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
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
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
	return len(genome.Synapses) > 0 || len(genome.SensorNeuronLinks) > 0 || len(genome.NeuronActuatorLinks) > 0
}

func (o *CutlinkFromElementToElement) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	type opCandidate struct {
		apply func(context.Context, model.Genome) (model.Genome, error)
	}
	candidates := make([]opCandidate, 0, 3)
	removeSynapse := &RemoveRandomSynapse{Rand: o.Rand}
	if removeSynapse.Applicable(genome, "") {
		candidates = append(candidates, opCandidate{apply: removeSynapse.Apply})
	}
	cutSensor := &CutlinkFromSensorToNeuron{Rand: o.Rand}
	if cutSensor.Applicable(genome, "") {
		candidates = append(candidates, opCandidate{apply: cutSensor.Apply})
	}
	cutActuator := &CutlinkFromNeuronToActuator{Rand: o.Rand}
	if cutActuator.Applicable(genome, "") {
		candidates = append(candidates, opCandidate{apply: cutActuator.Apply})
	}
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	selected := candidates[o.Rand.Intn(len(candidates))]
	return selected.apply(ctx, genome)
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
	allNeurons := filterNeuronIDs(genome, nil)
	addSynapse := hasAvailableDirectedPair(genome, allNeurons, allNeurons)
	addSensor := (&AddRandomSensorLink{Rand: o.Rand, ScapeName: ""}).Applicable(genome, "")
	addActuator := (&AddRandomActuatorLink{Rand: o.Rand, ScapeName: ""}).Applicable(genome, "")
	return addSynapse || addSensor || addActuator
}

func (o *LinkFromElementToElement) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	type opCandidate struct {
		apply func(context.Context, model.Genome) (model.Genome, error)
	}
	candidates := make([]opCandidate, 0, 3)
	allNeurons := filterNeuronIDs(genome, nil)
	if hasAvailableDirectedPair(genome, allNeurons, allNeurons) {
		candidates = append(candidates, opCandidate{apply: func(_ context.Context, g model.Genome) (model.Genome, error) {
			if o.MaxAbsWeight <= 0 {
				return model.Genome{}, errors.New("max abs weight must be > 0")
			}
			return addDirectedRandomSynapse(g, o.Rand, o.MaxAbsWeight, allNeurons, allNeurons)
		}})
	}
	addSensor := &AddRandomSensorLink{Rand: o.Rand, ScapeName: ""}
	if addSensor.Applicable(genome, "") {
		candidates = append(candidates, opCandidate{apply: addSensor.Apply})
	}
	addActuator := &AddRandomActuatorLink{Rand: o.Rand, ScapeName: ""}
	if addActuator.Applicable(genome, "") {
		candidates = append(candidates, opCandidate{apply: addActuator.Apply})
	}
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	selected := candidates[o.Rand.Intn(len(candidates))]
	return selected.apply(ctx, genome)
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
	allNeurons := filterNeuronIDs(genome, nil)
	return hasAvailableDirectedPair(genome, allNeurons, allNeurons)
}

func (o *LinkFromNeuronToNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if o.MaxAbsWeight <= 0 {
		return model.Genome{}, errors.New("max abs weight must be > 0")
	}
	allNeurons := filterNeuronIDs(genome, nil)
	return addDirectedRandomSynapse(genome, o.Rand, o.MaxAbsWeight, allNeurons, allNeurons)
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
	FeedForwardOnly bool
}

func (o *AddRandomOutsplice) Name() string {
	return "outsplice"
}

func (o *AddRandomOutsplice) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Synapses) == 0 {
		return false
	}
	outputSet := toIDSet(o.OutputNeuronIDs)
	layers := inferFeedforwardLayers(genome, nil, o.OutputNeuronIDs)
	for _, syn := range genome.Synapses {
		_, fromOutput := outputSet[syn.From]
		_, toOutput := outputSet[syn.To]
		if !fromOutput && toOutput && (!o.FeedForwardOnly || isFeedforwardEdge(layers, syn.From, syn.To)) {
			return true
		}
	}
	return false
}

func (o *AddRandomOutsplice) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	outputSet := toIDSet(o.OutputNeuronIDs)
	layers := inferFeedforwardLayers(genome, nil, o.OutputNeuronIDs)
	return addRandomNeuronWithSynapseCandidates(ctx, genome, o.Rand, o.Activations, func(s model.Synapse) bool {
		_, fromOutput := outputSet[s.From]
		_, toOutput := outputSet[s.To]
		return !fromOutput && toOutput && (!o.FeedForwardOnly || isFeedforwardEdge(layers, s.From, s.To))
	})
}

// AddRandomInsplice inserts a neuron by splitting a synapse biased toward
// input->non-input direction.
type AddRandomInsplice struct {
	Rand            *rand.Rand
	Activations     []string
	InputNeuronIDs  []string
	FeedForwardOnly bool
}

func (o *AddRandomInsplice) Name() string {
	return "insplice"
}

func (o *AddRandomInsplice) Applicable(genome model.Genome, _ string) bool {
	if len(genome.Synapses) == 0 {
		return false
	}
	inputSet := toIDSet(o.InputNeuronIDs)
	layers := inferFeedforwardLayers(genome, o.InputNeuronIDs, nil)
	for _, syn := range genome.Synapses {
		_, fromInput := inputSet[syn.From]
		_, toInput := inputSet[syn.To]
		if fromInput && !toInput && (!o.FeedForwardOnly || isFeedforwardEdge(layers, syn.From, syn.To)) {
			return true
		}
	}
	return false
}

func (o *AddRandomInsplice) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	inputSet := toIDSet(o.InputNeuronIDs)
	layers := inferFeedforwardLayers(genome, o.InputNeuronIDs, nil)
	return addRandomNeuronWithSynapseCandidates(ctx, genome, o.Rand, o.Activations, func(s model.Synapse) bool {
		_, fromInput := inputSet[s.From]
		_, toInput := inputSet[s.To]
		return fromInput && !toInput && (!o.FeedForwardOnly || isFeedforwardEdge(layers, s.From, s.To))
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
		return model.Genome{}, ErrNoMutationChoice
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
		return model.Genome{}, ErrNoMutationChoice
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
		return model.Genome{}, ErrNoMutationChoice
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
		return model.Genome{}, ErrNoMutationChoice
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
	return len(genome.Neurons) > 0
}

func (o *MutatePlasticityParameters) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	maxDelta := o.MaxDelta
	if maxDelta <= 0 {
		maxDelta = 0.15
	}
	idx := o.Rand.Intn(len(genome.Neurons))
	mutated := cloneGenome(genome)
	delta := (o.Rand.Float64()*2 - 1) * maxDelta
	rule := nn.NormalizePlasticityRuleName(neuronPlasticityRule(genome, idx))
	if width := selfModulationParameterWidth(rule); width > 0 {
		if selfModulationRuleUsesCoefficientMutation(rule) && o.Rand.Intn(2) == 0 {
			mutateNeuronPlasticityCoefficients(&mutated, genome, idx, delta, o.Rand)
			mutated.Neurons[idx].Generation = currentGenomeGeneration(mutated)
			return mutated, nil
		}
		if ok := mutateSelfModulationParameterVector(&mutated, genome, idx, width, delta, o.Rand); ok {
			mutated.Neurons[idx].Generation = currentGenomeGeneration(mutated)
			return mutated, nil
		}
	}
	if plasticityRuleUsesGeneralizedCoefficients(rule) {
		mutateNeuronPlasticityCoefficients(&mutated, genome, idx, delta, o.Rand)
	} else {
		baseRate := neuronPlasticityRate(genome, idx)
		mutated.Neurons[idx].PlasticityRate = math.Max(0, baseRate+delta)
	}
	mutated.Neurons[idx].Generation = currentGenomeGeneration(mutated)
	return mutated, nil
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
		return model.Genome{}, ErrNoMutationChoice
	}
	rules := o.Rules
	if len(rules) == 0 {
		rules = defaultPlasticityRules()
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
		return model.Genome{}, ErrNoMutationChoice
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
	if len(genome.Neurons) == 0 {
		return false
	}
	rules := append([]string(nil), o.Rules...)
	if len(rules) == 0 {
		rules = defaultPlasticityRules()
	}
	normalized := normalizePlasticityRuleOptions(rules)
	if len(normalized) == 0 {
		return false
	}
	for i := range genome.Neurons {
		current := neuronPlasticityRule(genome, i)
		for _, option := range normalized {
			if option != current {
				return true
			}
		}
	}
	return false
}

func (o *MutatePF) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	rules := append([]string(nil), o.Rules...)
	if len(rules) == 0 {
		rules = defaultPlasticityRules()
	}
	normalized := make([]string, 0, len(rules))
	for _, rule := range rules {
		name := nn.NormalizePlasticityRuleName(rule)
		if name == "" {
			continue
		}
		normalized = append(normalized, name)
	}
	if len(normalized) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}

	idx := o.Rand.Intn(len(genome.Neurons))
	current := neuronPlasticityRule(genome, idx)
	choices := filterOutString(normalized, current)
	if len(choices) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}

	mutated := cloneGenome(genome)
	mutated.Neurons[idx].PlasticityRule = choices[o.Rand.Intn(len(choices))]
	if mutated.Neurons[idx].PlasticityRate <= 0 {
		mutated.Neurons[idx].PlasticityRate = neuronPlasticityRate(genome, idx)
	}
	mutated.Neurons[idx].Generation = currentGenomeGeneration(mutated)
	return mutated, nil
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
		return model.Genome{}, ErrNoMutationChoice
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
		return model.Genome{}, ErrNoMutationChoice
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
	modes := append([]string(nil), o.Modes...)
	if len(modes) == 0 {
		return true
	}
	normalized := make([]string, 0, len(modes))
	seen := make(map[string]struct{}, len(modes))
	for _, mode := range modes {
		name := tuning.NormalizeCandidateSelectionName(mode)
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		normalized = append(normalized, name)
	}
	return len(normalized) > 1
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
			tuning.CandidateSelectActive,
			tuning.CandidateSelectActiveRnd,
			tuning.CandidateSelectRecent,
			tuning.CandidateSelectRecentRnd,
			tuning.CandidateSelectAll,
			tuning.CandidateSelectAllRandom,
			tuning.CandidateSelectCurrent,
			tuning.CandidateSelectCurrentRd,
			tuning.CandidateSelectLastGen,
			tuning.CandidateSelectLastGenRd,
		}
	}
	normalized := make([]string, 0, len(modes))
	seen := make(map[string]struct{}, len(modes))
	for _, mode := range modes {
		name := tuning.NormalizeCandidateSelectionName(mode)
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		normalized = append(normalized, name)
	}
	mutated := cloneGenome(genome)
	ensureStrategyConfig(&mutated)
	current := tuning.NormalizeCandidateSelectionName(mutated.Strategy.TuningSelection)
	choices := filterOutString(normalized, current)
	if len(choices) == 0 {
		return model.Genome{}, ErrNoMutationChoice
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
	values := append([]float64(nil), o.Values...)
	if len(values) == 0 {
		return true
	}
	seen := make(map[int64]struct{}, len(values))
	unique := 0
	for _, value := range values {
		if value <= 0 {
			continue
		}
		key := int64(value * 1e9)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		unique++
	}
	return unique > 1
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
		return model.Genome{}, ErrNoMutationChoice
	}
	mutated.Strategy.AnnealingFactor = choices[o.Rand.Intn(len(choices))]
	return mutated, nil
}

// MutateTotTopologicalMutations mirrors mutate_tot_topological_mutations.
type MutateTotTopologicalMutations struct {
	Rand     *rand.Rand
	Policies []string
	Choices  []TopologicalPolicyChoice
}

type TopologicalPolicyChoice struct {
	Name  string
	Param float64
}

func (o *MutateTotTopologicalMutations) Name() string {
	return "mutate_tot_topological_mutations"
}

func (o *MutateTotTopologicalMutations) Applicable(_ model.Genome, _ string) bool {
	if len(o.Choices) > 0 {
		unique := make(map[string]struct{}, len(o.Choices))
		for _, choice := range o.Choices {
			name := choice.Name
			if name == "" {
				continue
			}
			param := choice.Param
			if param <= 0 {
				param = defaultTopologicalParam(name)
			}
			key := fmt.Sprintf("%s:%0.9f", name, param)
			unique[key] = struct{}{}
		}
		return len(unique) > 1
	}
	policies := append([]string(nil), o.Policies...)
	if len(policies) == 0 {
		return true
	}
	normalized := make(map[string]struct{}, len(policies))
	for _, policy := range policies {
		if policy == "" {
			continue
		}
		normalized[policy] = struct{}{}
	}
	return len(normalized) > 1
}

func (o *MutateTotTopologicalMutations) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	mutated := cloneGenome(genome)
	ensureStrategyConfig(&mutated)

	choices := append([]TopologicalPolicyChoice(nil), o.Choices...)
	if len(choices) == 0 && len(o.Policies) > 0 {
		for _, name := range o.Policies {
			if name == "" {
				continue
			}
			choices = append(choices, TopologicalPolicyChoice{
				Name:  name,
				Param: defaultTopologicalParam(name),
			})
		}
	}
	if len(choices) == 0 {
		choices = []TopologicalPolicyChoice{
			{Name: "const", Param: 1.0},
			{Name: "ncount_linear", Param: 1.0},
			{Name: "ncount_exponential", Param: 0.5},
		}
	}
	filteredChoices := make([]TopologicalPolicyChoice, 0, len(choices))
	for _, choice := range choices {
		if choice.Name == "" || choice.Param <= 0 {
			continue
		}
		filteredChoices = append(filteredChoices, choice)
	}
	if len(filteredChoices) == 0 {
		return mutated, nil
	}

	current := mutated.Strategy.TopologicalMode
	if current == "" {
		current = "const"
	}
	currentParam := mutated.Strategy.TopologicalParam
	if currentParam <= 0 {
		currentParam = defaultTopologicalParam(current)
	}
	available := make([]TopologicalPolicyChoice, 0, len(filteredChoices))
	for _, choice := range filteredChoices {
		if choice.Name == current && math.Abs(choice.Param-currentParam) < 1e-9 {
			continue
		}
		available = append(available, choice)
	}
	if len(available) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	selected := available[o.Rand.Intn(len(available))]
	mutated.Strategy.TopologicalMode = selected.Name
	mutated.Strategy.TopologicalParam = selected.Param
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
	types := append([]string(nil), o.Types...)
	if len(types) == 0 {
		return true
	}
	normalized := make(map[string]struct{}, len(types))
	for _, item := range types {
		if item == "" {
			continue
		}
		normalized[item] = struct{}{}
	}
	return len(normalized) > 1
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
		return model.Genome{}, ErrNoMutationChoice
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
	return len(genome.Neurons) > 0 && len(sensorCandidates(genome, o.ScapeName)) > 0
}

func (o *AddRandomSensor) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	candidates := sensorCandidates(genome, o.ScapeName)
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	choice := candidates[o.Rand.Intn(len(candidates))]
	mutated := cloneGenome(genome)
	mutated.SensorIDs = append(mutated.SensorIDs, choice)
	targetNeuron := mutated.Neurons[o.Rand.Intn(len(mutated.Neurons))].ID
	mutated.SensorNeuronLinks = append(mutated.SensorNeuronLinks, model.SensorNeuronLink{
		SensorID: choice,
		NeuronID: targetNeuron,
	})
	syncIOLinkCounts(&mutated)
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
	return len(availableSensorNeuronPairs(genome)) > 0
}

func (o *AddRandomSensorLink) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.SensorIDs) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	candidates := availableSensorNeuronPairs(genome)
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	mutated := cloneGenome(genome)
	selected := candidates[o.Rand.Intn(len(candidates))]
	mutated.SensorNeuronLinks = append(mutated.SensorNeuronLinks, selected)
	syncIOLinkCounts(&mutated)
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
	return len(genome.Neurons) > 0 && len(actuatorCandidates(genome, o.ScapeName)) > 0
}

func (o *AddRandomActuator) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	candidates := actuatorCandidates(genome, o.ScapeName)
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	choice := candidates[o.Rand.Intn(len(candidates))]
	mutated := cloneGenome(genome)
	currentGeneration := currentGenomeGeneration(mutated)
	mutated.ActuatorIDs = append(mutated.ActuatorIDs, choice)
	touchActuatorGeneration(&mutated, choice, currentGeneration)
	sourceNeuron := mutated.Neurons[o.Rand.Intn(len(mutated.Neurons))].ID
	helperNeuronID := uniqueNeuronID(mutated, o.Rand)
	mutated.Neurons = append(mutated.Neurons, model.Neuron{
		ID:         helperNeuronID,
		Generation: currentGeneration,
		Activation: "tanh",
	})
	mutated.Synapses = append(mutated.Synapses, model.Synapse{
		ID:        uniqueSynapseID(mutated, o.Rand),
		From:      sourceNeuron,
		To:        helperNeuronID,
		Weight:    (o.Rand.Float64() * 2) - 1,
		Enabled:   true,
		Recurrent: sourceNeuron == helperNeuronID,
	})
	mutated.NeuronActuatorLinks = append(mutated.NeuronActuatorLinks, model.NeuronActuatorLink{
		NeuronID:   helperNeuronID,
		ActuatorID: choice,
	})
	syncIOLinkCounts(&mutated)
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
	return len(availableNeuronActuatorPairs(genome)) > 0
}

func (o *AddRandomActuatorLink) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.ActuatorIDs) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	if len(genome.Neurons) == 0 {
		return model.Genome{}, ErrNoNeurons
	}
	candidates := availableNeuronActuatorPairs(genome)
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	mutated := cloneGenome(genome)
	selected := candidates[o.Rand.Intn(len(candidates))]
	mutated.NeuronActuatorLinks = append(mutated.NeuronActuatorLinks, selected)
	syncIOLinkCounts(&mutated)
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
		return model.Genome{}, ErrNoMutationChoice
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
	filteredLinks := mutated.SensorNeuronLinks[:0]
	for _, link := range mutated.SensorNeuronLinks {
		if link.SensorID == selected {
			continue
		}
		filteredLinks = append(filteredLinks, link)
	}
	mutated.SensorNeuronLinks = filteredLinks
	syncIOLinkCounts(&mutated)
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
	if len(genome.SensorNeuronLinks) > 0 {
		return true
	}
	return genome.SensorLinks > 0
}

func (o *CutlinkFromSensorToNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.SensorNeuronLinks) == 0 && genome.SensorLinks <= 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	mutated := cloneGenome(genome)
	if len(mutated.SensorNeuronLinks) > 0 {
		idx := o.Rand.Intn(len(mutated.SensorNeuronLinks))
		mutated.SensorNeuronLinks = append(mutated.SensorNeuronLinks[:idx], mutated.SensorNeuronLinks[idx+1:]...)
		syncIOLinkCounts(&mutated)
		return mutated, nil
	}
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
		return model.Genome{}, ErrNoMutationChoice
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
	deleteActuatorGeneration(&mutated, selected)
	deleteActuatorTunable(&mutated, selected)
	filteredLinks := mutated.NeuronActuatorLinks[:0]
	for _, link := range mutated.NeuronActuatorLinks {
		if link.ActuatorID == selected {
			continue
		}
		filteredLinks = append(filteredLinks, link)
	}
	mutated.NeuronActuatorLinks = filteredLinks
	syncIOLinkCounts(&mutated)
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
	if len(genome.NeuronActuatorLinks) > 0 {
		return true
	}
	return genome.ActuatorLinks > 0
}

func (o *CutlinkFromNeuronToActuator) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if len(genome.NeuronActuatorLinks) == 0 && genome.ActuatorLinks <= 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	mutated := cloneGenome(genome)
	if len(mutated.NeuronActuatorLinks) > 0 {
		idx := o.Rand.Intn(len(mutated.NeuronActuatorLinks))
		mutated.NeuronActuatorLinks = append(mutated.NeuronActuatorLinks[:idx], mutated.NeuronActuatorLinks[idx+1:]...)
		syncIOLinkCounts(&mutated)
		return mutated, nil
	}
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
	return len(availableCPPChoices(genome)) > 0
}

func (o *AddRandomCPP) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if genome.Substrate == nil {
		return model.Genome{}, ErrNoMutationChoice
	}
	choices := availableCPPChoices(genome)
	if len(choices) == 0 {
		return model.Genome{}, ErrNoMutationChoice
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
	// In the simplified model, approximate CPP structural growth by adding one
	// extra sensor->neuron endpoint link when such a connection is available.
	if len(mutated.Neurons) > 0 && len(mutated.SensorIDs) > 0 {
		toCandidates := filterNeuronIDs(mutated, nil)
		sensorPairs := availableSensorToNeuronPairs(mutated, toCandidates)
		if len(sensorPairs) > 0 {
			mutated.SensorNeuronLinks = append(mutated.SensorNeuronLinks, sensorPairs[o.Rand.Intn(len(sensorPairs))])
			syncIOLinkCounts(&mutated)
		}
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
	return len(availableCEPChoices(genome)) > 0
}

func (o *AddRandomCEP) Apply(_ context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if genome.Substrate == nil {
		return model.Genome{}, ErrNoMutationChoice
	}
	choices := availableCEPChoices(genome)
	if len(choices) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	selected := choices[o.Rand.Intn(len(choices))]

	mutated := cloneGenome(genome)
	currentGeneration := currentGenomeGeneration(mutated)
	mutated.Substrate.CEPName = selected
	if mutated.Substrate.CPPName == "" {
		mutated.Substrate.CPPName = substrate.DefaultCPPName
	}
	if mutated.Substrate.Parameters == nil {
		mutated.Substrate.Parameters = map[string]float64{}
	}
	if len(mutated.Neurons) > 0 {
		sourceNeuron := mutated.Neurons[o.Rand.Intn(len(mutated.Neurons))].ID
		helperNeuronID := uniqueNeuronID(mutated, o.Rand)
		mutated.Neurons = append(mutated.Neurons, model.Neuron{
			ID:         helperNeuronID,
			Generation: currentGeneration,
			Activation: "tanh",
		})
		mutated.Synapses = append(mutated.Synapses, model.Synapse{
			ID:        uniqueSynapseID(mutated, o.Rand),
			From:      sourceNeuron,
			To:        helperNeuronID,
			Weight:    (o.Rand.Float64() * 2) - 1,
			Enabled:   true,
			Recurrent: sourceNeuron == helperNeuronID,
		})
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
	if genome.Substrate == nil || genome.Substrate.CPPName == "" {
		return model.Genome{}, ErrNoMutationChoice
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
	if genome.Substrate == nil || genome.Substrate.CEPName == "" {
		return model.Genome{}, ErrNoMutationChoice
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
	if genome.Substrate == nil || len(genome.Substrate.Dimensions) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	mutated := cloneGenome(genome)
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
		return model.Genome{}, ErrNoMutationChoice
	}
	candidates := make([]int, 0, len(genome.Substrate.Dimensions))
	for i, width := range genome.Substrate.Dimensions {
		if width > 1 {
			candidates = append(candidates, i)
		}
	}
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
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
		return model.Genome{}, ErrNoMutationChoice
	}
	mutated := cloneGenome(genome)
	dims := append([]int(nil), mutated.Substrate.Dimensions...)
	if len(dims) == 0 {
		return model.Genome{}, ErrNoMutationChoice
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
	mutated.Neurons[o.Index].Generation = currentGenomeGeneration(mutated)
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
	currentGeneration := currentGenomeGeneration(mutated)
	target := mutated.Synapses[o.SynapseIndex]
	mutated.Synapses = append(mutated.Synapses[:o.SynapseIndex], mutated.Synapses[o.SynapseIndex+1:]...)

	mutated.Neurons = append(mutated.Neurons, model.Neuron{
		ID:         o.NeuronID,
		Generation: currentGeneration,
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

func hasActuator(g model.Genome, id string) bool {
	for _, actuatorID := range g.ActuatorIDs {
		if actuatorID == id {
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

func inferFeedforwardLayers(genome model.Genome, inputNeuronIDs, outputNeuronIDs []string) map[string]int {
	layers := make(map[string]int, len(genome.Neurons))
	inputSet := toIDSet(inputNeuronIDs)
	outputSet := toIDSet(outputNeuronIDs)
	for _, n := range genome.Neurons {
		switch {
		case containsID(inputSet, n.ID):
			layers[n.ID] = 0
		case containsID(outputSet, n.ID):
			layers[n.ID] = 2
		default:
			layers[n.ID] = 1
		}
	}
	// Relax edge ordering to infer a monotonic feedforward layer ranking.
	for i := 0; i < len(genome.Neurons); i++ {
		changed := false
		for _, s := range genome.Synapses {
			fromLayer, okFrom := layers[s.From]
			toLayer, okTo := layers[s.To]
			if !okFrom || !okTo {
				continue
			}
			candidate := fromLayer + 1
			if candidate > toLayer {
				layers[s.To] = candidate
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return layers
}

func isFeedforwardEdge(layers map[string]int, fromID, toID string) bool {
	fromLayer, okFrom := layers[fromID]
	toLayer, okTo := layers[toID]
	if !okFrom || !okTo {
		return false
	}
	return fromLayer < toLayer
}

func filterDirectedFeedforwardCandidates(fromCandidates, toCandidates []string, layers map[string]int) ([]string, []string) {
	allowedFrom := make(map[string]struct{}, len(fromCandidates))
	allowedTo := make(map[string]struct{}, len(toCandidates))
	for _, from := range fromCandidates {
		for _, to := range toCandidates {
			if isFeedforwardEdge(layers, from, to) {
				allowedFrom[from] = struct{}{}
				allowedTo[to] = struct{}{}
			}
		}
	}
	filteredFrom := make([]string, 0, len(allowedFrom))
	for _, from := range fromCandidates {
		if containsID(allowedFrom, from) {
			filteredFrom = append(filteredFrom, from)
		}
	}
	filteredTo := make([]string, 0, len(allowedTo))
	for _, to := range toCandidates {
		if containsID(allowedTo, to) {
			filteredTo = append(filteredTo, to)
		}
	}
	return filteredFrom, filteredTo
}

func containsID(set map[string]struct{}, id string) bool {
	_, ok := set[id]
	return ok
}

func ensureStrategyConfig(g *model.Genome) {
	if g == nil {
		return
	}
	if g.Strategy == nil {
		g.Strategy = &model.StrategyConfig{
			TuningSelection:  tuning.CandidateSelectBestSoFar,
			AnnealingFactor:  1.0,
			TopologicalMode:  "const",
			TopologicalParam: 1.0,
			HeredityType:     "asexual",
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
	if g.Strategy.TopologicalParam <= 0 {
		g.Strategy.TopologicalParam = defaultTopologicalParam(g.Strategy.TopologicalMode)
	}
	if g.Strategy.HeredityType == "" {
		g.Strategy.HeredityType = "asexual"
	}
}

func defaultTopologicalParam(mode string) float64 {
	switch mode {
	case "const":
		return 1.0
	case "ncount_linear":
		return 1.0
	case "ncount_exponential":
		return 0.5
	default:
		return 0.5
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
		return model.Genome{}, ErrNoMutationChoice
	}
	type pair struct {
		from string
		to   string
	}
	candidates := make([]pair, 0, len(fromCandidates)*len(toCandidates))
	for _, from := range fromCandidates {
		for _, to := range toCandidates {
			if hasDirectedSynapse(genome, from, to) {
				continue
			}
			candidates = append(candidates, pair{from: from, to: to})
		}
	}
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
	}
	selected := candidates[rng.Intn(len(candidates))]
	id := uniqueSynapseID(genome, rng)
	weight := (rng.Float64()*2 - 1) * maxAbsWeight

	mutated := cloneGenome(genome)
	mutated.Synapses = append(mutated.Synapses, model.Synapse{
		ID:        id,
		From:      selected.from,
		To:        selected.to,
		Weight:    weight,
		Enabled:   true,
		Recurrent: selected.from == selected.to,
	})
	return mutated, nil
}

type directedNeuronPair struct {
	from string
	to   string
}

func availableInlinkNeuronPairs(genome model.Genome, fromCandidates, toCandidates []string) []directedNeuronPair {
	if len(fromCandidates) == 0 || len(toCandidates) == 0 {
		return nil
	}
	pairs := make([]directedNeuronPair, 0, len(fromCandidates)*len(toCandidates))
	for _, from := range fromCandidates {
		for _, to := range toCandidates {
			if hasDirectedSynapse(genome, from, to) {
				continue
			}
			pairs = append(pairs, directedNeuronPair{from: from, to: to})
		}
	}
	return pairs
}

func availableSensorToNeuronPairs(genome model.Genome, toCandidates []string) []model.SensorNeuronLink {
	if len(genome.SensorIDs) == 0 || len(toCandidates) == 0 {
		return nil
	}
	targetSet := make(map[string]struct{}, len(toCandidates))
	for _, id := range toCandidates {
		targetSet[id] = struct{}{}
	}
	pairs := make([]model.SensorNeuronLink, 0, len(genome.SensorIDs)*len(toCandidates))
	for _, sensorID := range uniqueStrings(genome.SensorIDs) {
		for _, neuronID := range toCandidates {
			if _, ok := targetSet[neuronID]; !ok {
				continue
			}
			if hasSensorNeuronLink(genome, sensorID, neuronID) {
				continue
			}
			pairs = append(pairs, model.SensorNeuronLink{
				SensorID: sensorID,
				NeuronID: neuronID,
			})
		}
	}
	return pairs
}

func hasDirectedSynapse(g model.Genome, from, to string) bool {
	for _, syn := range g.Synapses {
		if syn.From == from && syn.To == to {
			return true
		}
	}
	return false
}

func hasAvailableDirectedPair(g model.Genome, fromCandidates, toCandidates []string) bool {
	if len(fromCandidates) == 0 || len(toCandidates) == 0 {
		return false
	}
	for _, from := range fromCandidates {
		for _, to := range toCandidates {
			if !hasDirectedSynapse(g, from, to) {
				return true
			}
		}
	}
	return false
}

func removeDirectedRandomSynapse(genome model.Genome, rng *rand.Rand, keep func(s model.Synapse) bool) (model.Genome, error) {
	candidates := make([]int, 0, len(genome.Synapses))
	for i, syn := range genome.Synapses {
		if keep == nil || keep(syn) {
			candidates = append(candidates, i)
		}
	}
	if len(candidates) == 0 {
		return model.Genome{}, ErrNoMutationChoice
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

func neuronPlasticityRule(genome model.Genome, idx int) string {
	if idx < 0 || idx >= len(genome.Neurons) {
		return nn.PlasticityNone
	}
	if rule := nn.NormalizePlasticityRuleName(genome.Neurons[idx].PlasticityRule); rule != "" {
		return rule
	}
	if genome.Plasticity != nil {
		return nn.NormalizePlasticityRuleName(genome.Plasticity.Rule)
	}
	return nn.PlasticityNone
}

func neuronPlasticityRate(genome model.Genome, idx int) float64 {
	if idx >= 0 && idx < len(genome.Neurons) && genome.Neurons[idx].PlasticityRate > 0 {
		return genome.Neurons[idx].PlasticityRate
	}
	if genome.Plasticity != nil && genome.Plasticity.Rate > 0 {
		return genome.Plasticity.Rate
	}
	return 0.1
}

func neuronPlasticityA(genome model.Genome, idx int) float64 {
	if idx >= 0 && idx < len(genome.Neurons) && genome.Neurons[idx].PlasticityA != 0 {
		return genome.Neurons[idx].PlasticityA
	}
	if genome.Plasticity != nil && genome.Plasticity.CoeffA != 0 {
		return genome.Plasticity.CoeffA
	}
	return 0
}

func neuronPlasticityB(genome model.Genome, idx int) float64 {
	if idx >= 0 && idx < len(genome.Neurons) && genome.Neurons[idx].PlasticityB != 0 {
		return genome.Neurons[idx].PlasticityB
	}
	if genome.Plasticity != nil && genome.Plasticity.CoeffB != 0 {
		return genome.Plasticity.CoeffB
	}
	return 0
}

func neuronPlasticityC(genome model.Genome, idx int) float64 {
	if idx >= 0 && idx < len(genome.Neurons) && genome.Neurons[idx].PlasticityC != 0 {
		return genome.Neurons[idx].PlasticityC
	}
	if genome.Plasticity != nil && genome.Plasticity.CoeffC != 0 {
		return genome.Plasticity.CoeffC
	}
	return 0
}

func neuronPlasticityD(genome model.Genome, idx int) float64 {
	if idx >= 0 && idx < len(genome.Neurons) && genome.Neurons[idx].PlasticityD != 0 {
		return genome.Neurons[idx].PlasticityD
	}
	if genome.Plasticity != nil && genome.Plasticity.CoeffD != 0 {
		return genome.Plasticity.CoeffD
	}
	return 0
}

func plasticityRuleUsesGeneralizedCoefficients(rule string) bool {
	switch nn.NormalizePlasticityRuleName(rule) {
	case nn.PlasticitySelfModulationV1,
		nn.PlasticitySelfModulationV2,
		nn.PlasticitySelfModulationV3,
		nn.PlasticitySelfModulationV4,
		nn.PlasticitySelfModulationV5,
		nn.PlasticitySelfModulationV6,
		nn.PlasticityNeuromodulation:
		return true
	default:
		return false
	}
}

func selfModulationParameterWidth(rule string) int {
	switch nn.NormalizePlasticityRuleName(rule) {
	case nn.PlasticitySelfModulationV1, nn.PlasticitySelfModulationV2, nn.PlasticitySelfModulationV3:
		return 1
	case nn.PlasticitySelfModulationV4, nn.PlasticitySelfModulationV5:
		return 2
	case nn.PlasticitySelfModulationV6:
		return 5
	default:
		return 0
	}
}

func selfModulationRuleUsesCoefficientMutation(rule string) bool {
	switch nn.NormalizePlasticityRuleName(rule) {
	case nn.PlasticitySelfModulationV2, nn.PlasticitySelfModulationV3, nn.PlasticitySelfModulationV5:
		return true
	default:
		return false
	}
}

func mutateNeuronPlasticityCoefficients(mutated *model.Genome, base model.Genome, neuronIdx int, delta float64, rng *rand.Rand) {
	if mutated == nil || rng == nil || neuronIdx < 0 || neuronIdx >= len(mutated.Neurons) {
		return
	}
	switch rng.Intn(4) {
	case 0:
		mutated.Neurons[neuronIdx].PlasticityA = neuronPlasticityA(base, neuronIdx) + delta
	case 1:
		mutated.Neurons[neuronIdx].PlasticityB = neuronPlasticityB(base, neuronIdx) + delta
	case 2:
		mutated.Neurons[neuronIdx].PlasticityC = neuronPlasticityC(base, neuronIdx) + delta
	default:
		mutated.Neurons[neuronIdx].PlasticityD = neuronPlasticityD(base, neuronIdx) + delta
	}
}

func mutateSelfModulationParameterVector(
	mutated *model.Genome,
	base model.Genome,
	neuronIdx int,
	width int,
	delta float64,
	rng *rand.Rand,
) bool {
	if mutated == nil || width <= 0 || rng == nil || neuronIdx < 0 || neuronIdx >= len(base.Neurons) {
		return false
	}

	type vectorTarget struct {
		synapseIdx int
		bias       bool
	}

	neuronID := base.Neurons[neuronIdx].ID
	candidates := make([]vectorTarget, 0, 1)
	candidates = append(candidates, vectorTarget{bias: true})
	for i := range base.Synapses {
		if !base.Synapses[i].Enabled || base.Synapses[i].To != neuronID {
			continue
		}
		candidates = append(candidates, vectorTarget{synapseIdx: i})
	}

	target := candidates[rng.Intn(len(candidates))]
	var params []float64
	if target.bias {
		params = append([]float64(nil), mutated.Neurons[neuronIdx].PlasticityBiasParams...)
	} else {
		params = append([]float64(nil), mutated.Synapses[target.synapseIdx].PlasticityParams...)
	}
	if len(params) < width {
		params = append(params, make([]float64, width-len(params))...)
	}
	paramIdx := rng.Intn(width)
	params[paramIdx] += delta
	if target.bias {
		mutated.Neurons[neuronIdx].PlasticityBiasParams = params
	} else {
		mutated.Synapses[target.synapseIdx].PlasticityParams = params
	}
	return true
}

func normalizeNonEmptyStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func normalizePlasticityRuleOptions(rules []string) []string {
	seen := make(map[string]struct{}, len(rules))
	out := make([]string, 0, len(rules))
	for _, rule := range rules {
		name := nn.NormalizePlasticityRuleName(rule)
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	return out
}

func defaultPlasticityRules() []string {
	return []string{
		nn.PlasticityNone,
		nn.PlasticityHebbian,
		nn.PlasticityOja,
		nn.PlasticitySelfModulationV1,
		nn.PlasticitySelfModulationV2,
		nn.PlasticitySelfModulationV3,
		nn.PlasticitySelfModulationV4,
		nn.PlasticitySelfModulationV5,
		nn.PlasticitySelfModulationV6,
		nn.PlasticityNeuromodulation,
	}
}

type neuronSpreadTarget struct {
	id         string
	spread     float64
	sourceKind string
	sourceID   string
}

type tuningElementCandidate struct {
	kind       string
	id         string
	generation int
}

const (
	tuningElementNeuron   = "neuron"
	tuningElementActuator = "actuator"
)

func selectedNeuronSpreadsForMutateWeights(
	genome model.Genome,
	rng *rand.Rand,
	baseSpread float64,
	annealing float64,
) []neuronSpreadTarget {
	if (len(genome.Neurons) == 0 && len(genome.ActuatorIDs) == 0) || rng == nil {
		return nil
	}
	if baseSpread <= 0 {
		return nil
	}
	if annealing <= 0 {
		annealing = 1.0
	}

	mode := tuning.NormalizeCandidateSelectionName(genome.Strategy.TuningSelection)
	currentGeneration := currentGenomeGeneration(genome)
	candidates := tuningElementsForMutateWeights(genome, currentGeneration)
	selected := filterTuningElementsByMode(candidates, mode, currentGeneration, rng)
	targets := spreadTargetsFromElements(genome, selected, currentGeneration, baseSpread, annealing)
	if len(targets) == 0 {
		if shouldFallbackToFirstTuningTarget(mode) {
			targets = fallbackSpreadTargetsFromCandidates(genome, candidates, baseSpread)
		}
	}
	if len(targets) == 0 {
		return nil
	}
	if isRandomTuningSelectionMode(mode) {
		return randomNeuronSpreadSubset(targets, rng)
	}
	return targets
}

func shouldFallbackToFirstTuningTarget(mode string) bool {
	switch mode {
	case tuning.CandidateSelectDynamicA,
		tuning.CandidateSelectDynamic,
		tuning.CandidateSelectActiveRnd,
		tuning.CandidateSelectRecentRnd,
		tuning.CandidateSelectCurrent,
		tuning.CandidateSelectCurrentRd,
		tuning.CandidateSelectLastGen,
		tuning.CandidateSelectLastGenRd,
		tuning.CandidateSelectBestSoFar,
		tuning.CandidateSelectOriginal:
		return true
	default:
		return false
	}
}

func fallbackSpreadTargetsFromCandidates(
	genome model.Genome,
	candidates []tuningElementCandidate,
	spread float64,
) []neuronSpreadTarget {
	for _, candidate := range candidates {
		target := neuronSpreadTarget{
			spread:     spread,
			sourceKind: candidate.kind,
			sourceID:   candidate.id,
		}
		switch candidate.kind {
		case tuningElementNeuron:
			if candidate.id == "" || !hasNeuron(genome, candidate.id) {
				continue
			}
			target.id = candidate.id
		case tuningElementActuator:
			if candidate.id == "" || !hasActuator(genome, candidate.id) {
				continue
			}
			target.id = candidate.id
		default:
			continue
		}
		return []neuronSpreadTarget{target}
	}
	if len(genome.Neurons) > 0 {
		return []neuronSpreadTarget{{
			id:         genome.Neurons[0].ID,
			spread:     spread,
			sourceKind: tuningElementNeuron,
			sourceID:   genome.Neurons[0].ID,
		}}
	}
	if len(genome.ActuatorIDs) == 0 {
		return nil
	}
	fallback := genome.ActuatorIDs[0]
	return []neuronSpreadTarget{{
		id:         fallback,
		spread:     spread,
		sourceKind: tuningElementActuator,
		sourceID:   fallback,
	}}
}

func isRandomTuningSelectionMode(mode string) bool {
	switch mode {
	case tuning.CandidateSelectDynamic,
		tuning.CandidateSelectAllRandom,
		tuning.CandidateSelectActiveRnd,
		tuning.CandidateSelectRecentRnd,
		tuning.CandidateSelectCurrentRd,
		tuning.CandidateSelectLastGenRd:
		return true
	default:
		return false
	}
}

func nonRandomTuningSelectionMode(mode string) string {
	switch mode {
	case tuning.CandidateSelectDynamic:
		return tuning.CandidateSelectDynamicA
	case tuning.CandidateSelectAllRandom:
		return tuning.CandidateSelectAll
	case tuning.CandidateSelectActiveRnd:
		return tuning.CandidateSelectActive
	case tuning.CandidateSelectRecentRnd:
		return tuning.CandidateSelectRecent
	case tuning.CandidateSelectCurrentRd:
		return tuning.CandidateSelectCurrent
	case tuning.CandidateSelectLastGenRd:
		return tuning.CandidateSelectLastGen
	default:
		return mode
	}
}

func filterTuningElementsByMode(
	candidates []tuningElementCandidate,
	mode string,
	currentGeneration int,
	rng *rand.Rand,
) []tuningElementCandidate {
	if len(candidates) == 0 {
		return nil
	}
	baseMode := nonRandomTuningSelectionMode(mode)
	switch baseMode {
	case tuning.CandidateSelectDynamicA:
		u := rng.Float64()
		if u <= 0 {
			u = math.SmallestNonzeroFloat64
		}
		return filterTuningElementsByAge(candidates, currentGeneration, math.Sqrt(1/u))
	case tuning.CandidateSelectActive, tuning.CandidateSelectRecent:
		return filterTuningElementsByAge(candidates, currentGeneration, 3)
	case tuning.CandidateSelectCurrent, tuning.CandidateSelectLastGen:
		return filterTuningElementsByAge(candidates, currentGeneration, 0)
	case tuning.CandidateSelectAll, tuning.CandidateSelectBestSoFar, tuning.CandidateSelectOriginal:
		return append([]tuningElementCandidate(nil), candidates...)
	default:
		return append([]tuningElementCandidate(nil), candidates...)
	}
}

func filterTuningElementsByAge(candidates []tuningElementCandidate, currentGeneration int, maxAge float64) []tuningElementCandidate {
	filtered := make([]tuningElementCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		age := currentGeneration - candidate.generation
		if age < 0 {
			age = 0
		}
		if float64(age) <= maxAge {
			filtered = append(filtered, candidate)
		}
	}
	return filtered
}

func tuningElementsForMutateWeights(genome model.Genome, currentGeneration int) []tuningElementCandidate {
	out := make([]tuningElementCandidate, 0, len(genome.Neurons)+len(genome.ActuatorIDs))
	for _, neuron := range genome.Neurons {
		out = append(out, tuningElementCandidate{
			kind:       tuningElementNeuron,
			id:         neuron.ID,
			generation: effectiveNeuronGeneration(neuron, currentGeneration),
		})
	}
	for _, actuatorID := range uniqueStrings(genome.ActuatorIDs) {
		if actuatorID == "" {
			continue
		}
		out = append(out, tuningElementCandidate{
			kind:       tuningElementActuator,
			id:         actuatorID,
			generation: effectiveActuatorGeneration(genome, actuatorID, currentGeneration),
		})
	}
	return out
}

func spreadTargetsFromElements(
	genome model.Genome,
	selected []tuningElementCandidate,
	currentGeneration int,
	baseSpread float64,
	annealing float64,
) []neuronSpreadTarget {
	targets := make([]neuronSpreadTarget, 0, len(selected))
	for _, candidate := range selected {
		age := currentGeneration - candidate.generation
		if age < 0 {
			age = 0
		}
		spread := baseSpread * math.Pow(annealing, float64(age))
		if spread <= 0 {
			spread = baseSpread
		}
		target := neuronSpreadTarget{
			spread:     spread,
			sourceKind: candidate.kind,
			sourceID:   candidate.id,
		}
		switch candidate.kind {
		case tuningElementNeuron:
			if candidate.id == "" || !hasNeuron(genome, candidate.id) {
				continue
			}
			target.id = candidate.id
		case tuningElementActuator:
			if candidate.id == "" || !hasActuator(genome, candidate.id) {
				continue
			}
			target.id = candidate.id
		default:
			continue
		}
		targets = append(targets, target)
	}
	return targets
}

func currentGenomeGeneration(genome model.Genome) int {
	if gen, ok := inferGenerationFromTaggedID(genome.ID); ok {
		return gen
	}
	maxGen := 0
	for _, neuron := range genome.Neurons {
		if neuron.Generation > maxGen {
			maxGen = neuron.Generation
		}
	}
	for _, actuatorGen := range genome.ActuatorGenerations {
		if actuatorGen > maxGen {
			maxGen = actuatorGen
		}
	}
	for _, actuatorID := range genome.ActuatorIDs {
		if gen, ok := inferGenerationFromTaggedID(actuatorID); ok && gen > maxGen {
			maxGen = gen
		}
	}
	return maxGen
}

func effectiveNeuronGeneration(neuron model.Neuron, fallback int) int {
	switch {
	case neuron.Generation > 0:
		return neuron.Generation
	case neuron.ID != "":
		if gen, ok := inferGenerationFromTaggedID(neuron.ID); ok {
			return gen
		}
	}
	return fallback
}

func effectiveActuatorGeneration(genome model.Genome, actuatorID string, fallback int) int {
	if genome.ActuatorGenerations != nil {
		if generation, ok := genome.ActuatorGenerations[actuatorID]; ok && generation > 0 {
			return generation
		}
	}
	if generation, ok := inferGenerationFromTaggedID(actuatorID); ok {
		return generation
	}
	return fallback
}

func inferGenerationFromTaggedID(id string) (int, bool) {
	if id == "" {
		return 0, false
	}
	parts := strings.Split(id, "-")
	for _, part := range parts {
		if len(part) > 1 && part[0] == 'g' {
			gen, err := strconv.Atoi(part[1:])
			if err == nil {
				return gen, true
			}
		}
	}
	return 0, false
}

func randomNeuronSpreadSubset(targets []neuronSpreadTarget, rng *rand.Rand) []neuronSpreadTarget {
	if len(targets) == 0 {
		return nil
	}
	if len(targets) == 1 {
		return append([]neuronSpreadTarget(nil), targets...)
	}
	subset := make([]neuronSpreadTarget, 0, len(targets))
	mp := 1 / math.Sqrt(float64(len(targets)))
	for _, target := range targets {
		if rng.Float64() < mp {
			subset = append(subset, target)
		}
	}
	if len(subset) > 0 {
		return subset
	}
	return []neuronSpreadTarget{targets[rng.Intn(len(targets))]}
}

func perturbActuatorTunable(genome *model.Genome, actuatorID string, spread float64, rng *rand.Rand) bool {
	if genome == nil || actuatorID == "" || spread <= 0 || rng == nil {
		return false
	}
	if genome.ActuatorTunables == nil {
		genome.ActuatorTunables = map[string]float64{}
	}
	delta := (rng.Float64()*2 - 1) * spread
	genome.ActuatorTunables[actuatorID] += delta
	return true
}

func touchNeuronGeneration(neurons []model.Neuron, neuronID string, generation int) {
	if generation < 0 {
		generation = 0
	}
	for i := range neurons {
		if neurons[i].ID != neuronID {
			continue
		}
		neurons[i].Generation = generation
		return
	}
}

func touchActuatorGeneration(genome *model.Genome, actuatorID string, generation int) {
	if genome == nil || actuatorID == "" {
		return
	}
	if generation < 0 {
		generation = 0
	}
	if genome.ActuatorGenerations == nil {
		genome.ActuatorGenerations = map[string]int{}
	}
	genome.ActuatorGenerations[actuatorID] = generation
}

func deleteActuatorGeneration(genome *model.Genome, actuatorID string) {
	if genome == nil || genome.ActuatorGenerations == nil || actuatorID == "" {
		return
	}
	delete(genome.ActuatorGenerations, actuatorID)
	if len(genome.ActuatorGenerations) == 0 {
		genome.ActuatorGenerations = nil
	}
}

func deleteActuatorTunable(genome *model.Genome, actuatorID string) {
	if genome == nil || genome.ActuatorTunables == nil || actuatorID == "" {
		return
	}
	delete(genome.ActuatorTunables, actuatorID)
	if len(genome.ActuatorTunables) == 0 {
		genome.ActuatorTunables = nil
	}
}

func incomingSynapseIndexes(genome model.Genome, neuronID string) []int {
	indexes := make([]int, 0, len(genome.Synapses))
	for i, syn := range genome.Synapses {
		if syn.To == neuronID {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

func availableSensorNeuronPairs(genome model.Genome) []model.SensorNeuronLink {
	if len(genome.SensorIDs) == 0 || len(genome.Neurons) == 0 {
		return nil
	}
	sensors := uniqueStrings(genome.SensorIDs)
	pairs := make([]model.SensorNeuronLink, 0, len(sensors)*len(genome.Neurons))
	for _, sensorID := range sensors {
		for _, neuron := range genome.Neurons {
			if hasSensorNeuronLink(genome, sensorID, neuron.ID) {
				continue
			}
			pairs = append(pairs, model.SensorNeuronLink{
				SensorID: sensorID,
				NeuronID: neuron.ID,
			})
		}
	}
	return pairs
}

func availableNeuronActuatorPairs(genome model.Genome) []model.NeuronActuatorLink {
	if len(genome.ActuatorIDs) == 0 || len(genome.Neurons) == 0 {
		return nil
	}
	actuators := uniqueStrings(genome.ActuatorIDs)
	pairs := make([]model.NeuronActuatorLink, 0, len(actuators)*len(genome.Neurons))
	for _, neuron := range genome.Neurons {
		for _, actuatorID := range actuators {
			if hasNeuronActuatorLink(genome, neuron.ID, actuatorID) {
				continue
			}
			pairs = append(pairs, model.NeuronActuatorLink{
				NeuronID:   neuron.ID,
				ActuatorID: actuatorID,
			})
		}
	}
	return pairs
}

func hasSensorNeuronLink(genome model.Genome, sensorID, neuronID string) bool {
	for _, link := range genome.SensorNeuronLinks {
		if link.SensorID == sensorID && link.NeuronID == neuronID {
			return true
		}
	}
	return false
}

func hasNeuronActuatorLink(genome model.Genome, neuronID, actuatorID string) bool {
	for _, link := range genome.NeuronActuatorLinks {
		if link.NeuronID == neuronID && link.ActuatorID == actuatorID {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func syncIOLinkCounts(genome *model.Genome) {
	genome.SensorLinks = len(genome.SensorNeuronLinks)
	genome.ActuatorLinks = len(genome.NeuronActuatorLinks)
}

func availableCPPChoices(genome model.Genome) []string {
	if genome.Substrate == nil {
		return nil
	}
	return filterOutString(substrate.ListCPPs(), genome.Substrate.CPPName)
}

func availableCEPChoices(genome model.Genome) []string {
	if genome.Substrate == nil {
		return nil
	}
	return filterOutString(substrate.ListCEPs(), genome.Substrate.CEPName)
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
