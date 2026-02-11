package evo

import (
	"context"
	"errors"
	"fmt"
	"math/rand"

	"protogonos/internal/genotype"
	"protogonos/internal/model"
)

var (
	ErrNoSynapses = errors.New("genome has no synapses")
	ErrNoNeurons  = errors.New("genome has no neurons")
)

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

// AddRandomSynapse adds a random synapse between existing neurons.
type AddRandomSynapse struct {
	Rand         *rand.Rand
	MaxAbsWeight float64
}

func (o *AddRandomSynapse) Name() string {
	return "add_random_synapse"
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

// RemoveRandomSynapse removes a random synapse.
type RemoveRandomSynapse struct {
	Rand *rand.Rand
}

func (o *RemoveRandomSynapse) Name() string {
	return "remove_random_synapse"
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

// AddRandomNeuron inserts a neuron by splitting a random synapse.
type AddRandomNeuron struct {
	Rand        *rand.Rand
	Activations []string
}

func (o *AddRandomNeuron) Name() string {
	return "add_random_neuron"
}

func (o *AddRandomNeuron) Apply(ctx context.Context, genome model.Genome) (model.Genome, error) {
	if o == nil || o.Rand == nil {
		return model.Genome{}, errors.New("random source is required")
	}
	if len(genome.Synapses) == 0 {
		return model.Genome{}, ErrNoSynapses
	}

	activations := o.Activations
	if len(activations) == 0 {
		activations = []string{"identity", "relu", "tanh", "sigmoid"}
	}

	activation := activations[o.Rand.Intn(len(activations))]
	op := AddNeuronAtSynapse{
		SynapseIndex: o.Rand.Intn(len(genome.Synapses)),
		NeuronID:     uniqueNeuronID(genome, o.Rand),
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
