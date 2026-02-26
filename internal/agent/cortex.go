package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"protogonos/internal/genotype"
	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/nn"
	"protogonos/internal/substrate"
)

type CortexStatus string

const (
	CortexStatusActive     CortexStatus = "active"
	CortexStatusInactive   CortexStatus = "inactive"
	CortexStatusTerminated CortexStatus = "terminated"
)

var (
	ErrCortexInactive   = errors.New("cortex is inactive")
	ErrCortexTerminated = errors.New("cortex is terminated")
	ErrNoWeightBackup   = errors.New("no cortex weight backup available")
	ErrNoSynapses       = errors.New("no synapses available for perturbation")
)

type EvaluationReport struct {
	Fitness      []float64
	Cycles       int
	EndFlagTotal int
	GoalReached  bool
	Completed    bool
	Duration     time.Duration
}

type ActuatorSyncFeedback struct {
	Fitness     []float64
	EndFlag     int
	GoalReached bool
}

// ActuatorSyncReporter is an optional actuator capability used by
// reference-style cortex episode loops to aggregate per-cycle fitness/end flags.
type ActuatorSyncReporter interface {
	ConsumeSyncFeedback() (ActuatorSyncFeedback, bool)
}

type Cortex struct {
	id              string
	genome          model.Genome
	sensors         map[string]protoio.Sensor
	actuators       map[string]protoio.Actuator
	inputNeuronIDs  []string
	outputNeuronIDs []string
	substrate       substrate.Runtime
	nnState         *nn.ForwardState
	mu              sync.Mutex
	status          CortexStatus
	weightBackup    *model.Genome
}

func NewCortex(
	id string,
	genome model.Genome,
	sensors map[string]protoio.Sensor,
	actuators map[string]protoio.Actuator,
	inputNeuronIDs []string,
	outputNeuronIDs []string,
	substrateRuntime substrate.Runtime,
) (*Cortex, error) {
	if id == "" {
		return nil, fmt.Errorf("agent id is required")
	}
	if len(inputNeuronIDs) == 0 {
		return nil, fmt.Errorf("input neuron ids are required")
	}
	if len(outputNeuronIDs) == 0 {
		return nil, fmt.Errorf("output neuron ids are required")
	}

	return &Cortex{
		id:              id,
		genome:          genome,
		sensors:         sensors,
		actuators:       actuators,
		inputNeuronIDs:  append([]string(nil), inputNeuronIDs...),
		outputNeuronIDs: append([]string(nil), outputNeuronIDs...),
		substrate:       substrateRuntime,
		nnState:         nn.NewForwardState(),
		status:          CortexStatusActive,
	}, nil
}

func (c *Cortex) ID() string {
	return c.id
}

func (c *Cortex) RegisteredSensor(id string) (protoio.Sensor, bool) {
	if c.sensors == nil {
		return nil, false
	}
	s, ok := c.sensors[id]
	return s, ok
}

func (c *Cortex) RegisteredActuator(id string) (protoio.Actuator, bool) {
	if c.actuators == nil {
		return nil, false
	}
	a, ok := c.actuators[id]
	return a, ok
}

func (c *Cortex) Status() CortexStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.status
}

func (c *Cortex) Reactivate() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.status == CortexStatusTerminated {
		return ErrCortexTerminated
	}
	c.nnState = nn.NewForwardState()
	if managed, ok := c.substrate.(substrate.StatefulRuntime); ok {
		managed.Reset()
	}
	c.status = CortexStatusActive
	return nil
}

func (c *Cortex) Terminate() {
	c.mu.Lock()
	c.status = CortexStatusTerminated
	c.mu.Unlock()
}

func (c *Cortex) BackupWeights() {
	c.mu.Lock()
	backup := genotype.CloneGenome(c.genome)
	c.weightBackup = &backup
	if managed, ok := c.substrate.(substrate.StatefulRuntime); ok {
		managed.Backup()
	}
	c.mu.Unlock()
}

func (c *Cortex) SnapshotGenome() model.Genome {
	c.mu.Lock()
	defer c.mu.Unlock()
	return genotype.CloneGenome(c.genome)
}

func (c *Cortex) ApplyGenome(genome model.Genome) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.status == CortexStatusTerminated {
		return ErrCortexTerminated
	}
	c.genome = genotype.CloneGenome(genome)
	c.nnState = nn.NewForwardState()
	return nil
}

func (c *Cortex) RestoreWeights() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.weightBackup == nil {
		return ErrNoWeightBackup
	}
	if managed, ok := c.substrate.(substrate.StatefulRuntime); ok {
		if err := managed.Restore(); err != nil {
			return err
		}
	}
	c.genome = genotype.CloneGenome(*c.weightBackup)
	c.nnState = nn.NewForwardState()
	return nil
}

func (c *Cortex) ClearWeightBackup() {
	c.mu.Lock()
	c.weightBackup = nil
	c.mu.Unlock()
}

func (c *Cortex) PerturbWeights(rng *rand.Rand, spread float64) error {
	if rng == nil {
		return fmt.Errorf("random source is required")
	}
	if spread <= 0 {
		return fmt.Errorf("spread must be > 0")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.genome.Synapses) == 0 {
		return ErrNoSynapses
	}
	enabled := make([]int, 0, len(c.genome.Synapses))
	for i := range c.genome.Synapses {
		if c.genome.Synapses[i].Enabled {
			enabled = append(enabled, i)
		}
	}
	if len(enabled) == 0 {
		return ErrNoSynapses
	}
	mp := 1 / math.Sqrt(float64(len(enabled)))
	changed := 0
	for _, idx := range enabled {
		if rng.Float64() >= mp {
			continue
		}
		delta := (rng.Float64()*2 - 1) * spread
		next := c.genome.Synapses[idx].Weight + delta
		c.genome.Synapses[idx].Weight = saturateWeight(next)
		changed++
	}
	if changed == 0 {
		idx := enabled[rng.Intn(len(enabled))]
		delta := (rng.Float64()*2 - 1) * spread
		next := c.genome.Synapses[idx].Weight + delta
		c.genome.Synapses[idx].Weight = saturateWeight(next)
	}
	return nil
}

func (c *Cortex) Tick(ctx context.Context) ([]float64, error) {
	inputs := make([]float64, 0, len(c.genome.SensorIDs))
	for _, sensorID := range c.genome.SensorIDs {
		sensor, ok := c.sensors[sensorID]
		if !ok {
			return nil, fmt.Errorf("sensor not registered: %s", sensorID)
		}
		values, err := sensor.Read(ctx)
		if err != nil {
			return nil, err
		}
		inputs = append(inputs, values...)
	}

	return c.execute(ctx, inputs)
}

func (c *Cortex) RunStep(ctx context.Context, inputs []float64) ([]float64, error) {
	return c.execute(ctx, inputs)
}

func (c *Cortex) RunUntilEvaluationComplete(ctx context.Context, maxCycles int) (EvaluationReport, error) {
	report := EvaluationReport{}
	if maxCycles <= 0 {
		return report, fmt.Errorf("max cycles must be > 0")
	}
	if status := c.Status(); status == CortexStatusTerminated {
		return report, ErrCortexTerminated
	} else if status == CortexStatusInactive {
		return report, ErrCortexInactive
	}

	start := time.Now()
	for cycle := 0; cycle < maxCycles; cycle++ {
		if _, err := c.Tick(ctx); err != nil {
			return report, err
		}
		report.Cycles++

		cycleFitness, cycleEndFlag, cycleGoalReached := c.consumeActuatorSyncFeedback()
		report.Fitness = addFitnessVectors(report.Fitness, cycleFitness)
		report.EndFlagTotal += cycleEndFlag
		if cycleGoalReached {
			report.GoalReached = true
		}
		if report.EndFlagTotal > 0 || report.GoalReached {
			report.Completed = true
			report.Duration = time.Since(start)
			c.mu.Lock()
			if c.status != CortexStatusTerminated {
				c.status = CortexStatusInactive
			}
			c.mu.Unlock()
			return report, nil
		}
	}

	report.Duration = time.Since(start)
	return report, nil
}

func (c *Cortex) execute(ctx context.Context, inputs []float64) ([]float64, error) {
	status := c.Status()
	switch status {
	case CortexStatusTerminated:
		return nil, ErrCortexTerminated
	case CortexStatusInactive:
		return nil, ErrCortexInactive
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(inputs) != len(c.inputNeuronIDs) {
		return nil, fmt.Errorf("input size mismatch: got=%d want=%d", len(inputs), len(c.inputNeuronIDs))
	}

	inputByNeuron := make(map[string]float64, len(c.inputNeuronIDs))
	for i, neuronID := range c.inputNeuronIDs {
		inputByNeuron[neuronID] = inputs[i]
	}

	values, err := nn.ForwardWithState(c.genome, inputByNeuron, c.nnState)
	if err != nil {
		return nil, err
	}
	if c.genome.Plasticity != nil {
		if err := nn.ApplyPlasticity(&c.genome, values, *c.genome.Plasticity); err != nil {
			return nil, err
		}
	}

	outputs := make([]float64, len(c.outputNeuronIDs))
	for i, neuronID := range c.outputNeuronIDs {
		outputs[i] = values[neuronID]
	}
	if c.substrate != nil {
		substrateOutputs, err := c.substrate.Step(ctx, outputs)
		if err != nil {
			return nil, err
		}
		if len(substrateOutputs) >= len(outputs) {
			copy(outputs, substrateOutputs[:len(outputs)])
		}
	}

	if len(c.genome.ActuatorIDs) > 0 {
		chunks, err := splitOutputsForActuators(outputs, len(c.genome.ActuatorIDs))
		if err != nil {
			return nil, err
		}
		for i, actuatorID := range c.genome.ActuatorIDs {
			actuator, ok := c.actuators[actuatorID]
			if !ok {
				return nil, fmt.Errorf("actuator not registered: %s", actuatorID)
			}
			chunk := chunks[i]
			if c.genome.ActuatorTunables != nil {
				if offset, ok := c.genome.ActuatorTunables[actuatorID]; ok && offset != 0 {
					chunk = applyActuatorOffset(chunk, offset)
				}
			}
			if err := actuator.Write(ctx, chunk); err != nil {
				return nil, err
			}
		}
	}

	return outputs, nil
}

func splitOutputsForActuators(outputs []float64, actuatorCount int) ([][]float64, error) {
	if actuatorCount <= 0 {
		return nil, fmt.Errorf("actuator count must be > 0")
	}
	// Keep one-to-many compatibility: a single actuator receives the full output
	// vector, while N actuators receive equal contiguous slices.
	if actuatorCount == 1 {
		return [][]float64{append([]float64(nil), outputs...)}, nil
	}
	if len(outputs)%actuatorCount != 0 {
		return nil, fmt.Errorf("actuator/output shape mismatch: outputs=%d actuators=%d", len(outputs), actuatorCount)
	}
	chunkSize := len(outputs) / actuatorCount
	if chunkSize <= 0 {
		return nil, fmt.Errorf("actuator/output shape mismatch: outputs=%d actuators=%d", len(outputs), actuatorCount)
	}
	chunks := make([][]float64, 0, actuatorCount)
	for i := 0; i < actuatorCount; i++ {
		start := i * chunkSize
		end := start + chunkSize
		chunks = append(chunks, append([]float64(nil), outputs[start:end]...))
	}
	return chunks, nil
}

func applyActuatorOffset(values []float64, offset float64) []float64 {
	if offset == 0 {
		return values
	}
	out := append([]float64(nil), values...)
	for i := range out {
		out[i] += offset
	}
	return out
}

func saturateWeight(weight float64) float64 {
	const limit = math.Pi * 10
	if weight > limit {
		return limit
	}
	if weight < -limit {
		return -limit
	}
	return weight
}

func addFitnessVectors(acc, values []float64) []float64 {
	if len(values) == 0 {
		return acc
	}
	if len(acc) == 0 {
		return append([]float64(nil), values...)
	}
	out := append([]float64(nil), acc...)
	if len(values) > len(out) {
		out = append(out, make([]float64, len(values)-len(out))...)
	}
	for i, value := range values {
		out[i] += value
	}
	return out
}

func (c *Cortex) consumeActuatorSyncFeedback() ([]float64, int, bool) {
	fitness := []float64(nil)
	endFlag := 0
	goalReached := false
	for _, actuatorID := range c.genome.ActuatorIDs {
		actuator, ok := c.actuators[actuatorID]
		if !ok {
			continue
		}
		reporter, ok := actuator.(ActuatorSyncReporter)
		if !ok {
			continue
		}
		feedback, ok := reporter.ConsumeSyncFeedback()
		if !ok {
			continue
		}
		fitness = addFitnessVectors(fitness, feedback.Fitness)
		if feedback.EndFlag > 0 {
			endFlag += feedback.EndFlag
		}
		if feedback.GoalReached {
			goalReached = true
			if feedback.EndFlag <= 0 {
				endFlag++
			}
		}
	}
	return fitness, endFlag, goalReached
}
