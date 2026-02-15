package agent

import (
	"context"
	"fmt"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/nn"
	"protogonos/internal/substrate"
)

type Cortex struct {
	id              string
	genome          model.Genome
	sensors         map[string]protoio.Sensor
	actuators       map[string]protoio.Actuator
	inputNeuronIDs  []string
	outputNeuronIDs []string
	substrate       substrate.Runtime
	nnState         *nn.ForwardState
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

func (c *Cortex) execute(ctx context.Context, inputs []float64) ([]float64, error) {
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
			if err := actuator.Write(ctx, chunks[i]); err != nil {
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
