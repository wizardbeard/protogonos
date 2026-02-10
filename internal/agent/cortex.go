package agent

import (
	"context"
	"fmt"

	protoio "protogonos/internal/io"
	"protogonos/internal/model"
	"protogonos/internal/nn"
)

type Cortex struct {
	id              string
	genome          model.Genome
	sensors         map[string]protoio.Sensor
	actuators       map[string]protoio.Actuator
	inputNeuronIDs  []string
	outputNeuronIDs []string
}

func NewCortex(
	id string,
	genome model.Genome,
	sensors map[string]protoio.Sensor,
	actuators map[string]protoio.Actuator,
	inputNeuronIDs []string,
	outputNeuronIDs []string,
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

	values, err := nn.Forward(c.genome, inputByNeuron)
	if err != nil {
		return nil, err
	}

	outputs := make([]float64, len(c.outputNeuronIDs))
	for i, neuronID := range c.outputNeuronIDs {
		outputs[i] = values[neuronID]
	}

	if len(c.genome.ActuatorIDs) > 0 {
		if len(c.genome.ActuatorIDs) != len(outputs) {
			return nil, fmt.Errorf("actuator count mismatch: outputs=%d actuators=%d", len(outputs), len(c.genome.ActuatorIDs))
		}
		for i, actuatorID := range c.genome.ActuatorIDs {
			actuator, ok := c.actuators[actuatorID]
			if !ok {
				return nil, fmt.Errorf("actuator not registered: %s", actuatorID)
			}
			if err := actuator.Write(ctx, []float64{outputs[i]}); err != nil {
				return nil, err
			}
		}
	}

	return outputs, nil
}
