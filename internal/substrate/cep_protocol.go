package substrate

import (
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
)

var (
	ErrCEPProcessTerminated    = errors.New("cep process terminated")
	ErrUnexpectedCEPForwardPID = errors.New("unexpected cep fan-in sender")
	ErrInvalidCEPOutputWidth   = errors.New("invalid cep output width")
	ErrUnsupportedCEPCommand   = errors.New("unsupported cep command")
)

// CEPCommand mirrors the reference CEP->substrate message shape:
// {CEP_Pid, Command, Signal}.
type CEPCommand struct {
	FromPID string
	Command string
	Signal  []float64
}

// CEPProcess is a simplified Go analog of substrate_cep actor loop semantics:
// ordered fan-in accumulation, per-cycle command emission, and explicit
// terminate behavior.
type CEPProcess struct {
	id          string
	cepName     string
	parameters  map[string]float64
	faninPIDs   []string
	expectedIdx int
	acc         []float64
	terminated  bool
}

var cepProcessCounter uint64

func NewCEPProcess(cepName string, parameters map[string]float64, faninPIDs []string) (*CEPProcess, error) {
	return NewCEPProcessWithID("", cepName, parameters, faninPIDs)
}

func NewCEPProcessWithID(id string, cepName string, parameters map[string]float64, faninPIDs []string) (*CEPProcess, error) {
	if len(faninPIDs) == 0 {
		return nil, fmt.Errorf("fanin pids are required")
	}
	processID := strings.TrimSpace(id)
	if processID == "" {
		processID = fmt.Sprintf("cep_%d", atomic.AddUint64(&cepProcessCounter, 1))
	}
	out := &CEPProcess{
		id:         processID,
		cepName:    cepName,
		parameters: cloneFloatMap(parameters),
		faninPIDs:  append([]string(nil), faninPIDs...),
	}
	return out, nil
}

func (p *CEPProcess) ID() string {
	return p.id
}

func (p *CEPProcess) Terminate() {
	p.terminated = true
}

func (p *CEPProcess) Forward(fromPID string, input []float64) (CEPCommand, bool, error) {
	if p.terminated {
		return CEPCommand{}, false, ErrCEPProcessTerminated
	}
	if len(p.faninPIDs) == 0 {
		return CEPCommand{}, false, fmt.Errorf("fanin pids are required")
	}
	if p.expectedIdx >= len(p.faninPIDs) {
		p.expectedIdx = 0
	}
	expected := p.faninPIDs[p.expectedIdx]
	if fromPID != expected {
		return CEPCommand{}, false, fmt.Errorf("%w: expected=%s got=%s", ErrUnexpectedCEPForwardPID, expected, fromPID)
	}

	// Preserve the reference accumulation choreography:
	// lists:append(Input, Acc), then lists:reverse(Acc) at cycle end.
	chunk := append([]float64(nil), input...)
	p.acc = append(chunk, p.acc...)
	p.expectedIdx++
	if p.expectedIdx < len(p.faninPIDs) {
		return CEPCommand{}, false, nil
	}

	proper := reverseFloatSlice(p.acc)
	p.acc = p.acc[:0]
	p.expectedIdx = 0

	command, err := BuildCEPCommand(p.cepName, proper, p.parameters)
	if err != nil {
		return CEPCommand{}, false, err
	}
	command.FromPID = p.id
	return command, true, nil
}

func BuildCEPCommand(cepName string, output []float64, parameters map[string]float64) (CEPCommand, error) {
	switch cepName {
	case SetWeightCEPName:
		if len(output) != 1 {
			return CEPCommand{}, fmt.Errorf("%w: set_weight expects 1 signal, got=%d", ErrInvalidCEPOutputWidth, len(output))
		}
		weight := cepControlValue(output[0], parameters)
		return CEPCommand{
			Command: SetWeightCEPName,
			Signal:  []float64{weight},
		}, nil
	case SetABCNCEPName:
		return CEPCommand{
			Command: SetABCNCEPName,
			Signal:  append([]float64(nil), output...),
		}, nil
	case DefaultCEPName, SetIterativeCEPName:
		if len(output) != 1 {
			return CEPCommand{}, fmt.Errorf("%w: delta_weight expects 1 signal, got=%d", ErrInvalidCEPOutputWidth, len(output))
		}
		delta := cepControlValue(output[0], parameters)
		return CEPCommand{
			Command: SetIterativeCEPName,
			Signal:  []float64{delta},
		}, nil
	default:
		return CEPCommand{}, fmt.Errorf("%w: %s", ErrUnsupportedCEPCommand, cepName)
	}
}

func ApplyCEPCommand(current float64, command CEPCommand, parameters map[string]float64) (float64, error) {
	switch strings.TrimSpace(command.Command) {
	case SetWeightCEPName:
		if len(command.Signal) != 1 {
			return 0, fmt.Errorf("%w: set_weight expects 1 signal, got=%d", ErrInvalidCEPOutputWidth, len(command.Signal))
		}
		return saturateSubstrateWeight(command.Signal[0]), nil
	case SetIterativeCEPName, DefaultCEPName:
		if len(command.Signal) != 1 {
			return 0, fmt.Errorf("%w: set_iterative expects 1 signal, got=%d", ErrInvalidCEPOutputWidth, len(command.Signal))
		}
		return saturateSubstrateWeight(current + command.Signal[0]), nil
	case SetABCNCEPName:
		if len(command.Signal) == 0 {
			return 0, fmt.Errorf("%w: set_abcn expects at least 1 signal, got=0", ErrInvalidCEPOutputWidth)
		}
		params := cloneFloatMap(parameters)
		if len(command.Signal) >= 5 {
			if params == nil {
				params = map[string]float64{}
			}
			params["A"] = command.Signal[1]
			params["B"] = command.Signal[2]
			params["C"] = command.Signal[3]
			params["N"] = command.Signal[4]
		}
		return (SetABCNCEP{}).Apply(nil, current, command.Signal[0], params)
	default:
		return 0, fmt.Errorf("%w: %s", ErrUnsupportedCEPCommand, command.Command)
	}
}

func cloneFloatMap(in map[string]float64) map[string]float64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func reverseFloatSlice(values []float64) []float64 {
	if len(values) == 0 {
		return nil
	}
	out := append([]float64(nil), values...)
	for left, right := 0, len(out)-1; left < right; left, right = left+1, right-1 {
		out[left], out[right] = out[right], out[left]
	}
	return out
}
