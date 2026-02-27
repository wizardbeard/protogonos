package substrate

import "context"

// CPP computes a candidate weight update signal from substrate inputs.
type CPP interface {
	Name() string
	Compute(ctx context.Context, inputs []float64, params map[string]float64) (float64, error)
}

// VectorCPP is an optional CPP capability that produces an ordered control
// signal vector for CEP fan-in processing.
type VectorCPP interface {
	ComputeVector(ctx context.Context, inputs []float64, params map[string]float64) ([]float64, error)
}

// CEP applies a computed update signal to an existing weight.
type CEP interface {
	Name() string
	Apply(ctx context.Context, current float64, delta float64, params map[string]float64) (float64, error)
}

// Spec configures a substrate runtime instance.
type Spec struct {
	CPPName      string
	CEPName      string
	CEPNames     []string
	CEPFaninPIDs []string
	Dimensions   []int
	Parameters   map[string]float64
}

// Runtime executes substrate update steps over an internal weight vector.
type Runtime interface {
	Step(ctx context.Context, inputs []float64) ([]float64, error)
	Weights() []float64
}

// StatefulRuntime is an optional runtime capability mirroring substrate
// lifecycle controls used by the reference runtime (backup/revert/reset).
type StatefulRuntime interface {
	Runtime
	Backup()
	Restore() error
	Reset()
}

// TerminableRuntime is an optional runtime capability mirroring explicit
// terminate lifecycle semantics from process-oriented substrate actors.
type TerminableRuntime interface {
	Runtime
	Terminate()
}
