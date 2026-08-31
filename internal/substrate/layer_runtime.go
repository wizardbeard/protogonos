package substrate

import (
	"context"
	"fmt"
	"sync"
)

// LayerRuntimeSpec configures a typed coordinate-layer substrate runtime.
type LayerRuntimeSpec struct {
	Plasticity   string
	LinkForm     string
	StateMode    string
	Substrate    []CoordinateHyperlayer
	ABCN         ABCNSubstrate
	StaticCPP    CoordinateCPP
	IterativeCPP CoordinateIOWCPP
	CEPs         []CEP
	Parameters   map[string]float64
}

// LayerRuntime executes already-materialized substrate coordinate layers through
// the typed output lifecycle helpers.
type LayerRuntime struct {
	mu sync.Mutex

	plasticity   string
	linkForm     string
	stateMode    string
	substrate    []CoordinateHyperlayer
	abcn         ABCNSubstrate
	staticCPP    CoordinateCPP
	iterativeCPP CoordinateIOWCPP
	ceps         []CEP
	parameters   map[string]float64

	initialStateMode string
	initialSubstrate []CoordinateHyperlayer
	initialABCN      ABCNSubstrate
	backup           *layerRuntimeSnapshot
	terminated       bool
}

// LayerRuntimeSnapshot is a read-only copy of typed substrate runtime state.
type LayerRuntimeSnapshot struct {
	Plasticity string                 `json:"plasticity"`
	LinkForm   string                 `json:"link_form"`
	StateMode  string                 `json:"state_mode"`
	Terminated bool                   `json:"terminated"`
	Substrate  []CoordinateHyperlayer `json:"substrate,omitempty"`
	ABCN       ABCNSubstrate          `json:"abcn,omitempty"`
	Weights    []float64              `json:"weights,omitempty"`
}

type layerRuntimeSnapshot struct {
	stateMode string
	substrate []CoordinateHyperlayer
	abcn      ABCNSubstrate
}

// NewLayerRuntime creates a stateful runtime around materialized scalar or ABCN
// substrate layers.
func NewLayerRuntime(spec LayerRuntimeSpec) (*LayerRuntime, error) {
	plasticity := normalizeSubstratePlasticity(spec.Plasticity)
	if plasticity == "" {
		return nil, fmt.Errorf("%w: unsupported substrate plasticity %q", ErrInvalidSubstrateCoordinates, spec.Plasticity)
	}
	stateMode := normalizeSubstrateStateMode(spec.StateMode)
	if stateMode == "" {
		return nil, fmt.Errorf("%w: unsupported substrate state mode %q", ErrInvalidSubstrateCoordinates, spec.StateMode)
	}
	linkForm := spec.LinkForm
	if linkForm == "" {
		linkForm = LinkFormL2LFeedforward
	}

	scalar := cloneCoordinateHyperlayers(spec.Substrate)
	abcn := cloneABCNSubstrate(spec.ABCN)
	if plasticity == SubstratePlasticityABCN {
		if len(abcn.InputLayer) == 0 || len(abcn.Layers) == 0 {
			return nil, fmt.Errorf("%w: abcn substrate requires input and process layers", ErrInvalidSubstrateCoordinates)
		}
	} else if len(scalar) < 2 {
		return nil, fmt.Errorf("%w: substrate must include input and process/output layers", ErrInvalidSubstrateCoordinates)
	}

	return &LayerRuntime{
		plasticity:       plasticity,
		linkForm:         linkForm,
		stateMode:        stateMode,
		substrate:        scalar,
		abcn:             abcn,
		staticCPP:        spec.StaticCPP,
		iterativeCPP:     spec.IterativeCPP,
		ceps:             append([]CEP(nil), spec.CEPs...),
		parameters:       cloneFloatParams(spec.Parameters),
		initialStateMode: stateMode,
		initialSubstrate: cloneCoordinateHyperlayers(scalar),
		initialABCN:      cloneABCNSubstrate(abcn),
	}, nil
}

// Step advances the substrate state and returns the current output layer values.
func (r *LayerRuntime) Step(ctx context.Context, inputs []float64) ([]float64, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.terminated {
		return nil, ErrSubstrateRuntimeTerminated
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	result, err := CalculateOutputLifecycle(OutputLifecycleRequest{
		StateMode:    r.stateMode,
		Plasticity:   r.plasticity,
		LinkForm:     r.linkForm,
		InputValues:  [][]float64{append([]float64(nil), inputs...)},
		Substrate:    cloneCoordinateHyperlayers(r.substrate),
		ABCN:         cloneABCNSubstrate(r.abcn),
		StaticCPP:    r.staticCPP,
		IterativeCPP: r.iterativeCPP,
		CEPs:         append([]CEP(nil), r.ceps...),
		Parameters:   cloneFloatParams(r.parameters),
		Context:      ctx,
	})
	if err != nil {
		return nil, err
	}

	r.stateMode = result.StateMode
	r.substrate = cloneCoordinateHyperlayers(result.Substrate)
	r.abcn = cloneABCNSubstrate(result.ABCNSubstrate)
	return append([]float64(nil), result.Outputs...), nil
}

// Weights returns a flat copy of the substrate's current scalar weight surface.
func (r *LayerRuntime) Weights() []float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.weightsLocked()
}

// Snapshot returns a deep copy of the typed runtime state for diagnostics and
// artifact export paths.
func (r *LayerRuntime) Snapshot() LayerRuntimeSnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()

	return LayerRuntimeSnapshot{
		Plasticity: r.plasticity,
		LinkForm:   r.linkForm,
		StateMode:  r.stateMode,
		Terminated: r.terminated,
		Substrate:  cloneCoordinateHyperlayers(r.substrate),
		ABCN:       cloneABCNSubstrate(r.abcn),
		Weights:    r.weightsLocked(),
	}
}

func CloneLayerRuntimeSnapshot(snapshot *LayerRuntimeSnapshot) *LayerRuntimeSnapshot {
	if snapshot == nil {
		return nil
	}
	return &LayerRuntimeSnapshot{
		Plasticity: snapshot.Plasticity,
		LinkForm:   snapshot.LinkForm,
		StateMode:  snapshot.StateMode,
		Terminated: snapshot.Terminated,
		Substrate:  cloneCoordinateHyperlayers(snapshot.Substrate),
		ABCN:       cloneABCNSubstrate(snapshot.ABCN),
		Weights:    append([]float64(nil), snapshot.Weights...),
	}
}

// Backup saves the current runtime state for a later Restore.
func (r *LayerRuntime) Backup() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.backup = &layerRuntimeSnapshot{
		stateMode: r.stateMode,
		substrate: cloneCoordinateHyperlayers(r.substrate),
		abcn:      cloneABCNSubstrate(r.abcn),
	}
}

// Restore restores the most recent backup and revives a terminated runtime.
func (r *LayerRuntime) Restore() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.backup == nil {
		return ErrNoSubstrateBackup
	}
	r.stateMode = r.backup.stateMode
	r.substrate = cloneCoordinateHyperlayers(r.backup.substrate)
	r.abcn = cloneABCNSubstrate(r.backup.abcn)
	r.terminated = false
	return nil
}

// Reset restores the initial materialized state and revives a terminated runtime.
func (r *LayerRuntime) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stateMode = r.initialStateMode
	r.substrate = cloneCoordinateHyperlayers(r.initialSubstrate)
	r.abcn = cloneABCNSubstrate(r.initialABCN)
	r.terminated = false
}

// Terminate marks the runtime unavailable until Restore or Reset revives it.
func (r *LayerRuntime) Terminate() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.terminated = true
}

func (r *LayerRuntime) weightsLocked() []float64 {
	if r.plasticity == SubstratePlasticityABCN {
		return flattenABCNWeights(r.abcn)
	}
	return flattenCoordinateWeights(r.substrate)
}

func cloneCoordinateHyperlayers(layers []CoordinateHyperlayer) []CoordinateHyperlayer {
	out := make([]CoordinateHyperlayer, 0, len(layers))
	for _, layer := range layers {
		out = append(out, cloneCoordinateHyperlayer(layer))
	}
	return out
}

func cloneABCNSubstrate(state ABCNSubstrate) ABCNSubstrate {
	return ABCNSubstrate{
		InputLayer: cloneCoordinateHyperlayer(state.InputLayer),
		Layers:     cloneABCNCoordinateHyperlayers(state.Layers),
	}
}

func flattenCoordinateWeights(layers []CoordinateHyperlayer) []float64 {
	total := 0
	for _, layer := range layers {
		for _, neurode := range layer {
			total += len(neurode.Weights)
		}
	}
	out := make([]float64, 0, total)
	for _, layer := range layers {
		for _, neurode := range layer {
			out = append(out, neurode.Weights...)
		}
	}
	return out
}

func flattenABCNWeights(state ABCNSubstrate) []float64 {
	total := 0
	for _, layer := range state.Layers {
		for _, neurode := range layer {
			total += len(neurode.Weights)
		}
	}
	out := make([]float64, 0, total)
	for _, layer := range state.Layers {
		for _, neurode := range layer {
			for _, weight := range neurode.Weights {
				out = append(out, weight.Weight)
			}
		}
	}
	return out
}
