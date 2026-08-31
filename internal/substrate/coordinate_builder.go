package substrate

import (
	"errors"
	"fmt"
)

const (
	LinkFormL2LFeedforward      = "l2l_feedforward"
	LinkFormFullyInterconnected = "fully_interconnected"
	LinkFormJordanRecurrent     = "jordan_recurrent"
	LinkFormNeuronSelfRecurrent = "neuronself_recurrent"
)

var (
	ErrInvalidSubstrateCoordinates = errors.New("invalid substrate coordinates")
	ErrUnsupportedSubstrateLink    = errors.New("unsupported substrate link form")
)

// CoordinatePairBuildRequest carries the coordinate sources needed by each
// substrate.erl link-form population path.
type CoordinatePairBuildRequest struct {
	LinkForm            string
	PreviousLayerCoords [][]float64
	CurrentLayerCoords  [][]float64
	FlatSubstrateCoords [][]float64
	InputLayerCoords    [][]float64
	OutputLayerCoords   [][]float64
}

// BuildCoordinatePairsForLinkForm dispatches to the coordinate-pair builder
// matching the reference substrate link form.
func BuildCoordinatePairsForLinkForm(req CoordinatePairBuildRequest) ([]CoordinatePair, error) {
	switch req.LinkForm {
	case LinkFormL2LFeedforward:
		return BuildL2LFeedforwardCoordinatePairs(req.PreviousLayerCoords, req.CurrentLayerCoords)
	case LinkFormFullyInterconnected:
		return BuildFullyInterconnectedCoordinatePairs(req.FlatSubstrateCoords, req.CurrentLayerCoords)
	case LinkFormJordanRecurrent:
		return BuildJordanRecurrentCoordinatePairs(req.InputLayerCoords, req.OutputLayerCoords, req.CurrentLayerCoords)
	case LinkFormNeuronSelfRecurrent:
		return BuildNeuronSelfRecurrentCoordinatePairs(req.PreviousLayerCoords, req.CurrentLayerCoords)
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedSubstrateLink, req.LinkForm)
	}
}

// BuildL2LFeedforwardCoordinatePairs returns coordinate pairs in the same
// previous-layer/current-layer order used by substrate.erl's l2l_feedforward
// weight population path.
func BuildL2LFeedforwardCoordinatePairs(presynapticCoords [][]float64, postsynapticCoords [][]float64) ([]CoordinatePair, error) {
	if len(presynapticCoords) == 0 {
		return nil, fmt.Errorf("%w: missing presynaptic coordinates", ErrInvalidSubstrateCoordinates)
	}
	if len(postsynapticCoords) == 0 {
		return nil, fmt.Errorf("%w: missing postsynaptic coordinates", ErrInvalidSubstrateCoordinates)
	}

	pairs := make([]CoordinatePair, 0, len(presynapticCoords)*len(postsynapticCoords))
	for _, postsynaptic := range postsynapticCoords {
		for _, presynaptic := range presynapticCoords {
			pairs = append(pairs, CoordinatePair{
				PresynapticCoords:  append([]float64(nil), presynaptic...),
				PostsynapticCoords: append([]float64(nil), postsynaptic...),
			})
		}
	}
	return pairs, nil
}

// BuildFullyInterconnectedCoordinatePairs returns coordinate pairs for
// substrate.erl's fully_interconnected population path. The source list should
// be the flattened substrate coordinate list in reference traversal order.
func BuildFullyInterconnectedCoordinatePairs(flatSubstrateCoords [][]float64, currentLayerCoords [][]float64) ([]CoordinatePair, error) {
	if len(flatSubstrateCoords) == 0 {
		return nil, fmt.Errorf("%w: missing flat substrate coordinates", ErrInvalidSubstrateCoordinates)
	}
	if len(currentLayerCoords) == 0 {
		return nil, fmt.Errorf("%w: missing current-layer coordinates", ErrInvalidSubstrateCoordinates)
	}
	return BuildL2LFeedforwardCoordinatePairs(flatSubstrateCoords, currentLayerCoords)
}

// BuildNeuronSelfRecurrentCoordinatePairs returns coordinate pairs for
// substrate.erl's neuronself_recurrent population path. Each postsynaptic
// neurode receives its self-connection first, then all previous-layer inputs.
func BuildNeuronSelfRecurrentCoordinatePairs(previousLayerCoords [][]float64, currentLayerCoords [][]float64) ([]CoordinatePair, error) {
	if len(previousLayerCoords) == 0 {
		return nil, fmt.Errorf("%w: missing previous-layer coordinates", ErrInvalidSubstrateCoordinates)
	}
	if len(currentLayerCoords) == 0 {
		return nil, fmt.Errorf("%w: missing current-layer coordinates", ErrInvalidSubstrateCoordinates)
	}

	pairs := make([]CoordinatePair, 0, len(currentLayerCoords)*(len(previousLayerCoords)+1))
	for _, current := range currentLayerCoords {
		pairs = append(pairs, CoordinatePair{
			PresynapticCoords:  append([]float64(nil), current...),
			PostsynapticCoords: append([]float64(nil), current...),
		})
		for _, previous := range previousLayerCoords {
			pairs = append(pairs, CoordinatePair{
				PresynapticCoords:  append([]float64(nil), previous...),
				PostsynapticCoords: append([]float64(nil), current...),
			})
		}
	}
	return pairs, nil
}

// BuildJordanRecurrentCoordinatePairs returns coordinate pairs for
// substrate.erl's jordan_recurrent population path. The source list is the
// input hyperlayer followed by the output hyperlayer.
func BuildJordanRecurrentCoordinatePairs(inputLayerCoords [][]float64, outputLayerCoords [][]float64, currentLayerCoords [][]float64) ([]CoordinatePair, error) {
	if len(inputLayerCoords) == 0 {
		return nil, fmt.Errorf("%w: missing input-layer coordinates", ErrInvalidSubstrateCoordinates)
	}
	if len(outputLayerCoords) == 0 {
		return nil, fmt.Errorf("%w: missing output-layer coordinates", ErrInvalidSubstrateCoordinates)
	}
	sourceCoords := make([][]float64, 0, len(inputLayerCoords)+len(outputLayerCoords))
	sourceCoords = append(sourceCoords, inputLayerCoords...)
	sourceCoords = append(sourceCoords, outputLayerCoords...)
	return BuildL2LFeedforwardCoordinatePairs(sourceCoords, currentLayerCoords)
}
