package substrate

import (
	"errors"
	"fmt"
)

var ErrInvalidSubstrateCoordinates = errors.New("invalid substrate coordinates")

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
