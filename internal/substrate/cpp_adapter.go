package substrate

import "context"

// CoordinateCPPAdapter exposes scalar CPPs through coordinate CPP interfaces.
type CoordinateCPPAdapter struct {
	CPP CPP
}

// ComputeCoordinates flattens presynaptic and postsynaptic coordinates into the
// wrapped scalar CPP input vector.
func (a CoordinateCPPAdapter) ComputeCoordinates(ctx context.Context, presynaptic []float64, postsynaptic []float64, params map[string]float64) ([]float64, error) {
	if a.CPP == nil {
		return nil, ErrCPPNotFound
	}
	inputs := make([]float64, 0, len(presynaptic)+len(postsynaptic))
	inputs = append(inputs, presynaptic...)
	inputs = append(inputs, postsynaptic...)
	value, err := a.CPP.Compute(ctx, inputs, params)
	if err != nil {
		return nil, err
	}
	return []float64{value}, nil
}

// CoordinateIOWCPPAdapter exposes scalar CPPs through coordinate+IOW CPP
// interfaces.
type CoordinateIOWCPPAdapter struct {
	CPP CPP
}

// ComputeCoordinatesIOW flattens presynaptic, postsynaptic, and IOW values into
// the wrapped scalar CPP input vector.
func (a CoordinateIOWCPPAdapter) ComputeCoordinatesIOW(ctx context.Context, presynaptic []float64, postsynaptic []float64, iow []float64, params map[string]float64) ([]float64, error) {
	if a.CPP == nil {
		return nil, ErrCPPNotFound
	}
	inputs := make([]float64, 0, len(presynaptic)+len(postsynaptic)+len(iow))
	inputs = append(inputs, presynaptic...)
	inputs = append(inputs, postsynaptic...)
	inputs = append(inputs, iow...)
	value, err := a.CPP.Compute(ctx, inputs, params)
	if err != nil {
		return nil, err
	}
	return []float64{value}, nil
}

// AdaptCoordinateCPP returns a coordinate-capable wrapper for any scalar CPP.
func AdaptCoordinateCPP(cpp CPP) CoordinateCPP {
	if coordinate, ok := cpp.(CoordinateCPP); ok {
		return coordinate
	}
	return CoordinateCPPAdapter{CPP: cpp}
}

// AdaptCoordinateIOWCPP returns a coordinate+IOW-capable wrapper for any scalar
// CPP.
func AdaptCoordinateIOWCPP(cpp CPP) CoordinateIOWCPP {
	if coordinate, ok := cpp.(CoordinateIOWCPP); ok {
		return coordinate
	}
	return CoordinateIOWCPPAdapter{CPP: cpp}
}
