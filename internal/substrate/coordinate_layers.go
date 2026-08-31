package substrate

import "fmt"

// NeurodeCoordinate mirrors the coordinate/output/weights tuple used by the
// reference substrate hyperlayers while keeping runtime-specific state optional.
type NeurodeCoordinate struct {
	Coords  []float64
	Output  float64
	Weights []float64
}

// CoordinateHyperlayer is an ordered substrate layer of coordinate neurodes.
type CoordinateHyperlayer []NeurodeCoordinate

// Coordinates returns copied neurode coordinates in layer traversal order.
func (h CoordinateHyperlayer) Coordinates() [][]float64 {
	coords := make([][]float64, 0, len(h))
	for _, neurode := range h {
		coords = append(coords, append([]float64(nil), neurode.Coords...))
	}
	return coords
}

// FlattenCoordinateHyperlayers returns copied coordinates from all layers in
// the supplied order. It matches the flattened source-list shape used by
// substrate.erl's fully_interconnected path.
func FlattenCoordinateHyperlayers(layers ...CoordinateHyperlayer) [][]float64 {
	var total int
	for _, layer := range layers {
		total += len(layer)
	}

	coords := make([][]float64, 0, total)
	for _, layer := range layers {
		coords = append(coords, layer.Coordinates()...)
	}
	return coords
}

// CoordinateLayerPairBuildRequest carries typed hyperlayers for link-form
// coordinate-pair construction.
type CoordinateLayerPairBuildRequest struct {
	LinkForm      string
	PreviousLayer CoordinateHyperlayer
	CurrentLayer  CoordinateHyperlayer
	FlatSubstrate []CoordinateHyperlayer
	InputLayer    CoordinateHyperlayer
	OutputLayer   CoordinateHyperlayer
}

// BuildCoordinatePairsForLinkFormLayers builds coordinate pairs from typed
// hyperlayers, then delegates to the raw coordinate dispatcher.
func BuildCoordinatePairsForLinkFormLayers(req CoordinateLayerPairBuildRequest) ([]CoordinatePair, error) {
	return BuildCoordinatePairsForLinkForm(CoordinatePairBuildRequest{
		LinkForm:            req.LinkForm,
		PreviousLayerCoords: req.PreviousLayer.Coordinates(),
		CurrentLayerCoords:  req.CurrentLayer.Coordinates(),
		FlatSubstrateCoords: FlattenCoordinateHyperlayers(req.FlatSubstrate...),
		InputLayerCoords:    req.InputLayer.Coordinates(),
		OutputLayerCoords:   req.OutputLayer.Coordinates(),
	})
}

// BuildCoordList mirrors substrate.erl build_CoordList/1. Density 1 maps to a
// centered coordinate, otherwise coordinates span [-1, 1] in reference order.
func BuildCoordList(density int) ([]float64, error) {
	if density <= 0 {
		return nil, fmt.Errorf("%w: density must be > 0: %d", ErrInvalidSubstrateCoordinates, density)
	}
	if density == 1 {
		return []float64{0}, nil
	}

	dividers := density - 1
	step := 2.0 / float64(dividers)
	coords := make([]float64, density)
	for i := range coords {
		coords[i] = -1 + float64(i)*step
	}
	return coords, nil
}

// CreateCoordLists mirrors substrate.erl create_CoordLists/1. The supplied
// densities are consumed in order; each next dimension is prepended.
func CreateCoordLists(densities []int) ([][]float64, error) {
	if len(densities) == 0 {
		return nil, fmt.Errorf("%w: missing densities", ErrInvalidSubstrateCoordinates)
	}

	var coords [][]float64
	for _, density := range densities {
		coordList, err := BuildCoordList(density)
		if err != nil {
			return nil, err
		}
		if len(coords) == 0 {
			coords = make([][]float64, 0, len(coordList))
			for _, coord := range coordList {
				coords = append(coords, []float64{coord})
			}
			continue
		}

		next := make([][]float64, 0, len(coordList)*len(coords))
		for _, coord := range coordList {
			for _, existing := range coords {
				combined := make([]float64, 0, len(existing)+1)
				combined = append(combined, coord)
				combined = append(combined, existing...)
				next = append(next, combined)
			}
		}
		coords = next
	}
	return coords, nil
}

// BuildCoordinateHypercube mirrors substrate.erl cs/2: reverse the density
// vector, create coordinate lists, then attach zero output and initial weights.
func BuildCoordinateHypercube(densities []int, weights []float64) (CoordinateHyperlayer, error) {
	if len(densities) == 0 {
		return nil, fmt.Errorf("%w: missing densities", ErrInvalidSubstrateCoordinates)
	}

	reversed := make([]int, len(densities))
	for i := range densities {
		reversed[i] = densities[len(densities)-1-i]
	}
	coords, err := CreateCoordLists(reversed)
	if err != nil {
		return nil, err
	}

	layer := make(CoordinateHyperlayer, 0, len(coords))
	for _, coord := range coords {
		layer = append(layer, NeurodeCoordinate{
			Coords:  append([]float64(nil), coord...),
			Output:  0,
			Weights: append([]float64(nil), weights...),
		})
	}
	return layer, nil
}

// ExtrudeCoordinateHyperlayer mirrors substrate.erl extrude/2 by prepending a
// new coordinate dimension to every neurode in the layer.
func ExtrudeCoordinateHyperlayer(newDimensionCoord float64, layer CoordinateHyperlayer) CoordinateHyperlayer {
	out := make(CoordinateHyperlayer, 0, len(layer))
	for _, neurode := range layer {
		coords := make([]float64, 0, len(neurode.Coords)+1)
		coords = append(coords, newDimensionCoord)
		coords = append(coords, neurode.Coords...)
		out = append(out, NeurodeCoordinate{
			Coords:  coords,
			Output:  neurode.Output,
			Weights: append([]float64(nil), neurode.Weights...),
		})
	}
	return out
}
