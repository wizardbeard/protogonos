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

// SubstrateLayerBuildRequest carries the prebuilt input/output coordinate
// hyperlayers plus substrate densities needed by substrate.erl create_substrate.
type SubstrateLayerBuildRequest struct {
	InputLayer  CoordinateHyperlayer
	Densities   []int
	OutputLayer CoordinateHyperlayer
	LinkForm    string
}

const (
	CoordinateFormatUndefined = "undefined"
	CoordinateFormatNoGeo     = "no_geo"
	CoordinateFormatSymmetric = "symmetric"
	CoordinateFormatCoorded   = "coorded"
)

// IOCoordinateSpec describes the base substrate IO formats used by
// compose_ISubstrate and compose_OSubstrate.
type IOCoordinateSpec struct {
	Format      string
	Dim         int
	VL          int
	Resolutions []int
	Neurodes    CoordinateHyperlayer
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

// ComposeInputSubstrate builds the base input hyperlayer for undefined/no_geo
// IO specs before depth extrusion is applied.
func ComposeInputSubstrate(specs []IOCoordinateSpec) (CoordinateHyperlayer, error) {
	parts, err := composeBaseIOParts(specs, nil)
	if err != nil {
		return nil, err
	}
	return flattenCoordinateHyperlayerParts(parts), nil
}

// ComposeOutputSubstrate builds the base output hyperlayer for undefined/no_geo
// IO specs and attaches the supplied weight vector to every output neurode.
func ComposeOutputSubstrate(specs []IOCoordinateSpec, weights []float64) (CoordinateHyperlayer, error) {
	parts, err := composeBaseIOParts(specs, weights)
	if err != nil {
		return nil, err
	}
	return flattenCoordinateHyperlayerParts(parts), nil
}

// ComposeInputSubstrateForDimension mirrors substrate.erl compose_ISubstrate/2
// by padding IO coordinates to substrateDimension-2 and prepending input lead
// and IO-depth coordinates.
func ComposeInputSubstrateForDimension(specs []IOCoordinateSpec, substrateDimension int) (CoordinateHyperlayer, error) {
	return composeIOSubstrateForDimension(specs, nil, substrateDimension, -1)
}

// ComposeOutputSubstrateForDimension mirrors substrate.erl compose_OSubstrate/3
// by padding IO coordinates to substrateDimension-2 and prepending output lead
// and IO-depth coordinates.
func ComposeOutputSubstrateForDimension(specs []IOCoordinateSpec, substrateDimension int, weights []float64) (CoordinateHyperlayer, error) {
	return composeIOSubstrateForDimension(specs, weights, substrateDimension, 1)
}

func composeBaseIOParts(specs []IOCoordinateSpec, weights []float64) ([]CoordinateHyperlayer, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("%w: missing io specs", ErrInvalidSubstrateCoordinates)
	}
	parts := make([]CoordinateHyperlayer, 0, len(specs))
	for _, spec := range specs {
		format := spec.Format
		if format == "" {
			format = CoordinateFormatUndefined
		}
		switch format {
		case CoordinateFormatUndefined, CoordinateFormatNoGeo:
			if spec.VL <= 0 {
				return nil, fmt.Errorf("%w: vl must be > 0: %d", ErrInvalidSubstrateCoordinates, spec.VL)
			}
			coords, err := CreateCoordLists([]int{spec.VL})
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
			parts = append(parts, layer)
		case CoordinateFormatSymmetric:
			if len(spec.Resolutions) == 0 {
				return nil, fmt.Errorf("%w: missing symmetric resolutions", ErrInvalidSubstrateCoordinates)
			}
			coords, err := CreateCoordLists(spec.Resolutions)
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
			parts = append(parts, layer)
		case CoordinateFormatCoorded:
			layer, err := composeCoordedIOPart(spec, weights)
			if err != nil {
				return nil, err
			}
			parts = append(parts, layer)
		default:
			return nil, fmt.Errorf("%w: unsupported io coordinate format %q", ErrInvalidSubstrateCoordinates, spec.Format)
		}
	}
	return parts, nil
}

func composeIOSubstrateForDimension(specs []IOCoordinateSpec, weights []float64, substrateDimension int, leadCoord float64) (CoordinateHyperlayer, error) {
	requiredDim := substrateDimension - 2
	if requiredDim < 0 {
		return nil, fmt.Errorf("%w: substrate dimension must be >= 2: %d", ErrInvalidSubstrateCoordinates, substrateDimension)
	}

	parts, err := composeBaseIOParts(specs, weights)
	if err != nil {
		return nil, err
	}
	maxDim, err := maxIOCoordinateDim(specs)
	if err != nil {
		return nil, err
	}
	if requiredDim < maxDim {
		return nil, fmt.Errorf("%w: required io coordinate dimension %d is less than max io dimension %d", ErrInvalidSubstrateCoordinates, requiredDim, maxDim)
	}

	depthCoords, err := BuildCoordList(len(parts))
	if err != nil {
		return nil, err
	}
	total := 0
	for _, part := range parts {
		total += len(part)
	}
	out := make(CoordinateHyperlayer, 0, total)
	for i, part := range parts {
		out = append(out, advExtrudeIOPart(part, requiredDim, leadCoord, depthCoords[i])...)
	}
	return out, nil
}

func composeCoordedIOPart(spec IOCoordinateSpec, weights []float64) (CoordinateHyperlayer, error) {
	if spec.Dim <= 0 {
		return nil, fmt.Errorf("%w: coorded dim must be > 0: %d", ErrInvalidSubstrateCoordinates, spec.Dim)
	}
	if len(spec.Neurodes) == 0 {
		return nil, fmt.Errorf("%w: missing coorded neurodes", ErrInvalidSubstrateCoordinates)
	}

	layer := make(CoordinateHyperlayer, 0, len(spec.Neurodes))
	for i, neurode := range spec.Neurodes {
		if len(neurode.Coords) != spec.Dim {
			return nil, fmt.Errorf("%w: coorded neurode %d coordinate dimension %d does not match dim %d", ErrInvalidSubstrateCoordinates, i, len(neurode.Coords), spec.Dim)
		}

		copiedWeights := append([]float64(nil), neurode.Weights...)
		if weights != nil {
			copiedWeights = append([]float64(nil), weights...)
		}
		layer = append(layer, NeurodeCoordinate{
			Coords:  append([]float64(nil), neurode.Coords...),
			Output:  neurode.Output,
			Weights: copiedWeights,
		})
	}
	return layer, nil
}

func maxIOCoordinateDim(specs []IOCoordinateSpec) (int, error) {
	maxDim := 1
	for _, spec := range specs {
		dim, err := ioCoordinateDim(spec)
		if err != nil {
			return 0, err
		}
		if dim > maxDim {
			maxDim = dim
		}
	}
	return maxDim, nil
}

func ioCoordinateDim(spec IOCoordinateSpec) (int, error) {
	format := spec.Format
	if format == "" {
		format = CoordinateFormatUndefined
	}
	switch format {
	case CoordinateFormatUndefined, CoordinateFormatNoGeo:
		return 1, nil
	case CoordinateFormatSymmetric:
		if len(spec.Resolutions) == 0 {
			return 0, fmt.Errorf("%w: missing symmetric resolutions", ErrInvalidSubstrateCoordinates)
		}
		return len(spec.Resolutions), nil
	case CoordinateFormatCoorded:
		if spec.Dim <= 0 {
			return 0, fmt.Errorf("%w: coorded dim must be > 0: %d", ErrInvalidSubstrateCoordinates, spec.Dim)
		}
		return spec.Dim, nil
	default:
		return 0, fmt.Errorf("%w: unsupported io coordinate format %q", ErrInvalidSubstrateCoordinates, spec.Format)
	}
}

func advExtrudeIOPart(part CoordinateHyperlayer, requiredDim int, leadCoord float64, depthCoord float64) CoordinateHyperlayer {
	out := make(CoordinateHyperlayer, 0, len(part))
	for _, neurode := range part {
		coords := make([]float64, 0, requiredDim+2)
		coords = append(coords, leadCoord, depthCoord)
		for i := 0; i < requiredDim-len(neurode.Coords); i++ {
			coords = append(coords, 0)
		}
		coords = append(coords, neurode.Coords...)
		out = append(out, NeurodeCoordinate{
			Coords:  coords,
			Output:  neurode.Output,
			Weights: append([]float64(nil), neurode.Weights...),
		})
	}
	return out
}

func flattenCoordinateHyperlayerParts(parts []CoordinateHyperlayer) CoordinateHyperlayer {
	total := 0
	for _, part := range parts {
		total += len(part)
	}
	out := make(CoordinateHyperlayer, 0, total)
	for i := len(parts) - 1; i >= 0; i-- {
		out = append(out, cloneCoordinateHyperlayer(parts[i])...)
	}
	return out
}

// BuildSubstrateLayers mirrors the density/link-form layer shape of
// substrate.erl create_substrate after input/output coordinate layers have
// already been composed.
func BuildSubstrateLayers(req SubstrateLayerBuildRequest) ([]CoordinateHyperlayer, error) {
	if len(req.Densities) == 0 {
		return nil, fmt.Errorf("%w: missing densities", ErrInvalidSubstrateCoordinates)
	}
	if len(req.InputLayer) == 0 {
		return nil, fmt.Errorf("%w: missing input layer", ErrInvalidSubstrateCoordinates)
	}
	if len(req.OutputLayer) == 0 {
		return nil, fmt.Errorf("%w: missing output layer", ErrInvalidSubstrateCoordinates)
	}

	depth := req.Densities[0]
	if depth < 0 {
		return nil, fmt.Errorf("%w: depth must be >= 0: %d", ErrInvalidSubstrateCoordinates, depth)
	}
	subDensities := append([]int(nil), req.Densities[1:]...)
	if depth > 0 && len(subDensities) == 0 {
		return nil, fmt.Errorf("%w: missing hidden-layer densities", ErrInvalidSubstrateCoordinates)
	}

	iWeights, hWeights, err := substrateLayerWeightTemplates(req.LinkForm, depth, subDensities, len(req.InputLayer), len(req.OutputLayer))
	if err != nil {
		return nil, err
	}

	inputLayer := cloneCoordinateHyperlayer(req.InputLayer)
	switch depth {
	case 0:
		return []CoordinateHyperlayer{
			inputLayer,
			attachWeightsToLayer(req.OutputLayer, iWeights),
		}, nil
	case 1:
		recurrentLayer, err := BuildCoordinateHypercube(subDensities, iWeights)
		if err != nil {
			return nil, err
		}
		return []CoordinateHyperlayer{
			inputLayer,
			ExtrudeCoordinateHyperlayer(0, recurrentLayer),
			attachWeightsToLayer(req.OutputLayer, hWeights),
		}, nil
	default:
		recurrentBase, err := BuildCoordinateHypercube(subDensities, iWeights)
		if err != nil {
			return nil, err
		}
		hiddenBase, err := BuildCoordinateHypercube(subDensities, hWeights)
		if err != nil {
			return nil, err
		}
		depthCoords, err := BuildCoordList(depth + 1)
		if err != nil {
			return nil, err
		}
		recurrentCoord := depthCoords[1]
		hiddenCoords := append([]float64(nil), depthCoords[2:len(depthCoords)-1]...)

		layers := make([]CoordinateHyperlayer, 0, depth+2)
		layers = append(layers, inputLayer)
		layers = append(layers, ExtrudeCoordinateHyperlayer(recurrentCoord, recurrentBase))
		for _, hiddenCoord := range hiddenCoords {
			layers = append(layers, ExtrudeCoordinateHyperlayer(hiddenCoord, hiddenBase))
		}
		layers = append(layers, attachWeightsToLayer(req.OutputLayer, hWeights))
		return layers, nil
	}
}

func substrateLayerWeightTemplates(linkForm string, depth int, subDensities []int, inputCount int, outputCount int) ([]float64, []float64, error) {
	hiddenCount, err := multiplyDensities(subDensities)
	if err != nil {
		return nil, nil, err
	}

	switch linkForm {
	case LinkFormL2LFeedforward:
		return make([]float64, inputCount), make([]float64, hiddenCount), nil
	case LinkFormFullyInterconnected:
		if depth == 0 {
			return nil, nil, fmt.Errorf("%w: fully_interconnected requires depth > 0", ErrInvalidSubstrateCoordinates)
		}
		totalHidden, err := multiplyDensities(append([]int{depth - 1}, subDensities...))
		if err != nil {
			return nil, nil, err
		}
		totalWeights := totalHidden + inputCount + outputCount
		weights := make([]float64, totalWeights)
		return weights, append([]float64(nil), weights...), nil
	case LinkFormJordanRecurrent:
		return make([]float64, inputCount+outputCount), make([]float64, hiddenCount), nil
	case LinkFormNeuronSelfRecurrent:
		return make([]float64, inputCount+1), make([]float64, hiddenCount+1), nil
	default:
		return nil, nil, fmt.Errorf("%w: %q", ErrUnsupportedSubstrateLink, linkForm)
	}
}

func multiplyDensities(densities []int) (int, error) {
	if len(densities) == 0 {
		return 0, fmt.Errorf("%w: missing densities", ErrInvalidSubstrateCoordinates)
	}
	product := 1
	for _, density := range densities {
		if density <= 0 {
			return 0, fmt.Errorf("%w: density must be > 0: %d", ErrInvalidSubstrateCoordinates, density)
		}
		product *= density
	}
	return product, nil
}

func attachWeightsToLayer(layer CoordinateHyperlayer, weights []float64) CoordinateHyperlayer {
	out := cloneCoordinateHyperlayer(layer)
	for i := range out {
		out[i].Weights = append([]float64(nil), weights...)
	}
	return out
}

func cloneCoordinateHyperlayer(layer CoordinateHyperlayer) CoordinateHyperlayer {
	out := make(CoordinateHyperlayer, 0, len(layer))
	for _, neurode := range layer {
		out = append(out, NeurodeCoordinate{
			Coords:  append([]float64(nil), neurode.Coords...),
			Output:  neurode.Output,
			Weights: append([]float64(nil), neurode.Weights...),
		})
	}
	return out
}
