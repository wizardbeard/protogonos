package substrate

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
