package substrate

import (
	"context"
	"fmt"
	"math"
)

// NeurodeCoordinate mirrors the coordinate/output/weights tuple used by the
// reference substrate hyperlayers while keeping runtime-specific state optional.
type NeurodeCoordinate struct {
	Coords  []float64
	Output  float64
	Weights []float64
}

// CoordinateHyperlayer is an ordered substrate layer of coordinate neurodes.
type CoordinateHyperlayer []NeurodeCoordinate

// ABCNWeight mirrors the reference `{W, abcn, [A,B,C,N]}` plastic weight tuple.
type ABCNWeight struct {
	Weight float64
	A      float64
	B      float64
	C      float64
	N      float64
}

// ABCNNeurodeCoordinate carries coordinate/output state plus ABCN plastic
// weight tuples for typed plastic output calculations.
type ABCNNeurodeCoordinate struct {
	Coords  []float64
	Output  float64
	Weights []ABCNWeight
}

// ABCNCoordinateHyperlayer is an ordered layer of ABCN plastic coordinate
// neurodes.
type ABCNCoordinateHyperlayer []ABCNNeurodeCoordinate

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

// CreateSubstrateRequest carries IO specs, density vector, and link form for
// reference-style substrate layer construction.
type CreateSubstrateRequest struct {
	InputSpecs  []IOCoordinateSpec
	Densities   []int
	OutputSpecs []IOCoordinateSpec
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

// CountIONeurodes mirrors substrate.erl tot_ONeurodes/2 for the covered IO
// formats. The same cardinality rules apply to sensor and actuator specs.
func CountIONeurodes(specs []IOCoordinateSpec) (int, error) {
	if len(specs) == 0 {
		return 0, fmt.Errorf("%w: missing io specs", ErrInvalidSubstrateCoordinates)
	}

	total := 0
	for _, spec := range specs {
		count, err := ioNeurodeCount(spec)
		if err != nil {
			return 0, err
		}
		total += count
	}
	return total, nil
}

// FlattenInputValues mirrors lists:flatten(Input) for already typed input
// batches before input-hyperlayer population.
func FlattenInputValues(input ...[]float64) []float64 {
	total := 0
	for _, values := range input {
		total += len(values)
	}
	out := make([]float64, 0, total)
	for _, values := range input {
		out = append(out, values...)
	}
	return out
}

// PopulateInputHyperlayer mirrors substrate.erl populate_InputHyperlayer/3 by
// replacing each input neurode output with the next flattened input value.
func PopulateInputHyperlayer(layer CoordinateHyperlayer, input []float64) (CoordinateHyperlayer, error) {
	if len(layer) != len(input) {
		return nil, fmt.Errorf("%w: input length %d does not match input hyperlayer length %d", ErrInvalidSubstrateCoordinates, len(input), len(layer))
	}

	out := make(CoordinateHyperlayer, 0, len(layer))
	for i, neurode := range layer {
		out = append(out, NeurodeCoordinate{
			Coords:  append([]float64(nil), neurode.Coords...),
			Output:  input[i],
			Weights: append([]float64(nil), neurode.Weights...),
		})
	}
	return out, nil
}

// CalculateNeurodeOutputStd mirrors calculate_neurode_output_std/3 for the
// non-plastic substrate path: tanh(sum(previous_output * weight)).
func CalculateNeurodeOutputStd(previous CoordinateHyperlayer, neurode NeurodeCoordinate) (NeurodeCoordinate, error) {
	if len(previous) != len(neurode.Weights) {
		return NeurodeCoordinate{}, fmt.Errorf("%w: previous neurode count %d does not match weight count %d", ErrInvalidSubstrateCoordinates, len(previous), len(neurode.Weights))
	}

	sum := 0.0
	for i, prev := range previous {
		sum += prev.Output * neurode.Weights[i]
	}
	return NeurodeCoordinate{
		Coords:  append([]float64(nil), neurode.Coords...),
		Output:  math.Tanh(sum),
		Weights: append([]float64(nil), neurode.Weights...),
	}, nil
}

// CalculateOutputStd mirrors the l2l_feedforward calculate_output_std path for
// already-populated input and process/output hyperlayers with non-plastic weights.
func CalculateOutputStd(input CoordinateHyperlayer, layers []CoordinateHyperlayer) ([]float64, []CoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, nil, fmt.Errorf("%w: missing process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	previous := cloneCoordinateHyperlayer(input)
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current := make(CoordinateHyperlayer, 0, len(layer))
		for neurodeIdx, neurode := range layer {
			calculated, err := CalculateNeurodeOutputStd(previous, neurode)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d neurode %d: %w", layerIdx, neurodeIdx, err)
			}
			current = append(current, calculated)
		}
		updated = append(updated, current)
		previous = current
	}

	outputLayer := updated[len(updated)-1]
	outputs := make([]float64, 0, len(outputLayer))
	for _, neurode := range outputLayer {
		outputs = append(outputs, neurode.Output)
	}
	return outputs, updated, nil
}

// ABCNWeightUpdate mirrors substrate.erl abcn/4.
func ABCNWeightUpdate(input float64, output float64, weight ABCNWeight) ABCNWeight {
	updated := weight
	updated.Weight += weight.N * (weight.A*input*output + weight.B*input + weight.C*output)
	return updated
}

// CalculateNeurodeOutputABCN mirrors calculate_neurode_output_plast followed by
// update_neurode for one neurode.
func CalculateNeurodeOutputABCN(previous CoordinateHyperlayer, neurode ABCNNeurodeCoordinate) (ABCNNeurodeCoordinate, error) {
	if len(previous) != len(neurode.Weights) {
		return ABCNNeurodeCoordinate{}, fmt.Errorf("%w: previous neurode count %d does not match abcn weight count %d", ErrInvalidSubstrateCoordinates, len(previous), len(neurode.Weights))
	}

	sum := 0.0
	for i, prev := range previous {
		sum += prev.Output * neurode.Weights[i].Weight
	}
	output := math.Tanh(sum)

	updatedWeights := make([]ABCNWeight, 0, len(neurode.Weights))
	for i, weight := range neurode.Weights {
		updatedWeights = append(updatedWeights, ABCNWeightUpdate(previous[i].Output, output, weight))
	}
	return ABCNNeurodeCoordinate{
		Coords:  append([]float64(nil), neurode.Coords...),
		Output:  output,
		Weights: updatedWeights,
	}, nil
}

// CalculateOutputABCNStd mirrors the l2l calculate_output_std path when
// Plasticity is abcn.
func CalculateOutputABCNStd(input CoordinateHyperlayer, layers []ABCNCoordinateHyperlayer) ([]float64, []ABCNCoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, nil, fmt.Errorf("%w: missing abcn process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	previous := cloneCoordinateHyperlayer(input)
	updated := make([]ABCNCoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current := make(ABCNCoordinateHyperlayer, 0, len(layer))
		currentScalar := make(CoordinateHyperlayer, 0, len(layer))
		for neurodeIdx, neurode := range layer {
			calculated, err := CalculateNeurodeOutputABCN(previous, neurode)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d neurode %d: %w", layerIdx, neurodeIdx, err)
			}
			current = append(current, calculated)
			currentScalar = append(currentScalar, NeurodeCoordinate{
				Coords: append([]float64(nil), calculated.Coords...),
				Output: calculated.Output,
			})
		}
		updated = append(updated, current)
		previous = currentScalar
	}

	outputLayer := updated[len(updated)-1]
	outputs := make([]float64, 0, len(outputLayer))
	for _, neurode := range outputLayer {
		outputs = append(outputs, neurode.Output)
	}
	return outputs, updated, nil
}

// CalculateOutputForLinkForm dispatches the non-plastic typed-layer output
// calculation for the active substrate link forms.
func CalculateOutputForLinkForm(linkForm string, input CoordinateHyperlayer, layers []CoordinateHyperlayer) ([]float64, []CoordinateHyperlayer, error) {
	switch linkForm {
	case LinkFormL2LFeedforward:
		return CalculateOutputStd(input, layers)
	case LinkFormFullyInterconnected:
		return CalculateOutputFullyInterconnected(input, layers)
	case LinkFormJordanRecurrent:
		return CalculateOutputJordanRecurrent(input, layers)
	case LinkFormNeuronSelfRecurrent:
		return CalculateOutputNeuronSelfRecurrent(input, layers)
	default:
		return nil, nil, fmt.Errorf("%w: %q", ErrUnsupportedSubstrateLink, linkForm)
	}
}

// CalculateHoldOutput mirrors calculate_HoldOutput for the typed non-plastic
// substrate path. It uses the existing process/output hyperlayers as stored.
func CalculateHoldOutput(substrate []CoordinateHyperlayer, inputValues [][]float64, linkForm string) ([]float64, []CoordinateHyperlayer, error) {
	return calculateTypedOutputLifecycle(substrate, inputValues, linkForm)
}

// CalculateResetOutput mirrors calculate_ResetOutput for the typed non-plastic
// substrate path. CPP/CEP-driven weight repopulation is handled by runtime
// paths; this helper preserves the typed layer state shape and output flow.
func CalculateResetOutput(substrate []CoordinateHyperlayer, inputValues [][]float64, linkForm string) ([]float64, []CoordinateHyperlayer, error) {
	return calculateTypedOutputLifecycle(substrate, inputValues, linkForm)
}

// PopulateProcessHyperlayersStatic mirrors populate_PHyperlayers/get_weights
// for the static non-plastic path using typed CPP/CEP components.
func PopulateProcessHyperlayersStatic(ctx context.Context, substrate []CoordinateHyperlayer, linkForm string, cpp CoordinateCPP, ceps []CEP, params map[string]float64) ([]CoordinateHyperlayer, error) {
	if len(substrate) < 2 {
		return nil, fmt.Errorf("%w: substrate must include input and process/output layers", ErrInvalidSubstrateCoordinates)
	}
	if cpp == nil {
		return nil, fmt.Errorf("%w: missing coordinate cpp", ErrInvalidSubstrateCoordinates)
	}
	if len(ceps) == 0 {
		return nil, fmt.Errorf("%w: missing cep chain", ErrInvalidSubstrateCoordinates)
	}

	switch linkForm {
	case LinkFormL2LFeedforward:
		return populateStaticL2L(ctx, substrate[0], substrate[1:], cpp, ceps, params)
	case LinkFormFullyInterconnected:
		return populateStaticFullyInterconnected(ctx, substrate, cpp, ceps, params)
	case LinkFormJordanRecurrent:
		source := flattenCoordinateNeurodes(substrate[0], substrate[len(substrate)-1])
		return populateStaticL2L(ctx, source, substrate[1:], cpp, ceps, params)
	case LinkFormNeuronSelfRecurrent:
		return populateStaticNeuronSelfRecurrent(ctx, substrate[0], substrate[1:], cpp, ceps, params)
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedSubstrateLink, linkForm)
	}
}

// PopulateProcessHyperlayersIterative mirrors the iterative get_weights path by
// sending [input output previous_weight] with each coordinate pair.
func PopulateProcessHyperlayersIterative(ctx context.Context, substrate []CoordinateHyperlayer, linkForm string, cpp CoordinateIOWCPP, ceps []CEP, params map[string]float64) ([]CoordinateHyperlayer, error) {
	if len(substrate) < 2 {
		return nil, fmt.Errorf("%w: substrate must include input and process/output layers", ErrInvalidSubstrateCoordinates)
	}
	if cpp == nil {
		return nil, fmt.Errorf("%w: missing coordinate iow cpp", ErrInvalidSubstrateCoordinates)
	}
	if len(ceps) == 0 {
		return nil, fmt.Errorf("%w: missing cep chain", ErrInvalidSubstrateCoordinates)
	}

	switch linkForm {
	case LinkFormL2LFeedforward:
		return populateIterativeL2L(ctx, substrate[0], substrate[1:], cpp, ceps, params)
	case LinkFormFullyInterconnected:
		return populateIterativeFullyInterconnected(ctx, substrate, cpp, ceps, params)
	case LinkFormJordanRecurrent:
		source := flattenCoordinateNeurodes(substrate[0], substrate[len(substrate)-1])
		return populateIterativeL2L(ctx, source, substrate[1:], cpp, ceps, params)
	case LinkFormNeuronSelfRecurrent:
		return populateIterativeNeuronSelfRecurrent(ctx, substrate[0], substrate[1:], cpp, ceps, params)
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedSubstrateLink, linkForm)
	}
}

func calculateTypedOutputLifecycle(substrate []CoordinateHyperlayer, inputValues [][]float64, linkForm string) ([]float64, []CoordinateHyperlayer, error) {
	if len(substrate) < 2 {
		return nil, nil, fmt.Errorf("%w: substrate must include input and process/output layers", ErrInvalidSubstrateCoordinates)
	}

	populatedInput, err := PopulateInputHyperlayer(substrate[0], FlattenInputValues(inputValues...))
	if err != nil {
		return nil, nil, err
	}
	outputs, updatedLayers, err := CalculateOutputForLinkForm(linkForm, populatedInput, substrate[1:])
	if err != nil {
		return nil, nil, err
	}

	updatedSubstrate := make([]CoordinateHyperlayer, 0, len(substrate))
	updatedSubstrate = append(updatedSubstrate, cloneCoordinateHyperlayer(substrate[0]))
	updatedSubstrate = append(updatedSubstrate, updatedLayers...)
	return outputs, updatedSubstrate, nil
}

func populateStaticL2L(ctx context.Context, previous CoordinateHyperlayer, layers []CoordinateHyperlayer, cpp CoordinateCPP, ceps []CEP, params map[string]float64) ([]CoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("%w: missing process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	source := cloneCoordinateHyperlayer(previous)
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current, err := populateStaticLayer(ctx, source, layer, cpp, ceps, params)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", layerIdx, err)
		}
		updated = append(updated, current)
		source = current
	}
	return updated, nil
}

func populateStaticFullyInterconnected(ctx context.Context, substrate []CoordinateHyperlayer, cpp CoordinateCPP, ceps []CEP, params map[string]float64) ([]CoordinateHyperlayer, error) {
	source := flattenCoordinateNeurodes(substrate...)
	layers := substrate[1:]
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current, err := populateStaticLayer(ctx, source, layer, cpp, ceps, params)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", layerIdx, err)
		}
		updated = append(updated, current)
		source = replaceFlattenedLayer(source, len(substrate[0]), layers, updated, layerIdx)
	}
	return updated, nil
}

func populateStaticNeuronSelfRecurrent(ctx context.Context, previous CoordinateHyperlayer, layers []CoordinateHyperlayer, cpp CoordinateCPP, ceps []CEP, params map[string]float64) ([]CoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("%w: missing process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	source := cloneCoordinateHyperlayer(previous)
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current := make(CoordinateHyperlayer, 0, len(layer))
		for neurodeIdx, neurode := range layer {
			neurodeSource := make(CoordinateHyperlayer, 0, len(source)+1)
			neurodeSource = append(neurodeSource, cloneNeurodeCoordinate(neurode))
			neurodeSource = append(neurodeSource, cloneCoordinateHyperlayer(source)...)
			populated, err := populateStaticNeurode(ctx, neurodeSource, neurode, cpp, ceps, params)
			if err != nil {
				return nil, fmt.Errorf("layer %d neurode %d: %w", layerIdx, neurodeIdx, err)
			}
			current = append(current, populated)
		}
		updated = append(updated, current)
		source = current
	}
	return updated, nil
}

func populateStaticLayer(ctx context.Context, source CoordinateHyperlayer, layer CoordinateHyperlayer, cpp CoordinateCPP, ceps []CEP, params map[string]float64) (CoordinateHyperlayer, error) {
	if len(source) == 0 {
		return nil, fmt.Errorf("%w: missing source hyperlayer", ErrInvalidSubstrateCoordinates)
	}
	if len(layer) == 0 {
		return nil, fmt.Errorf("%w: missing target hyperlayer", ErrInvalidSubstrateCoordinates)
	}

	current := make(CoordinateHyperlayer, 0, len(layer))
	for neurodeIdx, neurode := range layer {
		populated, err := populateStaticNeurode(ctx, source, neurode, cpp, ceps, params)
		if err != nil {
			return nil, fmt.Errorf("neurode %d: %w", neurodeIdx, err)
		}
		current = append(current, populated)
	}
	return current, nil
}

func populateStaticNeurode(ctx context.Context, source CoordinateHyperlayer, neurode NeurodeCoordinate, cpp CoordinateCPP, ceps []CEP, params map[string]float64) (NeurodeCoordinate, error) {
	weights := make([]float64, 0, len(source))
	for sourceIdx, input := range source {
		if err := ctx.Err(); err != nil {
			return NeurodeCoordinate{}, err
		}
		weight, err := staticCoordinateWeight(ctx, input.Coords, neurode.Coords, cpp, ceps, params)
		if err != nil {
			return NeurodeCoordinate{}, fmt.Errorf("source %d: %w", sourceIdx, err)
		}
		weights = append(weights, weight)
	}

	populated := cloneNeurodeCoordinate(neurode)
	populated.Weights = weights
	return populated, nil
}

func staticCoordinateWeight(ctx context.Context, presynaptic []float64, postsynaptic []float64, cpp CoordinateCPP, ceps []CEP, params map[string]float64) (float64, error) {
	signal, err := cpp.ComputeCoordinates(ctx, presynaptic, postsynaptic, cloneFloatParams(params))
	if err != nil {
		return 0, err
	}
	if len(signal) == 0 {
		return 0, fmt.Errorf("%w: coordinate cpp returned empty signal", ErrInvalidSubstrateCoordinates)
	}

	weight := 0.0
	delta := signal[0]
	for _, cep := range ceps {
		if cep == nil {
			return 0, fmt.Errorf("%w: nil cep in chain", ErrInvalidSubstrateCoordinates)
		}
		weight, err = cep.Apply(ctx, weight, delta, cloneFloatParams(params))
		if err != nil {
			return 0, err
		}
		delta = weight
	}
	return weight, nil
}

func populateIterativeL2L(ctx context.Context, previous CoordinateHyperlayer, layers []CoordinateHyperlayer, cpp CoordinateIOWCPP, ceps []CEP, params map[string]float64) ([]CoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("%w: missing process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	source := cloneCoordinateHyperlayer(previous)
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current, err := populateIterativeLayer(ctx, source, layer, cpp, ceps, params)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", layerIdx, err)
		}
		updated = append(updated, current)
		source = current
	}
	return updated, nil
}

func populateIterativeFullyInterconnected(ctx context.Context, substrate []CoordinateHyperlayer, cpp CoordinateIOWCPP, ceps []CEP, params map[string]float64) ([]CoordinateHyperlayer, error) {
	source := flattenCoordinateNeurodes(substrate...)
	layers := substrate[1:]
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current, err := populateIterativeLayer(ctx, source, layer, cpp, ceps, params)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", layerIdx, err)
		}
		updated = append(updated, current)
		source = replaceFlattenedLayer(source, len(substrate[0]), layers, updated, layerIdx)
	}
	return updated, nil
}

func populateIterativeNeuronSelfRecurrent(ctx context.Context, previous CoordinateHyperlayer, layers []CoordinateHyperlayer, cpp CoordinateIOWCPP, ceps []CEP, params map[string]float64) ([]CoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("%w: missing process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	source := cloneCoordinateHyperlayer(previous)
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current := make(CoordinateHyperlayer, 0, len(layer))
		for neurodeIdx, neurode := range layer {
			neurodeSource := make(CoordinateHyperlayer, 0, len(source)+1)
			neurodeSource = append(neurodeSource, cloneNeurodeCoordinate(neurode))
			neurodeSource = append(neurodeSource, cloneCoordinateHyperlayer(source)...)
			populated, err := populateIterativeNeurode(ctx, neurodeSource, neurode, cpp, ceps, params)
			if err != nil {
				return nil, fmt.Errorf("layer %d neurode %d: %w", layerIdx, neurodeIdx, err)
			}
			current = append(current, populated)
		}
		updated = append(updated, current)
		source = current
	}
	return updated, nil
}

func populateIterativeLayer(ctx context.Context, source CoordinateHyperlayer, layer CoordinateHyperlayer, cpp CoordinateIOWCPP, ceps []CEP, params map[string]float64) (CoordinateHyperlayer, error) {
	if len(source) == 0 {
		return nil, fmt.Errorf("%w: missing source hyperlayer", ErrInvalidSubstrateCoordinates)
	}
	if len(layer) == 0 {
		return nil, fmt.Errorf("%w: missing target hyperlayer", ErrInvalidSubstrateCoordinates)
	}

	current := make(CoordinateHyperlayer, 0, len(layer))
	for neurodeIdx, neurode := range layer {
		populated, err := populateIterativeNeurode(ctx, source, neurode, cpp, ceps, params)
		if err != nil {
			return nil, fmt.Errorf("neurode %d: %w", neurodeIdx, err)
		}
		current = append(current, populated)
	}
	return current, nil
}

func populateIterativeNeurode(ctx context.Context, source CoordinateHyperlayer, neurode NeurodeCoordinate, cpp CoordinateIOWCPP, ceps []CEP, params map[string]float64) (NeurodeCoordinate, error) {
	if len(source) != len(neurode.Weights) {
		return NeurodeCoordinate{}, fmt.Errorf("%w: source neurode count %d does not match previous weight count %d", ErrInvalidSubstrateCoordinates, len(source), len(neurode.Weights))
	}

	weights := make([]float64, 0, len(source))
	for sourceIdx, input := range source {
		if err := ctx.Err(); err != nil {
			return NeurodeCoordinate{}, err
		}
		previousWeight := neurode.Weights[sourceIdx]
		weight, err := iterativeCoordinateWeight(ctx, input, neurode, previousWeight, cpp, ceps, params)
		if err != nil {
			return NeurodeCoordinate{}, fmt.Errorf("source %d: %w", sourceIdx, err)
		}
		weights = append(weights, weight)
	}

	populated := cloneNeurodeCoordinate(neurode)
	populated.Weights = weights
	return populated, nil
}

func iterativeCoordinateWeight(ctx context.Context, input NeurodeCoordinate, neurode NeurodeCoordinate, previousWeight float64, cpp CoordinateIOWCPP, ceps []CEP, params map[string]float64) (float64, error) {
	iow := []float64{input.Output, neurode.Output, previousWeight}
	signal, err := cpp.ComputeCoordinatesIOW(ctx, input.Coords, neurode.Coords, iow, cloneFloatParams(params))
	if err != nil {
		return 0, err
	}
	if len(signal) == 0 {
		return 0, fmt.Errorf("%w: coordinate iow cpp returned empty signal", ErrInvalidSubstrateCoordinates)
	}

	weight := previousWeight
	delta := signal[0]
	for _, cep := range ceps {
		if cep == nil {
			return 0, fmt.Errorf("%w: nil cep in chain", ErrInvalidSubstrateCoordinates)
		}
		weight, err = cep.Apply(ctx, weight, delta, cloneFloatParams(params))
		if err != nil {
			return 0, err
		}
		delta = weight
	}
	return weight, nil
}

// CalculateOutputFullyInterconnected mirrors calculate_output_fi for the
// non-plastic typed path by using the flattened substrate as each layer source.
func CalculateOutputFullyInterconnected(input CoordinateHyperlayer, layers []CoordinateHyperlayer) ([]float64, []CoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, nil, fmt.Errorf("%w: missing process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	source := flattenCoordinateNeurodes(append([]CoordinateHyperlayer{input}, layers...)...)
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current := make(CoordinateHyperlayer, 0, len(layer))
		for neurodeIdx, neurode := range layer {
			calculated, err := CalculateNeurodeOutputStd(source, neurode)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d neurode %d: %w", layerIdx, neurodeIdx, err)
			}
			current = append(current, calculated)
		}
		updated = append(updated, current)
		source = replaceFlattenedLayer(source, len(input), layers, updated, layerIdx)
	}
	return outputValues(updated), updated, nil
}

// CalculateOutputJordanRecurrent mirrors the non-plastic jordan_recurrent path:
// the first process layer receives input plus the previous output layer, then
// later layers use standard layer-to-layer propagation.
func CalculateOutputJordanRecurrent(input CoordinateHyperlayer, layers []CoordinateHyperlayer) ([]float64, []CoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, nil, fmt.Errorf("%w: missing process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	previous := flattenCoordinateNeurodes(input, layers[len(layers)-1])
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current := make(CoordinateHyperlayer, 0, len(layer))
		for neurodeIdx, neurode := range layer {
			calculated, err := CalculateNeurodeOutputStd(previous, neurode)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d neurode %d: %w", layerIdx, neurodeIdx, err)
			}
			current = append(current, calculated)
		}
		updated = append(updated, current)
		previous = current
	}
	return outputValues(updated), updated, nil
}

// CalculateOutputNeuronSelfRecurrent mirrors calculate_output_nsr for the
// non-plastic typed path by prepending each neurode's previous state to the
// previous layer source before calculating that neurode.
func CalculateOutputNeuronSelfRecurrent(input CoordinateHyperlayer, layers []CoordinateHyperlayer) ([]float64, []CoordinateHyperlayer, error) {
	if len(layers) == 0 {
		return nil, nil, fmt.Errorf("%w: missing process/output hyperlayers", ErrInvalidSubstrateCoordinates)
	}

	previous := cloneCoordinateHyperlayer(input)
	updated := make([]CoordinateHyperlayer, 0, len(layers))
	for layerIdx, layer := range layers {
		current := make(CoordinateHyperlayer, 0, len(layer))
		for neurodeIdx, neurode := range layer {
			source := make(CoordinateHyperlayer, 0, len(previous)+1)
			source = append(source, NeurodeCoordinate{
				Coords:  append([]float64(nil), neurode.Coords...),
				Output:  neurode.Output,
				Weights: append([]float64(nil), neurode.Weights...),
			})
			source = append(source, cloneCoordinateHyperlayer(previous)...)
			calculated, err := CalculateNeurodeOutputStd(source, neurode)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d neurode %d: %w", layerIdx, neurodeIdx, err)
			}
			current = append(current, calculated)
		}
		updated = append(updated, current)
		previous = current
	}
	return outputValues(updated), updated, nil
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

func ioNeurodeCount(spec IOCoordinateSpec) (int, error) {
	format := spec.Format
	if format == "" {
		format = CoordinateFormatUndefined
	}
	switch format {
	case CoordinateFormatUndefined, CoordinateFormatNoGeo:
		if spec.VL <= 0 {
			return 0, fmt.Errorf("%w: vl must be > 0: %d", ErrInvalidSubstrateCoordinates, spec.VL)
		}
		return spec.VL, nil
	case CoordinateFormatSymmetric:
		return multiplyDensities(spec.Resolutions)
	case CoordinateFormatCoorded:
		if spec.Dim <= 0 {
			return 0, fmt.Errorf("%w: coorded dim must be > 0: %d", ErrInvalidSubstrateCoordinates, spec.Dim)
		}
		if len(spec.Neurodes) == 0 {
			return 0, fmt.Errorf("%w: missing coorded neurodes", ErrInvalidSubstrateCoordinates)
		}
		for i, neurode := range spec.Neurodes {
			if len(neurode.Coords) != spec.Dim {
				return 0, fmt.Errorf("%w: coorded neurode %d coordinate dimension %d does not match dim %d", ErrInvalidSubstrateCoordinates, i, len(neurode.Coords), spec.Dim)
			}
		}
		return len(spec.Neurodes), nil
	default:
		return 0, fmt.Errorf("%w: unsupported io coordinate format %q", ErrInvalidSubstrateCoordinates, spec.Format)
	}
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

// CreateSubstrate mirrors substrate.erl create_substrate/4 for the covered IO
// formats and active link forms.
func CreateSubstrate(req CreateSubstrateRequest) ([]CoordinateHyperlayer, error) {
	if len(req.Densities) == 0 {
		return nil, fmt.Errorf("%w: missing densities", ErrInvalidSubstrateCoordinates)
	}
	substrateDimension := len(req.Densities)
	inputLayer, err := ComposeInputSubstrateForDimension(req.InputSpecs, substrateDimension)
	if err != nil {
		return nil, err
	}
	outputLayer, err := ComposeOutputSubstrateForDimension(req.OutputSpecs, substrateDimension, nil)
	if err != nil {
		return nil, err
	}
	outputCount, err := CountIONeurodes(req.OutputSpecs)
	if err != nil {
		return nil, err
	}
	if outputCount != len(outputLayer) {
		return nil, fmt.Errorf("%w: output neurode count %d does not match composed layer count %d", ErrInvalidSubstrateCoordinates, outputCount, len(outputLayer))
	}
	return BuildSubstrateLayers(SubstrateLayerBuildRequest{
		InputLayer:  inputLayer,
		Densities:   req.Densities,
		OutputLayer: outputLayer,
		LinkForm:    req.LinkForm,
	})
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
		out = append(out, cloneNeurodeCoordinate(neurode))
	}
	return out
}

func cloneNeurodeCoordinate(neurode NeurodeCoordinate) NeurodeCoordinate {
	return NeurodeCoordinate{
		Coords:  append([]float64(nil), neurode.Coords...),
		Output:  neurode.Output,
		Weights: append([]float64(nil), neurode.Weights...),
	}
}

func flattenCoordinateNeurodes(layers ...CoordinateHyperlayer) CoordinateHyperlayer {
	total := 0
	for _, layer := range layers {
		total += len(layer)
	}
	out := make(CoordinateHyperlayer, 0, total)
	for _, layer := range layers {
		out = append(out, cloneCoordinateHyperlayer(layer)...)
	}
	return out
}

func replaceFlattenedLayer(source CoordinateHyperlayer, inputCount int, originalLayers []CoordinateHyperlayer, updatedLayers []CoordinateHyperlayer, layerIdx int) CoordinateHyperlayer {
	out := cloneCoordinateHyperlayer(source)
	offset := inputCount
	for i := 0; i < layerIdx; i++ {
		offset += len(originalLayers[i])
	}
	copy(out[offset:offset+len(updatedLayers[layerIdx])], cloneCoordinateHyperlayer(updatedLayers[layerIdx]))
	return out
}

func outputValues(layers []CoordinateHyperlayer) []float64 {
	if len(layers) == 0 {
		return nil
	}
	outputLayer := layers[len(layers)-1]
	outputs := make([]float64, 0, len(outputLayer))
	for _, neurode := range outputLayer {
		outputs = append(outputs, neurode.Output)
	}
	return outputs
}

func cloneFloatParams(params map[string]float64) map[string]float64 {
	if params == nil {
		return nil
	}
	out := make(map[string]float64, len(params))
	for k, v := range params {
		out[k] = v
	}
	return out
}
