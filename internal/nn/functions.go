package nn

import (
	"fmt"
	"math"
)

const defaultSaturationLimit = 1000.0

// Saturation clamps values to the reference default range [-1000, 1000].
func Saturation(value float64) float64 {
	return SaturationWithSpread(value, defaultSaturationLimit)
}

// SaturationWithSpread clamps values to the symmetric range [-spread, spread].
func SaturationWithSpread(value, spread float64) float64 {
	if spread < 0 {
		spread = -spread
	}
	if value > spread {
		return spread
	}
	if value < -spread {
		return -spread
	}
	return value
}

// ScaleValue maps value from [min, max] to [-1, 1].
func ScaleValue(value, max, min float64) float64 {
	if max == min {
		return 0
	}
	return (value*2 - (max + min)) / (max - min)
}

// ScaleSlice maps each value from [min, max] to [-1, 1].
func ScaleSlice(values []float64, max, min float64) []float64 {
	out := make([]float64, len(values))
	for i, value := range values {
		out[i] = ScaleValue(value, max, min)
	}
	return out
}

// Sat clamps value to [min, max].
func Sat(value, max, min float64) float64 {
	if value > max {
		return max
	}
	if value < min {
		return min
	}
	return value
}

// SatDeadZone clamps value to [min, max] while zeroing values inside dead-zone bounds.
func SatDeadZone(value, max, min, deadZoneMax, deadZoneMin float64) float64 {
	if value < deadZoneMax && value > deadZoneMin {
		return 0
	}
	return Sat(value, max, min)
}

// Avg returns the arithmetic mean of values.
func Avg(values []float64) (float64, error) {
	if len(values) == 0 {
		return 0, fmt.Errorf("values must not be empty")
	}
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values)), nil
}

// Std returns population standard deviation.
func Std(values []float64) (float64, error) {
	mean, err := Avg(values)
	if err != nil {
		return 0, err
	}
	sum := 0.0
	for _, value := range values {
		diff := mean - value
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(values))), nil
}

func Cartesian(inCoord, coord []float64) []float64 {
	out := make([]float64, 0, len(inCoord)+len(coord))
	out = append(out, inCoord...)
	out = append(out, coord...)
	return out
}

func Polar(inCoord, coord []float64) ([]float64, error) {
	inPol, err := cartesianToPolarList(inCoord)
	if err != nil {
		return nil, err
	}
	coordPol, err := cartesianToPolarList(coord)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(inPol)+len(coordPol))
	out = append(out, inPol...)
	out = append(out, coordPol...)
	return out, nil
}

func Spherical(inCoord, coord []float64) ([]float64, error) {
	inSph, err := cartesianToSphericalList(inCoord)
	if err != nil {
		return nil, err
	}
	coordSph, err := cartesianToSphericalList(coord)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(inSph)+len(coordSph))
	out = append(out, inSph...)
	out = append(out, coordSph...)
	return out, nil
}

func CentripetalDistances(inCoord, coord []float64) []float64 {
	return []float64{centripetalDistance(inCoord), centripetalDistance(coord)}
}

func CartesianDistance(inCoord, coord []float64) ([]float64, error) {
	d, err := Distance(inCoord, coord)
	if err != nil {
		return nil, err
	}
	return []float64{d}, nil
}

func CartesianCoordDiffs(inCoord, coord []float64) ([]float64, error) {
	return VectorDifference(inCoord, coord)
}

func CartesianGaussedCoordDiffs(inCoord, coord []float64) ([]float64, error) {
	diffs, err := VectorDifference(inCoord, coord)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(diffs))
	for i, diff := range diffs {
		out[i] = gaussianActivation(diff)
	}
	return out, nil
}

func CartesianWithIOW(inCoord, coord, iow []float64) []float64 {
	base := Cartesian(inCoord, coord)
	out := make([]float64, 0, len(iow)+len(base))
	out = append(out, iow...)
	out = append(out, base...)
	return out
}

func PolarWithIOW(inCoord, coord, iow []float64) ([]float64, error) {
	base, err := Polar(inCoord, coord)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(iow)+len(base))
	out = append(out, iow...)
	out = append(out, base...)
	return out, nil
}

func SphericalWithIOW(inCoord, coord, iow []float64) ([]float64, error) {
	base, err := Spherical(inCoord, coord)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(iow)+len(base))
	out = append(out, iow...)
	out = append(out, base...)
	return out, nil
}

func CentripetalDistancesWithIOW(inCoord, coord, iow []float64) []float64 {
	base := CentripetalDistances(inCoord, coord)
	out := make([]float64, 0, len(iow)+len(base))
	out = append(out, iow...)
	out = append(out, base...)
	return out
}

func CartesianDistanceWithIOW(inCoord, coord, iow []float64) ([]float64, error) {
	base, err := CartesianDistance(inCoord, coord)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(iow)+len(base))
	out = append(out, iow...)
	out = append(out, base...)
	return out, nil
}

func CartesianCoordDiffsWithIOW(inCoord, coord, iow []float64) ([]float64, error) {
	base, err := CartesianCoordDiffs(inCoord, coord)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(iow)+len(base))
	out = append(out, iow...)
	out = append(out, base...)
	return out, nil
}

func CartesianGaussedCoordDiffsWithIOW(inCoord, coord, iow []float64) ([]float64, error) {
	base, err := CartesianGaussedCoordDiffs(inCoord, coord)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(iow)+len(base))
	out = append(out, iow...)
	out = append(out, base...)
	return out, nil
}

func IOW(_ []float64, _ []float64, iow []float64) []float64 {
	out := make([]float64, len(iow))
	copy(out, iow)
	return out
}

func ToCartesian(kind string, coordinates []float64) ([]float64, error) {
	switch kind {
	case "spherical":
		return SphericalToCartesianList(coordinates)
	case "polar":
		return PolarToCartesianList(coordinates)
	case "cartesian":
		out := make([]float64, len(coordinates))
		copy(out, coordinates)
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported coordinate kind: %s", kind)
	}
}

func Normalize(vector []float64) ([]float64, error) {
	normalizer := centripetalDistance(vector)
	if normalizer == 0 {
		return nil, fmt.Errorf("cannot normalize zero vector")
	}
	out := make([]float64, len(vector))
	for i, value := range vector {
		out[i] = value / normalizer
	}
	return out, nil
}

func SphericalToCartesianList(coordinates []float64) ([]float64, error) {
	if len(coordinates) != 3 {
		return nil, fmt.Errorf("spherical coordinates must have length 3")
	}
	p, theta, phi := coordinates[0], coordinates[1], coordinates[2]
	x := p * math.Sin(phi) * math.Cos(theta)
	y := p * math.Sin(phi) * math.Sin(theta)
	z := p * math.Cos(phi)
	return []float64{x, y, z}, nil
}

func CartesianToSphericalList(coordinates []float64) ([]float64, error) {
	if len(coordinates) == 2 {
		coordinates = []float64{coordinates[0], coordinates[1], 0}
	}
	if len(coordinates) != 3 {
		return nil, fmt.Errorf("cartesian coordinates must have length 2 or 3")
	}
	x, y, z := coordinates[0], coordinates[1], coordinates[2]
	preR := x*x + y*y
	r := math.Sqrt(preR)
	p := math.Sqrt(preR + z*z)
	theta := angleTheta(x, y, r)
	phi := 0.0
	if p != 0 {
		phi = math.Acos(z / p)
	}
	return []float64{p, theta, phi}, nil
}

func PolarToCartesianList(coordinates []float64) ([]float64, error) {
	if len(coordinates) != 2 {
		return nil, fmt.Errorf("polar coordinates must have length 2")
	}
	r, theta := coordinates[0], coordinates[1]
	x := r * math.Cos(theta)
	y := r * math.Sin(theta)
	return []float64{x, y, 0}, nil
}

func CartesianToPolarList(coordinates []float64) ([]float64, error) {
	if len(coordinates) == 2 {
		coordinates = []float64{coordinates[0], coordinates[1], 0}
	}
	if len(coordinates) != 3 {
		return nil, fmt.Errorf("cartesian coordinates must have length 2 or 3")
	}
	x, y := coordinates[0], coordinates[1]
	r := math.Sqrt(x*x + y*y)
	theta := angleTheta(x, y, r)
	return []float64{r, theta}, nil
}

func Distance(vector1, vector2 []float64) (float64, error) {
	if len(vector1) != len(vector2) {
		return 0, fmt.Errorf("vector length mismatch: %d != %d", len(vector1), len(vector2))
	}
	acc := 0.0
	for i := range vector1 {
		d := vector2[i] - vector1[i]
		acc += d * d
	}
	return math.Sqrt(acc), nil
}

func VectorDifference(vector1, vector2 []float64) ([]float64, error) {
	if len(vector1) != len(vector2) {
		return nil, fmt.Errorf("vector length mismatch: %d != %d", len(vector1), len(vector2))
	}
	out := make([]float64, len(vector1))
	for i := range vector1 {
		out[i] = vector2[i] - vector1[i]
	}
	return out, nil
}

// XORDemoStep mirrors the reference demo helper in functions.erl (s/1) but
// keeps recurrent state explicit instead of process-dictionary based.
func XORDemoStep(v1, v2, recurrentSignal float64) (output1, output2, nextRecurrentSignal float64) {
	output1 = math.Tanh(v1*-4.3986 + v2*-2.3223 + recurrentSignal*6.2832 + 1.3463)
	output2 = math.Tanh(output1*-4.9582 - 2.4443)
	return output1, output2, output1
}

func cartesianToPolarList(coordinates []float64) ([]float64, error) {
	if len(coordinates) != 2 {
		return nil, fmt.Errorf("polar conversion expects [Y,X] length 2 coordinates")
	}
	y, x := coordinates[0], coordinates[1]
	r := math.Sqrt(x*x + y*y)
	theta := angleTheta(x, y, r)
	return []float64{r, theta}, nil
}

func cartesianToSphericalList(coordinates []float64) ([]float64, error) {
	if len(coordinates) != 3 {
		return nil, fmt.Errorf("spherical conversion expects [Z,Y,X] length 3 coordinates")
	}
	z, y, x := coordinates[0], coordinates[1], coordinates[2]
	preR := x*x + y*y
	r := math.Sqrt(preR)
	p := math.Sqrt(preR + z*z)
	theta := angleTheta(x, y, r)
	phi := 0.0
	if p != 0 {
		phi = math.Acos(z / p)
	}
	return []float64{p, theta, phi}, nil
}

func centripetalDistance(coordinates []float64) float64 {
	acc := 0.0
	for _, value := range coordinates {
		acc += value * value
	}
	return math.Sqrt(acc)
}

func gaussianActivation(value float64) float64 {
	if value > 10 {
		value = 10
	} else if value < -10 {
		value = -10
	}
	return math.Exp(-(value * value))
}

func angleTheta(x, y, r float64) float64 {
	if r == 0 {
		return 0
	}
	switch {
	case x > 0 && y >= 0:
		return math.Atan(y / x)
	case x > 0 && y < 0:
		return math.Atan(y/x) + 2*math.Pi
	case x < 0:
		return math.Atan(y/x) + math.Pi
	case x == 0 && y > 0:
		return math.Pi / 2
	case x == 0 && y < 0:
		return 3 * math.Pi / 2
	default:
		return 0
	}
}
