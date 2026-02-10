package nn

import (
	"fmt"
	"math"
)

func Derivative(name string, x float64) (float64, error) {
	switch name {
	case "identity":
		return 1, nil
	case "relu":
		if x > 0 {
			return 1, nil
		}
		return 0, nil
	case "tanh":
		y := math.Tanh(x)
		return 1 - (y * y), nil
	case "sigmoid":
		s := 1 / (1 + math.Exp(-x))
		return s * (1 - s), nil
	case "sin":
		return math.Cos(x), nil
	case "cos":
		return -math.Sin(x), nil
	case "absolute":
		if x >= 0 {
			return 1, nil
		}
		return -1, nil
	case "quadratic":
		if x >= 0 {
			return 2 * x, nil
		}
		return -2 * x, nil
	case "gaussian":
		// d/dx exp(-x^2) = -2x*exp(-x^2)
		return -2 * x * math.Exp(-(x * x)), nil
	default:
		return 0, fmt.Errorf("unsupported derivative: %s", name)
	}
}
