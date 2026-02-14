package nn

import (
	"fmt"
	"math"
)

func Derivative(name string, x float64) (float64, error) {
	switch name {
	case "identity":
		return 1, nil
	case "linear":
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
		if x > 10 {
			x = 10
		} else if x < -10 {
			x = -10
		}
		s := 1 / (1 + math.Exp(-x))
		return s * (1 - s), nil
	case "sigmoid1":
		denom := 1 + math.Abs(x)
		return 1 / (denom * denom), nil
	case "sin":
		return math.Cos(x), nil
	case "cos":
		return -math.Sin(x), nil
	case "multiquadric":
		base := (x * x) + 0.01
		// Match reference derivative implementation.
		return -0.5 * x * math.Pow(base, -1.5), nil
	case "absolute":
		if x > 0 {
			return 1, nil
		}
		return -1, nil
	case "quadratic":
		if x >= 0 {
			return 2 * x, nil
		}
		return -2 * x, nil
	case "gaussian":
		if x > 10 {
			x = 10
		} else if x < -10 {
			x = -10
		}
		// d/dx exp(-x^2) = -2x*exp(-x^2)
		return -2 * x * math.Exp(-(x * x)), nil
	case "sqrt":
		if x == 0 {
			return 0, nil
		}
		return 1 / (2 * math.Sqrt(math.Abs(x))), nil
	case "log":
		if x == 0 {
			return 0, nil
		}
		return 1 / math.Abs(x), nil
	default:
		return 0, fmt.Errorf("unsupported derivative: %s", name)
	}
}
