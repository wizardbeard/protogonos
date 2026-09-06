package scape

import (
	"fmt"
	"math"
)

// Distance returns the Euclidean distance between two equal-width vectors.
// It mirrors the exported scape:distance/2 helper from the Erlang reference.
func Distance(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("distance vector width mismatch: %d != %d", len(a), len(b))
	}
	var sum float64
	for i := range a {
		delta := b[i] - a[i]
		sum += delta * delta
	}
	return math.Sqrt(sum), nil
}
