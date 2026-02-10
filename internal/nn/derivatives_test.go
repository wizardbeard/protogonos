package nn

import "testing"

func TestDerivativeBasics(t *testing.T) {
	cases := []struct {
		name string
		x    float64
	}{
		{"identity", 1},
		{"relu", 1},
		{"tanh", 0.5},
		{"sigmoid", 0.5},
		{"sin", 0.25},
		{"cos", 0.25},
		{"absolute", -1},
		{"quadratic", 2},
		{"gaussian", 0.1},
	}
	for _, c := range cases {
		if _, err := Derivative(c.name, c.x); err != nil {
			t.Fatalf("derivative %s failed: %v", c.name, err)
		}
	}
}

func TestDerivativeUnsupported(t *testing.T) {
	if _, err := Derivative("unknown", 1); err == nil {
		t.Fatal("expected unsupported derivative error")
	}
}
