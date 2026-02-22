package genotype

import (
	"math/rand"
	"testing"
)

func TestRandomElementDeterministicWithSeed(t *testing.T) {
	values := []string{"a", "b", "c", "d"}
	rng := rand.New(rand.NewSource(7))
	gotA, err := RandomElement(rng, values)
	if err != nil {
		t.Fatalf("random element first call: %v", err)
	}

	rng2 := rand.New(rand.NewSource(7))
	gotB, err := RandomElement(rng2, values)
	if err != nil {
		t.Fatalf("random element second call: %v", err)
	}
	if gotA != gotB {
		t.Fatalf("expected deterministic output with equal seeds, got %q != %q", gotA, gotB)
	}
}

func TestRandomElementValidatesEmptyInput(t *testing.T) {
	if _, err := RandomElement[int](rand.New(rand.NewSource(1)), nil); err == nil {
		t.Fatal("expected error for empty values")
	}
}

func TestCalculateOptimalSubstrateDimension(t *testing.T) {
	type shape struct {
		Dimensions []int
	}
	sensorFormats := []any{
		"no_geo",
		map[string]any{"dimensions": []any{1, 2, 3}},
		shape{Dimensions: []int{0, 0, 0, 0}},
	}
	actuatorFormats := []any{
		nil,
		[]int{1, 2},
		map[string]any{"dims": []float64{1, 2, 3}},
	}
	got := CalculateOptimalSubstrateDimension(sensorFormats, actuatorFormats)
	if got != 6 {
		t.Fatalf("expected optimal substrate dimension 6, got %d", got)
	}
}

func TestCalculateOptimalSubstrateDimensionDefaults(t *testing.T) {
	got := CalculateOptimalSubstrateDimension(nil, nil)
	if got != 3 {
		t.Fatalf("expected default optimal substrate dimension 3, got %d", got)
	}
}
