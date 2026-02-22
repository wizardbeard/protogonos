package genotype

import (
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"time"
)

// RandomElement mirrors genotype:random_element/1 with optional RNG injection.
func RandomElement[T any](rng *rand.Rand, values []T) (T, error) {
	var zero T
	if len(values) == 0 {
		return zero, fmt.Errorf("values are required")
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	return values[rng.Intn(len(values))], nil
}

// CalculateOptimalSubstrateDimension mirrors
// genotype:calculate_OptimalSubstrateDimension/2 by taking the maximum
// geometric format dimensionality and adding a +2 depth margin.
// Unknown/no-geometry formats default to dimensionality 1.
func CalculateOptimalSubstrateDimension(sensorFormats, actuatorFormats []any) int {
	maxDim := 1
	for _, format := range sensorFormats {
		if dim := formatDimensionCount(format); dim > maxDim {
			maxDim = dim
		}
	}
	for _, format := range actuatorFormats {
		if dim := formatDimensionCount(format); dim > maxDim {
			maxDim = dim
		}
	}
	return maxDim + 2
}

func formatDimensionCount(format any) int {
	if format == nil {
		return 1
	}
	switch f := format.(type) {
	case string:
		switch strings.ToLower(strings.TrimSpace(f)) {
		case "", "no_geo", "undefined":
			return 1
		default:
			return 1
		}
	case []int:
		if len(f) == 0 {
			return 1
		}
		return len(f)
	case []float64:
		if len(f) == 0 {
			return 1
		}
		return len(f)
	case map[string]any:
		if dims, ok := f["dimensions"]; ok {
			if dim := anySliceLen(dims); dim > 0 {
				return dim
			}
		}
		if dims, ok := f["dims"]; ok {
			if dim := anySliceLen(dims); dim > 0 {
				return dim
			}
		}
		return 1
	default:
		value := reflect.ValueOf(format)
		if !value.IsValid() {
			return 1
		}
		if value.Kind() == reflect.Pointer {
			if value.IsNil() {
				return 1
			}
			value = value.Elem()
		}
		if value.Kind() == reflect.Struct {
			field := value.FieldByName("Dimensions")
			if field.IsValid() && field.Kind() == reflect.Slice {
				if field.Len() == 0 {
					return 1
				}
				return field.Len()
			}
			return 1
		}
		if value.Kind() == reflect.Slice || value.Kind() == reflect.Array {
			if value.Len() == 0 {
				return 1
			}
			return value.Len()
		}
		return 1
	}
}

func anySliceLen(value any) int {
	if value == nil {
		return 0
	}
	switch raw := value.(type) {
	case []int:
		return len(raw)
	case []float64:
		return len(raw)
	case []any:
		return len(raw)
	default:
		rv := reflect.ValueOf(value)
		if !rv.IsValid() {
			return 0
		}
		if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
			return rv.Len()
		}
		return 0
	}
}
