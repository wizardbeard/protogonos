package substrate

import (
	"context"
	"reflect"
	"testing"
)

type recordingCPP struct {
	inputs []float64
}

func (c *recordingCPP) Name() string { return "recording_cpp" }

func (c *recordingCPP) Compute(_ context.Context, inputs []float64, _ map[string]float64) (float64, error) {
	c.inputs = append([]float64(nil), inputs...)
	return 0.75, nil
}

func TestCoordinateCPPAdapterFlattensCoordinates(t *testing.T) {
	cpp := &recordingCPP{}
	adapter := CoordinateCPPAdapter{CPP: cpp}

	got, err := adapter.ComputeCoordinates(context.Background(), []float64{0.1, 0.2}, []float64{0.3}, nil)
	if err != nil {
		t.Fatalf("compute coordinates: %v", err)
	}
	if !reflect.DeepEqual(got, []float64{0.75}) {
		t.Fatalf("unexpected coordinate output: %v", got)
	}
	if !reflect.DeepEqual(cpp.inputs, []float64{0.1, 0.2, 0.3}) {
		t.Fatalf("unexpected flattened inputs: %v", cpp.inputs)
	}
}

func TestCoordinateIOWCPPAdapterFlattensCoordinatesAndIOW(t *testing.T) {
	cpp := &recordingCPP{}
	adapter := CoordinateIOWCPPAdapter{CPP: cpp}

	got, err := adapter.ComputeCoordinatesIOW(context.Background(), []float64{0.1}, []float64{0.2}, []float64{0.3, 0.4}, nil)
	if err != nil {
		t.Fatalf("compute coordinates iow: %v", err)
	}
	if !reflect.DeepEqual(got, []float64{0.75}) {
		t.Fatalf("unexpected coordinate iow output: %v", got)
	}
	if !reflect.DeepEqual(cpp.inputs, []float64{0.1, 0.2, 0.3, 0.4}) {
		t.Fatalf("unexpected flattened inputs: %v", cpp.inputs)
	}
}
