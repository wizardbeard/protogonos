package substrate

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type coordinateRuntimeCPP struct{}

func (coordinateRuntimeCPP) Name() string { return "coordinate_runtime_cpp" }

func (coordinateRuntimeCPP) Compute(_ context.Context, _ []float64, _ map[string]float64) (float64, error) {
	return 0, nil
}

func (coordinateRuntimeCPP) ComputeCoordinates(_ context.Context, presynaptic []float64, postsynaptic []float64, params map[string]float64) ([]float64, error) {
	scale := 1.0
	if value, ok := params["scale"]; ok {
		scale = value
	}
	return []float64{
		(presynaptic[0] + postsynaptic[0]) * scale,
		(presynaptic[1] - postsynaptic[1]) * scale,
	}, nil
}

func (coordinateRuntimeCPP) ComputeCoordinatesIOW(_ context.Context, presynaptic []float64, postsynaptic []float64, iow []float64, params map[string]float64) ([]float64, error) {
	scale := 1.0
	if value, ok := params["scale"]; ok {
		scale = value
	}
	return []float64{
		(presynaptic[0] + postsynaptic[0] + iow[0]) * scale,
		iow[1],
		iow[2],
	}, nil
}

func TestCPPProcessComputeMessage(t *testing.T) {
	process, err := NewCPPProcess("cpp_test", "substrate_test", "exoself_test", SetWeightCPP{}, map[string]float64{"scale": 99})
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}

	output, err := process.HandleMessage(context.Background(), CPPComputeMessage{
		FromPID: "substrate_test",
		Input:   []float64{1, 3},
	})
	if err != nil {
		t.Fatalf("handle compute message: %v", err)
	}
	if len(output) != 1 || output[0] != 2 {
		t.Fatalf("unexpected cpp output: %v", output)
	}
}

func TestCPPProcessCoordinateMessage(t *testing.T) {
	process, err := NewCPPProcess("cpp_test", "substrate_test", "exoself_test", coordinateRuntimeCPP{}, map[string]float64{"scale": 2})
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}

	output, err := process.HandleMessage(context.Background(), CPPCoordinateMessage{
		FromPID:            "substrate_test",
		PresynapticCoords:  []float64{0.25, 0.75},
		PostsynapticCoords: []float64{0.5, 0.25},
	})
	if err != nil {
		t.Fatalf("handle coordinate message: %v", err)
	}
	want := []float64{1.5, 1.0}
	if !reflect.DeepEqual(output, want) {
		t.Fatalf("unexpected coordinate cpp output: got=%v want=%v", output, want)
	}
}

func TestCPPProcessCoordinateIOWMessage(t *testing.T) {
	process, err := NewCPPProcess("cpp_test", "substrate_test", "exoself_test", coordinateRuntimeCPP{}, map[string]float64{"scale": 2})
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}

	output, err := process.HandleMessage(context.Background(), CPPCoordinateMessage{
		FromPID:            "substrate_test",
		PresynapticCoords:  []float64{0.25, 0.75},
		PostsynapticCoords: []float64{0.5, 0.25},
		IOW:                []float64{0.25, 0.5, -0.5},
	})
	if err != nil {
		t.Fatalf("handle coordinate iow message: %v", err)
	}
	want := []float64{2.0, 0.5, -0.5}
	if !reflect.DeepEqual(output, want) {
		t.Fatalf("unexpected coordinate iow cpp output: got=%v want=%v", output, want)
	}
}

func TestCPPProcessCoordinateMessageRequiresCoordinateCPP(t *testing.T) {
	process, err := NewCPPProcess("cpp_test", "substrate_test", "exoself_test", SetWeightCPP{}, nil)
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}

	if _, err := process.HandleMessage(context.Background(), CPPCoordinateMessage{
		FromPID:            "substrate_test",
		PresynapticCoords:  []float64{0.25},
		PostsynapticCoords: []float64{0.5},
	}); !errors.Is(err, ErrCPPCoordinateNotSupported) {
		t.Fatalf("expected ErrCPPCoordinateNotSupported, got %v", err)
	}
}

func TestCPPProcessRejectsUnexpectedSubstrateSender(t *testing.T) {
	process, err := NewCPPProcess("cpp_test", "substrate_test", "exoself_test", SetWeightCPP{}, nil)
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}

	if _, err := process.HandleMessage(context.Background(), CPPComputeMessage{
		FromPID: "other_substrate",
		Input:   []float64{1},
	}); !errors.Is(err, ErrUnexpectedCPPComputePID) {
		t.Fatalf("expected ErrUnexpectedCPPComputePID, got %v", err)
	}
}

func TestCPPActorComputeCoordinatesFrom(t *testing.T) {
	process, err := NewCPPProcess(
		"cpp_test",
		runtimeSubstrateProcessID,
		runtimeExoSelfProcessID,
		coordinateRuntimeCPP{},
		map[string]float64{"scale": 2},
	)
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}
	actor := NewCPPActor(process)
	t.Cleanup(func() {
		_ = actor.TerminateFrom(runtimeExoSelfProcessID)
	})

	output, err := actor.ComputeCoordinatesFrom(
		context.Background(),
		runtimeSubstrateProcessID,
		[]float64{0.25, 0.75},
		[]float64{0.5, 0.25},
	)
	if err != nil {
		t.Fatalf("compute coordinates from actor: %v", err)
	}
	want := []float64{1.5, 1.0}
	if !reflect.DeepEqual(output, want) {
		t.Fatalf("unexpected coordinate actor output: got=%v want=%v", output, want)
	}
}

func TestCPPActorComputeCoordinatesIOWFrom(t *testing.T) {
	process, err := NewCPPProcess(
		"cpp_test",
		runtimeSubstrateProcessID,
		runtimeExoSelfProcessID,
		coordinateRuntimeCPP{},
		map[string]float64{"scale": 2},
	)
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}
	actor := NewCPPActor(process)
	t.Cleanup(func() {
		_ = actor.TerminateFrom(runtimeExoSelfProcessID)
	})

	output, err := actor.ComputeCoordinatesIOWFrom(
		context.Background(),
		runtimeSubstrateProcessID,
		[]float64{0.25, 0.75},
		[]float64{0.5, 0.25},
		[]float64{0.25, 0.5, -0.5},
	)
	if err != nil {
		t.Fatalf("compute coordinates iow from actor: %v", err)
	}
	want := []float64{2.0, 0.5, -0.5}
	if !reflect.DeepEqual(output, want) {
		t.Fatalf("unexpected coordinate iow actor output: got=%v want=%v", output, want)
	}
}

func TestCPPActorComputeVectorFrom(t *testing.T) {
	process, err := NewCPPProcess(
		"cpp_test",
		runtimeSubstrateProcessID,
		runtimeExoSelfProcessID,
		vectorRuntimeCPP{signals: []float64{1, 0.5, -0.25}},
		nil,
	)
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}
	actor := NewCPPActor(process)
	t.Cleanup(func() {
		_ = actor.TerminateFrom(runtimeExoSelfProcessID)
	})

	output, err := actor.ComputeVectorFrom(context.Background(), runtimeSubstrateProcessID, []float64{0})
	if err != nil {
		t.Fatalf("compute vector from actor: %v", err)
	}
	want := []float64{1, 0.5, -0.25}
	if len(output) != len(want) {
		t.Fatalf("unexpected vector width: got=%d want=%d output=%v", len(output), len(want), output)
	}
	for i := range want {
		if math.Abs(output[i]-want[i]) > 1e-9 {
			t.Fatalf("unexpected vector output[%d]: got=%v want=%v", i, output[i], want[i])
		}
	}
}

func TestCPPActorTerminateValidatesOwner(t *testing.T) {
	process, err := NewCPPProcess("cpp_test", runtimeSubstrateProcessID, runtimeExoSelfProcessID, SetWeightCPP{}, nil)
	if err != nil {
		t.Fatalf("new cpp process: %v", err)
	}
	actor := NewCPPActor(process)
	t.Cleanup(func() {
		if !process.terminated {
			_ = actor.TerminateFrom(runtimeExoSelfProcessID)
		}
	})

	if err := actor.TerminateFrom("other_exoself"); !errors.Is(err, ErrUnexpectedCPPTerminatePID) {
		t.Fatalf("expected ErrUnexpectedCPPTerminatePID, got %v", err)
	}
	if process.terminated {
		t.Fatal("unexpected cpp process termination from wrong owner")
	}
	if err := actor.TerminateFrom(runtimeExoSelfProcessID); err != nil {
		t.Fatalf("terminate from owner: %v", err)
	}
	if _, err := actor.ComputeFrom(context.Background(), runtimeSubstrateProcessID, []float64{1}); !errors.Is(err, ErrCPPActorTerminated) {
		t.Fatalf("expected ErrCPPActorTerminated after terminate, got %v", err)
	}
}
