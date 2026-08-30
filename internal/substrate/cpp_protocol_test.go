package substrate

import (
	"context"
	"errors"
	"math"
	"testing"
)

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
