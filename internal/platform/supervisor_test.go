package platform

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestSupervisorRestartsFailingTask(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{
		InitialBackoff: 1 * time.Millisecond,
		MaxBackoff:     2 * time.Millisecond,
		BackoffFactor:  1,
	})
	var calls atomic.Int32
	failures := int32(2)
	run := func(ctx context.Context) error {
		call := calls.Add(1)
		if call <= failures {
			return errors.New("boom")
		}
		<-ctx.Done()
		return ctx.Err()
	}
	if err := supervisor.Start("restarting", run); err != nil {
		t.Fatalf("start supervisor task: %v", err)
	}
	deadline := time.Now().Add(250 * time.Millisecond)
	for time.Now().Before(deadline) {
		if calls.Load() >= 3 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	if calls.Load() < 3 {
		t.Fatalf("expected task restarts to reach at least 3 calls, got=%d", calls.Load())
	}
	supervisor.StopAll()
	if len(supervisor.Tasks()) != 0 {
		t.Fatalf("expected no supervisor tasks after stop all, got=%v", supervisor.Tasks())
	}
}

func TestSupervisorStopsTaskByName(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{
		InitialBackoff: 1 * time.Millisecond,
		MaxBackoff:     2 * time.Millisecond,
		BackoffFactor:  1,
	})
	stopped := make(chan struct{})
	if err := supervisor.Start("named-stop", func(ctx context.Context) error {
		<-ctx.Done()
		close(stopped)
		return ctx.Err()
	}); err != nil {
		t.Fatalf("start supervisor task: %v", err)
	}
	supervisor.Stop("named-stop")
	select {
	case <-stopped:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("expected supervised task to stop after named stop")
	}
	if len(supervisor.Tasks()) != 0 {
		t.Fatalf("expected no supervisor tasks after named stop, got=%v", supervisor.Tasks())
	}
}

func TestSupervisorRejectsDuplicateTaskName(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{})
	if err := supervisor.Start("dup", func(ctx context.Context) error {
		<-ctx.Done()
		return nil
	}); err != nil {
		t.Fatalf("start supervisor task: %v", err)
	}
	if err := supervisor.Start("dup", func(context.Context) error { return nil }); err == nil {
		t.Fatal("expected duplicate task name to fail")
	}
	supervisor.StopAll()
}

func TestSupervisorPermanentFailureHook(t *testing.T) {
	failures := make(chan struct {
		name      string
		restarts  int
		errString string
	}, 1)
	supervisor := NewSupervisorWithHooks(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
		MaxRestarts:    1,
	}, SupervisorHooks{
		OnTaskPermanentFailure: func(name string, err error, restartCount int) {
			failures <- struct {
				name      string
				restarts  int
				errString string
			}{
				name:      name,
				restarts:  restartCount,
				errString: err.Error(),
			}
		},
	})
	if err := supervisor.Start("permanent", func(context.Context) error {
		return errors.New("boom")
	}); err != nil {
		t.Fatalf("start supervisor task: %v", err)
	}
	select {
	case failure := <-failures:
		if failure.name != "permanent" {
			t.Fatalf("unexpected failure task name: %s", failure.name)
		}
		if failure.restarts != 1 {
			t.Fatalf("expected restart count 1, got=%d", failure.restarts)
		}
		if failure.errString != "boom" {
			t.Fatalf("unexpected failure error string: %s", failure.errString)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("expected permanent failure hook callback")
	}
	supervisor.StopAll()
}

func TestSupervisorRestartHook(t *testing.T) {
	var restartCount atomic.Int32
	supervisor := NewSupervisorWithHooks(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
		MaxRestarts:    2,
	}, SupervisorHooks{
		OnTaskRestart: func(string, error, int) {
			restartCount.Add(1)
		},
	})
	if err := supervisor.Start("restart-hook", func(context.Context) error {
		return errors.New("boom")
	}); err != nil {
		t.Fatalf("start supervisor task: %v", err)
	}
	deadline := time.Now().Add(200 * time.Millisecond)
	for time.Now().Before(deadline) {
		if restartCount.Load() >= 2 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	if restartCount.Load() < 2 {
		t.Fatalf("expected at least 2 restart callbacks, got=%d", restartCount.Load())
	}
	supervisor.StopAll()
}
