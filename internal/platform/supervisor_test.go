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

func TestSupervisorOneForAllStopsSiblingTasksOnPermanentFailure(t *testing.T) {
	stopped := make(chan struct{}, 1)
	supervisor := NewSupervisorWithHooks(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
		MaxRestarts:    1,
		Strategy:       SupervisorStrategyOneForAll,
	}, SupervisorHooks{})

	if err := supervisor.Start("stable", func(ctx context.Context) error {
		<-ctx.Done()
		stopped <- struct{}{}
		return ctx.Err()
	}); err != nil {
		t.Fatalf("start stable supervisor task: %v", err)
	}
	if err := supervisor.Start("failing", func(context.Context) error {
		return errors.New("boom")
	}); err != nil {
		t.Fatalf("start failing supervisor task: %v", err)
	}

	select {
	case <-stopped:
	case <-time.After(300 * time.Millisecond):
		t.Fatal("expected stable sibling task to be stopped by one_for_all strategy")
	}
	deadline := time.Now().Add(200 * time.Millisecond)
	for time.Now().Before(deadline) {
		if len(supervisor.Tasks()) == 0 {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("expected no running tasks after one_for_all failure, got=%v", supervisor.Tasks())
}

func TestSupervisorOneForAllRestartsSiblingTasksOnRecoverableFailure(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
		MaxRestarts:    2,
		Strategy:       SupervisorStrategyOneForAll,
	})
	var stableRuns atomic.Int32
	var failingRuns atomic.Int32

	if err := supervisor.StartSpec(SupervisorChildSpec{
		Name:    "stable",
		Restart: SupervisorRestartPermanent,
	}, func(ctx context.Context) error {
		stableRuns.Add(1)
		<-ctx.Done()
		return ctx.Err()
	}); err != nil {
		t.Fatalf("start stable task: %v", err)
	}
	if err := supervisor.StartSpec(SupervisorChildSpec{
		Name:    "failing-once",
		Restart: SupervisorRestartPermanent,
	}, func(ctx context.Context) error {
		run := failingRuns.Add(1)
		if run == 1 {
			return errors.New("boom")
		}
		<-ctx.Done()
		return ctx.Err()
	}); err != nil {
		t.Fatalf("start failing task: %v", err)
	}
	defer supervisor.StopAll()

	deadline := time.Now().Add(300 * time.Millisecond)
	for time.Now().Before(deadline) {
		if stableRuns.Load() >= 2 && failingRuns.Load() >= 2 {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf(
		"expected one_for_all sibling restart for recoverable failure, stable_runs=%d failing_runs=%d tasks=%v",
		stableRuns.Load(),
		failingRuns.Load(),
		supervisor.Tasks(),
	)
}

func TestSupervisorOneForAllDoesNotRestartTemporarySibling(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
		MaxRestarts:    2,
		Strategy:       SupervisorStrategyOneForAll,
	})
	var temporaryRuns atomic.Int32
	var failingRuns atomic.Int32

	if err := supervisor.StartSpec(SupervisorChildSpec{
		Name:    "temporary",
		Restart: SupervisorRestartTemporary,
	}, func(ctx context.Context) error {
		temporaryRuns.Add(1)
		<-ctx.Done()
		return ctx.Err()
	}); err != nil {
		t.Fatalf("start temporary task: %v", err)
	}
	if err := supervisor.StartSpec(SupervisorChildSpec{
		Name:    "failing-once",
		Restart: SupervisorRestartPermanent,
	}, func(ctx context.Context) error {
		run := failingRuns.Add(1)
		if run == 1 {
			return errors.New("boom")
		}
		<-ctx.Done()
		return ctx.Err()
	}); err != nil {
		t.Fatalf("start failing task: %v", err)
	}
	defer supervisor.StopAll()

	deadline := time.Now().Add(300 * time.Millisecond)
	for time.Now().Before(deadline) {
		if failingRuns.Load() >= 2 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	if failingRuns.Load() < 2 {
		t.Fatalf("expected failing task to restart under one_for_all, runs=%d", failingRuns.Load())
	}
	time.Sleep(20 * time.Millisecond)
	if temporaryRuns.Load() != 1 {
		t.Fatalf("expected temporary sibling to avoid restart, runs=%d", temporaryRuns.Load())
	}
	tasks := supervisor.Tasks()
	if len(tasks) != 1 || tasks[0] != "failing-once" {
		t.Fatalf("expected only failing task to remain supervised, tasks=%v", tasks)
	}
}

func TestSupervisorChildrenExposeSpecAndStatus(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
		MaxRestarts:    2,
	})
	spec := SupervisorChildSpec{
		Name:    "child-status",
		Group:   "support",
		Restart: SupervisorRestartPermanent,
	}
	if err := supervisor.StartSpec(spec, func(context.Context) error {
		return errors.New("boom")
	}); err != nil {
		t.Fatalf("start child-status task: %v", err)
	}
	deadline := time.Now().Add(250 * time.Millisecond)
	for time.Now().Before(deadline) {
		children := supervisor.Children()
		if len(children) == 1 && children[0].RestartCount > 0 && children[0].LastError != "" {
			child := children[0]
			if child.Name != spec.Name {
				t.Fatalf("unexpected child name: %s", child.Name)
			}
			if child.Group != spec.Group {
				t.Fatalf("unexpected child group: %s", child.Group)
			}
			if child.RestartPolicy != spec.Restart {
				t.Fatalf("unexpected child restart policy: %s", child.RestartPolicy)
			}
			supervisor.StopAll()
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("expected child status with restart metadata, got=%+v", supervisor.Children())
}

func TestSupervisorPermanentRestartsOnNormalExit(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
	})
	var calls atomic.Int32
	if err := supervisor.StartSpec(SupervisorChildSpec{
		Name:    "permanent-normal",
		Restart: SupervisorRestartPermanent,
	}, func(ctx context.Context) error {
		call := calls.Add(1)
		if call <= 2 {
			return nil
		}
		<-ctx.Done()
		return ctx.Err()
	}); err != nil {
		t.Fatalf("start permanent-normal task: %v", err)
	}
	deadline := time.Now().Add(250 * time.Millisecond)
	for time.Now().Before(deadline) {
		if calls.Load() >= 3 {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	if calls.Load() < 3 {
		t.Fatalf("expected permanent task restart on normal exit, got calls=%d", calls.Load())
	}
	supervisor.StopAll()
}

func TestSupervisorTransientDoesNotRestartOnNormalExit(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
	})
	var calls atomic.Int32
	if err := supervisor.StartSpec(SupervisorChildSpec{
		Name:    "transient-normal",
		Restart: SupervisorRestartTransient,
	}, func(context.Context) error {
		calls.Add(1)
		return nil
	}); err != nil {
		t.Fatalf("start transient-normal task: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
	if calls.Load() != 1 {
		t.Fatalf("expected one transient call on normal exit, got=%d", calls.Load())
	}
	if len(supervisor.Tasks()) != 0 {
		t.Fatalf("expected no active transient task after normal exit, got=%v", supervisor.Tasks())
	}
}

func TestSupervisorTemporaryDoesNotRestartOnError(t *testing.T) {
	supervisor := NewSupervisor(SupervisorPolicy{
		InitialBackoff: time.Millisecond,
		MaxBackoff:     time.Millisecond,
		BackoffFactor:  1,
		MaxRestarts:    100,
	})
	var calls atomic.Int32
	if err := supervisor.StartSpec(SupervisorChildSpec{
		Name:    "temporary-error",
		Restart: SupervisorRestartTemporary,
	}, func(context.Context) error {
		calls.Add(1)
		return errors.New("boom")
	}); err != nil {
		t.Fatalf("start temporary-error task: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
	if calls.Load() != 1 {
		t.Fatalf("expected one temporary call on error, got=%d", calls.Load())
	}
	if len(supervisor.Tasks()) != 0 {
		t.Fatalf("expected no active temporary task after error exit, got=%v", supervisor.Tasks())
	}
}
