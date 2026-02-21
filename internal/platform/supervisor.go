package platform

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

type SupervisorPolicy struct {
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	BackoffFactor  float64
	MaxRestarts    int
}

func defaultSupervisorPolicy() SupervisorPolicy {
	return SupervisorPolicy{
		InitialBackoff: 10 * time.Millisecond,
		MaxBackoff:     200 * time.Millisecond,
		BackoffFactor:  2.0,
		MaxRestarts:    0,
	}
}

func normalizeSupervisorPolicy(policy SupervisorPolicy) SupervisorPolicy {
	def := defaultSupervisorPolicy()
	if policy.InitialBackoff <= 0 {
		policy.InitialBackoff = def.InitialBackoff
	}
	if policy.MaxBackoff <= 0 {
		policy.MaxBackoff = def.MaxBackoff
	}
	if policy.MaxBackoff < policy.InitialBackoff {
		policy.MaxBackoff = policy.InitialBackoff
	}
	if policy.BackoffFactor < 1 {
		policy.BackoffFactor = def.BackoffFactor
	}
	return policy
}

type Supervisor struct {
	policy SupervisorPolicy

	mu    sync.Mutex
	tasks map[string]*supervisorTask
}

type supervisorTask struct {
	cancel context.CancelFunc
	done   chan struct{}
}

func NewSupervisor(policy SupervisorPolicy) *Supervisor {
	return &Supervisor{
		policy: normalizeSupervisorPolicy(policy),
		tasks:  make(map[string]*supervisorTask),
	}
}

func (s *Supervisor) Start(name string, run func(ctx context.Context) error) error {
	if name == "" {
		return errors.New("task name is required")
	}
	if run == nil {
		return errors.New("task runner is required")
	}

	s.mu.Lock()
	if _, exists := s.tasks[name]; exists {
		s.mu.Unlock()
		return fmt.Errorf("task already running: %s", name)
	}
	ctx, cancel := context.WithCancel(context.Background())
	task := &supervisorTask{
		cancel: cancel,
		done:   make(chan struct{}),
	}
	s.tasks[name] = task
	s.mu.Unlock()

	go s.runTask(name, task, ctx, run)
	return nil
}

func (s *Supervisor) runTask(name string, task *supervisorTask, ctx context.Context, run func(ctx context.Context) error) {
	defer func() {
		s.mu.Lock()
		if current, ok := s.tasks[name]; ok && current == task {
			delete(s.tasks, name)
		}
		s.mu.Unlock()
		close(task.done)
	}()

	backoff := s.policy.InitialBackoff
	restarts := 0

	for {
		err := run(ctx)
		if ctx.Err() != nil {
			return
		}
		if err == nil {
			return
		}
		if s.policy.MaxRestarts > 0 && restarts >= s.policy.MaxRestarts {
			return
		}
		restarts++
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
		next := time.Duration(float64(backoff) * s.policy.BackoffFactor)
		if next > s.policy.MaxBackoff {
			next = s.policy.MaxBackoff
		}
		backoff = next
	}
}

func (s *Supervisor) Stop(name string) {
	s.mu.Lock()
	task, ok := s.tasks[name]
	s.mu.Unlock()
	if !ok {
		return
	}
	task.cancel()
	<-task.done
}

func (s *Supervisor) StopAll() {
	s.mu.Lock()
	tasks := make([]*supervisorTask, 0, len(s.tasks))
	for _, task := range s.tasks {
		tasks = append(tasks, task)
	}
	s.mu.Unlock()

	for _, task := range tasks {
		task.cancel()
	}
	for _, task := range tasks {
		<-task.done
	}
}

func (s *Supervisor) Tasks() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	names := make([]string, 0, len(s.tasks))
	for name := range s.tasks {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
