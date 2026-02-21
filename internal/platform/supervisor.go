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
	Strategy       SupervisorStrategy
}

type SupervisorStrategy string

const (
	SupervisorStrategyOneForOne SupervisorStrategy = "one_for_one"
	SupervisorStrategyOneForAll SupervisorStrategy = "one_for_all"
)

type SupervisorHooks struct {
	OnTaskRestart          func(name string, err error, restartCount int)
	OnTaskPermanentFailure func(name string, err error, restartCount int)
}

func defaultSupervisorPolicy() SupervisorPolicy {
	return SupervisorPolicy{
		InitialBackoff: 10 * time.Millisecond,
		MaxBackoff:     200 * time.Millisecond,
		BackoffFactor:  2.0,
		MaxRestarts:    0,
		Strategy:       SupervisorStrategyOneForOne,
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
	if policy.Strategy == "" {
		policy.Strategy = def.Strategy
	}
	switch policy.Strategy {
	case SupervisorStrategyOneForOne, SupervisorStrategyOneForAll:
	default:
		policy.Strategy = def.Strategy
	}
	return policy
}

type Supervisor struct {
	policy SupervisorPolicy
	hooks  SupervisorHooks

	mu    sync.Mutex
	tasks map[string]*supervisorTask
}

type supervisorTask struct {
	cancel context.CancelFunc
	done   chan struct{}
}

func NewSupervisor(policy SupervisorPolicy) *Supervisor {
	return NewSupervisorWithHooks(policy, SupervisorHooks{})
}

func NewSupervisorWithHooks(policy SupervisorPolicy, hooks SupervisorHooks) *Supervisor {
	return &Supervisor{
		policy: normalizeSupervisorPolicy(policy),
		hooks:  hooks,
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
			if s.hooks.OnTaskPermanentFailure != nil {
				go s.hooks.OnTaskPermanentFailure(name, err, restarts)
			}
			if s.policy.Strategy == SupervisorStrategyOneForAll {
				s.stopAllExcept(name)
			}
			return
		}
		restarts++
		if s.hooks.OnTaskRestart != nil {
			s.hooks.OnTaskRestart(name, err, restarts)
		}
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

func (s *Supervisor) stopAllExcept(excludedName string) {
	s.mu.Lock()
	entries := make([]*supervisorTask, 0, len(s.tasks))
	for name, task := range s.tasks {
		if name == excludedName {
			continue
		}
		entries = append(entries, task)
	}
	s.mu.Unlock()

	for _, task := range entries {
		task.cancel()
	}
	for _, task := range entries {
		<-task.done
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
