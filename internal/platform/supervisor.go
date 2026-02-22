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

type SupervisorRestartPolicy string

const (
	SupervisorRestartPermanent SupervisorRestartPolicy = "permanent"
	SupervisorRestartTransient SupervisorRestartPolicy = "transient"
	SupervisorRestartTemporary SupervisorRestartPolicy = "temporary"
)

type SupervisorChildSpec struct {
	Name    string
	Group   string
	Restart SupervisorRestartPolicy
}

type SupervisorChildStatus struct {
	Name            string                  `json:"name"`
	Group           string                  `json:"group,omitempty"`
	RestartPolicy   SupervisorRestartPolicy `json:"restart_policy"`
	RestartCount    int                     `json:"restart_count"`
	LastError       string                  `json:"last_error,omitempty"`
	PermanentFailed bool                    `json:"permanent_failed"`
}

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

	mu       sync.Mutex
	tasks    map[string]*supervisorTask
	finished map[string]SupervisorChildStatus
}

type supervisorTask struct {
	cancel context.CancelFunc
	done   chan struct{}
	spec   SupervisorChildSpec
	run    func(ctx context.Context) error

	restartCount    int
	lastErr         error
	permanentFailed bool
}

func NewSupervisor(policy SupervisorPolicy) *Supervisor {
	return NewSupervisorWithHooks(policy, SupervisorHooks{})
}

func NewSupervisorWithHooks(policy SupervisorPolicy, hooks SupervisorHooks) *Supervisor {
	return &Supervisor{
		policy:   normalizeSupervisorPolicy(policy),
		hooks:    hooks,
		tasks:    make(map[string]*supervisorTask),
		finished: make(map[string]SupervisorChildStatus),
	}
}

func (s *Supervisor) Start(name string, run func(ctx context.Context) error) error {
	spec := SupervisorChildSpec{
		Name:    name,
		Restart: SupervisorRestartPermanent,
	}
	return s.StartSpec(spec, run)
}

func (s *Supervisor) StartSpec(spec SupervisorChildSpec, run func(ctx context.Context) error) error {
	if spec.Name == "" {
		return errors.New("task name is required")
	}
	if run == nil {
		return errors.New("task runner is required")
	}
	if spec.Restart == "" {
		spec.Restart = SupervisorRestartPermanent
	}
	switch spec.Restart {
	case SupervisorRestartPermanent, SupervisorRestartTransient, SupervisorRestartTemporary:
	default:
		spec.Restart = SupervisorRestartPermanent
	}

	s.mu.Lock()
	if _, exists := s.tasks[spec.Name]; exists {
		s.mu.Unlock()
		return fmt.Errorf("task already running: %s", spec.Name)
	}
	delete(s.finished, spec.Name)
	ctx, cancel := context.WithCancel(context.Background())
	task := &supervisorTask{
		cancel: cancel,
		done:   make(chan struct{}),
		spec:   spec,
		run:    run,
	}
	s.tasks[spec.Name] = task
	s.mu.Unlock()

	go s.runTask(spec.Name, task, ctx, run)
	return nil
}

func (s *Supervisor) runTask(name string, task *supervisorTask, ctx context.Context, run func(ctx context.Context) error) {
	defer func() {
		s.mu.Lock()
		if current, ok := s.tasks[name]; ok && current == task {
			if shouldRetainFinishedStatus(task) {
				s.finished[name] = SupervisorChildStatus{
					Name:            task.spec.Name,
					Group:           task.spec.Group,
					RestartPolicy:   task.spec.Restart,
					RestartCount:    task.restartCount,
					LastError:       errString(task.lastErr),
					PermanentFailed: task.permanentFailed,
				}
			}
			delete(s.tasks, name)
		}
		s.mu.Unlock()
		close(task.done)
	}()

	backoff := s.policy.InitialBackoff

	for {
		err := run(ctx)
		if ctx.Err() != nil {
			return
		}
		restart := shouldRestart(specRestartPolicy(task), err)
		if !restart {
			return
		}
		s.mu.Lock()
		task.lastErr = err
		restarts := task.restartCount
		s.mu.Unlock()
		if s.policy.MaxRestarts > 0 && restarts >= s.policy.MaxRestarts {
			s.mu.Lock()
			task.permanentFailed = true
			task.restartCount = restarts
			s.mu.Unlock()
			if s.hooks.OnTaskPermanentFailure != nil {
				go s.hooks.OnTaskPermanentFailure(name, err, restarts)
			}
			if s.policy.Strategy == SupervisorStrategyOneForAll {
				s.stopAllExcept(name)
			}
			return
		}
		restarts++
		s.mu.Lock()
		task.restartCount = restarts
		s.mu.Unlock()
		if s.policy.Strategy == SupervisorStrategyOneForAll {
			s.restartSiblingsOneForAll(name, err)
		}
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

type oneForAllSiblingRestart struct {
	name         string
	previousTask *supervisorTask
	spec         SupervisorChildSpec
	run          func(ctx context.Context) error
	restarts     int
}

func (s *Supervisor) restartSiblingsOneForAll(excludedName string, triggeringErr error) {
	s.mu.Lock()
	restarts := make([]oneForAllSiblingRestart, 0, len(s.tasks))
	for name, task := range s.tasks {
		if name == excludedName {
			continue
		}
		restarts = append(restarts, oneForAllSiblingRestart{
			name:         name,
			previousTask: task,
			spec:         task.spec,
			run:          task.run,
			restarts:     task.restartCount,
		})
		task.cancel()
	}
	s.mu.Unlock()

	for _, sibling := range restarts {
		<-sibling.previousTask.done
	}

	restartErr := triggeringErr
	if restartErr == nil {
		restartErr = errors.New("one_for_all restart")
	}

	for _, sibling := range restarts {
		if !shouldRestart(sibling.spec.Restart, restartErr) {
			continue
		}
		ctx, cancel := context.WithCancel(context.Background())
		nextTask := &supervisorTask{
			cancel:       cancel,
			done:         make(chan struct{}),
			spec:         sibling.spec,
			run:          sibling.run,
			restartCount: sibling.restarts + 1,
			lastErr:      restartErr,
		}
		s.mu.Lock()
		current, exists := s.tasks[sibling.name]
		if exists && current != sibling.previousTask {
			s.mu.Unlock()
			cancel()
			continue
		}
		s.tasks[sibling.name] = nextTask
		s.mu.Unlock()
		if s.hooks.OnTaskRestart != nil {
			s.hooks.OnTaskRestart(sibling.name, restartErr, nextTask.restartCount)
		}
		go s.runTask(sibling.name, nextTask, ctx, sibling.run)
	}
}

func specRestartPolicy(task *supervisorTask) SupervisorRestartPolicy {
	if task == nil {
		return SupervisorRestartPermanent
	}
	if task.spec.Restart == "" {
		return SupervisorRestartPermanent
	}
	return task.spec.Restart
}

func shouldRestart(policy SupervisorRestartPolicy, err error) bool {
	switch policy {
	case SupervisorRestartPermanent:
		return true
	case SupervisorRestartTransient:
		return err != nil
	case SupervisorRestartTemporary:
		return false
	default:
		return true
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
	delete(s.finished, name)
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
	s.finished = make(map[string]SupervisorChildStatus)
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

func (s *Supervisor) Children() []SupervisorChildStatus {
	s.mu.Lock()
	defer s.mu.Unlock()

	names := make([]string, 0, len(s.tasks)+len(s.finished))
	for name := range s.tasks {
		names = append(names, name)
	}
	for name := range s.finished {
		if _, active := s.tasks[name]; active {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)

	out := make([]SupervisorChildStatus, 0, len(names))
	for _, name := range names {
		if task, ok := s.tasks[name]; ok {
			out = append(out, SupervisorChildStatus{
				Name:            task.spec.Name,
				Group:           task.spec.Group,
				RestartPolicy:   task.spec.Restart,
				RestartCount:    task.restartCount,
				LastError:       errString(task.lastErr),
				PermanentFailed: task.permanentFailed,
			})
			continue
		}
		if finished, ok := s.finished[name]; ok {
			out = append(out, finished)
		}
	}
	return out
}

func shouldRetainFinishedStatus(task *supervisorTask) bool {
	if task == nil {
		return false
	}
	return task.permanentFailed || task.restartCount > 0 || task.lastErr != nil
}
