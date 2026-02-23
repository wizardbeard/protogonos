package tuning

import (
	"context"

	"protogonos/internal/model"
)

type FitnessFn func(ctx context.Context, genome model.Genome) (float64, error)

// RuntimeEvaluateFn evaluates the currently materialized runtime agent in a
// mode-specific path (for example gt/validation/test). The returned trace is a
// generic payload that mirrors scape trace records.
type RuntimeEvaluateFn func(ctx context.Context, mode string) (fitness float64, trace map[string]any, goalReached bool, err error)

// RuntimeAgent exposes the runtime hooks exoself uses during in-place tuning.
type RuntimeAgent interface {
	SnapshotGenome() model.Genome
	ApplyGenome(genome model.Genome) error
	BackupWeights()
	RestoreWeights() error
	Reactivate() error
}

type RuntimeTuneResult struct {
	Genome  model.Genome
	Fitness float64
	Trace   map[string]any
	Report  TuneReport
}

type TuneReport struct {
	AttemptsPlanned      int  `json:"attempts_planned"`
	AttemptsExecuted     int  `json:"attempts_executed"`
	CandidateEvaluations int  `json:"candidate_evaluations"`
	AcceptedCandidates   int  `json:"accepted_candidates"`
	RejectedCandidates   int  `json:"rejected_candidates"`
	GoalReached          bool `json:"goal_reached"`
}

type Tuner interface {
	Name() string
	Tune(ctx context.Context, genome model.Genome, attempts int, fitness FitnessFn) (model.Genome, error)
}

type ReportingTuner interface {
	Tuner
	TuneWithReport(ctx context.Context, genome model.Genome, attempts int, fitness FitnessFn) (model.Genome, TuneReport, error)
}

type RuntimeTuner interface {
	Tuner
	TuneRuntime(ctx context.Context, runtime RuntimeAgent, attempts int, mode string, evaluate RuntimeEvaluateFn) (RuntimeTuneResult, error)
}

type RuntimeReportingTuner interface {
	RuntimeTuner
	TuneRuntimeWithReport(ctx context.Context, runtime RuntimeAgent, attempts int, mode string, evaluate RuntimeEvaluateFn) (RuntimeTuneResult, error)
}
