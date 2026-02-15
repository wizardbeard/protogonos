package tuning

import (
	"context"

	"protogonos/internal/model"
)

type FitnessFn func(ctx context.Context, genome model.Genome) (float64, error)

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
