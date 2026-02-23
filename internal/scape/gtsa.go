package scape

import (
	"context"
	"fmt"
	"math"
	"strings"
)

type GTSAScape struct{}

func (GTSAScape) Name() string {
	return "gtsa"
}

func (GTSAScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	return GTSAScape{}.EvaluateMode(ctx, agent, "gt")
}

func (GTSAScape) EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	cfg, err := gtsaConfigForMode(mode)
	if err != nil {
		return 0, nil, err
	}

	t := cfg.startT
	last := gtsaSignal(t)
	mse := 0.0

	for i := 0; i < cfg.steps; i++ {
		input := []float64{last}
		out, err := runner.RunStep(ctx, input)
		if err != nil {
			return 0, nil, err
		}
		if len(out) != 1 {
			return 0, nil, fmt.Errorf("gtsa requires one output, got %d", len(out))
		}
		t += 1.0
		next := gtsaSignal(t)
		delta := out[0] - next
		mse += delta * delta
		last = next
	}

	mse /= float64(cfg.steps)
	fitness := 1.0 / (1.0 + mse)
	return Fitness(fitness), Trace{
		"mse":     mse,
		"mode":    cfg.mode,
		"steps":   cfg.steps,
		"start_t": cfg.startT,
	}, nil
}

type gtsaModeConfig struct {
	mode   string
	steps  int
	startT float64
}

func gtsaConfigForMode(mode string) (gtsaModeConfig, error) {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "gt":
		return gtsaModeConfig{mode: "gt", steps: 40, startT: 0}, nil
	case "validation":
		return gtsaModeConfig{mode: "validation", steps: 32, startT: 120}, nil
	case "test":
		return gtsaModeConfig{mode: "test", steps: 32, startT: 240}, nil
	default:
		return gtsaModeConfig{}, fmt.Errorf("unsupported gtsa mode: %s", mode)
	}
}

func gtsaSignal(t float64) float64 {
	return math.Sin(t*0.2) + 0.5*math.Sin(t*0.05)
}
