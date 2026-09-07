package scape

import (
	"context"
	"fmt"
	"math"
	"testing"
)

type exampleTargetNavScape struct {
	maxSteps int
}

func (s exampleTargetNavScape) Name() string {
	return "example-target-nav"
}

func (s exampleTargetNavScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	maxSteps := s.maxSteps
	if maxSteps <= 0 {
		maxSteps = 8
	}

	const (
		target   = 1.0
		stepSize = 0.25
	)

	position := 0.0
	steps := 0
	reachedGoal := false
	for steps < maxSteps {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		distanceToTarget := target - position
		input := []float64{
			distanceToTarget,
			float64(steps) / float64(maxSteps),
		}
		output, err := runner.RunStep(ctx, input)
		if err != nil {
			return 0, nil, err
		}
		if len(output) != 1 {
			return 0, nil, fmt.Errorf("example-target-nav requires one output, got %d", len(output))
		}

		move := exampleClamp(output[0], -1, 1)
		position += move * stepSize
		steps++

		if math.Abs(target-position) <= 0.05 {
			reachedGoal = true
			break
		}
	}

	distance := math.Abs(target - position)
	progress := exampleClamp(1-distance, 0, 1)
	timeBonus := exampleClamp(1-float64(steps)/float64(maxSteps), 0, 1)
	fitness := 0.85*progress + 0.15*timeBonus
	if reachedGoal {
		fitness += 0.25
	}
	fitness = exampleClamp(fitness, 0, 1.25)

	trace := Trace{
		"steps":        steps,
		"position":     position,
		"distance":     distance,
		"reached_goal": reachedGoal,
		"fitness":      fitness,
	}
	return Fitness(fitness), trace, nil
}

type exampleNonStepAgent struct {
	id string
}

func (a exampleNonStepAgent) ID() string {
	return a.id
}

func TestCustomScapeExampleRewardsGoalProgress(t *testing.T) {
	sc := exampleTargetNavScape{maxSteps: 8}
	forager := scriptedStepAgent{
		id: "forager",
		fn: func([]float64) []float64 {
			return []float64{1}
		},
	}
	stuck := scriptedStepAgent{
		id: "stuck",
		fn: func([]float64) []float64 {
			return []float64{0}
		},
	}

	foragerFitness, foragerTrace, err := sc.Evaluate(context.Background(), forager)
	if err != nil {
		t.Fatalf("forager evaluate: %v", err)
	}
	stuckFitness, _, err := sc.Evaluate(context.Background(), stuck)
	if err != nil {
		t.Fatalf("stuck evaluate: %v", err)
	}

	if foragerFitness <= stuckFitness {
		t.Fatalf("expected target progress to score higher: forager=%f stuck=%f trace=%+v", foragerFitness, stuckFitness, foragerTrace)
	}
	if reached, _ := foragerTrace["reached_goal"].(bool); !reached {
		t.Fatalf("expected forager to reach goal, trace=%+v", foragerTrace)
	}
	if distance, ok := foragerTrace["distance"].(float64); !ok || distance > 0.05 {
		t.Fatalf("expected terminal distance trace, got %+v", foragerTrace)
	}
}

func TestCustomScapeExampleRequiresStepAgent(t *testing.T) {
	sc := exampleTargetNavScape{}
	_, _, err := sc.Evaluate(context.Background(), exampleNonStepAgent{id: "metadata-only"})
	if err == nil {
		t.Fatal("expected non-step agent to fail")
	}
}

func TestCustomScapeExampleValidatesOutputWidth(t *testing.T) {
	sc := exampleTargetNavScape{}
	badAgent := scriptedStepAgent{
		id: "bad-output",
		fn: func([]float64) []float64 {
			return []float64{1, 0}
		},
	}

	_, _, err := sc.Evaluate(context.Background(), badAgent)
	if err == nil {
		t.Fatal("expected malformed output to fail")
	}
}

func exampleClamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
