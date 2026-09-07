# Custom Scape Developer Guide

This guide explains how to add a new scape to `protogonos`.

A scape is an environment or task. It gives an agent inputs, reads the agent outputs, and returns a numeric fitness score.

## Required Idea

A genetic TWEANN run needs selection pressure.

In this project, selection pressure comes from `scape.Fitness`. The population monitor ranks genomes by fitness, keeps stronger candidates, and mutates offspring.

You can use novelty, diversity, or curriculum scoring, but the current evolution loop still needs a numeric score. If you want a system with no fitness function, you must change the evolution loop and selection model first.

## Main Files

Add or update these areas:

- `internal/scape`: the environment and scoring logic.
- `internal/io`: sensor and actuator IDs, aliases, and optional process adapters.
- `internal/morphology`: sensor/actuator compatibility for the scape.
- `internal/genotype`: seed construction support if the default seed builder needs scape-specific widths.
- `pkg/protogonos`: public API request/config fields if the scape needs user options.
- `cmd/protogonosctl`: CLI flags and default scape registration.
- `docs`: user and parity documentation.

## Minimal Scape Contract

Every scape implements this interface:

```go
type Scape interface {
	Name() string
	Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error)
}
```

Most neural agents implement `StepAgent`:

```go
type StepAgent interface {
	Agent
	RunStep(ctx context.Context, input []float64) ([]float64, error)
}
```

If the scape has training, validation, test, or benchmark windows, implement `ModeAwareScape`:

```go
type ModeAwareScape interface {
	Scape
	EvaluateMode(ctx context.Context, agent Agent, mode string) (Fitness, Trace, error)
}
```

## Step 1: Define The Goal

Write the goal first.

Good examples:
- reach a target position,
- predict the next value,
- classify a sequence,
- collect resources,
- survive while avoiding hazards,
- minimize runtime or control cost.

Bad goals are vague. For example, "behave intelligently" is not a useful scape goal. It does not tell the fitness function what to reward.

## Step 2: Define Inputs

Inputs are sensor values sent to the neural network.

Keep these rules:
- Use a fixed vector order.
- Normalize values when possible.
- Keep the vector width stable.
- Document every channel.
- Add trace fields that expose important sensor values.

Example input vector:

```go
input := []float64{
	targetDistance,
	agentEnergy,
	hazardProximity,
	progress,
}
```

## Step 3: Define Outputs

Outputs are actuator values returned by the neural network.

Keep these rules:
- Clamp or normalize unsafe values.
- Define what happens when output is missing or malformed.
- Keep the output width stable.
- Record decoded actions in the trace.

Example output decode:

```go
move := clamp(output[0], -1, 1)
turn := clamp(output[1], -1, 1)
```

## Step 4: Define Fitness

Fitness must match the goal.

Good fitness functions usually combine:
- progress reward,
- terminal success bonus,
- bounded failure penalty,
- resource or time cost,
- simple normalization.

Example:

```go
fitness := 0.0
fitness += 0.60 * progressToGoal
fitness += 0.20 * survivalRatio
fitness += 0.20 * energyRatio
if reachedGoal {
	fitness += 0.25
}
fitness = clamp(fitness, 0, 1.25)
```

Avoid fitness functions where most agents get the same score. Evolution needs score differences to rank genomes.

## Step 5: Implement The Scape

Create a file in `internal/scape`, for example `target_nav.go`.

```go
package scape

import (
	"context"
	"fmt"
)

type TargetNavScape struct{}

func (TargetNavScape) Name() string {
	return "target-nav"
}

func (TargetNavScape) Evaluate(ctx context.Context, agent Agent) (Fitness, Trace, error) {
	runner, ok := agent.(StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	state := newTargetNavEpisode()
	for !state.done {
		if err := ctx.Err(); err != nil {
			return 0, nil, err
		}

		input := state.sense()
		output, err := runner.RunStep(ctx, input)
		if err != nil {
			return 0, nil, err
		}
		state.apply(output)
	}

	fitness := state.fitness()
	trace := Trace{
		"steps":        state.steps,
		"distance":     state.distance,
		"reached_goal": state.reachedGoal,
		"fitness":      float64(fitness),
	}
	return fitness, trace, nil
}
```

Add tests in `internal/scape/target_nav_test.go`.

Test at least:
- valid fitness is returned,
- malformed output fails or uses a documented fallback,
- terminal conditions work,
- trace fields explain the score.

## Step 6: Add IO IDs

Add sensor and actuator names in `internal/io`.

Use `internal/io/scalar_components.go` for simple scalar components. Add aliases if you need reference-style names.

Example shape:

```go
const TargetNavDistanceSensorName = "target_nav_distance"
const TargetNavMoveActuatorName = "target_nav_move"
```

Add registry tests so the names stay available.

## Step 7: Add Morphology

Add a morphology file in `internal/morphology`.

```go
type TargetNavMorphology struct{}

func (TargetNavMorphology) Name() string {
	return "target-nav-v1"
}

func (TargetNavMorphology) Sensors() []string {
	return []string{
		protoio.TargetNavDistanceSensorName,
	}
}

func (TargetNavMorphology) Actuators() []string {
	return []string{
		protoio.TargetNavMoveActuatorName,
	}
}

func (TargetNavMorphology) Compatible(scape string) bool {
	return scape == "target-nav"
}
```

Then wire it into the morphology constructor and compatibility checks in `internal/morphology/morphology.go`.

Add tests for:
- construction,
- compatibility,
- sensor list,
- actuator list.

## Step 8: Add Seed Support

Seed support makes the first genomes match the scape IO width.

Check:
- `internal/genotype/lifecycle.go`
- `internal/genotype/construct.go`
- `internal/genotype/agent_construct.go`

If your scape uses normal scalar sensors and actuators, existing seed helpers may be enough. If it uses vector outputs, scanner surfaces, or special IO metadata, add explicit tests for seed input/output width.

## Step 9: Register In CLI And API

The CLI registers built-in scapes in `cmd/protogonosctl/main.go`.

Add your scape to `registerDefaultScapes`:

```go
if err := p.RegisterScape(scape.TargetNavScape{}); err != nil {
	return err
}
```

If the scape needs user options, add fields to `pkg/protogonos.RunRequest`, config loading in `cmd/protogonosctl/config.go`, and CLI flags in `cmd/protogonosctl/main.go`.

## Step 10: Run It

Initialize storage:

```bash
protogonosctl init --store sqlite --db-path ./protogonos.db
```

Run evolution:

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape target-nav \
  --pop 50 \
  --gens 100 \
  --seed 1
```

Run a benchmark:

```bash
protogonosctl benchmark \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape target-nav \
  --pop 50 \
  --gens 100 \
  --seed 1 \
  --min-improvement 0.01
```

Export artifacts:

```bash
protogonosctl export --store sqlite --db-path ./protogonos.db --latest
```

## Config File Example

```json
{
  "scape": "target-nav",
  "population": 50,
  "generations": 100,
  "seed": 1,
  "workers": 4,
  "fitness_goal": 1.0,
  "evaluations_limit": 10000
}
```

Run with config:

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --config ./target-nav.json
```

## Test Checklist

Before treating a custom scape as stable:

- Run `go test ./...`.
- Run `go test -tags sqlite ./...`.
- Add at least one direct scape unit test.
- Add at least one morphology compatibility test.
- Add at least one seed construction width test.
- Add at least one CLI or API integration test if the scape has user-facing options.
- Run a bounded benchmark and inspect `benchmark_summary.json`.
- Export the run and confirm required artifacts exist.

## Design Notes

Prefer simple scoring first. Add complexity only when basic agents can receive different scores.

A useful first version has:
- one clear objective,
- a small input vector,
- one or two outputs,
- deterministic seed behavior,
- visible trace diagnostics,
- a bounded episode length.

Do not hide task state that is required to learn the task. If the agent needs target direction, distance, or recent reward to solve the problem, expose it as a sensor unless the goal is specifically partial observability.

Do not use exact reference internals as a requirement unless they matter to behavior. The Go rewrite keeps DXNN2 concepts, but it replaces OTP, mnesia, and ETS mechanics with Go lifecycle, storage, and artifacts.
