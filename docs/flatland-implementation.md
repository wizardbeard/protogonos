# Flatland Implementation

Flatland is the richest scape in the current Go rewrite. It can be used as a normal evolutionary evaluation environment or as a public world with process-style calls.

## What Flatland Tests

Flatland evaluates an agent that must survive and forage in a small world.

The agent receives sensor values, produces movement or command outputs, and receives fitness from the scape. Evolution then uses that fitness to rank genomes, keep better candidates, and mutate the next generation.

In practical terms, Flatland rewards agents that:
- stay alive longer,
- preserve energy,
- collect food,
- avoid poison,
- avoid walls,
- handle prey and predator signals,
- use scanner data effectively,
- avoid predator pressure,
- make useful public-world actions when those actuators are enabled.

## Run Flatland

Assume `protogonosctl` is available on `PATH`.

```bash
protogonosctl init --store sqlite --db-path ./protogonos.db
```

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 50 \
  --gens 100 \
  --seed 1
```

For the benchmark path:

```bash
protogonosctl benchmark \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 50 \
  --gens 100 \
  --seed 1 \
  --min-improvement -0.2
```

Export the latest run:

```bash
protogonosctl export --latest
```

## Useful Flatland Options

Flatland supports run-level overrides for scanner shape, layout selection, episode length, and benchmark trial count.

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 50 \
  --gens 100 \
  --seed 7 \
  --flatland-scanner-profile balanced5 \
  --flatland-max-age 300 \
  --flatland-forage-goal 12
```

Force a deterministic layout variant:

```bash
protogonosctl benchmark \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 40 \
  --gens 50 \
  --seed 12 \
  --flatland-layout-variants 4 \
  --flatland-force-layout-variant 2 \
  --flatland-benchmark-trials 5 \
  --min-improvement -0.2
```

Randomize among layout variants:

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 40 \
  --gens 50 \
  --seed 12 \
  --flatland-layout-randomize \
  --flatland-layout-variants 4
```

Scanner profiles currently used by the CLI are:
- `balanced5`
- `core3`
- `forward5`

Scanner spread and offset can also be overridden:

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 40 \
  --gens 50 \
  --seed 13 \
  --flatland-scanner-profile forward5 \
  --flatland-scanner-spread 0.25 \
  --flatland-scanner-offset 0.1
```

## Runtime Model

Flatland has two main execution paths.

The first path is normal scape evaluation. The population monitor creates or loads genomes, builds cortex instances, evaluates each agent in Flatland, ranks fitness, and mutates the next population.

The second path is the public-world process API. This mirrors the reference Erlang process style more closely. It supports:
- `start`
- `stop`
- `sync`
- `enter`
- `leave`
- `sense`
- `act`
- `tick`
- `update_agents`
- `get_all`

The Go type behind that public path is `FlatlandPublicProcess`.

## Sensors

The default Flatland sensor surface includes scalar state and scanner values.

Core scalar channels include:
- food distance,
- energy,
- prey signal,
- predator signal,
- poison signal,
- wall signal,
- food proximity,
- prey proximity,
- predator proximity,
- poison proximity,
- wall proximity,
- resource balance.

Scanner channels include distance, color, and energy bins. The scanner helpers are exposed as direct Go functions for reference-style tests and reuse:
- `FlatlandDistanceScanner`
- `FlatlandColorScanner`
- `FlatlandEnergyScaner`
- `FlatlandShortestDistance`
- `FlatlandShortestIntrLine`
- `FlatlandIntr`

The miss value for distance scanning is `-1`. Color and energy scanner outputs are normalized values designed for neural input.

## Actuators

The default Flatland morphology uses movement. The scanner morphology uses a two-wheel output surface.

Public command actuators are also available:
- `speak`
- `gestalt_output`
- `spear`
- `shoot`
- `create_offspring`

These command actuators exist to match the reference public scape protocol. They are not required for a basic Flatland run.

## Fitness

Flatland computes a shaped fitness for episode evaluation. The score combines:
- survival,
- normalized energy,
- food versus poison balance,
- accumulated reward,
- wall penalties,
- resource respawn activity,
- prey/predator interaction quality,
- predator pressure penalties,
- forage-goal completion bonus.

The current episode fitness is clamped to a bounded range.

Public-process actuator calls also expose a reference-style immediate feedback value. That value is intentionally small while the avatar is alive and zero after terminal paths. The richer shaped score remains available in trace diagnostics as `shaped_fitness`.

Useful trace fields include:
- `age`
- `max_age`
- `forage_goal`
- `energy`
- `energy_norm`
- `food_collected`
- `poison_hits`
- `prey_collected`
- `predator_hits`
- `wall_collisions`
- `reward_total`
- `terminal_reason`
- `shaped_fitness`
- `reference_fitness`

## Goals And Fitness Functions

For TWEANN systems using genetic algorithms, you normally need a fitness signal. Without one, selection has no objective basis for deciding which genomes should reproduce.

The reference DXNN2 implementation follows that model. Scapes produce fitness feedback. `population_monitor` ranks evaluated agents, applies optional fitness postprocessing, selects parents, replicates genomes, and mutates offspring. The reference also has `fitness_postprocessor:novelty_proportional/1`, but it is a placeholder returning `void`, not a complete novelty-search replacement for fitness-driven selection.

The current Go implementation follows the same pattern. Every `scape.Scape` returns:

```go
Evaluate(ctx, agent) (Fitness, Trace, error)
```

So the scape must still produce a numeric fitness. You can implement novelty, diversity, or curriculum behavior, but it should still be converted into a fitness or selection score unless the evolution loop is changed.

## Creating A Custom Scape

A custom scape needs four design decisions.

First, define the goal. Examples:
- reach a target,
- classify a sequence,
- maximize resource collection,
- minimize control cost,
- survive for a fixed episode,
- complete a task before timeout.

Second, define the sensor surface. This is the input vector the cortex receives. Keep it stable and documented. If the vector width changes, seed construction and morphology compatibility must also change.

Third, define the actuator surface. This is the output vector the cortex writes. Map neural outputs to concrete actions with explicit bounds and defaults.

Fourth, define the fitness function. It should reward the real goal directly enough that evolution gets useful selection pressure.

Bad fitness functions usually:
- reward side effects instead of task completion,
- make most agents receive the same score,
- only reward success at the final step when early gradients would help,
- hide penalties that make survival impossible,
- change meaning between runs without being recorded in artifacts.

Better fitness functions usually:
- combine a terminal goal bonus with step-level progress,
- include bounded penalties for unsafe or wasteful behavior,
- normalize inputs and scores,
- expose diagnostics in the trace,
- use fixed seeds or recorded seeds for repeatability.

## Minimal Go Shape

A scape implements `internal/scape.Scape`.

```go
type MyScape struct{}

func (MyScape) Name() string {
	return "my-scape"
}

func (MyScape) Evaluate(ctx context.Context, agent scape.Agent) (scape.Fitness, scape.Trace, error) {
	runner, ok := agent.(scape.StepAgent)
	if !ok {
		return 0, nil, fmt.Errorf("agent %s does not implement step runner", agent.ID())
	}

	input := []float64{0.0, 1.0}
	output, err := runner.RunStep(ctx, input)
	if err != nil {
		return 0, nil, err
	}

	fitness := scoreOutput(output)
	trace := scape.Trace{
		"goal":    "example",
		"fitness": float64(fitness),
	}
	return fitness, trace, nil
}
```

After that, register the scape with the platform/API path, add morphology sensor and actuator IDs, add seed-construction support if needed, and add tests that prove a hand-built or seeded agent can receive a meaningful fitness signal.

## Suggested Custom-Scape Checklist

- Name the scape and add alias normalization if it needs reference-style aliases.
- Define sensors and actuator IDs in the IO registry.
- Define morphology compatibility.
- Add seed construction support.
- Implement `Evaluate`.
- Add `EvaluateMode` if `gt`, `validation`, `test`, or `benchmark` should use different windows.
- Return trace fields that explain the score.
- Add unit tests for scoring and bounds.
- Add an integration test through `PopulationMonitor`.
- Add a bounded benchmark smoke command before treating it as production-ready.

## Common Flatland Experiments

Short smoke run:

```bash
protogonosctl benchmark \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 6 \
  --gens 3 \
  --seed 404 \
  --w-substrate 0.02 \
  --min-improvement -0.2
```

Longer evolution run:

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 100 \
  --gens 250 \
  --seed 1001 \
  --workers 4 \
  --flatland-max-age 300 \
  --flatland-forage-goal 12
```

Inspect generated artifacts:

```bash
protogonosctl export --latest
```

Run the full parity gate:

```bash
./scripts/done_check.sh
```
