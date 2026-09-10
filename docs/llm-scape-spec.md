# LLM Scape Spec

This document defines the first real LLM scape target.

The goal is to keep the first version small enough to train, test, replay, and compare.

## Scope

Build one scape:

- name: `comm-grid-mentor`,
- base environment: `comm-grid`,
- learner: evolved Go agent,
- LLM role: mentor actor,
- training target: learner policy, not the LLM,
- provider mode: fixture by default, OpenAI-compatible HTTP as an opt-in path,
- streaming: not supported in this version.

## Actor Roles

Learner agent:

- receives local grid state,
- receives the latest mentor message,
- chooses one bounded action per turn,
- earns numeric fitness from task progress.

LLM mentor:

- receives a compact text state,
- sends one short hint per learner turn,
- does not move in the grid,
- does not receive fitness directly,
- is not mutated by the genetic algorithm.

Scape:

- owns the grid,
- applies bounded actions,
- stores messages,
- computes fitness,
- records replayable trace data.

## Action Schema

The learner action stays bounded:

```txt
move_north
move_south
move_east
move_west
pickup
drop
wait
```

The mentor response is text, but the scape limits it:

- maximum message length,
- one mentor message per learner turn,
- invalid or late provider output becomes an empty hint,
- token use is charged as a fitness cost.

## Fitness Function

Use deterministic numeric scoring:

- `+100` when the learner carries the key to the goal,
- `+10` when the learner picks up the key,
- `-1` per learner step,
- `-5` per invalid learner action,
- `-0.01` per mentor token,
- `-2` when a mentor call fails or times out.

The evolution loop ranks genomes by final numeric fitness.

The LLM can influence behavior through hints, but selection still depends on the learner result.

## Episode Rules

- Fixed grid size for the first version: `3x3`.
- Fixed key position: `(1,0)`.
- Fixed goal position: `(2,0)`.
- Fixed learner start: `(0,0)`.
- Maximum learner turns: `8`.
- One mentor hint before each learner action.
- Provider timeout: bounded by config.
- Replay uses stored mentor outputs, not new provider calls.

## Minimal Implementation Needs

Implement only these additions:

- a scape wrapper that calls an `llm.Provider` for mentor hints,
- a learner observation that includes the latest hint,
- artifact fields for mentor request, response, token count, and failure kind,
- replay support that reuses stored mentor outputs,
- one integration test with the fixture provider,
- one CLI example that runs the fixture path.

Do not add more suite, CSV, dashboard, or report features for this spec.

## Current Slice

Implemented:

- `CommGridMentorScape` as a thin wrapper around `comm-grid`,
- fixture-backed mentor calls through `llm.Provider`,
- numeric learner observation extension for the latest mentor hint,
- raw mentor request, response, hint, token count, and failure data in trace output,
- bounded provider failure handling with empty hints and a final fitness penalty,
- fixture-provider tests for completion, failure handling, and hint encoding,
- `protogonosctl comm-grid-mentor` as a fixture-only console runner with text and JSON output,
- file-backed `comm_grid_mentor.json` artifacts for mentor runs,
- replay from stored mentor artifacts with final trace matching,
- no-mentor baseline comparison through `protogonosctl comm-grid-mentor --compare-baseline`,
- a focused `Polis.RunEvolution` test that compares fixture-mentor and no-hint baseline scapes end to end,
- fixture-backed `protogonosctl run --scape comm-grid-mentor` support through the normal scape registry,
- configurable normal-run fixture plans through `--comm-grid-mentor-plan solve|silent`,
- normal-run mentor baseline comparison through `--comm-grid-mentor-compare-baseline`,
- stored plan display in `protogonosctl runs` text and JSON output,
- history filtering with `protogonosctl runs --scape comm-grid-mentor --comm-grid-mentor-plan solve|silent`,
- benchmark artifacts through `protogonosctl benchmark --scape comm-grid-mentor --comm-grid-mentor-plan solve|silent`,
- live OpenAI-compatible mentor population runs through `protogonosctl run --scape comm-grid-mentor`.

Not implemented yet:

- streaming mentor calls.

## CLI Example

Run the fixture mentor path:

```bash
protogonosctl comm-grid-mentor --plan solve
```

Print JSON for trace inspection:

```bash
protogonosctl comm-grid-mentor --plan solve --json
```

Replay a stored mentor artifact:

```bash
protogonosctl comm-grid-mentor --replay-run-id comm-grid-mentor-fixture-001
```

Run fixture-backed mentor evolution through the normal CLI path:

```bash
protogonosctl run --scape comm-grid-mentor --pop 4 --gens 1 --seed 91
protogonosctl run --scape comm-grid-mentor --comm-grid-mentor-plan silent --pop 4 --gens 1 --seed 91
protogonosctl run --scape comm-grid-mentor --comm-grid-mentor-compare-baseline --pop 4 --gens 1 --seed 91
protogonosctl runs --json
protogonosctl runs --scape comm-grid-mentor --comm-grid-mentor-plan solve
```

Write normal benchmark artifacts for a mentor or no-hint run:

```bash
protogonosctl benchmark --scape comm-grid-mentor --comm-grid-mentor-plan solve --pop 4 --gens 1 --seed 91 --min-improvement 0
protogonosctl benchmark --scape comm-grid-mentor --comm-grid-mentor-plan silent --pop 4 --gens 1 --seed 91 --min-improvement 0
```

Run the mentor scape with a live OpenAI-compatible endpoint:

```bash
protogonosctl run \
  --scape comm-grid-mentor \
  --comm-grid-mentor-provider openai-compatible \
  --comm-grid-mentor-base-url http://192.168.1.50:1234/v1 \
  --comm-grid-mentor-model local-model \
  --comm-grid-mentor-api-key-env PROTOGONOS_LLM_API_KEY \
  --comm-grid-mentor-max-tokens 8 \
  --pop 4 \
  --gens 1 \
  --seed 91
```

Compare the fixture mentor with the no-hint baseline:

```bash
protogonosctl comm-grid-mentor --plan solve --compare-baseline
```

## Training Question

This scape tests one question:

Can an evolved learner make better bounded choices when an LLM mentor supplies short language hints?

Useful comparisons:

- learner with no mentor,
- learner with fixture mentor,
- learner with OpenAI-compatible mentor,
- same seed and population settings across all runs.

## Non-Goals

- training the LLM,
- free-form learner actions,
- multi-agent negotiation,
- LLM-as-judge scoring,
- streaming responses,
- tool calls,
- long-term memory,
- provider-specific prompt tuning.

## Exit Criteria

The first version is complete when:

- fixture mentor runs are deterministic,
- replay matches the original trace,
- the no-mentor and mentor paths can be compared with the same seed,
- artifacts include all provider inputs and outputs,
- at least one bounded evolutionary run completes end-to-end.
