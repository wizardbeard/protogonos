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
