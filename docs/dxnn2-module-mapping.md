# DXNN2 -> Protogonos Module Mapping (Phase 1)

This document is the Phase 1 inventory artifact described in `AGENTS.md`.
It maps the Erlang DXNN2 conceptual modules to the Go rewrite packages and tracks current implementation status.

## Scope and Source of Truth

The authoritative responsibilities come from `AGENTS.md` (derived from upstream DXNN2 README):

- `polis`: top-level platform lifecycle and persistent store ownership
- `population_monitor`: generation loop (spawn/evaluate/select/replicate/mutate)
- `cortex`: synchronize sensors, neurons, actuators
- `exoself`: bounded synaptic weight tuning (memetic loop)
- `scapes`: environment/problem evaluation
- `morphology`: sensor/actuator compatibility with scapes
- `records.hrl` extensibility intent: add IO modules, operators, activations via shared schemas

## Module Mapping

| DXNN2 concept/module | Responsibility in Erlang | Go package | Current status |
|---|---|---|---|
| `polis` | Platform lifecycle + mnesia integration | `internal/platform` + `internal/storage` | `implemented` |
| `population_monitor` | Population evolution lifecycle | `internal/evo` | `implemented` |
| `cortex` | Per-agent execution coordination | `internal/agent` + `internal/nn` + `internal/io` | `implemented` |
| `exoself` | Weight tuning loop | `internal/tuning` | `implemented` |
| `scapes` | Agent evaluation environments | `internal/scape` | `implemented (xor, regression-mimic)` |
| `morphology` | Sensor/actuator compatibility rules | `internal/morphology` | `implemented (xor + regression-mimic baselines)` |
| records/schema definitions | Shared extensible record contracts | `internal/model` + `internal/storage/codec.go` | `implemented baseline` |
| benchmark output flow | Run artifact outputs | `cmd/protogonosctl` + `internal/stats` | `implemented` |
| stable downstream API | Public surface | `pkg/protogonos` | `implemented baseline` |

## Data Schema Inventory (Phase 1)

Implemented core struct set in `internal/model/types.go`:

- `VersionedRecord`
- `Genome`
- `Neuron`
- `Synapse`
- `Agent`
- `Population`
- `ScapeSummary`

Phase 1 schema rules currently enforced:

- Every persisted record shape embeds version metadata (`schema_version`, `codec_version`)
- Initial constants in codec layer:
  - `CurrentSchemaVersion = 1`
  - `CurrentCodecVersion = 1`

Current codec coverage:

- Implemented for `Genome` in `internal/storage/codec.go`
- Fixture-backed decode and round-trip tests in `internal/storage/codec_test.go`

## Runtime Contract Inventory

Current interface definitions (scaffold-level):

- `storage.Store` in `internal/storage/store.go`
- `scape.Scape` in `internal/scape/scape.go`
- `morphology.Morphology` in `internal/morphology/morphology.go`
- `io.Sensor` / `io.Actuator` in `internal/io/interfaces.go`
- `evo.Operator` in `internal/evo/operator.go`
- `tuning.Tuner` in `internal/tuning/tuner.go`

## Current Phase 1 Coverage

Completed:

- Target package skeleton created (`cmd`, `internal`, `pkg`, `testdata`)
- Noop in-memory persistence backend (`MemoryStore`)
- Noop `Polis` init + scape registration path
- Golden fixture for minimal genome:
  - `testdata/fixtures/minimal_genome_v1.json`
- Tests proving:
  - codec decode of fixture
  - genome codec round trip
  - noop polis init/register lifecycle

Remaining hardening:

- Add a formal API compatibility policy and changelog discipline for `pkg/protogonos`.
- Add module-by-module checklist evidence against upstream DXNN2 `src/` for residual gaps.
- Expand golden fixture coverage for additional operator and morphology combinations.

## Mapping Principles (to preserve parity)

- Preserve actor-style lifecycle semantics with goroutines + context cancellation
- Keep storage behind `storage.Store` so mnesia replacement remains swappable
- Keep extension points registry-driven to mirror records.hrl extensibility intent
- Keep `pkg/protogonos` minimal/stable while implementation evolves in `internal`

## Immediate Follow-ups

1. Add a full upstream `src/` module-to-module audit table as direct parity evidence.
2. Add fixture tests for additional deterministic mutation and IO combinations.
