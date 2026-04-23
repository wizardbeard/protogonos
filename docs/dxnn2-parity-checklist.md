# DXNN2 Parity Checklist

This checklist tracks module-level parity evidence against the DXNN2 responsibilities in `AGENTS.md`.

## Core responsibilities

- `polis` lifecycle + store ownership: `done`
  - Evidence: `internal/platform/polis.go`, `internal/storage/*`, CLI `init/start`.
- `population_monitor` generation loop: `done`
  - Evidence: `internal/evo/population_monitor.go` and tests.
- `cortex` orchestration (`sensor -> nn -> actuator`): `done`
  - Evidence: `internal/agent/cortex.go`, `internal/agent/cortex_test.go`.
- `exoself` bounded weight tuning: `done`
  - Evidence: `internal/tuning/exoself.go`, tuning integration tests.
- `scapes` evaluation environments: `done (baseline)`
  - Evidence: `internal/scape/xor.go`, `internal/scape/regression_mimic.go`.
- morphology compatibility checks: `done (baseline)`
  - Evidence: `internal/morphology/regression_mimic.go`, `internal/morphology/xor.go`.

## Extension points parity

- activation registry: `done`
  - Evidence: `internal/nn/registry.go`.
- mutation operator registry/policy: `done`
  - Evidence: `internal/evo/registry.go`, `internal/evo/policy.go`.
- sensor/actuator registries: `done`
  - Evidence: `internal/io/registry.go`, `internal/io/scalar_components.go`.

## Persistence parity

- memory store: `done`
  - Evidence: `internal/storage/memory.go`.
- sqlite store: `done`
  - Evidence: `internal/storage/sqlite.go`, `internal/storage/sqlite_test.go`.
- versioned codec coverage for core records: `done`
  - Evidence: `internal/storage/codec.go`, `internal/storage/codec_test.go`, `testdata/fixtures/minimal_*.json`, `testdata/fixtures/io_links_genome_v1.json`, `testdata/fixtures/vector_io_genome_v1.json`, `testdata/fixtures/substrate_io_genome_v1.json`.

## Artifact and workflow parity

- benchmark artifacts + export: `done`
  - Evidence: `internal/stats/artifacts.go`, CLI `benchmark/export`.
- reproducible acceptance script: `done`
  - Evidence: `scripts/done_check.sh`.

## Upstream coverage

- upstream `src/` module mapping: `done`
  - Evidence: `docs/dxnn2-src-module-audit.md` maps every audited reference module to a Go equivalent or explicit parity gap note.

## Remaining parity hardening

- Expand golden fixture coverage further only for mutation families outside the current IO and generic link-helper operator set.
