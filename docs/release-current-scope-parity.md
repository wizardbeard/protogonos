# Current-Scope DXNN2 Parity Release Note

Date: 2026-09-07

## Status

Current-scope DXNN2 parity is ready to declare.

The final pre-declaration gate passed from a clean working tree:

```bash
./scripts/done_check.sh
```

The gate was rerun after the Flatland/custom-scape documentation updates and the executable custom-scape fixture were added. It passed again on 2026-09-07.

## Gate Result

- Active reference modules in `.ref/src`: 33
- Documented audit rows: 33
- Implemented/current-scope rows: 31
- Partial rows: 0
- Missing rows: 0
- Explicitly out-of-scope rows: 2
- Out-of-scope modules: `dxnn2_app.erl`, `visor.erl`
- Default Go test suite: passed
- SQLite-tagged Go test suite: passed
- Core benchmark/export checks: passed
- Richer scape smoke benchmark/export checks: passed

## Latest Verification

Command:

```bash
./scripts/done_check.sh
```

Result:

- Parity docs summary: `active_ref_modules=33 implemented=31 partial=0 missing=0 out_of_scope=2`
- Default Go test suite: passed
- SQLite-tagged Go test suite: passed
- Benchmark/export checks: passed for `xor`, `regression-mimic`, and `cart-pole-lite`
- Richer scape benchmark/export checks: passed for `flatland`, `gtsa`, `fx`, `epitopes`, `dtm`, `pole2-balancing`, and `llvm-phase-ordering`
- Final gate result: `PASS`

## Included Scope

- Polis lifecycle over Go storage.
- Population evolution, selection, replication, mutation, species handling, lineage, and diagnostics.
- Genotype/schema materialization, record conversion, and run artifacts.
- Cortex, neuron, sensor, and actuator runtime execution.
- Exoself tuning and runtime weight backup/restore/perturb flows.
- Substrate CPP/CEP runtime behavior and process-style mailbox surfaces.
- Benchmark and data-extraction workflows.
- Current scape set: `xor`, `regression-mimic`, `cart-pole-lite`, `pole2-balancing`, `dtm`, `flatland`, `gtsa`, `fx`, `epitopes`, and `llvm-phase-ordering`.

## Explicit Exclusions

- `dxnn2_app.erl`: OTP application boot is replaced by CLI/API lifecycle.
- `visor.erl`: visualization/UI drawing remains out of scope.
- Exact Erlang process scheduling is not required for current-scope functional parity.
- Mnesia/ETS ownership mechanics are replaced by Go storage and versioned artifacts.
- GS canvas behavior is out of scope.
- Reference delegate-only scapes absent from `.ref/src` are not active requirements.
- `mnist` morphology clauses are outside the current AGENTS target scape set.
- Upstream TODO/stub helper surfaces are documented as non-gaps.

## Pre-Release Requirement

Before tagging or announcing parity, rerun:

```bash
./scripts/done_check.sh
```

The working tree should be clean before the run. Generated benchmark/export outputs should not be committed unless explicitly needed for a release artifact.
