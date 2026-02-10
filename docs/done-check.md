# Done Check

This document defines the reproducible acceptance check for the AGENTS.md done criteria.

## Preconditions

- Go toolchain installed.
- SQLite driver dependency available.
- Run from repository root.

## One-command verification

```bash
./scripts/done_check.sh
```

## What this validates

1. `go test ./...` passes.
2. `go test -tags sqlite ./...` passes.
3. End-to-end benchmark run on XOR completes with sqlite backend.
4. End-to-end benchmark run on regression-mimic completes with sqlite backend.
5. For each benchmark run, artifacts exist:
   - `config.json`
   - `fitness_history.json`
   - `top_genomes.json`
   - `lineage.json`
   - `benchmark_summary.json`
6. `export --latest` succeeds for each benchmark run.
7. Exported artifacts contain the same required files.

## Acceptance thresholds

- XOR benchmark uses `--min-improvement 0.0001`.
- Regression-mimic benchmark uses `--min-improvement 0.0001`.
- Benchmark pass/fail is written in `benchmark_summary.json`.

## Notes

- The script uses fixed seeds for reproducibility:
  - XOR seed: `101`
  - Regression-mimic seed: `202`
- The script validates both benchmark artifact generation and exportability.
