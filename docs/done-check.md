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
3. End-to-end benchmark runs complete with sqlite backend for:
   - core acceptance scapes: `xor`, `regression-mimic`, `cart-pole-lite`
   - parity smoke scapes: `flatland`, `gtsa`, `fx`, `epitopes`, `dtm`, `pole2-balancing`, `llvm-phase-ordering`
4. For each benchmark run, artifacts exist:
   - `config.json`
   - `fitness_history.json`
   - `top_genomes.json`
   - `lineage.json`
   - `generation_diagnostics.json`
   - `species_history.json`
   - `benchmark_summary.json`
5. `benchmark_summary.json` must have `"passed": true` for each run (both benchmark and export copies).
6. `export --latest` succeeds for each benchmark run.
7. Exported artifacts contain the same required files.

## Acceptance thresholds

- Core acceptance scapes:
  - XOR benchmark uses `--min-improvement 0.0001`.
  - Regression-mimic benchmark uses `--min-improvement 0.0` (plateau-safe guard).
  - Cart-pole-lite benchmark uses `--min-improvement 0.0001`.
- Parity smoke scapes use `--min-improvement -0.2` to catch severe regressions while keeping bounded smoke runtime.
- Parity smoke scapes run with `--w-substrate 0` to keep IO surface structure stable during smoke checks.

## Notes

- The script uses fixed seeds for reproducibility:
  - core: `xor=101`, `regression-mimic=202`, `cart-pole-lite=303`
  - smoke: `flatland=404`, `gtsa=405`, `fx=406`, `epitopes=407`, `dtm=408`, `pole2-balancing=409`, `llvm-phase-ordering=410`
- The script validates benchmark artifact generation, benchmark summary pass/fail semantics, and exportability.
