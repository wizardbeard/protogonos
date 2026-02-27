#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[done-check] Running default test suite"
go test ./...

echo "[done-check] Running sqlite-tagged test suite"
go test -tags sqlite ./...

run_benchmark_and_verify() {
  local scape="$1"
  local seed="$2"
  local min_improvement="$3"
  local pop_size="$4"
  local generations="$5"
  local workers="$6"
  local w_substrate="$7"

  echo "[done-check] Running benchmark command (sqlite backend) for scape=$scape pop=$pop_size gens=$generations seed=$seed min_improvement=$min_improvement w_substrate=$w_substrate"
  local run_output
  run_output="$(go run -tags sqlite ./cmd/protogonosctl benchmark \
    --store sqlite \
    --db-path ./protogonos.donecheck.db \
    --scape "$scape" \
    --pop "$pop_size" \
    --gens "$generations" \
    --seed "$seed" \
    --workers "$workers" \
    --w-substrate "$w_substrate" \
    --min-improvement "$min_improvement")"

  echo "$run_output"
  local run_id
  run_id="$(echo "$run_output" | sed -n 's/.*run_id=\([^ ]*\).*/\1/p' | head -n1)"
  if [[ -z "$run_id" ]]; then
    echo "[done-check] ERROR: could not extract run_id from benchmark output for scape=$scape" >&2
    exit 1
  fi

  local artifact_dir="benchmarks/$run_id"
  for file in config.json fitness_history.json top_genomes.json lineage.json generation_diagnostics.json species_history.json benchmark_summary.json; do
    if [[ ! -f "$artifact_dir/$file" ]]; then
      echo "[done-check] ERROR: missing artifact $artifact_dir/$file" >&2
      exit 1
    fi
  done
  if ! grep -q '"speciation_threshold"' "$artifact_dir/generation_diagnostics.json"; then
    echo "[done-check] ERROR: missing speciation diagnostics fields in $artifact_dir/generation_diagnostics.json" >&2
    exit 1
  fi
  if ! grep -q '"species"' "$artifact_dir/species_history.json"; then
    echo "[done-check] ERROR: missing species history content in $artifact_dir/species_history.json" >&2
    exit 1
  fi
  if ! grep -Eq '"passed"[[:space:]]*:[[:space:]]*true' "$artifact_dir/benchmark_summary.json"; then
    echo "[done-check] ERROR: benchmark_summary did not pass for scape=$scape ($artifact_dir/benchmark_summary.json)" >&2
    exit 1
  fi

  echo "[done-check] Exporting latest run for scape=$scape"
  go run -tags sqlite ./cmd/protogonosctl export --latest >/dev/null

  for file in config.json fitness_history.json top_genomes.json lineage.json generation_diagnostics.json species_history.json benchmark_summary.json; do
    if [[ ! -f "exports/$run_id/$file" ]]; then
      echo "[done-check] ERROR: missing exported artifact exports/$run_id/$file" >&2
      exit 1
    fi
  done
  if ! grep -Eq '"passed"[[:space:]]*:[[:space:]]*true' "exports/$run_id/benchmark_summary.json"; then
    echo "[done-check] ERROR: exported benchmark_summary did not pass for scape=$scape (exports/$run_id/benchmark_summary.json)" >&2
    exit 1
  fi

  echo "[done-check] Verified scape=$scape run_id=$run_id"
}

# Scape 1: XOR
run_benchmark_and_verify "xor" "101" "0.0001" "12" "6" "2" "0.02"

# Scape 2: Regression mimic
run_benchmark_and_verify "regression-mimic" "202" "0.0" "12" "6" "2" "0.02"

# Scape 3: Cart-pole-lite
run_benchmark_and_verify "cart-pole-lite" "303" "0.0001" "12" "6" "2" "0.02"

# Expanded parity smoke scapes.
# For these, we gate against severe regression while keeping run time bounded.
# IO-structure mutators are disabled to keep seed IO surfaces stable for smoke runs.
run_benchmark_and_verify "flatland" "404" "-0.2" "6" "3" "2" "0"
run_benchmark_and_verify "gtsa" "405" "-0.2" "6" "3" "2" "0"
run_benchmark_and_verify "fx" "406" "-0.2" "6" "3" "2" "0"
run_benchmark_and_verify "epitopes" "407" "-0.2" "6" "3" "2" "0"
run_benchmark_and_verify "dtm" "408" "-0.2" "6" "3" "2" "0"
run_benchmark_and_verify "pole2-balancing" "409" "-0.2" "6" "3" "2" "0"
run_benchmark_and_verify "llvm-phase-ordering" "410" "-0.2" "6" "3" "2" "0"

echo "[done-check] PASS"
