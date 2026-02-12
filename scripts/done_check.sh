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

  echo "[done-check] Running benchmark command (sqlite backend) for scape=$scape"
  local run_output
  run_output="$(go run -tags sqlite ./cmd/protogonosctl benchmark \
    --store sqlite \
    --db-path ./protogonos.donecheck.db \
    --scape "$scape" \
    --pop 12 \
    --gens 6 \
    --seed "$seed" \
    --workers 2 \
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

  echo "[done-check] Exporting latest run for scape=$scape"
  go run -tags sqlite ./cmd/protogonosctl export --latest >/dev/null

  for file in config.json fitness_history.json top_genomes.json lineage.json generation_diagnostics.json species_history.json benchmark_summary.json; do
    if [[ ! -f "exports/$run_id/$file" ]]; then
      echo "[done-check] ERROR: missing exported artifact exports/$run_id/$file" >&2
      exit 1
    fi
  done

  echo "[done-check] Verified scape=$scape run_id=$run_id"
}

# Scape 1: XOR
run_benchmark_and_verify "xor" "101" "0.0001"

# Scape 2: Regression mimic
run_benchmark_and_verify "regression-mimic" "202" "0.0001"

# Scape 3: Cart-pole-lite
run_benchmark_and_verify "cart-pole-lite" "303" "0.0001"

echo "[done-check] PASS"
