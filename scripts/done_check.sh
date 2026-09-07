#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

check_parity_docs() {
  local audit_doc="docs/dxnn2-src-module-audit.md"
  local checklist_doc="docs/dxnn2-full-parity-checklist.md"

  echo "[done-check] Checking DXNN2 parity summary docs"

  local ref_modules
  ref_modules="$(find .ref/src -maxdepth 1 -type f -name '*.erl' -printf '%f\n' | sort)"

  local audit_modules
  audit_modules="$(grep '^| `.*\.erl` |' "$audit_doc" | sed 's/^| `//; s/` |.*//' | sort)"

  local checklist_modules
  checklist_modules="$(grep '^| `.*\.erl` |' "$checklist_doc" | sed 's/^| `//; s/` |.*//' | sort)"

  local ref_count audit_count checklist_count
  ref_count="$(printf '%s\n' "$ref_modules" | grep -c '.erl$' || true)"
  audit_count="$(printf '%s\n' "$audit_modules" | grep -c '.erl$' || true)"
  checklist_count="$(printf '%s\n' "$checklist_modules" | grep -c '.erl$' || true)"

  if [[ "$ref_count" != "33" ]]; then
    echo "[done-check] ERROR: expected 33 active .ref/src Erlang modules, got $ref_count" >&2
    exit 1
  fi
  if [[ "$audit_count" != "$ref_count" ]]; then
    echo "[done-check] ERROR: audit row count $audit_count does not match reference module count $ref_count" >&2
    exit 1
  fi
  if [[ "$checklist_count" != "$ref_count" ]]; then
    echo "[done-check] ERROR: checklist row count $checklist_count does not match reference module count $ref_count" >&2
    exit 1
  fi

  local missing_from_audit extra_in_audit missing_from_checklist extra_in_checklist
  missing_from_audit="$(comm -23 <(printf '%s\n' "$ref_modules") <(printf '%s\n' "$audit_modules"))"
  extra_in_audit="$(comm -13 <(printf '%s\n' "$ref_modules") <(printf '%s\n' "$audit_modules"))"
  missing_from_checklist="$(comm -23 <(printf '%s\n' "$ref_modules") <(printf '%s\n' "$checklist_modules"))"
  extra_in_checklist="$(comm -13 <(printf '%s\n' "$ref_modules") <(printf '%s\n' "$checklist_modules"))"

  if [[ -n "$missing_from_audit" || -n "$extra_in_audit" ]]; then
    echo "[done-check] ERROR: audit module rows do not match .ref/src" >&2
    echo "[done-check] missing_from_audit: ${missing_from_audit:-none}" >&2
    echo "[done-check] extra_in_audit: ${extra_in_audit:-none}" >&2
    exit 1
  fi
  if [[ -n "$missing_from_checklist" || -n "$extra_in_checklist" ]]; then
    echo "[done-check] ERROR: checklist module rows do not match .ref/src" >&2
    echo "[done-check] missing_from_checklist: ${missing_from_checklist:-none}" >&2
    echo "[done-check] extra_in_checklist: ${extra_in_checklist:-none}" >&2
    exit 1
  fi

  local duplicate_audit duplicate_checklist
  duplicate_audit="$(printf '%s\n' "$audit_modules" | uniq -d)"
  duplicate_checklist="$(printf '%s\n' "$checklist_modules" | uniq -d)"
  if [[ -n "$duplicate_audit" || -n "$duplicate_checklist" ]]; then
    echo "[done-check] ERROR: duplicate module rows found" >&2
    echo "[done-check] duplicate_audit: ${duplicate_audit:-none}" >&2
    echo "[done-check] duplicate_checklist: ${duplicate_checklist:-none}" >&2
    exit 1
  fi

  local implemented_count partial_count missing_count out_count done_count na_count
  implemented_count="$(grep '^| `.*\.erl` |' "$audit_doc" | grep -F -c '| `implemented` |' || true)"
  partial_count="$(grep '^| `.*\.erl` |' "$audit_doc" | grep -F -c '| `partial` |' || true)"
  missing_count="$(grep '^| `.*\.erl` |' "$audit_doc" | grep -F -c '| `missing` |' || true)"
  out_count="$(grep '^| `.*\.erl` |' "$audit_doc" | grep -F -c '| `out-of-scope-now` |' || true)"
  done_count="$(grep '^| `.*\.erl` |' "$checklist_doc" | grep -F -c '| `done` |' || true)"
  na_count="$(grep '^| `.*\.erl` |' "$checklist_doc" | grep -F -c '| `n/a` |' || true)"

  if [[ "$implemented_count" != "31" || "$partial_count" != "0" || "$missing_count" != "0" || "$out_count" != "2" ]]; then
    echo "[done-check] ERROR: unexpected audit status counts implemented=$implemented_count partial=$partial_count missing=$missing_count out-of-scope-now=$out_count" >&2
    exit 1
  fi
  if [[ "$done_count" != "31" || "$na_count" != "2" ]]; then
    echo "[done-check] ERROR: unexpected checklist status counts done=$done_count n/a=$na_count" >&2
    exit 1
  fi

  grep -F -q -- '- `implemented`: 31' "$audit_doc"
  grep -F -q -- '- `partial`: 0' "$audit_doc"
  grep -F -q -- '- `missing`: 0' "$audit_doc"
  grep -F -q -- '- `out-of-scope-now`: 2' "$audit_doc"

  local stale_pattern
  stale_pattern='remaining strict runtime-depth gaps|were still under review|partial`: 7|implemented`: 32|Run a final repository-wide parity audit|remaining strict-parity work is audit'
  if grep -RniE "$stale_pattern" "$audit_doc" "$checklist_doc" >/dev/null; then
    echo "[done-check] ERROR: stale parity gap wording found in parity docs" >&2
    grep -RniE "$stale_pattern" "$audit_doc" "$checklist_doc" >&2
    exit 1
  fi

  echo "[done-check] Parity docs summary: active_ref_modules=$ref_count implemented=$implemented_count partial=$partial_count missing=$missing_count out_of_scope=$out_count"
}

check_parity_docs

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
# Substrate/IO mutators remain enabled to exercise mutable topology/runtime paths.
run_benchmark_and_verify "flatland" "404" "-0.2" "6" "3" "2" "0.02"
run_benchmark_and_verify "gtsa" "405" "-0.2" "6" "3" "2" "0.02"
run_benchmark_and_verify "fx" "406" "-0.2" "6" "3" "2" "0.02"
run_benchmark_and_verify "epitopes" "407" "-0.2" "6" "3" "2" "0.02"
run_benchmark_and_verify "dtm" "408" "-0.2" "6" "3" "2" "0.02"
run_benchmark_and_verify "pole2-balancing" "409" "-0.2" "6" "3" "2" "0.02"
run_benchmark_and_verify "llvm-phase-ordering" "410" "-0.2" "6" "3" "2" "0.02"

echo "[done-check] PASS"
