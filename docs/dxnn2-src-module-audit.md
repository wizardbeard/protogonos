# DXNN2 `src/` Module Audit (Verified Against `.ref/src`)

This audit is based on the local reference implementation at `.ref/src`.

## Audit date

- 2026-02-10

## Verification source

- Reference root: `.ref/src`
- Files audited: all `.erl` plus `records.hrl`

## Module matrix

Status legend:
- `implemented`: direct functional equivalent exists in current Go codebase.
- `partial`: concept present, but significant behaviors from reference module are missing.
- `missing`: no meaningful equivalent found yet.
- `out-of-scope-now`: not targeted in current AGENTS phase scope.

| Ref module | Primary role (inferred from source) | Go mapping | Status | Evidence |
|---|---|---|---|---|
| `polis.erl` | platform lifecycle, DB bootstrap/start/stop | `internal/platform/polis.go`, `internal/storage/*` with idempotent init/start state, explicit stop lifecycle, and scape lookup semantics | `partial` | `internal/platform/polis.go`, `internal/platform/polis_test.go`, `.ref/src/polis.erl` |
| `population_monitor.erl` | eval loop, evolutionary control, species/population ops | `internal/evo/population_monitor.go` with species-aware evolution loop, lineage/diagnostics, early-stop controls (`fitness_goal`, `evaluations_limit`), `survival_percentage`-driven elite retention derivation, `specie_size_limit` parent-pool capping, pause/continue/stop control-channel semantics via platform/API/CLI (`run_id` live controls), continuation from persisted population snapshots via `continue_population_id`, fail-fast IO compatibility validation, generation-offset continuation semantics, persisted `initial_generation` run metadata, default continuation identity reuse (`run_id <- continue_population_id`), append-on-continue history persistence for same-run continuation, map2rec `pmp.population_id` continuation-default materialization, and explicit `delete_population` lifecycle deletion command path | `partial` | `internal/evo/population_monitor.go`, `internal/evo/population_monitor_test.go`, `internal/platform/polis_evolution_test.go`, `internal/platform/polis.go`, `internal/genotype/store.go`, `internal/morphology/regression_mimic.go`, `internal/stats/artifacts.go`, `cmd/protogonosctl/config.go`, `cmd/protogonosctl/main_integration_test.go`, `cmd/protogonosctl/main.go`, `pkg/protogonos/api_test.go`, `.ref/src/population_monitor.erl` |
| `cortex.erl` | sensor/neuron/actuator runtime coordination | `internal/agent/cortex.go` with sensor->forward->actuator orchestration and vector/chunked actuator dispatch support | `partial` | `internal/agent/cortex.go`, `internal/agent/cortex_test.go`, `.ref/src/cortex.erl` |
| `exoself.erl` | agent lifecycle + tuning + eval mode orchestration | `internal/tuning/exoself.go` + monitor integration | `partial` | `internal/tuning/exoself.go`, `internal/evo/population_monitor_tuning_test.go`, `.ref/src/exoself.erl` |
| `neuron.erl` | neuron actor runtime, integrator/plasticity hooks | `internal/nn/network.go` | `partial` | `internal/nn/network.go`, `.ref/src/neuron.erl` |
| `sensor.erl` | sensor actor implementations + scape IO | `internal/io/*` | `partial` | `internal/io/interfaces.go`, `internal/io/scalar_components.go`, `.ref/src/sensor.erl` |
| `actuator.erl` | actuator actor implementations + scape IO | `internal/io/*` | `partial` | `internal/io/interfaces.go`, `internal/io/scalar_components.go`, `.ref/src/actuator.erl` |
| `scape.erl` | benchmark environments (xor, pole balancing, fx, dtm, etc.) | `internal/scape/*` | `partial` | `internal/scape/xor.go`, `internal/scape/regression_mimic.go`, `.ref/src/scape.erl` |
| `morphology.erl` | morphology catalogs + init/full sensor/actuator selection | `internal/morphology/*` | `partial` | `internal/morphology/regression_mimic.go`, `internal/morphology/xor.go`, `.ref/src/morphology.erl` |
| `genome_mutator.erl` | mutation operator suite | `internal/evo/mutations.go` with dedicated bias add/remove, activation/aggregator, and plasticity-rule mutation operators, and reference `add_bias`/`remove_bias`/`mutate_af`/`mutate_aggrf`/`mutate_pf` config/profile mapping | `partial` | `internal/evo/mutations.go`, `internal/evo/mutations_test.go`, `cmd/protogonosctl/config.go`, `cmd/protogonosctl/profiles.go`, `pkg/protogonos/api.go`, `.ref/src/genome_mutator.erl` |
| `selection_algorithm.erl` | selection strategy variants | `internal/evo/selection.go` + monitor integration + CLI/API alias mapping (`hof_competition`/`hof_rank`/`hof_top3`/`hof_efficiency`/`hof_random`/`competition`/`top3`) and map2rec config materialization via `constraint.population_selection_f` | `implemented` | `internal/evo/selection.go`, `internal/evo/population_monitor.go`, `cmd/protogonosctl/main.go`, `cmd/protogonosctl/config.go`, `cmd/protogonosctl/profiles.go`, `pkg/protogonos/api.go`, `.ref/src/selection_algorithm.erl` |
| `tuning_selection.erl` | exoself selection schedule variants | `internal/tuning/exoself.go` (`best_so_far`/`original`/`dynamic_random` + `all`/`all_random`/`recent`/`recent_random`/`lastgen`/`lastgen_random`) with CLI/API/config normalization | `partial` | `internal/tuning/exoself.go`, `cmd/protogonosctl/main.go`, `pkg/protogonos/api.go`, `cmd/protogonosctl/config.go`, `.ref/src/tuning_selection.erl` |
| `tuning_duration.erl` | tuning-attempt duration policies | `internal/tuning/policy.go` + API/CLI/config alias normalization (`const`/`nsize_proportional`/`wsize_proportional`) | `partial` | `internal/tuning/policy.go`, `pkg/protogonos/api.go`, `cmd/protogonosctl/config.go`, `.ref/src/tuning_duration.erl` |
| `tot_topological_mutations.erl` | mutation-count policy functions | `internal/evo/topological_mutations.go` + monitor integration + map2rec constraint materialization into run config | `partial` | `internal/evo/topological_mutations.go`, `internal/evo/population_monitor.go`, `cmd/protogonosctl/config.go`, `.ref/src/tot_topological_mutations.erl` |
| `functions.erl` | activation/math utility set | `internal/nn/registry.go` built-ins | `partial` | `internal/nn/registry.go`, `.ref/src/functions.erl` |
| `derivatives.erl` | derivative functions for activations | `internal/nn/derivatives.go` | `partial` | `internal/nn/derivatives.go`, `.ref/src/derivatives.erl` |
| `plasticity.erl` | Hebbian/Oja/etc plasticity rules | `internal/nn/plasticity.go` + cortex integration | `partial` | `internal/nn/plasticity.go`, `internal/agent/cortex.go`, `.ref/src/plasticity.erl` |
| `signal_aggregator.erl` | dot/mult/diff aggregation modes | selectable per-neuron aggregation in forward path | `partial` | `internal/nn/network.go`, `internal/nn/network_test.go`, `.ref/src/signal_aggregator.erl` |
| `genotype.erl` | construction/cloning/deletion/fingerprint topologies | seed builders + model/store basics + topology fingerprint/signature + genotype utility package | `partial` | `cmd/protogonosctl/main.go`, `pkg/protogonos/api.go`, `internal/model/types.go`, `internal/genotype/*`, `.ref/src/genotype.erl` |
| `specie_identifier.erl` | species topology identifiers | `internal/evo/specie_identifier.go` + species tournament selector with explicit `tot_n` parity mapping (and topology fallback), wired through config/API/runtime selection | `implemented` | `internal/evo/specie_identifier.go`, `internal/evo/specie_identifier_test.go`, `internal/evo/selection.go`, `cmd/protogonosctl/config.go`, `pkg/protogonos/api.go`, `.ref/src/specie_identifier.erl` |
| `fitness_postprocessor.erl` | post-fitness adjustment (size/novelty) | `internal/evo/fitness_postprocessor.go` + monitor integration + CLI/API alias mapping (`nsize_proportional`) + map2rec config materialization (`constraint.population_fitness_postprocessor_f`) + reference `size_proportional` exponent parity (`EFF=0.05`) + reference `novelty_proportional` placeholder parity (no-op) | `implemented` | `internal/evo/fitness_postprocessor.go`, `internal/evo/fitness_postprocessor_test.go`, `internal/evo/population_monitor.go`, `cmd/protogonosctl/main.go`, `cmd/protogonosctl/config.go`, `pkg/protogonos/api.go`, `.ref/src/fitness_postprocessor.erl` |
| `benchmarker.erl` | experiment orchestration/reporting/graphs | `benchmark` command + artifacts + profile resolution views | `partial` | `cmd/protogonosctl/main.go`, `cmd/protogonosctl/profiles.go`, `internal/stats/artifacts.go`, `.ref/src/benchmarker.erl` |
| `map2rec.erl` | map->record conversion helpers | `internal/map2rec` (broad baseline parity across core records including `constraint`/`pmp`/`sensor`/`actuator`/`neuron`/`agent`/`cortex`/`specie`/`population`/`trace`/`stat`/`topology_summary`/`signature`/`champion`) + runtime use in parity profile ingestion and `run/benchmark --config` ingestion | `partial` | `internal/map2rec/convert.go`, `internal/map2rec/convert_test.go`, `cmd/protogonosctl/profiles.go`, `cmd/protogonosctl/config.go`, `cmd/protogonosctl/main.go`, `cmd/protogonosctl/main_integration_test.go`, `.ref/src/map2rec.erl` |
| `records.hrl` | central schema/extensibility surface | `internal/model/types.go`, registries | `partial` | `internal/model/types.go`, `internal/io/registry.go`, `internal/evo/registry.go`, `internal/nn/registry.go`, `.ref/src/records.hrl` |
| `dxnn2_app.erl` | OTP application entrypoint | `cmd/protogonosctl/main.go` | `out-of-scope-now` | `cmd/protogonosctl/main.go`, `.ref/src/dxnn2_app.erl` |
| `flatland.erl` | interactive multi-agent world/scape server | `internal/scape/flatland.go` | `partial` | `internal/scape/flatland.go`, `.ref/src/flatland.erl` |
| `fx.erl` | financial simulation/evaluation tooling | `internal/scape/fx.go` | `partial` | `internal/scape/fx.go`, `.ref/src/fx.erl` |
| `scape_GTSA.erl` | time-series scape process | `internal/scape/gtsa.go` | `partial` | `internal/scape/gtsa.go`, `.ref/src/scape_GTSA.erl` |
| `substrate.erl` | substrate encoding runtime | `internal/substrate/runtime.go` + cortex/monitor integration | `partial` | `internal/substrate/runtime.go`, `internal/agent/cortex.go`, `internal/evo/population_monitor.go`, `.ref/src/substrate.erl` |
| `substrate_cpp.erl` | substrate CPP runtime | `internal/substrate/components.go` + registry | `partial` | `internal/substrate/components.go`, `internal/substrate/registry.go`, `.ref/src/substrate_cpp.erl` |
| `substrate_cep.erl` | substrate CEP runtime | `internal/substrate/components.go` + registry | `partial` | `internal/substrate/components.go`, `internal/substrate/registry.go`, `.ref/src/substrate_cep.erl` |
| `visor.erl` | visualization loop/UI drawing | none | `out-of-scope-now` | `.ref/src/visor.erl` |
| `epitopes.erl` | experiment/sim utility tooling | none | `out-of-scope-now` | `.ref/src/epitopes.erl` |
| `data_extractor.erl` | dataset ingestion and conversion tooling | none | `out-of-scope-now` | `.ref/src/data_extractor.erl` |

## Summary

- `implemented`: 3
- `partial`: 26
- `missing`: 0
- `out-of-scope-now`: 4

Most core AGENTS responsibilities are present but still `partial` versus reference breadth. The largest parity gaps are:

1. Species-level behavior and advanced selection/competition variants.
2. Substrate encoding family (`substrate*`) and plasticity.
3. Full scape catalog and environment-specific sensor/actuator implementations.
4. Rich records parity (many `records.hrl` fields not yet modeled).

## Recommended next parity increments

1. Implement `fitness_postprocessor` + `selection_algorithm` equivalents in `internal/evo`.
2. Add `tot_topological_mutations` policy support in `PopulationMonitor`.
3. Add substrate feature gate with minimal `substrate` skeleton and tests.
4. Expand scapes beyond XOR/regression-mimic (at least one dynamic control scape).
