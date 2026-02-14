# DXNN2 Full Parity Checklist (Reference `.ref/src`)

Snapshot date: 2026-02-12

Status keys:
- `done`: implemented and covered by tests in current Go rewrite.
- `partial`: implemented baseline behavior, but reference breadth/edge semantics still incomplete.
- `todo`: not yet implemented for parity.
- `n/a`: intentionally out of scope for strict core parity in this phase.

## Module parity matrix

| Reference module | Status | Notes |
|---|---|---|
| `polis.erl` | `partial` | Core lifecycle + persistence orchestration implemented, including idempotent init/start state, explicit stop lifecycle, and scape lookup semantics; OTP process/supervision semantics still simplified. |
| `population_monitor.erl` | `partial` | Generation loop, species-aware selection, quotas, diagnostics, lineage implemented, including early-stop lifecycle controls for fitness-goal/evaluation-limit, `survival_percentage`-driven elite retention semantics, `specie_size_limit` parent-pool capping, pause/continue/stop control-channel semantics with live run control via platform/API and explicit `run_id`, and continuation from persisted population snapshots (`continue_population_id`); remaining fine-grained lifecycle parity pending. |
| `genotype.erl` | `partial` | Seed/clone/lifecycle/store flows implemented; full reference operation surface still broader. |
| `records.hrl` | `partial` | Core record equivalents implemented with versioned codecs; full field-level parity still pending. |
| `map2rec.erl` | `partial` | Baseline conversion parity implemented for `constraint`/`pmp`/`sensor`/`actuator`/`neuron`/`agent`/`cortex`/`specie`/`population` plus `trace`/`stat`/`topology_summary`/`signature`/`champion`; broader record coverage still pending. |
| `genome_mutator.erl` | `partial` | Core topology/weight operators plus plasticity/substrate parameter mutations implemented, with dedicated `add_bias`/`remove_bias` parity via random bias perturb/remove operators, dedicated `mutate_af`/`mutate_aggrf` parity via random activation/aggregator mutations, and dedicated `mutate_pf` parity via plasticity-rule mutation; full operator breadth pending. |
| `selection_algorithm.erl` | `done` | Elite/tournament/species/shared species plus reference HOF strategy surface (`hof_competition`/`hof_rank`/`hof_top3`/`hof_efficiency`/`hof_random`) and aliases (`competition`/`top3`) implemented across CLI/API and map2rec `constraint.population_selection_f` config materialization. |
| `fitness_postprocessor.erl` | `done` | Reference postprocessor surface implemented: `none`, `size_proportional` (with `EFF=0.05` scaling), and `novelty_proportional` placeholder semantics aligned as no-op; `nsize_proportional` alias and map2rec `constraint.population_fitness_postprocessor_f` materialization wired through CLI/API. |
| `specie_identifier.erl` | `done` | Reference `tot_n` specie distinguisher parity implemented, plus topology-based identifier support and adaptive species assignment continuity/threshold diagnostics. |
| `tot_topological_mutations.erl` | `done` | Constant + count-scaled topological mutation count policies implemented/tested, with map2rec `constraint.tot_topological_mutations_fs` config materialization support. |
| `tuning_selection.erl` | `partial` | `best_so_far`/`original`/`dynamic_random` plus reference mode set (`all`, `all_random`, `recent`, `recent_random`, `lastgen`, `lastgen_random`) implemented; full reference policy set pending. |
| `tuning_duration.erl` | `done` | Fixed/linear decay/topology-scaled attempt policies implemented and tested, including reference aliases (`const`, `nsize_proportional`, `wsize_proportional`). |
| `cortex.erl` | `partial` | Per-step orchestrator with sensor->nn->actuator loop implemented, including vector/chunked actuator dispatch semantics; distributed/OTP-specific process lifecycle/sync semantics remain simplified. |
| `neuron.erl` | `partial` | Runtime neuron/synapse eval implemented; complete reference semantics breadth pending. |
| `functions.erl` | `partial` | Core activation/math set implemented; full function catalog parity pending. |
| `derivatives.erl` | `done` | Derivative registry and tests implemented. |
| `plasticity.erl` | `partial` | Runtime plasticity integration + mutation hooks implemented; full rule set parity pending. |
| `signal_aggregator.erl` | `partial` | Aggregator support present; full reference aggregator set pending. |
| `exoself.erl` | `partial` | Bounded exoself tuning loop integrated with attempts/policies; full orchestration parity still pending. |
| `scape.erl` | `partial` | XOR/regression/cart-pole-lite/flatland/gtsa scapes implemented; full reference scape family breadth pending. |
| `flatland.erl` | `partial` | Baseline Go flatland scape added; not full behavioral parity with reference world simulation yet. |
| `scape_GTSA.erl` | `partial` | Baseline GTSA scape added; not full ETS/time-series workflow parity yet. |
| `fx.erl` | `partial` | Baseline FX scape/morphology/IO implemented; full market/workflow parity pending. |
| `morphology.erl` | `partial` | Morphology compatibility + validation implemented for current scapes; full constructor parity pending. |
| `sensor.erl` | `partial` | Registry + scalar families implemented for current scapes; broader sensor family parity pending. |
| `actuator.erl` | `partial` | Registry + scalar output families implemented for current scapes; broader actuator family parity pending. |
| `substrate.erl` | `partial` | Substrate scaffolding/runtime integrated; expanded behavior parity pending. |
| `substrate_cpp.erl` | `partial` | CPP registry/runtime hooks implemented in baseline form. |
| `substrate_cep.erl` | `partial` | CEP registry/runtime hooks implemented in baseline form. |
| `benchmarker.erl` | `partial` | CLI benchmark workflow + run artifacts + parity profile loader implemented; full report/stat format parity pending. |
| `data_extractor.erl` | `todo` | No dedicated ingestion utility module yet. |
| `epitopes.erl` | `todo` | No equivalent utility module yet. |
| `dxnn2_app.erl` | `n/a` | OTP app boot parity intentionally replaced by CLI/API runtime model. |
| `visor.erl` | `n/a` | Visualization loop currently out of scope. |

## Completed in latest iterations

- Species history persisted in store (memory/sqlite), exported in artifacts, queryable via API and CLI (`species` command).
- Adaptive speciation upgraded with representative continuity across generations.
- Mutation operator gating by genome/scape compatibility added.
- New parity scapes added: `flatland`, `gtsa` with morphology/IO/seed wiring.
- Parity profile loader added for `.ref` benchmark profiles (`--profile`).
- Reference strategy aliases added: `hof_competition`, `dynamic_random`.
- `done_check` tightened to require `species_history.json` and speciation diagnostics fields.
- Added `species-diff` API/CLI command for generation-level species dynamics diffs.
- Centralized typed run-config materialization with stricter validation in API.
- Implemented true `dynamic_random` exoself candidate-selection semantics.
- Added baseline `map2rec` package for permissive `constraint`/`pmp` record materialization.
- Expanded `map2rec` parity with permissive `sensor`/`actuator` record materialization and tests.
- Added permissive `map2rec` materialization for `neuron` and `agent` records with defaults + malformed-field fallback tests.
- Added permissive `map2rec` materialization for `cortex`, `specie`, and `population` records with mapping tests.
- Added permissive `map2rec` materialization for `trace`, `stat`, `topology_summary`, `signature`, and `champion` records with mapping tests.
- Wired `map2rec` into parity profile ingestion so profile-to-runtime mapping uses record materialization semantics.
- Added `benchmark --config <json>` map2rec-backed config ingestion path with explicit-flag override behavior.
- Added `run --config <json>` map2rec-backed config ingestion path with the same explicit-flag override behavior.
- Added sqlite integration tests for `run --config` and `benchmark --config` map2rec ingestion and flag overrides.
- Added `profile show` (text + JSON) to inspect resolved map2rec-backed parity profile materialization.
- Added reference selection aliases `competition` and `top3` across CLI/API selection resolution.
- Added reference fitness-postprocessor alias `nsize_proportional` across CLI/API resolution.
- Added reference tuning-duration aliases (`const`, `nsize_proportional`, `wsize_proportional`) across CLI/API/config materialization.
- Added reference tuning-selection mode set (`all`, `all_random`, `recent`, `recent_random`, `lastgen`, `lastgen_random`) across tuning/API/CLI/config normalization.
- Added map2rec materialization for `constraint.tot_topological_mutations_fs` into `RunRequest` with run-config override integration coverage.
- Added map2rec materialization for `constraint.population_fitness_postprocessor_f` into `RunRequest` with run-config override integration coverage.
- Preserved `constraint.population_selection_f` aliases (`competition`, `top3`) during map2rec config materialization instead of collapsing to one default.
- Added selection-strategy parity aliases `hof_rank`, `hof_efficiency`, and `hof_random` with CLI/API resolution and selector tests.
- Aligned `size_proportional` fitness postprocessing with reference efficiency scaling exponent (`EFF=0.05`) and added dedicated postprocessor tests.
- Added dedicated bias mutation operator (`perturb_random_bias`) and mapped reference `add_bias` config/profile operators to a distinct bias-mutation weight path.
- Added dedicated activation/aggregator mutation operators (`change_random_activation`, `change_random_aggregator`) and mapped reference `mutate_af`/`mutate_aggrf` config/profile operators to distinct mutation-weight paths.
- Added dedicated bias-removal mutation operator (`remove_random_bias`) and mapped reference `remove_bias` config/profile operators to a distinct remove-bias mutation-weight path.
- Added dedicated plasticity-function mutation operator (`change_plasticity_rule`) and mapped reference `mutate_pf` config/profile operators to a distinct plasticity-rule mutation-weight path.
- Hardened `polis` lifecycle semantics with idempotent init, explicit stop/reset of in-memory runtime registration, and direct registered-scape lookup coverage.
- Added population-monitor lifecycle stop controls (`fitness_goal`, `evaluations_limit`) with early-termination coverage and fixed persisted generation to reflect executed generations under early stop.
- Added `survival_percentage` parity semantics in `PopulationMonitor` to derive elite retention count when `elite_count` is unset, with validation and lineage-backed behavioral tests.
- Added map2rec/CLI/API parity wiring for lifecycle controls (`survival_percentage`, `fitness_goal`, `evaluations_limit`) through `run --config` and persisted run artifacts, with sqlite integration and API validation coverage.
- Added map2rec/CLI/API parity wiring for `specie_size_limit` with monitor-level per-species parent-pool capping semantics and dedicated unit/integration validation coverage.
- Added population-monitor control-channel semantics (`pause`, `continue`, `stop`) with generation-boundary synchronization tests to mirror reference op-tag control flow.
- Exposed monitor pause-control flow through platform/API/CLI via `start_paused`/`auto_continue_ms` (`--start-paused`, `--auto-continue-ms`) with config-materialization and integration coverage.
- Added live run control registry in `polis` keyed by `run_id` and exposed API-level `PauseRun`/`ContinueRun`/`StopRun` controls with controlled-run integration tests.
- Added population continuation flow (`continue/1` parity path) by loading persisted population snapshots via `continue_population_id`/`--continue-pop-id` and resuming evolution from stored genomes.
- Added fail-fast IO compatibility validation for continued populations against the target scape (`EnsurePopulationIOCompatibility`) to prevent deferred worker-time mismatches.
- Added generation-offset continuation semantics so resumed runs continue absolute generation numbering and persisted population generations from snapshot state.
- Added `initial_generation` artifact/config persistence for continued runs to make generation-offset provenance explicit in exported run metadata.
- Aligned continuation identity semantics closer to reference `continue/1`: when `continue_population_id` is set and `run_id` is omitted, the continued run reuses the population ID as run identity.
- Added append-on-continue persistence semantics: when a continuation reuses an existing run identity, prior fitness/diagnostics/species/lineage/top-genome history is merged and extended instead of being replaced.
- Added CLI live-control command surface `monitor pause|continue|stop --run-id ...` mapped to platform/API run-control registry semantics.
- Added map2rec `pmp.population_id` continuation-default materialization so config-driven runs can map directly into continued population/run identity when explicit IDs are omitted.
- Added explicit population lifecycle deletion path (`population delete --id ...`) mapped to genotype/store snapshot deletion for `delete_population` parity coverage.
- Added dedicated `hof_top3` selection parity path and wired it across evo selector resolution, API/CLI strategy parsing, and alias-acceptance tests.
- Aligned `novelty_proportional` with reference placeholder behavior as no-op while preserving clone isolation and postprocessor alias wiring (`nsize_proportional`).
- Added `tot_n` specie identifier parity and wired specie-identifier selection (`topology|tot_n`) through config/API/runtime and run artifacts.
- Expanded cortex actuator dispatch parity to support single-actuator vector writes and even chunking across multiple actuators.

## Highest-priority remaining gaps to reach strict parity

1. Full `map2rec`/record-materialization parity semantics across additional record families.
2. Full scape behavior parity for `flatland`, `gtsa`, and `fx`.
3. Remaining operator/policy breadth from `genome_mutator`, `selection_algorithm`, and `fitness_postprocessor`.
4. Full substrate CPP/CEP behavioral parity beyond baseline scaffolding.
5. Utility ecosystem parity (`data_extractor`, `epitopes`) if required by target workflows.
