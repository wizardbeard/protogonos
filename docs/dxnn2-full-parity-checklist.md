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
| `genome_mutator.erl` | `partial` | Core topology/weight operators plus plasticity/substrate parameter mutations implemented, with dedicated `add_bias`/`remove_bias` parity via random bias perturb/remove operators, dedicated `mutate_af`/`mutate_aggrf` parity via random activation/aggregator mutations, dedicated `mutate_pf` parity via plasticity-rule mutation, reference-closer `mutate_weights` behavior via proportional multi-weight perturbation (`1/sqrt(total_weights)` with non-empty fallback), directional add/remove link operators (`add_inlink`/`add_outlink`/`remove_inlink`/`remove_outlink`) wired through mutation policy, explicit splice operators for `outsplice`/`insplice`, explicit sensor/actuator add/remove mutation operators (`add_sensor`/`add_sensorlink`/`add_actuator`/`add_actuatorlink`/`remove_sensor`/`remove_actuator`) backed by scape-compatible IO registry filtering, explicit substrate-structure operators for `add_cpp`/`add_cep`, substrate-dimension circuit mutations for `add_CircuitNode`/`add_CircuitLayer`, and operator-name parity aligned to reference mutation tags for lineage/diagnostics reporting; full operator breadth pending. |
| `selection_algorithm.erl` | `done` | Elite/tournament/species/shared species plus reference HOF strategy surface (`hof_competition`/`hof_rank`/`hof_top3`/`hof_efficiency`/`hof_random`) and aliases (`competition`/`top3`) implemented across CLI/API and map2rec `constraint.population_selection_f` config materialization. |
| `fitness_postprocessor.erl` | `done` | Reference postprocessor surface implemented: `none`, `size_proportional` (with `EFF=0.05` scaling), and `novelty_proportional` placeholder semantics aligned as no-op; `nsize_proportional` alias and map2rec `constraint.population_fitness_postprocessor_f` materialization wired through CLI/API. |
| `specie_identifier.erl` | `done` | Reference `tot_n` specie distinguisher parity implemented, plus topology-based identifier support and adaptive species assignment continuity/threshold diagnostics. |
| `tot_topological_mutations.erl` | `done` | Constant + count-scaled topological mutation count policies implemented/tested, including stochastic `ncount_exponential` range semantics aligned to reference behavior, with map2rec `constraint.tot_topological_mutations_fs` config materialization support. |
| `tuning_selection.erl` | `partial` | Reference mode-name surface implemented (`dynamic`, `dynamic_random`, `active`, `active_random`, `current`, `current_random`, `all`, `all_random`) plus legacy aliases (`recent`, `recent_random`, `lastgen`, `lastgen_random`); random modes use reference-style probabilistic subset selection (`1/sqrt(N)` with non-empty fallback), age modes use generation-aware filtering inferred from genome IDs, and `lastgen*` aliases now resolve to current-generation semantics; full per-neuron spread semantics remain simplified versus source. |
| `tuning_duration.erl` | `done` | Fixed/linear decay/topology-scaled attempt policies implemented and tested, with distinct reference-style `nsize_proportional` and `wsize_proportional` attempt formulas (instead of a shared alias) and `const` alias support. |
| `cortex.erl` | `partial` | Per-step orchestrator with sensor->nn->actuator loop implemented, including vector/chunked actuator dispatch semantics; distributed/OTP-specific process lifecycle/sync semantics remain simplified. |
| `neuron.erl` | `partial` | Runtime neuron/synapse eval implemented with activation, aggregator modes, and reference-aligned output saturation (`[-1,1]`); complete reference semantics breadth (OTP actor lifecycle, weight backup/restore protocol) pending. |
| `functions.erl` | `partial` | Expanded activation/math catalog implemented (`tanh`/`sigmoid`/`sigmoid1`/`sin`/`cos`/`gaussian`/`sqrt`/`log`/threshold families, etc.); broader non-activation utility helper surface still pending. |
| `derivatives.erl` | `done` | Derivative registry and tests implemented. |
| `plasticity.erl` | `partial` | Runtime plasticity integration + mutation hooks implemented with Hebbian/Oja updates, saturation bounds, and reference rule-alias compatibility (`hebbian_w`, `ojas`, `ojas_w`); full self-modulation rule set parity pending. |
| `signal_aggregator.erl` | `done` | Aggregator support present (`dot_product`, `mult_product`, `diff_product`) with reference-style multiplicative `mult_product` semantics and stateful `diff_product` previous-input differencing parity across cortex steps. |
| `exoself.erl` | `partial` | Bounded exoself tuning loop integrated with attempts/policies, minimum-improvement acceptance threshold (`MIN_PIMPROVEMENT`-style gating), goal-aware early stop (mapped to `fitness_goal`), perturbation controls (`perturbation_range`, `annealing_factor`), and per-generation tuning telemetry surfaced in diagnostics/artifacts; full orchestration parity still pending. |
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
- Aligned neuron forward-path output clipping with reference neuron saturation semantics (`OUTPUT_SAT_LIMIT=1`).
- Expanded NN activation registry toward reference `functions.erl` with additional built-ins and behavior tests (sin/cos/gaussian/sqrt/log/threshold-style activations).
- Expanded derivative parity toward reference `derivatives.erl` (including `linear`, `sigmoid1`, `multiquadric`, `sqrt`, `log`, and clipping parity in `sigmoid`/`gaussian` derivatives).
- Added plasticity rule-name alias parity so reference-style PF names (`hebbian_w`, `ojas`, `ojas_w`) map onto runtime Hebbian/Oja updates.
- Aligned `mult_product` aggregator semantics with reference multiplicative bias behavior and added focused NN parity coverage.
- Added exoself minimum-improvement gating (`tune_min_improvement`) across tuner runtime, CLI/config/API wiring, and validation coverage.
- Added tuning-selection alias parity for reference mode names (`dynamic`, `active`, `active_random`, `current`, `current_random`) with CLI/API/config normalization and runtime support.
- Aligned tuning `*_random` selection behavior with reference probabilistic subset semantics (`1/sqrt(N)` + non-empty fallback) over the available candidate base pool.
- Added goal-aware exoself short-circuit semantics so tuning attempts stop early when best fitness already reaches the configured run goal.
- Added exoself perturbation controls (`tune_perturbation_range`, `tune_annealing_factor`) across runtime, CLI/config/API validation, and persisted run config artifacts.
- Added generation-aware candidate-pool filtering for `dynamic`/`active`/`current` tuning-selection families by inferring generation from genome IDs.
- Added per-generation tuning telemetry (`invocations`, attempts, candidate evaluations, accepted/rejected counts, goal hits) to diagnostics and persisted artifacts via optional reporting tuner support.
- Exposed tuning telemetry fields in `protogonosctl diagnostics` output for direct CLI parity introspection.
- Added derived tuning efficiency ratios in `protogonosctl diagnostics` (`tuning_accept_rate`, `tuning_evals_per_attempt`) to aid parity comparisons against reference runs.
- Promoted tuning efficiency ratios to first-class diagnostics fields persisted through store/API/export (`generation_diagnostics.json`), with CLI consuming stored values directly.
- Extended `species-diff` output to include tuning telemetry deltas between compared generations for combined speciation+tuning analysis.
- Added `species-diff --show-diagnostics` to print full from/to diagnostics snapshots (including tuning telemetry and ratios) alongside deltas.
- Added `species-diff --json` for machine-readable diff output suitable for scripting/CI parity checks.
- Added `diagnostics --json` for machine-readable per-generation diagnostics (including tuning telemetry and ratios) in scripts/CI.
- Added `runs --json` for machine-readable run index/listing output, keeping analysis/reporting commands automation-friendly.
- Added `lineage --json` and `top --json` for machine-readable lineage/top-genome analysis outputs.
- Added `fitness --json` and `species --json` for machine-readable history output parity across reporting commands.
- Aligned default `mutate_weights` path closer to reference by perturbing a proportional subset of weights (`1/sqrt(total_weights)`) with guaranteed non-empty mutation fallback.
- Added directional add-link mutation paths (`add_inlink`/`add_outlink`) via dedicated operators (`add_random_inlink`/`add_random_outlink`) in default mutation policy.
- Added directional remove-link mutation paths (`remove_inlink`/`remove_outlink`) via dedicated operators (`remove_random_inlink`/`remove_random_outlink`) in default mutation policy.
- Added explicit sensor/actuator mutation paths (`add_sensor`/`add_sensorlink`/`add_actuator`/`add_actuatorlink`) via dedicated scape-compatible IO operators.
- Added explicit substrate-structure mutation paths (`add_cpp`/`add_cep`) via dedicated operators (`add_random_cpp`/`add_random_cep`) in default mutation policy.
- Added substrate-dimension circuit mutation paths for `add_CircuitNode`/`add_CircuitLayer` via dedicated operators (`add_circuit_node`/`add_circuit_layer`) in default mutation policy.
- Added explicit splice mutation paths (`outsplice`/`insplice`) via dedicated operators (`add_random_outsplice`/`add_random_insplice`) in the add-neuron policy bucket.
- Aligned mutation operator names with reference tags (for example: `add_inlink`, `outsplice`, `add_cpp`, `add_CircuitNode`) to improve lineage parity.
- Added explicit sensor/actuator removal mutation paths (`remove_sensor`/`remove_actuator`) and mapped sensor/actuator cutlink-style names into substrate mutation weighting.

## Highest-priority remaining gaps to reach strict parity

1. Full `map2rec`/record-materialization parity semantics across additional record families.
2. Full scape behavior parity for `flatland`, `gtsa`, and `fx`.
3. Remaining operator/policy breadth from `genome_mutator`, `selection_algorithm`, and `fitness_postprocessor`.
4. Full substrate CPP/CEP behavioral parity beyond baseline scaffolding.
5. Utility ecosystem parity (`data_extractor`, `epitopes`) if required by target workflows.
