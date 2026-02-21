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
| `population_monitor.erl` | `partial` | Generation loop, species-aware selection, quotas, diagnostics, lineage implemented, including early-stop lifecycle controls for fitness-goal/evaluation-limit, `survival_percentage`-driven elite retention semantics, `specie_size_limit` parent-pool capping, pause/continue/stop control-channel semantics with live run control via platform/API and explicit `run_id`, continuation from persisted population snapshots (`continue_population_id`), and success-only mutation-step retry semantics (failed/inapplicable mutators no longer consume mutation count); remaining fine-grained lifecycle parity pending. |
| `genotype.erl` | `partial` | Seed/clone/lifecycle/store flows implemented; full reference operation surface still broader. |
| `records.hrl` | `partial` | Core record equivalents implemented with versioned codecs; full field-level parity still pending. |
| `map2rec.erl` | `done` | Reference conversion surface implemented (`constraint`/`pmp`/`sensor`/`actuator`) and extended with broader record-materialization coverage (`neuron`/`agent`/`cortex`/`substrate`/`polis`/`scape`/`sector`/`avatar`/`object`/`circle`/`square`/`line`/`e`/`a`/`specie`/`population`/`trace`/`stat`/`topology_summary`/`signature`/`champion`) with mapping and fallback tests. |
| `genome_mutator.erl` | `done` | Core topology/weight operators plus plasticity/substrate parameter mutations implemented, with explicit reference-name mutators for `mutate_weights`, `add_bias`/`remove_bias`, `mutate_af`/`mutate_aggrf`, `mutate_pf`, `mutate_plasticity_parameters`, `add_neuron`, and `remove_neuron`, while preserving reference-closer `mutate_weights` behavior via tuning-selection-aware neuron targeting and proportional per-neuron weight perturbation (`1/sqrt(incoming_weights)` with non-empty fallback); directional add/remove link operators (`add_inlink`/`add_outlink`/`remove_inlink`/`remove_outlink`) wired through mutation policy with no non-directional fallback when directional candidates are absent (mutation canceled/retried) and with feedforward layer-order gating enabled in default policies, and `add_inlink` now sourcing from both input-neuron and sensor endpoint pools in the simplified model; explicit splice operators for `outsplice`/`insplice` with directional candidate gating and feedforward edge filtering in feedforward mode and exhaustion cancellation (`ErrNoMutationChoice`) when no directional edge remains or no synapses exist, explicit sensor/actuator add/remove mutation operators (`add_sensor`/`add_sensorlink`/`add_actuator`/`add_actuatorlink`/`remove_sensor`/`remove_actuator`) backed by scape-compatible IO registry filtering and split semantics (`add_sensor` requires available neuron endpoints and establishes an initial endpoint link, `add_actuator` now builds a helper-neuron scaffold with one incoming synapse and uses that helper as the initial actuator endpoint link, `add_*link` adds additional explicit endpoint links, `remove_*` removes selected component with its endpoint links and cancellation on empty/exhausted choices), explicit cutlink alias operators (`cutlink_FromElementToElement`/`cutlink_FromNeuronToNeuron`/`cutlink_FromSensorToNeuron`/`cutlink_FromNeuronToActuator`) and link helper aliases (`link_FromElementToElement`/`link_FromNeuronToNeuron`/`link_FromSensorToNeuron`/`link_FromNeuronToActuator`) mapped in shared config/profile canonical weight-bucket mapping paths, with generic element-link helpers now selecting across synapse/sensor-link/actuator-link candidate pools and canceling when all pools are exhausted, and with explicit neuron-only helper aliases (`link_FromNeuronToNeuron`/`cutlink_FromNeuronToNeuron`) now using candidate-aware applicability and cancellation on exhausted pools; explicit substrate-structure operators for `add_cpp`/`remove_cpp`/`add_cep`/`remove_cep` with cancellation when CPP/CEP registry alternatives are exhausted, with `add_cpp` now also appending one sensor->neuron scaffold link when compatible endpoint candidates exist and `add_cep` appending a helper-neuron synapse scaffold when neurons are present, substrate-dimension circuit mutations for `add_circuit_node`/`delete_circuit_node`/`add_circuit_layer` now canceled when substrate/dimension prerequisites are missing and gated to substrate-configured genomes, explicit search-parameter mutator paths (`mutate_tuning_selection`/`mutate_tuning_annealing`/`mutate_tot_topological_mutations`/`mutate_heredity_type`) backed by genome strategy metadata with expanded reference mode-surface coverage for tuning selection (`dynamic`/`dynamic_random`/`active`/`active_random`/`recent`/`recent_random`/`all`/`all_random`/`current`/`current_random`/`lastgen`/`lastgen_random`) and cancellation when no alternative strategy choice exists, with `Applicable` now reflecting alternative-choice availability when mutator choice sets are singular, and operator-name parity aligned to reference mutation tags for lineage/diagnostics reporting with exhaustive mapping tests for the must-have operator surface; legacy mutation name aliases are canonicalized during ingestion (for example `add_CircuitNode` and `remove_outLink`), duplicate directed-edge creation is now prevented for add-link mutators, directional add/remove link exhausted candidate pools now cancel with `ErrNoMutationChoice`, cutlink IO aliases now cancel with `ErrNoMutationChoice` on empty endpoint-link pools, sensor/actuator link mutators now stop at full connectivity and cancel with `ErrNoMutationChoice` when component pools are exhausted or empty while persisting explicit endpoint link records with synchronized legacy link counters, and function-choice mutators (`mutate_af`/`mutate_aggrf`/`mutate_pf`) plus plasticity/substrate-parameter mutators (`perturb_plasticity_rate`/`change_plasticity_rule`/`perturb_substrate_parameter`) now cancel when prerequisites or alternative choices are unavailable, with function mutator `Applicable` now also reflecting alternative-choice availability; `remove_neuron` now cancels when all neurons are protected/non-removable; `mutate_tot_topological_mutations` now mutates a full mode+parameter pair (`topological_mode` + `topological_param`) instead of mode-only, and `mutate_pf`/`mutate_plasticity_parameters` now target per-neuron plasticity metadata instead of genome-global fields. |
| `selection_algorithm.erl` | `done` | Elite/tournament/species/shared species plus reference HOF strategy surface (`hof_competition`/`hof_rank`/`hof_top3`/`hof_efficiency`/`hof_random`) and aliases (`competition`/`top3`) implemented across CLI/API and map2rec `constraint.population_selection_f` config materialization. |
| `fitness_postprocessor.erl` | `done` | Reference postprocessor surface implemented: `none`, `size_proportional` (with `EFF=0.05` scaling), and `novelty_proportional` placeholder semantics aligned as no-op; `nsize_proportional` alias and map2rec `constraint.population_fitness_postprocessor_f` materialization wired through CLI/API. |
| `specie_identifier.erl` | `done` | Reference `tot_n` specie distinguisher parity implemented, plus topology-based identifier support and adaptive species assignment continuity/threshold diagnostics. |
| `tot_topological_mutations.erl` | `done` | Constant + count-scaled topological mutation count policies implemented/tested, including stochastic `ncount_exponential` range semantics aligned to reference behavior, with map2rec `constraint.tot_topological_mutations_fs` config materialization support. |
| `tuning_selection.erl` | `done` | Reference mode-name surface implemented (`dynamic`, `dynamic_random`, `active`, `active_random`, `current`, `current_random`, `all`, `all_random`) plus legacy aliases (`recent`, `recent_random`, `lastgen`, `lastgen_random`); random modes use reference-style probabilistic subset selection (`1/sqrt(N)` with non-empty fallback), age modes use generation-aware filtering inferred from genome IDs, `all`/`all_random` now operate over the full candidate pool (not current-only), and `lastgen*` aliases now resolve to current-generation semantics; exoself and `mutate_weights` mirror reference mode-specific empty-pool fallback behavior (`active` no fallback/no-op, `active_random`/`current*`/`dynamic*` fallback to first element), and both exoself perturbation and `mutate_weights` apply age-annealed spread with direct actuator-local perturbation (per-actuator tunables + generation touch) alongside neuron-target perturbation. |
| `tuning_duration.erl` | `done` | Fixed/linear decay/topology-scaled attempt policies implemented and tested, with distinct reference-style `nsize_proportional` and `wsize_proportional` attempt formulas (instead of a shared alias) and `const` alias support. |
| `cortex.erl` | `done` | Per-step orchestrator with sensor->nn->actuator loop implemented, including vector/chunked actuator dispatch semantics; lifecycle/session parity now includes explicit `active`/`inactive`/`terminated` cortex states, reactivation/reset behavior, actuator sync-feedback aggregation (`fitness_acc` vector addition + end-flag accumulation), and episode-completion reporting with cycle/time/goal metadata via `RunUntilEvaluationComplete`. |
| `neuron.erl` | `partial` | Runtime neuron/synapse eval implemented with activation, aggregator modes, and reference-aligned output saturation (`[-1,1]`), with per-target-neuron plasticity rule/rate precedence now supported in runtime weight updates (post-neuron metadata overrides genome defaults); complete reference semantics breadth (OTP actor lifecycle, weight backup/restore protocol, inbox/reset message choreography) pending. |
| `functions.erl` | `partial` | Expanded activation/math catalog implemented (`tanh`/`sigmoid`/`sigmoid1`/`sin`/`cos`/`gaussian`/`sqrt`/`log`/threshold families, etc.); broader non-activation utility helper surface still pending. |
| `derivatives.erl` | `done` | Derivative registry and tests implemented. |
| `plasticity.erl` | `partial` | Runtime plasticity integration + mutation hooks implemented with Hebbian/Oja updates, saturation bounds, and reference rule-alias compatibility (`hebbian_w`, `ojas`, `ojas_w`), now including per-target-neuron rule/rate override precedence over genome defaults; full self-modulation rule set parity pending. |
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
- Tightened genome mutator directional add-link parity by rejecting duplicate directed neuron edges (including directional inlink/outlink candidate exhaustion).
- Tightened genome mutator sensor/actuator link parity by enforcing full-connectivity ceilings for `add_sensorlink`/`add_actuatorlink` applicability and mutation cancellation.
- Tightened genome mutator IO endpoint-link parity by storing explicit sensor->neuron and neuron->actuator links for `add_*link`/`cutlink_*`/`remove_*`, while keeping legacy link counters synchronized.
- Tightened genome mutator add-IO parity by having `add_sensor` and `add_actuator` immediately establish one initial endpoint link, matching reference-style add-and-connect semantics.
- Tightened `mutate_weights` parity by targeting synapse perturbations through strategy tuning-selection neuron subsets with per-neuron proportional/fallback mutation behavior.
- Expanded `mutate_tuning_selection` mutation mode coverage to the full reference-style tuning selection surface and legacy aliases.
- Tightened substrate mutation cancellation semantics so `add_cpp` and `add_cep` are only applicable when alternative registry choices exist and return a cancellation error when exhausted.
- Tightened `add_sensor` and `add_actuator` parity semantics to require available neuron endpoints for add-and-connect behavior (inapplicable/error when none exist).
- Tightened search-parameter mutator parity by canceling `mutate_tuning_selection`/`mutate_tuning_annealing`/`mutate_tot_topological_mutations`/`mutate_heredity_type` when no alternative choice exists.
- Tightened IO component mutation cancellation semantics so `add_sensor`/`add_actuator` and `remove_sensor`/`remove_actuator` return cancellation errors on exhausted or empty choice sets instead of no-op success.
- Tightened function mutator cancellation semantics so `mutate_af`, `mutate_aggrf`, and `mutate_pf` cancel when no alternative function choice exists.
- Tightened substrate/circuit mutator cancellation semantics so `add_cpp`/`add_cep`/`remove_cpp`/`remove_cep` and `add_circuit_node`/`delete_circuit_node`/`add_circuit_layer` cancel when substrate prerequisites are missing or exhausted.
- Tightened plasticity/substrate-parameter mutator cancellation semantics so `perturb_plasticity_rate`, `change_plasticity_rule`, and `perturb_substrate_parameter` cancel when required config or alternative choices are unavailable.
- Tightened `remove_neuron` cancellation semantics so mutation is canceled when all neurons are protected (no removable candidates).
- Tightened search mutator applicability semantics so `Applicable` reflects alternative-choice availability for constrained mutator choice sets.
- Tightened function mutator applicability semantics so `mutate_af`/`mutate_aggrf`/`mutate_pf` are inapplicable when configured choices provide no alternative.
- Tightened genome mutator search-parameter parity for `mutate_tot_topological_mutations` by mutating topological function mode and parameter together (`topological_mode` + `topological_param`).
- Tightened genome mutator plasticity parity by mutating `mutate_pf` and `mutate_plasticity_parameters` on per-neuron plasticity metadata rather than genome-global plasticity state.
- Tightened directional/add-link cancellation semantics so exhausted `add_inlink`/`add_outlink`/`add_sensorlink`/`add_actuatorlink` pools now return `ErrNoMutationChoice`.
- Tightened directional remove-link cancellation semantics so exhausted `remove_inlink`/`remove_outlink` pools now return `ErrNoMutationChoice`.
- Tightened IO cutlink cancellation semantics so `cutlink_FromSensorToNeuron` and `cutlink_FromNeuronToActuator` return `ErrNoMutationChoice` when no endpoint links exist.
- Tightened splice cancellation semantics so exhausted directional `outsplice`/`insplice` pools return `ErrNoMutationChoice`.
- Tightened splice cancellation semantics so `outsplice`/`insplice` return `ErrNoMutationChoice` for directional exhaustion and no-synapse cases.
- Tightened IO link-add cancellation semantics so `add_sensorlink`/`add_actuatorlink` return `ErrNoMutationChoice` when required component pools are empty.
- Tightened structural add-actuator/add-cep parity so `add_actuator` builds a helper-neuron scaffold (`source -> helper -> actuator`) and `add_cep` appends a helper-neuron synapse scaffold when neurons are present.
- Tightened `add_inlink` source-pool parity by allowing sensor->neuron endpoint-link candidates in addition to input-neuron synapse candidates.
- Tightened `add_cpp` structural parity by appending one sensor->neuron scaffold link when compatible endpoint candidates exist.
- Tightened generic element-link helper parity so `link_FromElementToElement` and `cutlink_FromElementToElement` now operate across synapse and endpoint-link pools with cancellation on full exhaustion.
- Tightened neuron-only link helper parity so `link_FromNeuronToNeuron` and `cutlink_FromNeuronToNeuron` now use candidate-aware applicability and cancellation on exhausted pools.
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
- Corrected `all`/`all_random` tuning-selection semantics to use the full candidate pool instead of current-generation-only filtering.
- Aligned `mutate_weights` tuning-selection semantics with age-filtered mode pools and age-annealed per-neuron spread (`max_delta * annealing^age`) using neuron generation metadata with ID fallback.
- Aligned exoself perturbation with tuning-selection mode-specific neuron pools and age-annealed per-neuron spread (`perturbation_range * pi * annealing^age`) with generation metadata/ID fallback.
- Added actuator-generation metadata tracking and actuator-aware tuning target projection (actuator candidates map to linked neurons) across exoself and `mutate_weights`.
- Aligned exoself empty-pool fallback semantics by mode to match reference behavior (`active` can return empty; `active_random`/`current*`/`dynamic*` fallback to first candidate with max spread).
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
- Added substrate-dimension circuit mutation paths for `add_circuit_node`/`add_circuit_layer` via dedicated operators in default mutation policy.
- Added explicit splice mutation paths (`outsplice`/`insplice`) via dedicated operators (`add_random_outsplice`/`add_random_insplice`) in the add-neuron policy bucket.
- Aligned mutation operator names with reference tags (for example: `add_inlink`, `outsplice`, `add_cpp`, `add_circuit_node`) to improve lineage parity.
- Added explicit sensor/actuator removal mutation paths (`remove_sensor`/`remove_actuator`) and mapped sensor/actuator cutlink-style names into substrate mutation weighting.
- Canonicalized legacy circuit mutation aliases (`add_CircuitNode`/`delete_CircuitNode`/`add_CircuitLayer`) during config/profile ingestion while preserving compatibility.
- Added direct actuator-local tuning perturbation support so exoself and `mutate_weights` mutate per-actuator tunables, with cortex applying actuator-local offsets during actuator dispatch.
- Aligned `mutate_weights` empty-pool behavior to reference mode-specific tuning-selection fallbacks (`active` no fallback; `active_random`/`current*`/`dynamic*` fallback to first element).
- Aligned active-mode empty-pool semantics to reference no-op behavior in both exoself perturbation and `mutate_weights` (no random fallback mutation when no eligible targets exist).
- Completed cortex lifecycle/synchronization parity with explicit active/inactive/terminated state handling, reactivation reset semantics, actuator sync-feedback accumulation, and episode-level completion reporting.
- Added per-neuron plasticity precedence in runtime updates so incoming synapses use post-neuron plasticity rule/rate overrides before genome-level defaults.

## Highest-priority remaining gaps to reach strict parity

1. Full `records.hrl` field-level parity semantics beyond current map2rec materialization surface.
2. Full scape behavior parity for `flatland`, `gtsa`, and `fx`.
3. Residual operator/policy breadth around substrate-runtime interaction helpers.
4. Full substrate CPP/CEP behavioral parity beyond baseline scaffolding.
5. Utility ecosystem parity (`data_extractor`, `epitopes`) if required by target workflows.
