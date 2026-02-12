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
| `polis.erl` | `partial` | Core lifecycle + persistence orchestration implemented; OTP-specific semantics still simplified. |
| `population_monitor.erl` | `partial` | Generation loop, species-aware selection, quotas, diagnostics, lineage implemented; remaining fine-grained lifecycle parity pending. |
| `genotype.erl` | `partial` | Seed/clone/lifecycle/store flows implemented; full reference operation surface still broader. |
| `records.hrl` | `partial` | Core record equivalents implemented with versioned codecs; full field-level parity still pending. |
| `map2rec.erl` | `partial` | Baseline `constraint`/`pmp` map-to-record conversion layer implemented in `internal/map2rec`; broader record coverage still pending. |
| `genome_mutator.erl` | `partial` | Core topology/weight operators plus plasticity/substrate parameter mutations implemented; full operator breadth pending. |
| `selection_algorithm.erl` | `partial` | Elite/tournament/species/shared species + `hof_competition` alias implemented; additional reference strategies pending. |
| `fitness_postprocessor.erl` | `partial` | None/size/novelty postprocessors implemented; full policy breadth pending. |
| `specie_identifier.erl` | `partial` | Topology/speciation support implemented with adaptive threshold + representatives; full reference behavior tuning pending. |
| `tot_topological_mutations.erl` | `done` | Constant + count-scaled topological mutation count policies implemented and tested. |
| `tuning_selection.erl` | `partial` | `best_so_far`/`original`/`dynamic_random` implemented; full reference policy set pending. |
| `tuning_duration.erl` | `done` | Fixed/linear decay/topology-scaled attempt policies implemented and tested. |
| `cortex.erl` | `partial` | Per-step orchestrator with sensor->nn->actuator loop implemented; distributed/OTP-specific behaviors simplified. |
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

## Highest-priority remaining gaps to reach strict parity

1. Full `map2rec`/record-materialization parity semantics across additional record families.
2. Full scape behavior parity for `flatland`, `gtsa`, and `fx`.
3. Remaining operator/policy breadth from `genome_mutator`, `selection_algorithm`, and `fitness_postprocessor`.
4. Full substrate CPP/CEP behavioral parity beyond baseline scaffolding.
5. Utility ecosystem parity (`data_extractor`, `epitopes`) if required by target workflows.
