# DXNN2 Full Parity Checklist (Reference `.ref/src`)

This checklist defines a strict path from current baseline parity to full parity with the reference Erlang source in `.ref/src`.

## Scope definition

- `must-have`: required to claim full reference parity.
- `nice-to-have`: useful but not required for strict parity sign-off.

## Must-have parity items

### Core platform and lifecycle

- [ ] `polis.erl`: complete platform lifecycle parity (init/create/reset/sync/start/stop semantics and persistence orchestration).
- [ ] `population_monitor.erl`: full generation loop parity including all population/specie lifecycle operations exposed in reference.
- [ ] `genotype.erl`: full agent/genotype lifecycle operations and compatibility with reference data flow.
- [ ] `records.hrl`: schema-level parity for all behaviorally relevant records/fields currently used by reference modules.
- [ ] `map2rec.erl`: equivalent conversion/validation behavior for record materialization paths.

### Evolution operators and policies

- [ ] `genome_mutator.erl`: full operator surface parity (not only current baseline set).
- [ ] `selection_algorithm.erl`: full selection strategy parity.
- [ ] `fitness_postprocessor.erl`: full postprocessing policy parity.
- [ ] `specie_identifier.erl`: full specie-ID behavior parity.
- [ ] `tot_topological_mutations.erl`: full topological mutation count policy parity.
- [ ] `tuning_selection.erl`: full tuning selection policy parity.
- [ ] `tuning_duration.erl`: full tuning duration/attempt policy parity.

### Agent runtime and NN behavior

- [ ] `cortex.erl`: full orchestration semantics parity (timing/flow edge cases included).
- [ ] `neuron.erl`: complete neuron runtime behavior parity.
- [ ] `functions.erl`: activation/math function parity (all reference-used functions).
- [ ] `derivatives.erl`: derivative function parity.
- [ ] `plasticity.erl`: full plasticity rules and integration parity.
- [ ] `signal_aggregator.erl`: full aggregator behavior parity.

### Exoself behavior

- [ ] `exoself.erl`: full orchestration parity between genetic/memetic modes and evaluation attempts.

### Scapes, morphology, and IO modules

- [ ] `scape.erl`: parity across all reference scape modes used for benchmarks/training.
- [ ] `morphology.erl`: full morphology constructor/selection parity.
- [ ] `sensor.erl`: sensor family parity for reference scapes.
- [ ] `actuator.erl`: actuator family parity for reference scapes.

### Substrate family

- [ ] `substrate.erl`: functional substrate runtime parity.
- [ ] `substrate_cpp.erl`: CPP behavior parity.
- [ ] `substrate_cep.erl`: CEP behavior parity.

### Benchmarker workflow

- [ ] `benchmarker.erl`: workflow/reporting parity sufficient to reproduce comparable benchmark runs and outputs.

## Nice-to-have parity items

These are in reference source but can be treated as non-blocking for strict ANN platform parity if your target excludes UI/tooling ecosystems.

- [ ] `visor.erl`: visualization loop parity.
- [ ] `epitopes.erl`: auxiliary experiment utility parity.
- [ ] `data_extractor.erl`: dataset ingestion/transform utility parity.
- [ ] `dxnn2_app.erl`: OTP app boot parity beyond current CLI-based startup model.

## Scape-specific parity backlog (must-have for strict reference parity)

- [ ] `flatland.erl` scape parity.
- [ ] `fx.erl` scape parity.
- [ ] `scape_GTSA.erl` scape parity.

## Current highest-risk parity gaps

1. Reference scape breadth and corresponding morphology/IO families.
2. Exoself policy depth (`tuning_selection`, `tuning_duration`) beyond current hill-climb baseline.
3. Full operator/policy breadth in `genome_mutator`, `selection_algorithm`, and `fitness_postprocessor`.
4. Complete record/schema parity with `records.hrl` where behavior depends on omitted fields.
5. Substrate module family behavior breadth (`substrate*`).

## Suggested execution order

1. Finish must-have evolutionary policy/operator surface.
2. Expand scape+morphology+IO family parity.
3. Complete exoself policy parity.
4. Complete substrate family parity.
5. Close records/schema parity gaps and verify module-by-module acceptance.
