# DXNN2 Source Inventory and Go Mapping

This is the source inventory companion to `docs/dxnn2-module-mapping.md`, focused on module-level parity targets.
Detailed evidence table: `docs/dxnn2-src-module-audit.md`.

## Upstream reference

- Repository: `CorticalComputer/DXNN2`
- Primary behavioral source in this project: upstream README responsibilities (cortex/population_monitor/exoself/scapes/morphology/polis).

## Inventory (README-grounded module set)

The following DXNN2 modules are treated as required parity anchors:

- `polis`
- `population_monitor`
- `cortex`
- `exoself`
- `scape` modules
- `morphology`
- `sensors`
- `actuators`
- `modular_constructor`
- `records.hrl`

## Go mapping (implemented)

- `polis` -> `internal/platform`
- `population_monitor` -> `internal/evo/population_monitor.go`
- `cortex` -> `internal/agent/cortex.go`
- `exoself` -> `internal/tuning/exoself.go`
- `scape` -> `internal/scape`
- `morphology` -> `internal/morphology`
- `sensors/actuators` contracts -> `internal/io`
- extensible records intent -> `internal/model` + `internal/storage/codec.go`
- mutation operators -> `internal/evo`

## Parity status snapshot

- Core lifecycle (`polis`) over persistent store: implemented (memory + sqlite backend selection).
- Evolution loop: implemented with selection/replication/mutation.
- Agent execution path: implemented (`sensor -> nn -> actuator`, plus scape step execution).
- Exoself tuning: implemented and integrated.
- Scapes: XOR and regression-mimic implemented with benchmark pathway.
- Morphology families: XOR and regression-mimic compatibility checks implemented.
- Exportable artifacts: implemented (`config`, `fitness_history`, `top_genomes`, `lineage`, optional `compare_tuning`, optional `benchmark_summary`).

## Remaining module-level parity work

- Add richer morphology families and additional concrete sensor/actuator components.
- Extend mutation/operator baseline further toward broader DXNN2 module breadth.
- Add explicit parity fixtures for modular constructor behavior and additional records-style extension sets.
- Add direct upstream `src/` module-by-module mapping evidence.
