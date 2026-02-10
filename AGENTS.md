# AGENTS.md — DXNN2 (Erlang) → Go rewrite guide (called: protogonos)

Repo: https://github.com/CorticalComputer/DXNN2 :contentReference[oaicite:0]{index=0}  
DXNN2 is an Erlang implementation of a distributed Topology and Weight Evolving ANN platform (“DXNN MK2”). :contentReference[oaicite:1]{index=1}

## 0) Goal

Rebuild DXNN2 in Go with functional parity for:
- Core platform lifecycle (“polis”) over a persistent store
- Population evolution loop (selection/replication/variation)
- Agent execution model: cortex + neurons + sensors + actuators
- “Exoself” weight tuning loop (memetic search)
- “Scapes” as environments that evaluate agents

The rewrite should preserve the conceptual architecture described in the README: cortex orchestration, population_monitor, exoself, scapes, morphology, sensors/actuators, records.hrl extension points, and a top-level “polis” platform over an Erlang mnesia database. :contentReference[oaicite:2]{index=2}

## 1) Non-goals (initially)

- UI, web dashboards, distributed multi-node clustering
- Perfect behavioral determinism across runs
- Feature additions beyond parity (opt-in later)

## 2) Source-of-truth constraints

The README describes these key responsibilities:
- `cortex` synchronizes neurons, sensors, actuators. :contentReference[oaicite:3]{index=3}
- `population_monitor` spawns agents, waits for evaluation, then performs selection/replication/variation (mutation operators). :contentReference[oaicite:4]{index=4}
- `exoself` performs synaptic weight tuning (memetic algorithm capability). :contentReference[oaicite:5]{index=5}
- “Scapes” present problems/environments; “morphology” defines sensors/actuators compatibility with scapes. :contentReference[oaicite:6]{index=6}
- Platform is called `polis`; it manages the (mnesia) database and top-level system infrastructure. :contentReference[oaicite:7]{index=7}
- Extensibility: sensors/actuators, mutation operators, activation functions are added via record definitions + implementations. :contentReference[oaicite:8]{index=8}

Treat these as invariants to preserve in Go, even if internal structure changes.

## 3) Go architecture mapping

### Erlang ↔ Go mental model

| Erlang concept | DXNN2 usage | Go equivalent |
|---|---|---|
| process / gen_server | long-lived actors (polis, population monitor, agent cortex, scape) | goroutine + mailbox channel, explicit lifecycle |
| message passing | coordination | typed events over channels |
| supervision trees | fault containment | `errgroup`, structured cancellation, restart loops with backoff |
| mnesia | persistent state | embedded KV/SQL store (see §6) |
| records (.hrl) | shared schemas | Go structs + versioned codec + schema tests |

### Proposed Go package layout

```txt
/cmd
protogonosctl/ # CLI: init, start, run benchmarks, etc.
/internal
platform/ # polis: lifecycle, config, process registry
storage/ # persistence abstraction + concrete backend
evo/ # selection/replication/variation, operators
agent/ # agent runtime, cortex orchestration
nn/ # neurons, synapses, activation, network eval
io/ # sensors + actuators interfaces + implementations
morphology/ # morphology specs, compatibility rules
scape/ # environments, evaluation harness
tuning/ # exoself: weight optimization
stats/ # logging, metrics, fitness history
/pkg
protogonos/ # minimal public API (stable surface)
/testdata
fixtures/ # golden tests, serialized genomes, etc.
```

Guiding rule: keep `/internal` cohesive and allow `/pkg/protogonos` to remain stable for downstream use.

## 4) Interfaces (core contracts)

### Agent lifecycle
- Construct genotype (topology + parameters)
- Materialize phenotype (runtime network + IO wiring)
- Run in scape until terminal condition
- Return fitness + diagnostics

### Minimal Go interfaces

- `storage.Store`: transaction-ish API for genomes/agents/populations
- `scape.Scape`: `Evaluate(ctx, agent) (Fitness, Trace, error)`
- `morphology.Morphology`: enumerates available sensors/actuators and constraints
- `io.Sensor` / `io.Actuator`: typed signal interfaces
- `evo.Operator`: mutation / crossover (even if crossover starts disabled)
- `tuning.Tuner`: exoself weight tuning, bounded by attempts/steps

## 5) Execution model

### Orchestration roles (match README semantics)
- `platform.Polis`: owns config, store, scape registry, and top-level run loop. :contentReference[oaicite:9]{index=9}
- `evo.PopulationMonitor`: owns one population run: spawn agents, evaluate, then select/replicate/mutate. :contentReference[oaicite:10]{index=10}
- `agent.Cortex`: per-agent coordinator that advances sensors → NN → actuators per tick/step. :contentReference[oaicite:11]{index=11}
- `tuning.Exoself`: optional inner-loop tuner that adjusts weights per agent between evaluations. :contentReference[oaicite:12]{index=12}

### Concurrency policy
- One goroutine per evaluated agent (bounded worker pool)
- Scapes may run concurrently; each evaluation must be cancellation-aware
- Determinism controls:
  - explicit `rand.Source` per run, derived from seed + agent id
  - store seed alongside run metadata

## 6) Persistence strategy (mnesia replacement)

DXNN2 uses mnesia under “polis”. :contentReference[oaicite:13]{index=13}  
In Go, pick a storage backend that supports:
- fast local dev
- transactions or atomic batches
- simple backups

Recommended options (choose one early and encapsulate behind `storage.Store`):
- SQLite (via modernc.org/sqlite for pure-Go) for relational queries and durability
- Pebble / Badger for KV speed

Hard requirement: store formats must be versioned (schema version + codec version in every record).

## 7) Porting plan (phased, test-first)

### Phase 1 — Inventory and golden fixtures
Deliverables:
- Enumerate modules and data schemas in `src/` (Erlang) and document mapping
- Define Go structs mirroring core records (genome, neuron, synapse, agent, population, scape summary)
- Create golden test fixtures:
  - serialized minimal genomes
  - expected forward-pass outputs for fixed weights
  - expected mutation operator effects

Success criteria:
- `go test ./...` validates codecs and invariants
- A “noop” polis run can init store and register scapes

### Phase 2 — NN runtime parity
Deliverables:
- Implement network evaluation: neuron activation, synapses, step scheduling
- Implement sensor/actuator plumbing
- Implement `agent.Cortex` tick loop

Success criteria:
- A hand-constructed agent can run one episode in a trivial scape and produce expected outputs

### Phase 3 — Evolution loop parity
Deliverables:
- `PopulationMonitor` lifecycle: spawn N agents, evaluate, rank, select, replicate, mutate :contentReference[oaicite:14]{index=14}
- Baseline operators: add/remove neuron, add/remove synapse, perturb weights, change activation, IO modifications

Success criteria:
- A small population improves fitness on a toy scape over several generations

### Phase 4 — Exoself tuning (memetic inner loop)
Deliverables:
- Implement bounded weight-tuning per agent (stochastic hill climb / CMA-lite / SGD-like)
- Integrate with evaluation attempts (README notes memetic vs genetic controlled by attempts) :contentReference[oaicite:15]{index=15}

Success criteria:
- With tuning enabled, fitness improves faster on a stationary toy task

### Phase 5 — Benchmarks + CLI
Deliverables:
- CLI commands analogous to README workflow: init store, start polis, run population, export results :contentReference[oaicite:16]{index=16}
- Benchmarks folder output equivalent (write run artifacts)

Success criteria:
- Reproducible run artifacts with seed and config embedded

## 8) Testing strategy

- Unit tests:
  - activation functions
  - mutation operators (property tests: invariants preserved)
  - codecs round-trip
- Integration tests:
  - toy scapes: XOR, cart-pole-lite, regression mimic
- Golden tests:
  - fixed seed runs for N generations with expected summary statistics bounds (not exact values)

## 9) Coding standards

- Go 1.22+ (or current stable your org uses)
- `golangci-lint` with:
  - `errcheck`, `staticcheck`, `govet`, `gosec` (tune as needed)
- Explicit context usage for all long-lived operations
- No panics across package boundaries (panic allowed only in `main` for truly fatal config issues)

## 10) Extension points (match “records.hrl” intent)

DXNN2 allows adding:
- sensors and actuators via morphology + IO modules :contentReference[oaicite:17]{index=17}
- mutation operators and activation functions via shared definitions + implementations :contentReference[oaicite:18]{index=18}

In Go:
- register via init-time registries:
  - `nn.RegisterActivation(name, fn)`
  - `evo.RegisterOperator(name, op)`
  - `io.RegisterSensor(name, factory)`
  - `io.RegisterActuator(name, factory)`
- enforce versioning and compatibility checks in registry

## 11) “Done” definition

A Go rewrite counts as complete when:
- A polis run can initialize storage, register a scape, create a population, and evolve agents end-to-end
- At least one reference scape shows measurable fitness gain over generations
- Run artifacts are exportable (genomes + lineage + fitness history)
- Public API is stable enough for downstream experiments

## 12) Quick repo notes (from upstream)

The upstream README indicates:
- compilation and startup steps in Erlang (`make:all()`, `polis:create()`, `polis:start()`) :contentReference[oaicite:19]{index=19}
- a “benchmarks” folder is expected for outputs :contentReference[oaicite:20]{index=20}
- morphology and modular constructor influence initial sensors/actuators selection :contentReference[oaicite:21]{index=21}

Mirror these flows in `dxnnctl`:
- `dxnnctl init`
- `dxnnctl start`
- `dxnnctl run --scape xor --pop 50 --gens 100 --seed 1`
- `dxnnctl export --run-id ...`

Our application is called `protogonos` and the `cmd` binary is called `protogonosctl`.

---
Owner: wizardbeard
Last updated: 2026-02-08
