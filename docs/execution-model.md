# Execution Model

This document describes how `protogonos` maps Erlang actor primitives to Go.

## Short Answer

`protogonos` uses native Go primitives, not a generic actor framework.

The main primitives are:

- goroutines,
- channels,
- typed message structs,
- `context.Context`,
- mutexes for local state protection,
- synchronous method calls where a full mailbox is not needed.

## Erlang To Go Mapping

| Erlang primitive | Go replacement |
|---|---|
| process | goroutine-backed actor, process-state wrapper, or direct object |
| mailbox | typed channel |
| receive loop | goroutine loop over a channel |
| selective receive | pending-message queue plus typed message handling |
| gen_server call | request struct plus reply channel |
| process pid | stable string ID |
| supervisor cancellation | `context.Context` cancellation and explicit termination calls |

## Population Monitor

The population monitor is Go-native.

It does not create one long-lived actor per genome. It evaluates genomes through a bounded worker pool:

- `jobs` channel sends genomes to workers.
- `results` channel returns fitness and trace data.
- `sync.WaitGroup` waits for workers to finish.
- `context.Context` cancels the run.
- `MonitorCommand` handles pause, continue, stop, goal-reached, and print-trace control.

This is a practical replacement for the reference population monitor process. It preserves the main behavior: evaluate candidates, collect fitness, rank, select, replicate, mutate, and repeat.

Key file:

- `internal/evo/population_monitor.go`

## Cortex

The cortex is the per-agent coordinator.

It supports three IO execution modes:

- `direct`: sensors and actuators are plain method calls.
- `process`: sensors and actuators use synchronous process-state wrappers.
- `actor`: sensors and actuators run behind goroutine-backed mailboxes.

The `actor` mode is the closest match to Erlang process behavior. The `direct` mode is faster and simpler for most runs.

Key file:

- `internal/agent/cortex.go`

## Sensor And Actuator Actors

Sensor and actuator actors use this shape:

- one goroutine per actor,
- one mailbox channel,
- one reply channel per synchronous call,
- owner PID checks for init and terminate messages,
- explicit terminated-state errors.

This models the Erlang call/receive pattern without bringing in an external actor library.

Key files:

- `internal/io/sensor_process.go`
- `internal/io/actuator_process.go`

## Substrate CEP Actors

The substrate CEP path has the strongest actor model in the codebase.

CEP actors use:

- an inbox for typed CEP messages,
- an outbox for emitted commands,
- an error mailbox,
- sync-marker messages for deterministic drain points,
- pending-message storage for selective-receive-like behavior,
- owner-scoped init and terminate handling.

This is closer to Erlang actor behavior than the main evolution loop. It exists where mailbox ordering and staged message flow matter.

Key file:

- `internal/substrate/cep_protocol.go`

## Scape Processes

Scape process wrappers model reference-style command APIs.

Examples include:

- `XORProcess`,
- `FlatlandPublicProcess`,
- `EpitopesProcess`,
- `GTSAProcess`,
- `FXProcess`,
- `DTMProcess`,
- `Pole2Process`,
- `LLVMPhaseOrderingProcess`.

Most of these wrappers are synchronous state machines. They preserve command semantics, state transitions, and trace fields. They are not always goroutine-backed actors.

## Design Rule

Use the simplest execution surface that preserves behavior:

- use direct calls for normal fast evaluation,
- use process wrappers for reference-style command surfaces,
- use goroutine-backed actors where mailbox ordering, ownership, and async posting matter.

This keeps the Go implementation testable while preserving the DXNN2 concepts that affect behavior.
