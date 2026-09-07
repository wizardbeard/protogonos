# protogonos

`protogonos` is a Go rewrite of DXNN2, an Erlang TWEANN system.

The CLI binary is `protogonosctl`.

## Status

Current-scope DXNN2 parity is declared for the supported runtime, evolution loop, scapes, exoself tuning, substrate runtime, storage, benchmarks, and data extraction.

See [release-current-scope-parity.md](docs/release-current-scope-parity.md) for the exact scope and exclusions.

## Dependencies

- Go `1.24.0` or newer.
- Standard Go toolchain commands: `go test`, `go build`, and `gofmt`.
- SQLite support is built with the `sqlite` build tag.
- SQLite uses the pure-Go `modernc.org/sqlite` dependency from `go.mod`.
- Bash is needed for helper scripts such as [done-check.md](docs/done-check.md).

Main Go module dependencies are listed in [go.mod](go.mod).

## Build

Build the default CLI:

```bash
go build -o protogonosctl ./cmd/protogonosctl
```

Build the SQLite-enabled CLI:

```bash
go build -tags sqlite -o protogonosctl ./cmd/protogonosctl
```

Run tests:

```bash
go test ./...
```

Run SQLite-tagged tests:

```bash
go test -tags sqlite ./...
```

## Run

Initialize a local SQLite store:

```bash
protogonosctl init --store sqlite --db-path ./protogonos.db
```

Run a small XOR evolution job:

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape xor \
  --pop 50 \
  --gens 100 \
  --seed 1
```

Run Flatland:

```bash
protogonosctl run \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 50 \
  --gens 100 \
  --seed 1
```

Run a benchmark:

```bash
protogonosctl benchmark \
  --store sqlite \
  --db-path ./protogonos.db \
  --scape flatland \
  --pop 50 \
  --gens 100 \
  --seed 1 \
  --min-improvement -0.2
```

Export the latest run artifacts:

```bash
protogonosctl export --latest
```

Artifacts are written under `benchmarks/` and `exports/`.

## Useful Docs

- [Flatland implementation](docs/flatland-implementation.md)
- [Custom scape guide](docs/custom-scape-guide.md)
- [Execution model](docs/execution-model.md)
- [Done check](docs/done-check.md)
- [API stability](docs/api-stability.md)
- [DXNN2 module mapping](docs/dxnn2-module-mapping.md)
- [DXNN2 source module audit](docs/dxnn2-src-module-audit.md)
- [Full parity checklist](docs/dxnn2-full-parity-checklist.md)

## Release Gate

Use the local parity gate before release work:

```bash
./scripts/done_check.sh
```

This script is intentionally not run by GitHub Actions. It is heavier than normal CI due to its tests, benchmarks, exports, and parity-doc checks.
