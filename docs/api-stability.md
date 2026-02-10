# API Stability Policy (`pkg/protogonos`)

This project treats `pkg/protogonos` as the public, downstream-facing API surface.

## Stability level

- Current level: `baseline-stable`.
- Breaking changes are allowed only with an explicit changelog entry and version bump.

## Compatibility rules

- Additive changes are preferred:
  - adding request fields with sane defaults,
  - adding response fields without changing existing meanings.
- Behavioral changes that alter defaults must be gated behind explicit request options.
- Removed fields, renamed fields, and semantic re-interpretation are breaking.

## Change process

1. Update `pkg/protogonos/api.go` and `pkg/protogonos/api_test.go`.
2. Update this document if compatibility expectations change.
3. Add release note entry summarizing compatibility impact.
4. Run:
   - `go test ./...`
   - `go test -tags sqlite ./...`

## Current guaranteed methods

- `New`, `Close`
- `Init`, `Start`
- `Run`, `Runs`, `Export`

These methods are expected to remain available with compatible behavior across baseline releases.
