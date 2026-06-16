# Contributing

## Prerequisites

- Go 1.22+
- API keys only needed for paid integration tests (not required for a normal PR)

## Running tests

Unit tests (no network, no keys):

```
go test ./...
```

Paid integration tests (requires `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`):

```
go test -tags=paid_integration ./...
```

## Adding a driver

1. Create `drivers/<name>/` with at least `driver.go` and `client.go`.
2. Implement `ai.Provider`. Implement `ai.BatchProvider` and `ai.AgentProvider` if the provider supports it.
3. Register in an `init()`:
   ```go
   func init() {
       ai.Register("<name>", New)
   }
   ```
4. Add unit tests in `driver_test.go` (no build tag). Use `net/http/httptest` to mock the API.
5. Add paid integration tests in `driver_paid_test.go` under `//go:build paid_integration`.
6. Record token usage via `ai.UsageRecorderFromContext(ctx)` after each successful completion.
7. Update the driver table in `README.md`.

## Code style

- Standard `gofmt` / `goimports` formatting.
- No external dependencies beyond the Go standard library in the root package.
- Driver packages may import only what they need (HTTP client, JSON, etc.). Avoid pulling in large SDK dependencies — a thin HTTP client is preferred.
- All exported types and functions must have a doc comment.

## Pull requests

- Keep each PR focused on one driver or one feature.
- All unit tests must pass: `go test ./...`
- No new linter warnings: `go vet ./...`
- If you're adding a new driver, include at least one unit test covering a successful completion and one covering an API error response.
