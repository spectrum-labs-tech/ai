# github.com/spectrum-labs-tech/ai

A minimal Go library for structured AI completions. Uses a `database/sql`-style driver registry so you import only the providers you need.

## Install

```
go get github.com/spectrum-labs-tech/ai
```

Import the driver(s) you want alongside the root package:

```go
import (
    "github.com/spectrum-labs-tech/ai"
    _ "github.com/spectrum-labs-tech/ai/drivers/openai"
    // _ "github.com/spectrum-labs-tech/ai/drivers/anthropic"
    // _ "github.com/spectrum-labs-tech/ai/drivers/gemini"
)
```

## Quick start

```go
p, err := ai.New(&ai.Config{
    Provider: "openai",
    APIKey:   os.Getenv("OPENAI_API_KEY"),
    Model:    "gpt-4o-mini",
})
if err != nil {
    log.Fatal(err)
}
defer p.Close()

const schema = `{
    "type": "object",
    "properties": {
        "summary": {"type": "string"}
    },
    "required": ["summary"],
    "additionalProperties": false
}`

result, err := p.Complete(ctx, "You are a summarizer.", "Summarize: "+text, schema, ai.Options{})
```

## Structured output

Every `Complete` call accepts a JSON Schema string. The provider enforces it via its native structured-output API (OpenAI response format, Anthropic tool use, Gemini controlled generation). The return value is always a raw JSON string matching your schema.

## Token usage

Wrap your context with a `UsageRecorder` to capture tokens and estimated cost:

```go
rec := &ai.UsageRecorder{}
ctx = ai.WithUsageRecorder(ctx, rec)

result, err := p.Complete(ctx, system, user, schema, ai.Options{})

if u := rec.Usage(); u != nil {
    fmt.Printf("tokens: %d prompt / %d completion, cost: $%.6f\n",
        u.PromptTokens, u.CompletionTokens, u.Cost)
}
```

## Async batch

Drivers implement `BatchProvider` for provider-managed async batches (typically ~50 % cheaper):

```go
bp := p.(ai.BatchProvider)

job, err := bp.SubmitBatch(ctx, []ai.BatchRequest{
    {CustomID: "row-1", SystemPrompt: system, UserPrompt: user1, JSONSchema: schema},
    {CustomID: "row-2", SystemPrompt: system, UserPrompt: user2, JSONSchema: schema},
}, ai.BatchOptions{})

// Poll until job.Done, then:
results, err := bp.GetBatchResults(ctx, job.ID)
```

## Agent / tool-calling loop

Drivers that implement `AgentProvider` run a multi-turn tool loop:

```go
ap := p.(ai.AgentProvider)

result, err := ap.CompleteWithTools(ctx, system, user, schema,
    []ai.ToolDefinition{{Name: "lookup", Description: "...", Parameters: paramsSchema}},
    ai.Options{},
    func(ctx context.Context, calls []ai.ToolCallRequest) ([]ai.ToolCallResult, error) {
        // execute each tool call and return results
    },
)
```

## Drivers

| Import path | Provider |
|---|---|
| `drivers/openai` | OpenAI (GPT-4o, o-series) |
| `drivers/anthropic` | Anthropic (Claude) |
| `drivers/gemini` | Google Gemini |

## Dynamic factory

`NewProviderFactory` returns a `ProviderFactory` that constructs a `BatchProvider` on demand, useful when different tasks use different models:

```go
factory := ai.NewProviderFactory("openai", os.Getenv("OPENAI_API_KEY"))

bp, err := factory("gpt-4.1-mini")
```

## Testing

Unit tests require no API keys:

```
go test ./...
```

Paid integration tests hit live APIs and require the relevant `*_API_KEY` environment variable:

```
go test -tags=paid_integration ./...
```

## License

MIT — see [LICENSE](LICENSE).
