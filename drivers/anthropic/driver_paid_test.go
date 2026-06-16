//go:build paid_integration

package anthropic_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/spectrum-labs-tech/ai"
	_ "github.com/spectrum-labs-tech/ai/drivers/anthropic" // register driver
)

// TestComplete_JSONSchema verifies that the Anthropic driver can send a complete
// request with a JSON schema in the system prompt and returns parseable JSON.
//
// Run with:
//
//	ANTHROPIC_API_KEY=sk-ant-... task go:integration-paid
func TestComplete_JSONSchema(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "anthropic",
		APIKey:   apiKey,
		Model:    "claude-haiku-4-5-20251001", // cheapest model for plumbing tests
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	system := `You are a test assistant. Always respond with valid JSON matching the schema.`
	user := `Return a JSON object with a single boolean field "ok" set to true.`
	schema := `{
		"type": "object",
		"required": ["ok"],
		"properties": {
			"ok": { "type": "boolean" }
		}
	}`

	content, err := provider.Complete(ctx(t), system, user, schema, ai.Options{MaxTokens: 256})
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}

	if content == "" {
		t.Fatal("expected non-empty response content")
	}

	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(content), &raw); err != nil {
		t.Fatalf("response is not valid JSON: %v\nContent: %s", err, content)
	}

	t.Logf("response: %s", content)
}

// TestComplete_NoSchema verifies that Complete works when no JSON schema is provided
// (schema parameter is empty string).
func TestComplete_NoSchema(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "anthropic",
		APIKey:   apiKey,
		Model:    "claude-haiku-4-5-20251001",
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	content, err := provider.Complete(ctx(t),
		`You are a test assistant.`,
		`Say only the word "hello".`,
		"", // no schema
		ai.Options{MaxTokens: 16},
	)
	if err != nil {
		t.Fatalf("Complete with no schema failed: %v", err)
	}

	if content == "" {
		t.Fatal("expected non-empty response")
	}

	t.Logf("response: %s", content)
}

// TestBatch_SubmitAndPoll verifies the full batch lifecycle:
// SubmitBatch → GetBatch (poll until ended) → GetBatchResults.
// This confirms that all three BatchProvider methods work end-to-end and
// that result parsing produces at least one non-empty Output.
//
// Note: Anthropic batch processing can take several minutes — ctx has a 5-minute timeout.
func TestBatch_SubmitAndPoll(t *testing.T) {
	// Batch tests take minutes; do not parallelize to avoid unnecessary API cost.
	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "anthropic",
		APIKey:   apiKey,
		Model:    "claude-haiku-4-5-20251001",
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	batchProvider, ok := provider.(ai.BatchProvider)
	if !ok {
		t.Fatal("anthropic driver does not implement ai.BatchProvider")
	}

	requests := []ai.BatchRequest{
		{
			CustomID:     "test-plumbing-1",
			SystemPrompt: `You are a test assistant. Always respond with valid JSON.`,
			UserPrompt:   `Return {"ok": true}.`,
			JSONSchema: `{
				"type": "object",
				"required": ["ok"],
				"properties": {"ok": {"type": "boolean"}}
			}`,
			Options: ai.Options{MaxTokens: 64},
		},
	}

	job, err := batchProvider.SubmitBatch(ctx(t), requests, ai.BatchOptions{})
	if err != nil {
		t.Fatalf("SubmitBatch failed: %v", err)
	}

	if job.ID == "" {
		t.Fatal("SubmitBatch returned empty batch ID")
	}
	if job.Provider != "anthropic" {
		t.Errorf("Provider = %q, want %q", job.Provider, "anthropic")
	}

	t.Logf("submitted batch %s (status=%s)", job.ID, job.Status)

	// Poll until the batch is done, up to the context deadline.
	pollCtx := batchCtx(t)
	var finalJob *ai.BatchJob
	for {
		select {
		case <-pollCtx.Done():
			t.Fatalf("timed out waiting for batch %s to complete (last status=%s)", job.ID, job.Status)
		default:
		}

		finalJob, err = batchProvider.GetBatch(pollCtx, job.ID)
		if err != nil {
			t.Fatalf("GetBatch failed: %v", err)
		}

		t.Logf("batch %s status=%s", finalJob.ID, finalJob.Status)

		if finalJob.Done {
			break
		}

		time.Sleep(15 * time.Second)
	}

	if finalJob.Status != "completed" {
		t.Errorf("expected status %q, got %q", "completed", finalJob.Status)
	}

	results, err := batchProvider.GetBatchResults(pollCtx, finalJob.ID)
	if err != nil {
		t.Fatalf("GetBatchResults failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("GetBatchResults returned no results")
	}

	for _, r := range results {
		t.Logf("result custom_id=%s output=%q error=%q", r.CustomID, r.Output, r.Error)
		if r.Error != "" {
			t.Errorf("result %s has unexpected error: %s", r.CustomID, r.Error)
		}
		if r.Output == "" {
			t.Errorf("result %s has empty output", r.CustomID)
		}
	}
}

// TestBatch_DuplicateCustomID verifies that submitting a batch with duplicate
// custom IDs returns an error without making an API call (client-side validation).
func TestBatch_DuplicateCustomID(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "anthropic",
		APIKey:   apiKey,
		Model:    "claude-haiku-4-5-20251001",
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	batchProvider := provider.(ai.BatchProvider)

	requests := []ai.BatchRequest{
		{CustomID: "dup-1", SystemPrompt: "test", UserPrompt: "test", Options: ai.Options{MaxTokens: 16}},
		{CustomID: "dup-1", SystemPrompt: "test", UserPrompt: "test", Options: ai.Options{MaxTokens: 16}},
	}

	_, err = batchProvider.SubmitBatch(ctx(t), requests, ai.BatchOptions{})
	if err == nil {
		t.Fatal("expected error for duplicate custom_id, got nil")
	}

	t.Logf("got expected error: %v", err)
}

// TestBatch_EmptyRequests verifies that submitting an empty batch returns an error.
func TestBatch_EmptyRequests(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "anthropic",
		APIKey:   apiKey,
		Model:    "claude-haiku-4-5-20251001",
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	batchProvider := provider.(ai.BatchProvider)

	_, err = batchProvider.SubmitBatch(ctx(t), nil, ai.BatchOptions{})
	if err == nil {
		t.Fatal("expected error for empty batch requests, got nil")
	}

	t.Logf("got expected error: %v", err)
}

// TestRawAPIResponse makes a direct HTTP call and logs the full response body.
// Diagnostic helper — skip unless explicitly needed.
func TestRawAPIResponse(t *testing.T) {
	t.Skip("diagnostic only — run manually when debugging raw API responses")

	apiKey := mustAPIKey(t)

	reqBody := map[string]interface{}{
		"model":      "claude-haiku-4-5-20251001",
		"max_tokens": 64,
		"system":     "You are a test assistant. Always respond with valid JSON.",
		"messages": []map[string]string{
			{"role": "user", "content": `Return {"ok": true}.`},
		},
	}

	bodyBytes, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx(t), http.MethodPost, "https://api.anthropic.com/v1/messages", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("HTTP request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	rawBody, _ := io.ReadAll(resp.Body)
	t.Logf("HTTP status: %d", resp.StatusCode)
	t.Logf("Raw response body:\n%s", string(rawBody))

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d\nBody: %s", resp.StatusCode, string(rawBody))
	}
}

func mustAPIKey(t *testing.T) string {
	t.Helper()
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Fatal("ANTHROPIC_API_KEY env var is required for paid integration tests")
	}
	return apiKey
}

// ctx returns a context with a 60-second timeout for single-call tests.
func ctx(t *testing.T) context.Context {
	t.Helper()
	c, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	t.Cleanup(cancel)
	return c
}

// batchCtx returns a context with a 5-minute timeout for batch polling tests.
func batchCtx(t *testing.T) context.Context {
	t.Helper()
	c, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	t.Cleanup(cancel)

	// Print a friendly reminder so it's obvious when this is the bottleneck.
	t.Log("batch context: 5 minute timeout (Anthropic batch processing can take several minutes)")

	return c
}

// TestMapStatus_DriverLevel is a compile-time check that the driver is importable
// and registers itself. It validates the driver name without making any API call.
func TestDriverRegistered(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	p, err := ai.New(&ai.Config{
		Provider: "anthropic",
		APIKey:   apiKey,
		Model:    "claude-haiku-4-5-20251001",
	})
	if err != nil {
		t.Fatalf("ai.New: %v", err)
	}
	defer func() { _ = p.Close() }()

	if name := p.ProviderName(); name != "anthropic" {
		t.Errorf("ProviderName() = %q, want %q", name, "anthropic")
	}

	if model := p.ModelName(); model != "claude-haiku-4-5-20251001" {
		t.Errorf("ModelName() = %q, want %q", model, "claude-haiku-4-5-20251001")
	}

	if _, ok := p.(ai.BatchProvider); !ok {
		t.Error("driver does not implement ai.BatchProvider")
	}

	t.Logf("driver registered: provider=%s model=%s implements_batch=true", p.ProviderName(), p.ModelName())
}
