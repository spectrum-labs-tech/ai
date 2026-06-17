//go:build paid_integration

package gemini_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/spectrum-labs-tech/ai"
	"github.com/spectrum-labs-tech/ai/drivers/gemini"
)

// TestComplete_JSONSchema verifies that the Gemini driver can send a complete
// request with a JSON schema and returns parseable JSON.
func TestComplete_JSONSchema(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "gemini",
		APIKey:   apiKey,
		Model:    gemini.ModelGemini31FlashLite,
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
		"additionalProperties": false,
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

// TestComplete_NoSchema verifies that Complete works when no JSON schema is provided.
func TestComplete_NoSchema(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "gemini",
		APIKey:   apiKey,
		Model:    gemini.ModelGemini31FlashLite,
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
func TestBatch_SubmitAndPoll(t *testing.T) {
	// Batch tests take minutes; do not parallelize to avoid unnecessary API cost.
	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "gemini",
		APIKey:   apiKey,
		Model:    gemini.ModelGemini31FlashLite,
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	batchProvider, ok := provider.(ai.BatchProvider)
	if !ok {
		t.Fatal("gemini driver does not implement ai.BatchProvider")
	}

	requests := []ai.BatchRequest{
		{
			CustomID:     "test-plumbing-1",
			SystemPrompt: `You are a test assistant. Always respond with valid JSON.`,
			UserPrompt:   `Return {"ok": true}.`,
			JSONSchema: `{
				"type": "object",
				"required": ["ok"],
				"additionalProperties": false,
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
	if job.Provider != "gemini" {
		t.Errorf("Provider = %q, want %q", job.Provider, "gemini")
	}

	t.Logf("submitted batch %s (status=%s)", job.ID, job.Status)

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

		time.Sleep(30 * time.Second)
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

// TestBatch_DuplicateCustomID verifies client-side duplicate custom_id validation.
func TestBatch_DuplicateCustomID(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	provider, err := ai.New(&ai.Config{
		Provider: "gemini",
		APIKey:   apiKey,
		Model:    gemini.ModelGemini31FlashLite,
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
		Provider: "gemini",
		APIKey:   apiKey,
		Model:    gemini.ModelGemini31FlashLite,
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

// TestDriverRegistered verifies the driver registers correctly and reports its name.
func TestDriverRegistered(t *testing.T) {
	t.Parallel()

	apiKey := mustAPIKey(t)

	p, err := ai.New(&ai.Config{
		Provider: "gemini",
		APIKey:   apiKey,
		Model:    gemini.ModelGemini31FlashLite,
	})
	if err != nil {
		t.Fatalf("ai.New: %v", err)
	}
	defer func() { _ = p.Close() }()

	if name := p.ProviderName(); name != "gemini" {
		t.Errorf("ProviderName() = %q, want %q", name, "gemini")
	}

	if model := p.ModelName(); model != gemini.ModelGemini31FlashLite {
		t.Errorf("ModelName() = %q, want %q", model, gemini.ModelGemini31FlashLite)
	}

	if _, ok := p.(ai.BatchProvider); !ok {
		t.Error("driver does not implement ai.BatchProvider")
	}

	t.Logf("driver registered: provider=%s model=%s implements_batch=true", p.ProviderName(), p.ModelName())
}

func mustAPIKey(t *testing.T) string {
	t.Helper()
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Fatal("GEMINI_API_KEY env var is required for paid integration tests")
	}
	return apiKey
}

func ctx(t *testing.T) context.Context {
	t.Helper()
	c, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	t.Cleanup(cancel)
	return c
}

func batchCtx(t *testing.T) context.Context {
	t.Helper()
	c, cancel := context.WithTimeout(t.Context(), 15*time.Minute)
	t.Cleanup(cancel)
	t.Log("batch context: 15 minute timeout (Gemini batch processing can take several minutes)")
	return c
}
