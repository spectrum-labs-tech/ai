//go:build paid_integration

package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"

	"github.com/spectrum-labs-tech/ai"
	_ "github.com/spectrum-labs-tech/ai/drivers/openai" // register driver
)

// TestComplete_ParameterCompatibility verifies that the OpenAI driver sends the
// correct request parameters for the production model (gpt-5-nano):
//   - Uses max_completion_tokens (not the legacy max_tokens)
//   - Omits temperature when not set (gpt-5-nano only accepts the default)
//
// This catches API-breaking changes before they reach production.
//
// Run with:
//
//	OPENAI_API_KEY=sk-... task go:integration-paid
func TestComplete_ParameterCompatibility(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY env var is required for paid integration tests")
	}

	provider, err := ai.New(&ai.Config{
		Provider: "openai",
		APIKey:   apiKey,
		Model:    "gpt-5-nano",
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	// Minimal system + user prompt. Must contain "json" for json_object response_format.
	system := `You are a test assistant. Always respond with valid JSON.`
	user := `Return a JSON object with a single boolean field "ok" set to true.`
	schema := `{
		"type": "object",
		"required": ["ok"],
		"properties": {
			"ok": { "type": "boolean" }
		}
	}`

	// Use MaxTokens only — no Temperature — matching the production duplicate-review call.
	content, err := provider.Complete(ctx(t), system, user, schema, ai.Options{MaxTokens: 500})
	if err != nil {
		t.Fatalf("Complete failed: %v\n\nThis likely means a request parameter was rejected by the model.\nCheck that max_completion_tokens is used (not max_tokens) and that temperature is omitted.", err)
	}

	if content == "" {
		t.Fatal("expected non-empty response content")
	}

	// Verify the response is valid JSON.
	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(content), &raw); err != nil {
		t.Fatalf("response is not valid JSON: %v\nContent: %s", err, content)
	}

	t.Logf("response: %s", content)
}

// TestComplete_TemperatureDefault verifies that calling Complete without a
// Temperature in Options does not trigger a model rejection (gpt-5-nano only
// accepts the default temperature of 1).
func TestComplete_TemperatureDefault(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY env var is required for paid integration tests")
	}

	provider, err := ai.New(&ai.Config{
		Provider: "openai",
		APIKey:   apiKey,
		Model:    "gpt-5-nano",
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	// Nil temperature must not be sent as a non-default value.
	opts := ai.Options{MaxTokens: 500} // Temperature is nil (zero value)
	content, err := provider.Complete(ctx(t),
		`You are a test assistant. Always respond with valid JSON.`,
		`Return a JSON object with a single string field "answer" set to "yes".`,
		`{"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}}}`,
		opts,
	)
	if err != nil {
		t.Fatalf("Complete with nil temperature failed: %v\n\nIf the error mentions temperature, the driver is sending a non-default value for a model that does not support it.", err)
	}

	if content == "" {
		t.Fatal("expected non-empty response")
	}

	t.Logf("response: %s", content)
}

// TestRawAPIResponse makes a direct HTTP call and logs the full response body.
// It is a diagnostic helper, not a standard CI test — skip unless explicitly needed.
func TestRawAPIResponse(t *testing.T) {
	t.Skip("diagnostic only — run manually when debugging raw API responses")
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY env var is required for paid integration tests")
	}

	reqBody := map[string]interface{}{
		"model": "gpt-5-nano",
		"messages": []map[string]string{
			{"role": "system", "content": "You are a test assistant. Always respond with valid JSON."},
			{"role": "user", "content": `Return a JSON object with a single boolean field "ok" set to true.`},
		},
		"response_format":       map[string]string{"type": "json_object"},
		"max_completion_tokens": 200,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx(t), "POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("HTTP request failed: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	rawBody, _ := io.ReadAll(resp.Body)
	t.Logf("HTTP status: %d", resp.StatusCode)
	t.Logf("Raw response body:\n%s", string(rawBody))

	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(rawBody, &parsed); err != nil {
		t.Fatalf("response not valid JSON: %v", err)
	}
}

func ctx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 60_000_000_000) // 60s
	t.Cleanup(cancel)
	return ctx
}
