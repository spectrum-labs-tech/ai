//go:build paid_integration

package otium_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/spectrum-labs-tech/ai"
	"github.com/spectrum-labs-tech/ai/drivers/otium"
)

// TestComplete_Live hits a real Otium endpoint and verifies a structured completion round-trips.
// Both an API key and the endpoint are required (the driver never defaults the base URL):
//
//	OTIUM_API_KEY=... OTIUM_BASE_URL=https://<your-otium>/v1 go test -tags=paid_integration ./drivers/otium/
func TestComplete_Live(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("OTIUM_API_KEY")
	baseURL := os.Getenv("OTIUM_BASE_URL")
	if apiKey == "" || baseURL == "" {
		t.Fatal("OTIUM_API_KEY and OTIUM_BASE_URL are required for paid integration tests")
	}

	provider, err := ai.New(&ai.Config{
		Provider: "otium",
		APIKey:   apiKey,
		BaseURL:  baseURL,
		Model:    otium.ModelOtiumMedium,
	})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}
	defer func() { _ = provider.Close() }()

	system := `You are a test assistant. Always respond with valid JSON.`
	user := `Return a JSON object with a single boolean field "ok" set to true.`
	schema := `{"type":"object","required":["ok"],"additionalProperties":false,"properties":{"ok":{"type":"boolean"}}}`

	content, err := provider.Complete(ctx(t), system, user, schema, ai.Options{MaxTokens: 200})
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

func ctx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 60_000_000_000) // 60s
	t.Cleanup(cancel)
	return ctx
}
