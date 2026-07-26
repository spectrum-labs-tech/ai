package otium

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/spectrum-labs-tech/ai"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func jsonResp(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// newTestDriver builds a driver whose HTTP transport is replaced by rt (no network).
func newTestDriver(t *testing.T, model string, rt roundTripFunc) *Driver {
	t.Helper()
	provider, err := New(&ai.Config{APIKey: "test-key", BaseURL: "https://example.invalid/v1", Model: model})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	d := provider.(*Driver)
	d.client.httpClient.Transport = rt
	return d
}

func TestNew_RequiresKeyAndBaseURL(t *testing.T) {
	t.Parallel()
	if _, err := New(&ai.Config{BaseURL: "https://x/v1"}); err == nil {
		t.Fatal("expected error when API key is missing")
	}
	// No base URL is defaulted — the endpoint must be caller-supplied.
	if _, err := New(&ai.Config{APIKey: "k"}); err == nil {
		t.Fatal("expected error when base URL is missing (must not default)")
	}
	p, err := New(&ai.Config{APIKey: "k", BaseURL: "https://x/v1"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if p.ProviderName() != "otium" {
		t.Fatalf("ProviderName = %q, want otium", p.ProviderName())
	}
	if p.ModelName() != ModelOtiumMedium {
		t.Fatalf("default ModelName = %q, want %q", p.ModelName(), ModelOtiumMedium)
	}
}

func TestComplete_Success(t *testing.T) {
	t.Parallel()
	d := newTestDriver(t, ModelOtiumMedium, func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		return jsonResp(`{"id":"c1","model":"otium-medium","choices":[{"index":0,"message":{"role":"assistant","content":"{\"ok\":true}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`), nil
	})

	rec := &ai.UsageRecorder{}
	ctx := ai.WithUsageRecorder(t.Context(), rec)
	out, err := d.Complete(ctx, "system", "user", `{"type":"object"}`, ai.Options{})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if out != `{"ok":true}` {
		t.Fatalf("Complete() = %q", out)
	}
	u := rec.Usage()
	if u == nil || u.TotalTokens != 15 {
		t.Fatalf("recorded usage = %+v", u)
	}
	// Cost is derived from the otium-medium list rate, so it must be positive and non-zero.
	if u.Cost <= 0 {
		t.Fatalf("recorded cost = %v, want > 0", u.Cost)
	}
}

func TestComplete_APIError(t *testing.T) {
	t.Parallel()
	d := newTestDriver(t, ModelOtiumMedium, func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusInternalServerError,
			Body:       io.NopCloser(strings.NewReader(`{"error":"boom"}`)),
		}, nil
	})
	if _, err := d.Complete(t.Context(), "s", "u", "", ai.Options{}); err == nil {
		t.Fatal("expected an error on a 500 response")
	}
}

func TestSubmitBatch(t *testing.T) {
	t.Parallel()
	d := newTestDriver(t, ModelOtiumMedium, func(r *http.Request) (*http.Response, error) {
		switch r.URL.Path {
		case "/v1/files":
			body, _ := io.ReadAll(r.Body)
			_ = r.Body.Close()
			if !strings.Contains(string(body), "req-1") {
				t.Fatalf("multipart body missing custom id: %s", body)
			}
			return jsonResp(`{"id":"file-batch","object":"file","bytes":10,"created_at":1,"filename":"otium-batch.jsonl","purpose":"batch"}`), nil
		case "/v1/batches":
			return jsonResp(`{"id":"batch-1","object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-batch","completion_window":"24h","status":"validating","created_at":1,"request_counts":{"total":1,"completed":0,"failed":0}}`), nil
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
			return nil, nil
		}
	})

	job, err := d.SubmitBatch(t.Context(), []ai.BatchRequest{{CustomID: "req-1", SystemPrompt: "s", UserPrompt: "u", JSONSchema: `{"type":"object"}`}}, ai.BatchOptions{})
	if err != nil {
		t.Fatalf("SubmitBatch() error = %v", err)
	}
	if job.ID != "batch-1" || job.InputFileID != "file-batch" || job.Provider != "otium" {
		t.Fatalf("SubmitBatch() job = %+v", job)
	}
}

func TestSubmitBatch_DuplicateCustomID(t *testing.T) {
	t.Parallel()
	d := newTestDriver(t, ModelOtiumMedium, func(_ *http.Request) (*http.Response, error) {
		t.Fatal("should not hit the network on a validation error")
		return nil, nil
	})
	_, err := d.SubmitBatch(t.Context(), []ai.BatchRequest{
		{CustomID: "dup", UserPrompt: "a"},
		{CustomID: "dup", UserPrompt: "b"},
	}, ai.BatchOptions{})
	if err == nil {
		t.Fatal("expected an error for a duplicate custom_id")
	}
}

func TestGetBatchResults(t *testing.T) {
	t.Parallel()
	d := newTestDriver(t, ModelOtiumMedium, func(r *http.Request) (*http.Response, error) {
		switch r.URL.Path {
		case "/v1/batches/batch-1":
			return jsonResp(`{"id":"batch-1","object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-batch","output_file_id":"file-out","completion_window":"24h","status":"completed","created_at":1,"request_counts":{"total":1,"completed":1,"failed":0}}`), nil
		case "/v1/files/file-out/content":
			return jsonResp("{\"id\":\"r1\",\"custom_id\":\"req-1\",\"response\":{\"status_code\":200,\"request_id\":\"rq-1\",\"body\":{\"model\":\"otium-medium\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"{\\\"ok\\\":true}\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}},\"error\":null}\n"), nil
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
			return nil, nil
		}
	})

	results, err := d.GetBatchResults(t.Context(), "batch-1")
	if err != nil {
		t.Fatalf("GetBatchResults() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	got := results[0]
	if got.CustomID != "req-1" || got.Output != `{"ok":true}` || got.RequestID != "rq-1" {
		t.Fatalf("result = %+v", got)
	}
	if got.Usage == nil || got.Usage.TotalTokens != 15 || got.Usage.Cost <= 0 {
		t.Fatalf("usage = %+v", got.Usage)
	}
}

func TestCostPerMTokens(t *testing.T) {
	t.Parallel()
	// Known tiers priced; unknown returns 0,0,0 ("not available").
	if in, _, out := CostPerMTokens(ModelOtiumMedium); in <= 0 || out <= 0 {
		t.Fatalf("medium rate = %v/%v, want positive", in, out)
	}
	if in, cached, out := CostPerMTokens("otium-unknown"); in != 0 || cached != 0 || out != 0 {
		t.Fatalf("unknown tier rate = %v/%v/%v, want 0,0,0", in, cached, out)
	}
}
