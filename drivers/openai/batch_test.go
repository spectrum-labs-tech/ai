package openai

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/spectrum-labs-tech/ai"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestSubmitBatch(t *testing.T) {
	t.Parallel()

	provider, err := New(&ai.Config{
		APIKey:  "test-key",
		BaseURL: "https://example.invalid/v1",
		Model:   "gpt-4o-mini",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	driver := provider.(*Driver)
	driver.client.httpClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch r.URL.Path {
		case "/v1/files":
			body, _ := io.ReadAll(r.Body)
			_ = r.Body.Close()
			if !strings.Contains(string(body), "purpose") || !strings.Contains(string(body), "batch") {
				t.Fatalf("multipart body missing purpose=batch: %s", string(body))
			}
			if !strings.Contains(string(body), "req-1") {
				t.Fatalf("multipart body missing custom id: %s", string(body))
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"id":"file-batch","object":"file","bytes":123,"created_at":1,"filename":"openai-batch.jsonl","purpose":"batch"}`)),
			}, nil
		case "/v1/batches":
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"id":"batch-123","object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-batch","completion_window":"24h","status":"validating","created_at":1,"request_counts":{"total":1,"completed":0,"failed":0}}`)),
			}, nil
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
			return nil, nil
		}
	})

	job, err := driver.SubmitBatch(t.Context(), []ai.BatchRequest{
		{
			CustomID:     "req-1",
			SystemPrompt: "system",
			UserPrompt:   "user",
			JSONSchema:   `{"type":"object"}`,
		},
	}, ai.BatchOptions{})
	if err != nil {
		t.Fatalf("SubmitBatch() error = %v", err)
	}
	if job.ID != "batch-123" || job.InputFileID != "file-batch" || job.Provider != "openai" {
		t.Fatalf("SubmitBatch() job = %+v", job)
	}
}

func TestGetBatchResults(t *testing.T) {
	t.Parallel()

	provider, err := New(&ai.Config{
		APIKey:  "test-key",
		BaseURL: "https://example.invalid/v1",
		Model:   "gpt-4o-mini",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	driver := provider.(*Driver)
	driver.client.httpClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch r.URL.Path {
		case "/v1/batches/batch-123":
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"id":"batch-123","object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-batch","output_file_id":"file-out","completion_window":"24h","status":"completed","created_at":1,"request_counts":{"total":1,"completed":1,"failed":0}}`)),
			}, nil
		case "/v1/files/file-out/content":
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/jsonl"}},
				Body:       io.NopCloser(strings.NewReader("{\"id\":\"batch_req_1\",\"custom_id\":\"req-1\",\"response\":{\"status_code\":200,\"request_id\":\"req-123\",\"body\":{\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"{\\\"ok\\\":true}\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}},\"error\":null}\n")),
			}, nil
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
			return nil, nil
		}
	})

	results, err := driver.GetBatchResults(t.Context(), "batch-123")
	if err != nil {
		t.Fatalf("GetBatchResults() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].CustomID != "req-1" || results[0].Output != `{"ok":true}` || results[0].RequestID != "req-123" {
		t.Fatalf("result = %+v", results[0])
	}
	if results[0].Usage == nil || results[0].Usage.TotalTokens != 15 {
		t.Fatalf("usage = %+v", results[0].Usage)
	}
}
