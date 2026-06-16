package gemini

import (
	"encoding/json"
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

func TestNew_DefaultModel(t *testing.T) {
	t.Parallel()

	provider, err := New(&ai.Config{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	if provider.ModelName() != defaultModel {
		t.Fatalf("ModelName() = %q, want %q", provider.ModelName(), defaultModel)
	}
	if provider.ProviderName() != "gemini" {
		t.Fatalf("ProviderName() = %q, want %q", provider.ProviderName(), "gemini")
	}
}

func TestDriverComplete(t *testing.T) {
	t.Parallel()

	var gotReq GenerateContentRequest
	provider, err := New(&ai.Config{
		APIKey:  "test-key",
		BaseURL: "https://example.invalid/v1beta",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	driver := provider.(*Driver)
	driver.client.httpClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1beta/models/"+defaultModel+":generateContent" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		if got := r.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Fatalf("x-goog-api-key = %q, want %q", got, "test-key")
		}

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read request body: %v", err)
		}
		_ = r.Body.Close()
		if err := json.Unmarshal(bodyBytes, &gotReq); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(`{
			"candidates": [
				{
					"content": {
						"parts": [
							{"text": "{\"name\":\"example\"}"}
						]
					},
					"finishReason": "STOP"
				}
			],
			"usageMetadata": {
				"promptTokenCount": 120,
				"candidatesTokenCount": 30,
				"totalTokenCount": 150,
				"cachedContentTokenCount": 20
			}
		}`)),
		}, nil
	})

	recorder := &ai.UsageRecorder{}
	ctx := ai.WithUsageRecorder(t.Context(), recorder)
	out, err := provider.Complete(ctx, "system rules", "user prompt", `{"type":"object"}`, ai.Options{MaxTokens: 222})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}

	if out != `{"name":"example"}` {
		t.Fatalf("Complete() = %q", out)
	}
	if gotReq.SystemInstruction == nil || len(gotReq.SystemInstruction.Parts) != 1 || gotReq.SystemInstruction.Parts[0].Text != "system rules" {
		t.Fatalf("system instruction not forwarded: %+v", gotReq.SystemInstruction)
	}
	if len(gotReq.Contents) != 1 || gotReq.Contents[0].Parts[0].Text != "user prompt" {
		t.Fatalf("contents not forwarded: %+v", gotReq.Contents)
	}
	if gotReq.GenerationConfig == nil {
		t.Fatal("GenerationConfig = nil")
	}
	if gotReq.GenerationConfig.ResponseMIMEType != "application/json" {
		t.Fatalf("ResponseMIMEType = %q, want application/json", gotReq.GenerationConfig.ResponseMIMEType)
	}
	if gotReq.GenerationConfig.MaxOutputTokens != 222 {
		t.Fatalf("MaxOutputTokens = %d, want 222", gotReq.GenerationConfig.MaxOutputTokens)
	}
	if string(gotReq.GenerationConfig.ResponseJSONSchema) != `{"type":"object"}` {
		t.Fatalf("ResponseJSONSchema = %s", string(gotReq.GenerationConfig.ResponseJSONSchema))
	}

	usage := recorder.Usage()
	if usage == nil {
		t.Fatal("usage recorder was not populated")
		return
	}
	if usage.PromptTokens != 120 || usage.CompletionTokens != 30 || usage.TotalTokens != 150 || usage.CachedTokens != 20 {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestDriverCompleteAPIError(t *testing.T) {
	t.Parallel()

	provider, err := New(&ai.Config{
		APIKey:  "bad-key",
		BaseURL: "https://example.invalid/v1beta",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	driver := provider.(*Driver)
	driver.client.httpClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusUnauthorized,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"error":{"code":401,"message":"bad key","status":"UNAUTHENTICATED"}}`)),
		}, nil
	})

	_, err = provider.Complete(t.Context(), "system", "user", `{"type":"object"}`, ai.Options{})
	if err == nil {
		t.Fatal("Complete() error = nil, want error")
	}
}

func TestSubmitBatchInlineAndGetResults(t *testing.T) {
	t.Parallel()

	provider, err := New(&ai.Config{
		APIKey:  "test-key",
		BaseURL: "https://example.invalid/v1beta",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	driver := provider.(*Driver)
	driver.client.httpClient.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch r.URL.Path {
		case "/v1beta/models/" + defaultModel + ":batchGenerateContent":
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(`{
					"name":"batches/123",
					"done":false,
					"metadata":{"state":"JOB_STATE_PENDING"}
				}`)),
			}, nil
		case "/v1beta/batches/123":
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(`{
					"name":"batches/123",
					"done":true,
					"metadata":{"state":"JOB_STATE_SUCCEEDED"},
					"response":{"inlinedResponses":[
						{
							"key":"req-1",
							"response":{
								"candidates":[{"content":{"parts":[{"text":"{\"name\":\"batch\"}"}]}}],
								"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":3,"totalTokenCount":11}
							}
						}
					]}
				}`)),
			}, nil
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
			return nil, nil
		}
	})

	job, err := driver.SubmitBatch(t.Context(), []ai.BatchRequest{
		{CustomID: "req-1", SystemPrompt: "system", UserPrompt: "user", JSONSchema: `{"type":"object"}`},
	}, ai.BatchOptions{DisplayName: "refresh"})
	if err != nil {
		t.Fatalf("SubmitBatch() error = %v", err)
	}
	if job.ID != "batches/123" || job.Provider != "gemini" {
		t.Fatalf("job = %+v", job)
	}

	results, err := driver.GetBatchResults(t.Context(), "batches/123")
	if err != nil {
		t.Fatalf("GetBatchResults() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}
	if results[0].CustomID != "req-1" || results[0].Output != `{"name":"batch"}` {
		t.Fatalf("result = %+v", results[0])
	}
}
