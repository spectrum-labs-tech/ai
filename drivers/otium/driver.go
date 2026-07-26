// Package otium is a driver for Otium, a batch-first inference service that runs open
// models on low-cost compute behind an OpenAI-compatible API. Because the wire format is
// OpenAI-compatible, this driver speaks that protocol against a caller-supplied base URL;
// what it adds over pointing the openai driver at a different URL is correct provider
// attribution ("otium" everywhere the caller keys on it) and Otium's own published rate
// card, so usage and cost are recorded against Otium rather than mislabeled as OpenAI.
//
// It implements ai.Provider (synchronous structured completions) and ai.BatchProvider
// (asynchronous batches — the primary path for large enrichment workloads).
package otium

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/spectrum-labs-tech/ai"
)

// Model aliases. Otium's contract is "declare intent, not a model": a caller names a
// capability tier and Otium picks the concrete open model behind it, so these are the stable,
// customer-facing ids — not underlying model names.
const (
	// ModelOtiumSmall is the small/cheap capability tier.
	ModelOtiumSmall = "otium-small"
	// ModelOtiumMedium is the mid capability tier — the default and launch tier.
	ModelOtiumMedium = "otium-medium"
	// ModelOtiumLarge is the large/high-quality capability tier.
	ModelOtiumLarge = "otium-large"
)

func init() {
	ai.Register("otium", New)
}

// Driver implements ai.Provider and ai.BatchProvider for Otium.
type Driver struct {
	client *Client
	model  string
}

// New creates a new Otium driver. Both an API key and a base URL are required — the base URL
// is never defaulted, so the endpoint is always caller-supplied configuration (e.g. from an
// environment variable), never embedded here.
func New(cfg *ai.Config) (ai.Provider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("otium: API key is required")
	}
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("otium: base URL is required (set Config.BaseURL to your Otium endpoint)")
	}
	model := cfg.Model
	if model == "" {
		model = ModelOtiumMedium
	}
	return &Driver{
		client: NewClient(cfg.APIKey, cfg.BaseURL, cfg.MaxRetries),
		model:  model,
	}, nil
}

// Complete sends a chat completion request and returns the raw JSON content string.
func (d *Driver) Complete(ctx context.Context, systemPrompt, userPrompt, jsonSchema string, opts ai.Options) (string, error) {
	resp, err := d.client.CreateChatCompletion(ctx, d.buildChatCompletionRequest(systemPrompt, userPrompt, jsonSchema, opts))
	if err != nil {
		return "", fmt.Errorf("otium API call failed: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("otium: no choices returned")
	}
	d.recordUsage(ctx, resp.Usage, d.model)
	content, _ := resp.Choices[0].Message.Content.(string)
	return content, nil
}

// ProviderName returns "otium".
func (d *Driver) ProviderName() string { return "otium" }

// ModelName returns the capability tier / model being used.
func (d *Driver) ModelName() string { return d.model }

// Close releases resources (no-op for the HTTP client).
func (d *Driver) Close() error { return nil }

// SubmitBatch submits a file-backed batch of structured chat completions.
func (d *Driver) SubmitBatch(ctx context.Context, requests []ai.BatchRequest, opts ai.BatchOptions) (*ai.BatchJob, error) {
	if len(requests) == 0 {
		return nil, fmt.Errorf("otium: batch requires at least one request")
	}

	var payload bytes.Buffer
	enc := json.NewEncoder(&payload)
	seen := make(map[string]struct{}, len(requests))
	for i, req := range requests {
		if strings.TrimSpace(req.CustomID) == "" {
			return nil, fmt.Errorf("otium: batch request %d missing custom_id", i)
		}
		if _, dup := seen[req.CustomID]; dup {
			return nil, fmt.Errorf("otium: duplicate custom_id %q", req.CustomID)
		}
		seen[req.CustomID] = struct{}{}

		bodyBytes, err := json.Marshal(d.buildChatCompletionRequest(req.SystemPrompt, req.UserPrompt, req.JSONSchema, req.Options))
		if err != nil {
			return nil, fmt.Errorf("otium: marshal batch request %q: %w", req.CustomID, err)
		}
		var body map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &body); err != nil {
			return nil, fmt.Errorf("otium: decode batch request %q: %w", req.CustomID, err)
		}
		if err := enc.Encode(BatchRequestLine{
			CustomID: req.CustomID,
			Method:   "POST",
			URL:      "/v1/chat/completions",
			Body:     body,
		}); err != nil {
			return nil, fmt.Errorf("otium: encode batch line %q: %w", req.CustomID, err)
		}
	}

	file, err := d.client.UploadFile(ctx, "otium-batch.jsonl", payload.Bytes(), "batch")
	if err != nil {
		return nil, fmt.Errorf("otium: upload batch input file: %w", err)
	}

	window := opts.CompletionWindow
	if window == "" {
		window = "24h"
	}
	batch, err := d.client.CreateBatch(ctx, &CreateBatchRequest{
		InputFileID:      file.ID,
		Endpoint:         "/v1/chat/completions",
		CompletionWindow: window,
		Metadata:         opts.Metadata,
	})
	if err != nil {
		return nil, fmt.Errorf("otium: create batch: %w", err)
	}
	job := mapBatchObject(batch, d.model)
	job.InputFileID = file.ID
	return job, nil
}

// GetBatch retrieves one batch job's status.
func (d *Driver) GetBatch(ctx context.Context, batchID string) (*ai.BatchJob, error) {
	batch, err := d.client.GetBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("otium: get batch: %w", err)
	}
	return mapBatchObject(batch, d.model), nil
}

// CancelBatch attempts to cancel one in-flight batch job.
func (d *Driver) CancelBatch(ctx context.Context, batchID string) (*ai.BatchJob, error) {
	batch, err := d.client.CancelBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("otium: cancel batch: %w", err)
	}
	return mapBatchObject(batch, d.model), nil
}

// GetBatchResults downloads and parses the available output and error files for a batch.
func (d *Driver) GetBatchResults(ctx context.Context, batchID string) ([]ai.BatchResult, error) {
	batch, err := d.client.GetBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("otium: get batch: %w", err)
	}

	var results []ai.BatchResult
	if batch.OutputFileID != "" {
		body, err := d.client.DownloadFileContent(ctx, batch.OutputFileID)
		if err != nil {
			if isNotFound(err) {
				return nil, fmt.Errorf("otium: output file %s: %w", batch.OutputFileID, ai.ErrBatchOutputExpired)
			}
			return nil, fmt.Errorf("otium: download batch output file: %w", err)
		}
		parsed, err := parseBatchResults(body)
		if err != nil {
			return nil, err
		}
		results = append(results, parsed...)
	}
	if batch.ErrorFileID != "" {
		body, err := d.client.DownloadFileContent(ctx, batch.ErrorFileID)
		if err != nil {
			if isNotFound(err) {
				return results, nil // output was present; error file gone — return what we have
			}
			return nil, fmt.Errorf("otium: download batch error file: %w", err)
		}
		parsed, err := parseBatchResults(body)
		if err != nil {
			return nil, err
		}
		results = append(results, parsed...)
	}
	return results, nil
}

// buildChatCompletionRequest assembles a chat request, folding the JSON schema into the
// system prompt (belt-and-suspenders with response_format) and attaching an image when set.
func (d *Driver) buildChatCompletionRequest(systemPrompt, userPrompt, jsonSchema string, opts ai.Options) *ChatCompletionRequest {
	temperature := 0.0
	if opts.Temperature != nil {
		temperature = *opts.Temperature
	}

	fullSystem := systemPrompt
	if jsonSchema != "" {
		fullSystem += "\n\nJSON schema for your response:\n" + jsonSchema
	}

	var userContent any = userPrompt
	if opts.ImageURL != "" {
		userContent = []MessageContentPart{
			{Type: "text", Text: userPrompt},
			{Type: "image_url", ImageURL: &MessageImageURLRef{URL: opts.ImageURL}},
		}
	}

	req := &ChatCompletionRequest{
		Model: d.model,
		Messages: []Message{
			{Role: "system", Content: fullSystem},
			{Role: "user", Content: userContent},
		},
		Temperature: temperature,
	}
	if strings.TrimSpace(jsonSchema) != "" {
		req.ResponseFormat = &ResponseFormat{
			Type: "json_schema",
			JSONSchema: &ResponseFormatJSONSchema{
				Name:   "response",
				Strict: true,
				Schema: json.RawMessage(jsonSchema),
			},
		}
	} else {
		req.ResponseFormat = &ResponseFormat{Type: "json_object"}
	}
	if opts.MaxTokens > 0 {
		req.MaxTokens = opts.MaxTokens
	}
	return req
}

// recordUsage records token usage + cost for one response via the context UsageRecorder.
func (d *Driver) recordUsage(ctx context.Context, usage Usage, model string) {
	r := ai.UsageRecorderFromContext(ctx)
	if r == nil {
		return
	}
	cached := usage.CachedTokens
	if usage.PromptTokensDetails != nil && usage.PromptTokensDetails.CachedTokens != 0 {
		cached = usage.PromptTokensDetails.CachedTokens
	}
	inputPerM, cachedPerM, outputPerM := CostPerMTokens(model)
	uncached := usage.PromptTokens - cached
	cost := float64(uncached)/1e6*inputPerM +
		float64(cached)/1e6*cachedPerM +
		float64(usage.CompletionTokens)/1e6*outputPerM
	r.Record(ai.CompletionUsage{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
		CachedTokens:     cached,
		Cost:             cost,
	})
}

func mapBatchObject(batch *BatchObject, model string) *ai.BatchJob {
	var raw json.RawMessage
	if b, err := json.Marshal(batch); err == nil {
		raw = b
	}
	job := &ai.BatchJob{
		ID:           batch.ID,
		Provider:     "otium",
		Model:        model,
		Status:       batch.Status,
		InputFileID:  batch.InputFileID,
		OutputFileID: batch.OutputFileID,
		ErrorFileID:  batch.ErrorFileID,
		RequestCounts: ai.BatchRequestCounts{
			Total:     batch.RequestCounts.Total,
			Completed: batch.RequestCounts.Completed,
			Failed:    batch.RequestCounts.Failed,
		},
		Metadata:         batch.Metadata,
		Done:             isTerminalBatchStatus(batch.Status),
		ProviderResponse: raw,
	}
	job.CreatedAt = unixTimePtr(batch.CreatedAt)
	job.StartedAt = unixTimePtrFromPtr(batch.InProgressAt)
	job.CompletedAt = unixTimePtrFromPtr(batch.CompletedAt)
	job.FailedAt = unixTimePtrFromPtr(batch.FailedAt)
	job.CancelledAt = unixTimePtrFromPtr(batch.CancelledAt)
	return job
}

func isTerminalBatchStatus(status string) bool {
	switch status {
	case "completed", "failed", "cancelled", "expired":
		return true
	default:
		return false
	}
}

func unixTimePtr(ts int64) *time.Time {
	if ts == 0 {
		return nil
	}
	t := time.Unix(ts, 0).UTC()
	return &t
}

func unixTimePtrFromPtr(ts *int64) *time.Time {
	if ts == nil || *ts == 0 {
		return nil
	}
	t := time.Unix(*ts, 0).UTC()
	return &t
}

func parseBatchResults(data []byte) ([]ai.BatchResult, error) {
	lines := bytes.Split(data, []byte("\n"))
	results := make([]ai.BatchResult, 0, len(lines))
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var row BatchResultLine
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("otium: parse batch result line: %w", err)
		}
		result := ai.BatchResult{
			CustomID:         row.CustomID,
			ProviderResponse: append(json.RawMessage(nil), line...),
		}
		if row.Response != nil {
			result.StatusCode = row.Response.StatusCode
			result.RequestID = row.Response.RequestID
			result.Output = extractChatContent(row.Response.Body)
			result.Usage = extractUsage(row.Response.Body)
		}
		if len(row.Error) > 0 {
			if b, err := json.Marshal(row.Error); err == nil {
				result.Error = string(b)
			}
		}
		results = append(results, result)
	}
	return results, nil
}

func extractChatContent(body json.RawMessage) string {
	var parsed struct {
		Choices []Choice `json:"choices"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil || len(parsed.Choices) == 0 {
		return ""
	}
	content, _ := parsed.Choices[0].Message.Content.(string)
	return content
}

func extractUsage(body json.RawMessage) *ai.CompletionUsage {
	var parsed struct {
		Model string `json:"model"`
		Usage Usage  `json:"usage"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil
	}
	u := parsed.Usage
	if u.TotalTokens == 0 && u.PromptTokens == 0 && u.CompletionTokens == 0 {
		return nil
	}
	cached := u.CachedTokens
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens != 0 {
		cached = u.PromptTokensDetails.CachedTokens
	}
	inputPerM, cachedPerM, outputPerM := CostPerMTokens(parsed.Model)
	uncached := u.PromptTokens - cached
	cost := float64(uncached)/1e6*inputPerM +
		float64(cached)/1e6*cachedPerM +
		float64(u.CompletionTokens)/1e6*outputPerM
	return &ai.CompletionUsage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
		CachedTokens:     cached,
		Cost:             cost,
	}
}

// CostPerMTokens returns input/cached/output list price per million tokens for an Otium
// capability tier. These are Otium's published list rates: deliberately set as a clear
// discount under the comparable frontier tiers while staying a premium product — not a
// race-to-the-bottom price, and not a signal of the underlying cost basis. Unknown tiers
// return 0,0,0 ("not available"), which callers must treat as unknown rather than free.
func CostPerMTokens(model string) (inputPerM, cachedPerM, outputPerM float64) {
	switch model {
	case ModelOtiumSmall:
		return 0.15, 0.015, 0.75
	case ModelOtiumMedium:
		return 0.40, 0.04, 2.00
	case ModelOtiumLarge:
		return 1.20, 0.12, 6.00
	default:
		return 0, 0, 0
	}
}
