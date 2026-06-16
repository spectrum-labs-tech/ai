package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/spectrum-labs-tech/ai"
)

const (
	ModelClaudeHaiku45  = "claude-haiku-4-5-20251001"
	ModelClaudeSonnet46 = "claude-sonnet-4-6"
	ModelClaudeOpus47   = "claude-opus-4-7"
)

func init() {
	ai.Register("anthropic", New)
}

// Driver implements the ai.BatchProvider interface for Anthropic.
type Driver struct {
	client *Client
	model  string
}

// New creates a new Anthropic driver from the provided config.
func New(cfg *ai.Config) (ai.Provider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("anthropic: API key is required")
	}

	if cfg.Model == "" {
		cfg.Model = ModelClaudeSonnet46
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	}

	return &Driver{
		client: NewClient(cfg.APIKey, baseURL),
		model:  cfg.Model,
	}, nil
}

// Complete sends a single message to Anthropic and returns the text content.
func (d *Driver) Complete(ctx context.Context, systemPrompt, userPrompt, jsonSchema string, opts ai.Options) (string, error) {
	fullSystem := systemPrompt
	if jsonSchema != "" {
		fullSystem += "\n\nJSON schema for your response:\n" + jsonSchema
	}

	maxTokens := 4096
	if opts.MaxTokens > 0 {
		maxTokens = opts.MaxTokens
	}

	req := &messageRequest{
		Model:     d.model,
		MaxTokens: maxTokens,
		System: []systemBlock{
			{Type: "text", Text: fullSystem, CacheControl: &cacheControl{Type: "ephemeral"}},
		},
		Messages: []messageItem{
			{Role: "user", Content: userPrompt},
		},
	}

	resp, err := d.client.CreateMessage(ctx, req)
	if err != nil {
		return "", fmt.Errorf("anthropic API call failed: %w", err)
	}

	text := extractTextContent(resp.Content)
	if text == "" {
		return "", fmt.Errorf("anthropic: no text content in response")
	}
	if jsonSchema != "" {
		text = unwrapMarkdownJSONFence(text)
	}

	// Record token usage via context if a recorder is present.
	if r := ai.UsageRecorderFromContext(ctx); r != nil {
		inputPerM, cacheReadPerM, outputPerM := CostPerMTokens(d.model)
		uncached := resp.Usage.InputTokens - resp.Usage.CacheCreationInputTokens - resp.Usage.CacheReadInputTokens
		cost := float64(uncached)/1e6*inputPerM +
			float64(resp.Usage.CacheCreationInputTokens)/1e6*(inputPerM*1.25) +
			float64(resp.Usage.CacheReadInputTokens)/1e6*cacheReadPerM +
			float64(resp.Usage.OutputTokens)/1e6*outputPerM
		r.Record(ai.CompletionUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
			CachedTokens:     resp.Usage.CacheReadInputTokens,
			Cost:             cost,
		})
	}

	return text, nil
}

// SubmitBatch submits an Anthropic Messages Batch job.
func (d *Driver) SubmitBatch(ctx context.Context, requests []ai.BatchRequest, opts ai.BatchOptions) (*ai.BatchJob, error) {
	if len(requests) == 0 {
		return nil, fmt.Errorf("anthropic: batch requires at least one request")
	}

	seen := make(map[string]struct{}, len(requests))
	lines := make([]batchRequestLine, 0, len(requests))
	for i, req := range requests {
		if strings.TrimSpace(req.CustomID) == "" {
			return nil, fmt.Errorf("anthropic: batch request %d missing custom_id", i)
		}
		if _, exists := seen[req.CustomID]; exists {
			return nil, fmt.Errorf("anthropic: duplicate custom_id %q", req.CustomID)
		}
		seen[req.CustomID] = struct{}{}

		fullSystem := req.SystemPrompt
		if req.JSONSchema != "" {
			fullSystem += "\n\nJSON schema for your response:\n" + req.JSONSchema
		}

		maxTokens := 4096
		if req.Options.MaxTokens > 0 {
			maxTokens = req.Options.MaxTokens
		}

		lines = append(lines, batchRequestLine{
			CustomID: req.CustomID,
			Params: messageRequest{
				Model:     d.model,
				MaxTokens: maxTokens,
				System: []systemBlock{
					{Type: "text", Text: fullSystem, CacheControl: &cacheControl{Type: "ephemeral"}},
				},
				Messages: []messageItem{
					{Role: "user", Content: req.UserPrompt},
				},
			},
		})
	}

	batchResp, err := d.client.SubmitBatch(ctx, &batchSubmitRequest{Requests: lines})
	if err != nil {
		return nil, fmt.Errorf("anthropic: submit batch: %w", err)
	}

	return mapBatchStatus(batchResp, d.model), nil
}

// GetBatch retrieves the current status of an Anthropic batch.
func (d *Driver) GetBatch(ctx context.Context, batchID string) (*ai.BatchJob, error) {
	resp, err := d.client.GetBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("anthropic: get batch: %w", err)
	}
	return mapBatchStatus(resp, d.model), nil
}

// CancelBatch attempts to cancel an Anthropic batch.
func (d *Driver) CancelBatch(ctx context.Context, batchID string) (*ai.BatchJob, error) {
	resp, err := d.client.CancelBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("anthropic: cancel batch: %w", err)
	}
	return mapBatchStatus(resp, d.model), nil
}

// GetBatchResults downloads and parses the JSONL results for a completed batch.
func (d *Driver) GetBatchResults(ctx context.Context, batchID string) ([]ai.BatchResult, error) {
	data, err := d.client.GetBatchResults(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("anthropic: get batch results: %w", err)
	}
	return parseAnthropicBatchResults(data, d.model)
}

// ProviderName returns "anthropic".
func (d *Driver) ProviderName() string { return "anthropic" }

// ModelName returns the model being used.
func (d *Driver) ModelName() string { return d.model }

// Close releases resources (no-op for Anthropic HTTP client).
func (d *Driver) Close() error { return nil }

// mapBatchStatus maps an Anthropic batch status response to ai.BatchJob.
func mapBatchStatus(resp *batchStatusResponse, model string) *ai.BatchJob {
	var raw json.RawMessage
	if b, err := json.Marshal(resp); err == nil {
		raw = b
	}

	status := mapStatus(resp.ProcessingStatus)
	done := resp.ProcessingStatus == "ended"

	failed := resp.RequestCounts.Errored + resp.RequestCounts.Canceled + resp.RequestCounts.Expired

	job := &ai.BatchJob{
		ID:       resp.ID,
		Provider: "anthropic",
		Model:    model,
		Status:   status,
		Done:     done,
		RequestCounts: ai.BatchRequestCounts{
			Total:     resp.RequestCounts.Processing + resp.RequestCounts.Succeeded + failed,
			Completed: resp.RequestCounts.Succeeded,
			Failed:    failed,
		},
		ProviderResponse: raw,
	}

	if t, err := time.Parse(time.RFC3339, resp.CreatedAt); err == nil {
		job.CreatedAt = &t
	}

	return job
}

// mapStatus maps Anthropic processing_status to a normalized status string.
func mapStatus(processingStatus string) string {
	switch processingStatus {
	case "in_progress":
		return "in_progress"
	case "ended":
		return "completed"
	default:
		return processingStatus
	}
}

// parseAnthropicBatchResults parses the JSONL results stream into ai.BatchResult slice.
func parseAnthropicBatchResults(data []byte, model string) ([]ai.BatchResult, error) {
	lines := bytes.Split(data, []byte("\n"))
	results := make([]ai.BatchResult, 0, len(lines))

	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		var row batchResultLine
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("anthropic: parse batch result line: %w", err)
		}

		result := ai.BatchResult{
			CustomID:         row.CustomID,
			ProviderResponse: append(json.RawMessage(nil), line...),
		}

		switch row.Result.Type {
		case "succeeded":
			if row.Result.Message != nil {
				result.Output = unwrapMarkdownJSONFence(extractTextContent(row.Result.Message.Content))
				if row.Result.Message.Usage.InputTokens > 0 || row.Result.Message.Usage.OutputTokens > 0 {
					inputPerM, cacheReadPerM, outputPerM := CostPerMTokens(model)
					u := row.Result.Message.Usage
					uncached := u.InputTokens - u.CacheCreationInputTokens - u.CacheReadInputTokens
					cost := float64(uncached)/1e6*inputPerM +
						float64(u.CacheCreationInputTokens)/1e6*(inputPerM*1.25) +
						float64(u.CacheReadInputTokens)/1e6*cacheReadPerM +
						float64(u.OutputTokens)/1e6*outputPerM
					result.Usage = &ai.CompletionUsage{
						PromptTokens:     u.InputTokens,
						CompletionTokens: u.OutputTokens,
						TotalTokens:      u.InputTokens + u.OutputTokens,
						CachedTokens:     u.CacheReadInputTokens,
						Cost:             cost,
					}
				}
			}
		case "errored":
			if row.Result.Error != nil {
				result.Error = row.Result.Error.Message
			} else {
				result.Error = "errored"
			}
		case "canceled":
			result.Error = "canceled"
		case "expired":
			result.Error = "expired"
		}

		results = append(results, result)
	}
	return results, nil
}

// extractTextContent returns the first text block from a content slice.
func extractTextContent(blocks []contentBlock) string {
	for _, b := range blocks {
		if b.Type == "text" {
			return b.Text
		}
	}
	return ""
}

// unwrapMarkdownJSONFence converts fenced JSON markdown into raw JSON text.
// It only unwraps when the inner content parses as valid JSON.
func unwrapMarkdownJSONFence(text string) string {
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "```") || !strings.HasSuffix(trimmed, "```") {
		return text
	}

	withoutClose := strings.TrimSuffix(trimmed, "```")
	withoutClose = strings.TrimRight(withoutClose, " \t\r\n")

	newline := strings.IndexByte(withoutClose, '\n')
	if newline < 0 {
		return text
	}

	opener := strings.TrimSpace(withoutClose[:newline])
	if !strings.HasPrefix(opener, "```") {
		return text
	}

	lang := strings.TrimSpace(strings.TrimPrefix(opener, "```"))
	if lang != "" && !strings.EqualFold(lang, "json") {
		return text
	}

	inner := strings.TrimSpace(withoutClose[newline+1:])
	if inner == "" || !json.Valid([]byte(inner)) {
		return text
	}

	return inner
}
