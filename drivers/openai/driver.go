package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/spectrum-labs-tech/ai"
)

// isNotFound reports whether err is an HTTP 404 from the OpenAI client.
func isNotFound(err error) bool {
	return err != nil && strings.Contains(err.Error(), "status 404")
}

const (
	ModelGPT5Nano  = "gpt-5-nano"
	ModelGPT54Nano = "gpt-5.4-nano"
	ModelGPT5Mini  = "gpt-5-mini"
	ModelGPT5      = "gpt-5"
	ModelGPT51     = "gpt-5.1"
	ModelGPT54     = "gpt-5.4"
)

func init() {
	ai.Register("openai", New)
}

// Driver implements the ai.Provider interface for OpenAI.
type Driver struct {
	client *Client
	model  string
}

// New creates a new OpenAI driver.
func New(cfg *ai.Config) (ai.Provider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("openai: API key is required")
	}

	if cfg.Model == "" {
		cfg.Model = ModelGPT5Nano
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	return &Driver{
		client: NewClient(cfg.APIKey, baseURL),
		model:  cfg.Model,
	}, nil
}

// Complete sends a chat completion request to OpenAI and returns the raw JSON content string.
func (d *Driver) Complete(ctx context.Context, systemPrompt, userPrompt, jsonSchema string, opts ai.Options) (string, error) {
	// Estimate token size and refuse if too large.
	// gpt-5-nano has 272k input limit; we target 240k to leave buffer for response.
	totalTokens := len(systemPrompt+userPrompt) / 4
	if totalTokens > 240000 {
		return "", fmt.Errorf("openai: request too large (~%d tokens); reduce prompt size", totalTokens)
	}

	chatReq := d.buildChatCompletionRequest(systemPrompt, userPrompt, jsonSchema, opts)

	resp, err := d.client.CreateChatCompletion(ctx, chatReq)
	if err != nil {
		return "", fmt.Errorf("openai API call failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("openai: no choices returned")
	}

	// Record token usage via context if a recorder is present.
	if r := ai.UsageRecorderFromContext(ctx); r != nil {
		cachedTokens := 0
		if resp.Usage.PromptTokensDetails != nil {
			cachedTokens = resp.Usage.PromptTokensDetails.CachedTokens
		}
		if cachedTokens == 0 {
			cachedTokens = resp.Usage.CachedTokens
		}
		inputPerM, cachedPerM, outputPerM := CostPerMTokens(d.model)
		uncached := resp.Usage.PromptTokens - cachedTokens
		cost := float64(uncached)/1e6*inputPerM +
			float64(cachedTokens)/1e6*cachedPerM +
			float64(resp.Usage.CompletionTokens)/1e6*outputPerM
		r.Record(ai.CompletionUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
			CachedTokens:     cachedTokens,
			Cost:             cost,
		})
	}

	content, _ := resp.Choices[0].Message.Content.(string)
	return content, nil
}

// ProviderName returns "openai".
func (d *Driver) ProviderName() string { return "openai" }

// ModelName returns the model being used.
func (d *Driver) ModelName() string { return d.model }

// Close releases resources (no-op for OpenAI HTTP client).
func (d *Driver) Close() error { return nil }

// SubmitBatch submits a file-backed OpenAI batch job for structured chat completions.
func (d *Driver) SubmitBatch(ctx context.Context, requests []ai.BatchRequest, opts ai.BatchOptions) (*ai.BatchJob, error) {
	if len(requests) == 0 {
		return nil, fmt.Errorf("openai: batch requires at least one request")
	}

	lines := make([]OpenAIBatchRequestLine, 0, len(requests))
	seen := make(map[string]struct{}, len(requests))
	for i, req := range requests {
		if strings.TrimSpace(req.CustomID) == "" {
			return nil, fmt.Errorf("openai: batch request %d missing custom_id", i)
		}
		if _, exists := seen[req.CustomID]; exists {
			return nil, fmt.Errorf("openai: duplicate custom_id %q", req.CustomID)
		}
		seen[req.CustomID] = struct{}{}

		bodyBytes, err := json.Marshal(d.buildChatCompletionRequest(req.SystemPrompt, req.UserPrompt, req.JSONSchema, req.Options))
		if err != nil {
			return nil, fmt.Errorf("openai: marshal batch request %q: %w", req.CustomID, err)
		}
		var body map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &body); err != nil {
			return nil, fmt.Errorf("openai: decode batch request %q: %w", req.CustomID, err)
		}

		lines = append(lines, OpenAIBatchRequestLine{
			CustomID: req.CustomID,
			Method:   "POST",
			URL:      "/v1/chat/completions",
			Body:     body,
		})
	}

	var payload bytes.Buffer
	enc := json.NewEncoder(&payload)
	for _, line := range lines {
		if err := enc.Encode(line); err != nil {
			return nil, fmt.Errorf("openai: encode batch line: %w", err)
		}
	}

	file, err := d.client.UploadFile(ctx, "openai-batch.jsonl", payload.Bytes(), "batch")
	if err != nil {
		return nil, fmt.Errorf("openai: upload batch input file: %w", err)
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
		return nil, fmt.Errorf("openai: create batch: %w", err)
	}
	job := mapBatchObject(batch, d.model)
	job.InputFileID = file.ID
	return job, nil
}

// GetBatch retrieves one OpenAI batch job.
func (d *Driver) GetBatch(ctx context.Context, batchID string) (*ai.BatchJob, error) {
	batch, err := d.client.GetBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("openai: get batch: %w", err)
	}
	return mapBatchObject(batch, d.model), nil
}

// CancelBatch attempts to cancel one OpenAI batch job.
func (d *Driver) CancelBatch(ctx context.Context, batchID string) (*ai.BatchJob, error) {
	batch, err := d.client.CancelBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("openai: cancel batch: %w", err)
	}
	return mapBatchObject(batch, d.model), nil
}

// GetBatchResults downloads and parses the available output and error files for a batch.
func (d *Driver) GetBatchResults(ctx context.Context, batchID string) ([]ai.BatchResult, error) {
	batch, err := d.client.GetBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("openai: get batch: %w", err)
	}

	var results []ai.BatchResult
	if batch.OutputFileID != "" {
		body, err := d.client.DownloadFileContent(ctx, batch.OutputFileID)
		if err != nil {
			if isNotFound(err) {
				return nil, fmt.Errorf("openai: output file %s: %w", batch.OutputFileID, ai.ErrBatchOutputExpired)
			}
			return nil, fmt.Errorf("openai: download batch output file: %w", err)
		}
		parsed, err := parseOpenAIBatchResults(body)
		if err != nil {
			return nil, err
		}
		results = append(results, parsed...)
	}
	if batch.ErrorFileID != "" {
		body, err := d.client.DownloadFileContent(ctx, batch.ErrorFileID)
		if err != nil {
			if isNotFound(err) {
				// Error file gone but output file was present and downloaded — continue with what we have.
				return results, nil
			}
			return nil, fmt.Errorf("openai: download batch error file: %w", err)
		}
		parsed, err := parseOpenAIBatchResults(body)
		if err != nil {
			return nil, err
		}
		results = append(results, parsed...)
	}
	return results, nil
}

// CompleteWithTools runs an agentic loop where the model may call tools before
// producing a final JSON response. Usage is accumulated across all iterations
// and recorded via the context UsageRecorder on the final response.
func (d *Driver) CompleteWithTools(
	ctx context.Context,
	systemPrompt, userPrompt, jsonSchema string,
	tools []ai.ToolDefinition,
	opts ai.Options,
	execTools func(context.Context, []ai.ToolCallRequest) ([]ai.ToolCallResult, error),
) (string, error) {
	fullSystem := systemPrompt
	if jsonSchema != "" {
		fullSystem += "\n\nJSON schema for your response:\n" + jsonSchema
	}

	temperature := 0.0
	if opts.Temperature != nil {
		temperature = *opts.Temperature
	}

	// Build initial user message — multimodal when an image is attached.
	var userContent any
	if opts.ImageURL != "" {
		userContent = []MessageContentPart{
			{Type: "text", Text: userPrompt},
			{Type: "image_url", ImageURL: &MessageImageURLRef{URL: opts.ImageURL}},
		}
	} else {
		userContent = userPrompt
	}

	messages := []Message{
		{Role: "system", Content: fullSystem},
		{Role: "user", Content: userContent},
	}

	// Convert tool definitions to the OpenAI wire format.
	oaiTools := make([]Tool, len(tools))
	for i, t := range tools {
		oaiTools[i] = Tool{
			Type: "function",
			Function: Function{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		}
	}

	var (
		totalPromptTokens     int
		totalCompletionTokens int
		totalCachedTokens     int
	)

	const maxIter = 12
	for iter := 0; iter < maxIter; iter++ {
		req := &ChatCompletionRequest{
			Model:          d.model,
			Messages:       messages,
			Temperature:    temperature,
			ResponseFormat: &ResponseFormat{Type: "json_object"},
		}
		if len(oaiTools) > 0 {
			req.Tools = oaiTools
			req.ToolChoice = "auto"
		}
		if opts.MaxTokens > 0 {
			req.MaxTokens = opts.MaxTokens
		}

		resp, err := d.client.CreateChatCompletion(ctx, req)
		if err != nil {
			return "", fmt.Errorf("openai agent loop (iter %d): %w", iter, err)
		}
		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("openai: no choices returned (iter %d)", iter)
		}

		// Accumulate token usage across all iterations.
		cached := 0
		if resp.Usage.PromptTokensDetails != nil {
			cached = resp.Usage.PromptTokensDetails.CachedTokens
		}
		if cached == 0 {
			cached = resp.Usage.CachedTokens
		}
		totalPromptTokens += resp.Usage.PromptTokens
		totalCompletionTokens += resp.Usage.CompletionTokens
		totalCachedTokens += cached

		choice := resp.Choices[0]

		if choice.FinishReason != "tool_calls" {
			// Final response — record cumulative usage and return.
			if r := ai.UsageRecorderFromContext(ctx); r != nil {
				inputPerM, cachedPerM, outputPerM := CostPerMTokens(d.model)
				uncached := totalPromptTokens - totalCachedTokens
				cost := float64(uncached)/1e6*inputPerM +
					float64(totalCachedTokens)/1e6*cachedPerM +
					float64(totalCompletionTokens)/1e6*outputPerM
				r.Record(ai.CompletionUsage{
					PromptTokens:     totalPromptTokens,
					CompletionTokens: totalCompletionTokens,
					TotalTokens:      totalPromptTokens + totalCompletionTokens,
					CachedTokens:     totalCachedTokens,
					Cost:             cost,
				})
			}
			content, _ := choice.Message.Content.(string)
			return content, nil
		}

		// Tool call round: append assistant message then execute and append results.
		messages = append(messages, choice.Message)

		calls := make([]ai.ToolCallRequest, len(choice.Message.ToolCalls))
		for i, tc := range choice.Message.ToolCalls {
			calls[i] = ai.ToolCallRequest{
				ID:       tc.ID,
				Name:     tc.Function.Name,
				ArgsJSON: tc.Function.Arguments,
			}
		}

		results, err := execTools(ctx, calls)
		if err != nil {
			return "", fmt.Errorf("openai: tool execution failed: %w", err)
		}
		for _, r := range results {
			messages = append(messages, Message{
				Role:       "tool",
				ToolCallID: r.ID,
				Content:    r.Content,
			})
		}
	}

	return "", fmt.Errorf("openai: agentic loop exceeded %d iterations without a final response", maxIter)
}

func (d *Driver) buildChatCompletionRequest(systemPrompt, userPrompt, jsonSchema string, opts ai.Options) *ChatCompletionRequest {
	temperature := 0.0
	if opts.Temperature != nil {
		temperature = *opts.Temperature
	}

	fullSystem := systemPrompt
	if jsonSchema != "" {
		fullSystem += "\n\nJSON schema for your response:\n" + jsonSchema
	}

	// Build the user message content. When an image URL is present, use the
	// multimodal content array format required by the OpenAI vision API.
	var userContent any
	if opts.ImageURL != "" {
		userContent = []MessageContentPart{
			{Type: "text", Text: userPrompt},
			{Type: "image_url", ImageURL: &MessageImageURLRef{URL: opts.ImageURL}},
		}
	} else {
		userContent = userPrompt
	}

	chatReq := &ChatCompletionRequest{
		Model: d.model,
		Messages: []Message{
			{Role: "system", Content: fullSystem},
			{Role: "user", Content: userContent},
		},
		Temperature: temperature,
		ResponseFormat: &ResponseFormat{
			Type: "json_object",
		},
	}
	if opts.MaxTokens > 0 {
		chatReq.MaxTokens = opts.MaxTokens
	}
	return chatReq
}

func mapBatchObject(batch *BatchObject, model string) *ai.BatchJob {
	var raw json.RawMessage
	if b, err := json.Marshal(batch); err == nil {
		raw = b
	}

	job := &ai.BatchJob{
		ID:           batch.ID,
		Provider:     "openai",
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
		Done:             batch.Status == "completed" || batch.Status == "failed" || batch.Status == "cancelled" || batch.Status == "expired",
		ProviderResponse: raw,
	}
	job.CreatedAt = unixTimePtr(batch.CreatedAt)
	job.StartedAt = unixTimePtrFromPtr(batch.InProgressAt)
	job.CompletedAt = unixTimePtrFromPtr(batch.CompletedAt)
	job.FailedAt = unixTimePtrFromPtr(batch.FailedAt)
	job.CancelledAt = unixTimePtrFromPtr(batch.CancelledAt)
	return job
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

func parseOpenAIBatchResults(data []byte) ([]ai.BatchResult, error) {
	lines := bytes.Split(data, []byte("\n"))
	results := make([]ai.BatchResult, 0, len(lines))
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		var row OpenAIBatchResultLine
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("openai: parse batch result line: %w", err)
		}

		result := ai.BatchResult{
			CustomID:         row.CustomID,
			ProviderResponse: append(json.RawMessage(nil), line...),
		}
		if row.Response != nil {
			result.StatusCode = row.Response.StatusCode
			result.RequestID = row.Response.RequestID
			result.Output = extractBatchChatCompletionContent(row.Response.Body)
			result.Usage = extractBatchUsage(row.Response.Body)
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

func extractBatchChatCompletionContent(body json.RawMessage) string {
	var parsed struct {
		Choices []Choice `json:"choices"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil || len(parsed.Choices) == 0 {
		return ""
	}
	content, _ := parsed.Choices[0].Message.Content.(string)
	return content
}

func extractBatchUsage(body json.RawMessage) *ai.CompletionUsage {
	var parsed struct {
		Usage Usage `json:"usage"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil
	}
	if parsed.Usage.TotalTokens == 0 && parsed.Usage.PromptTokens == 0 && parsed.Usage.CompletionTokens == 0 {
		return nil
	}
	cachedTokens := 0
	if parsed.Usage.PromptTokensDetails != nil {
		cachedTokens = parsed.Usage.PromptTokensDetails.CachedTokens
	}
	if cachedTokens == 0 {
		cachedTokens = parsed.Usage.CachedTokens
	}
	inputPerM, cachedPerM, outputPerM := CostPerMTokens(parsedModelOrDefault(body, ""))
	uncached := parsed.Usage.PromptTokens - cachedTokens
	cost := float64(uncached)/1e6*inputPerM +
		float64(cachedTokens)/1e6*cachedPerM +
		float64(parsed.Usage.CompletionTokens)/1e6*outputPerM
	return &ai.CompletionUsage{
		PromptTokens:     parsed.Usage.PromptTokens,
		CompletionTokens: parsed.Usage.CompletionTokens,
		TotalTokens:      parsed.Usage.TotalTokens,
		CachedTokens:     cachedTokens,
		Cost:             cost,
	}
}

func parsedModelOrDefault(body json.RawMessage, fallback string) string {
	var parsed struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil || parsed.Model == "" {
		return fallback
	}
	return parsed.Model
}

// CostPerMTokens returns input/cached/output cost per million tokens for a model.
func CostPerMTokens(model string) (inputPerM, cachedPerM, outputPerM float64) {
	switch model {
	case "gpt-5-nano":
		return 0.05, 0.025, 0.40
	case "gpt-4o-mini":
		return 0.15, 0.075, 0.60
	case "gpt-4.1-nano":
		return 0.10, 0.025, 0.40
	default:
		return 0.15, 0.075, 0.60
	}
}
