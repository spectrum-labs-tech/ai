package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/spectrum-labs-tech/ai"
)

const defaultModel = "gemini-3.1-flash-lite-preview"

func init() {
	ai.Register("gemini", New)
}

// Driver implements the ai.Provider interface for Gemini.
type Driver struct {
	client *Client
	model  string
}

// New creates a new Gemini driver.
func New(cfg *ai.Config) (ai.Provider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("gemini: API key is required")
	}

	if cfg.Model == "" {
		cfg.Model = defaultModel
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta"
	}

	return &Driver{
		client: NewClient(cfg.APIKey, baseURL, cfg.MaxRetries),
		model:  cfg.Model,
	}, nil
}

// Complete sends a structured generation request to Gemini and returns the raw JSON content string.
func (d *Driver) Complete(ctx context.Context, systemPrompt, userPrompt, jsonSchema string, opts ai.Options) (string, error) {
	// Keep a safety buffer for large HTML prompts. Gemini models usually tolerate large inputs,
	// but we still reject obviously oversized requests before incurring network cost.
	totalTokens := (len(systemPrompt) + len(userPrompt)) / 4
	if totalTokens > 900000 {
		return "", fmt.Errorf("gemini: request too large (~%d tokens); reduce prompt size", totalTokens)
	}

	req := d.buildGenerateContentRequest(systemPrompt, userPrompt, jsonSchema, opts)

	resp, err := d.client.GenerateContent(ctx, d.model, req)
	if err != nil {
		return "", fmt.Errorf("gemini API call failed: %w", err)
	}
	if len(resp.Candidates) == 0 {
		if resp.PromptFeedback.BlockReason != "" {
			return "", fmt.Errorf("gemini: no candidates returned (block reason: %s)", resp.PromptFeedback.BlockReason)
		}
		return "", fmt.Errorf("gemini: no candidates returned")
	}

	content := strings.TrimSpace(extractCandidateText(resp.Candidates[0]))
	if content == "" {
		return "", fmt.Errorf("gemini: empty candidate content")
	}

	if r := ai.UsageRecorderFromContext(ctx); r != nil {
		prompt := resp.UsageMetadata.PromptTokenCount
		completion := resp.UsageMetadata.CandidatesTokenCount
		total := resp.UsageMetadata.TotalTokenCount
		if total == 0 {
			total = prompt + completion
		}
		r.Record(ai.CompletionUsage{
			PromptTokens:     prompt,
			CompletionTokens: completion,
			TotalTokens:      total,
			CachedTokens:     resp.UsageMetadata.CachedContentTokenCount,
			Cost:             estimateCost(d.model, prompt, completion, resp.UsageMetadata.CachedContentTokenCount),
		})
	}

	return content, nil
}

func extractCandidateText(candidate Candidate) string {
	parts := make([]string, 0, len(candidate.Content.Parts))
	for _, part := range candidate.Content.Parts {
		if strings.TrimSpace(part.Text) == "" {
			continue
		}
		parts = append(parts, part.Text)
	}
	return strings.Join(parts, "\n")
}

// ProviderName returns "gemini".
func (d *Driver) ProviderName() string { return "gemini" }

// ModelName returns the model being used.
func (d *Driver) ModelName() string { return d.model }

// Close releases resources (no-op for Gemini HTTP client).
func (d *Driver) Close() error { return nil }

// SubmitBatch submits a Gemini batch job. It uses inline requests for smaller batches
// and file-backed submission when requested or when the inline payload grows too large.
func (d *Driver) SubmitBatch(ctx context.Context, requests []ai.BatchRequest, opts ai.BatchOptions) (*ai.BatchJob, error) {
	if len(requests) == 0 {
		return nil, fmt.Errorf("gemini: batch requires at least one request")
	}

	items := make([]GeminiBatchRequestItem, 0, len(requests))
	seen := make(map[string]struct{}, len(requests))
	for i, req := range requests {
		if strings.TrimSpace(req.CustomID) == "" {
			return nil, fmt.Errorf("gemini: batch request %d missing custom_id", i)
		}
		if _, exists := seen[req.CustomID]; exists {
			return nil, fmt.Errorf("gemini: duplicate custom_id %q", req.CustomID)
		}
		seen[req.CustomID] = struct{}{}

		items = append(items, GeminiBatchRequestItem{
			Request:  d.buildGenerateContentRequest(req.SystemPrompt, req.UserPrompt, req.JSONSchema, req.Options),
			Metadata: &GeminiBatchMetadata{Key: req.CustomID},
		})
	}

	displayName := opts.DisplayName
	if displayName == "" {
		displayName = "gemini-batch"
	}

	createReq := &BatchGenerateContentRequest{
		Batch: GeminiBatchConfig{
			DisplayName: displayName,
		},
	}

	inlineWrapper := &GeminiInlineRequests{Requests: items}
	if !opts.ForceFile {
		createReq.Batch.InputConfig = GeminiInputConfig{Requests: inlineWrapper}
		body, err := json.Marshal(createReq)
		if err == nil && len(body) <= 20*1024*1024 {
			op, err := d.client.CreateBatch(ctx, d.model, createReq)
			if err == nil {
				return mapGeminiBatch(op, d.model), nil
			}
		}
	}

	jsonl, err := buildGeminiJSONL(items)
	if err != nil {
		return nil, err
	}
	fileName, err := d.client.UploadJSONLFile(ctx, displayName, jsonl)
	if err != nil {
		return nil, fmt.Errorf("gemini: upload batch input file: %w", err)
	}
	createReq.Batch.InputConfig = GeminiInputConfig{FileName: fileName}
	op, err := d.client.CreateBatch(ctx, d.model, createReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: create batch: %w", err)
	}
	job := mapGeminiBatch(op, d.model)
	job.InputFileID = fileName
	return job, nil
}

// GetBatch retrieves one Gemini batch job.
func (d *Driver) GetBatch(ctx context.Context, batchID string) (*ai.BatchJob, error) {
	op, err := d.client.GetBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("gemini: get batch: %w", err)
	}
	return mapGeminiBatch(op, d.model), nil
}

// CancelBatch attempts to cancel a Gemini batch job.
func (d *Driver) CancelBatch(ctx context.Context, batchID string) (*ai.BatchJob, error) {
	op, err := d.client.CancelBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("gemini: cancel batch: %w", err)
	}
	return mapGeminiBatch(op, d.model), nil
}

// GetBatchResults retrieves inline results or downloads a result file for a Gemini batch.
func (d *Driver) GetBatchResults(ctx context.Context, batchID string) ([]ai.BatchResult, error) {
	op, err := d.client.GetBatch(ctx, batchID)
	if err != nil {
		return nil, fmt.Errorf("gemini: get batch: %w", err)
	}

	if len(op.Response.InlinedResponses) > 0 {
		return parseGeminiBatchEnvelopes(op.Response.InlinedResponses)
	}
	if op.Response.ResponsesFile == "" {
		return nil, nil
	}

	body, err := d.client.DownloadFile(ctx, op.Response.ResponsesFile)
	if err != nil {
		return nil, fmt.Errorf("gemini: download batch result file: %w", err)
	}
	return parseGeminiBatchFile(body)
}

func estimateCost(model string, promptTokens, completionTokens, cachedTokens int) float64 {
	inputPerM, cachedPerM, outputPerM := costPerMTokens(model)
	uncached := promptTokens - cachedTokens
	if uncached < 0 {
		uncached = 0
	}
	return float64(uncached)/1e6*inputPerM +
		float64(cachedTokens)/1e6*cachedPerM +
		float64(completionTokens)/1e6*outputPerM
}

func (d *Driver) buildGenerateContentRequest(systemPrompt, userPrompt, jsonSchema string, opts ai.Options) *GenerateContentRequest {
	temperature := 0.0
	if opts.Temperature != nil {
		temperature = *opts.Temperature
	}

	req := &GenerateContentRequest{
		Contents: []Content{
			{
				Role: "user",
				Parts: []Part{
					{Text: userPrompt},
				},
			},
		},
		GenerationConfig: &GenerationConfig{
			Temperature:      temperature,
			ResponseMIMEType: "application/json",
		},
	}
	if systemPrompt != "" {
		req.SystemInstruction = &Content{
			Parts: []Part{{Text: systemPrompt}},
		}
	}
	if opts.MaxTokens > 0 {
		req.GenerationConfig.MaxOutputTokens = opts.MaxTokens
	}
	if strings.TrimSpace(jsonSchema) != "" {
		req.GenerationConfig.ResponseJSONSchema = json.RawMessage(jsonSchema)
	}
	return req
}

func buildGeminiJSONL(items []GeminiBatchRequestItem) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, item := range items {
		line := struct {
			Key     string                  `json:"key"`
			Request *GenerateContentRequest `json:"request"`
		}{
			Key:     item.Metadata.Key,
			Request: item.Request,
		}
		if err := enc.Encode(line); err != nil {
			return nil, fmt.Errorf("gemini: encode batch input line: %w", err)
		}
	}
	return buf.Bytes(), nil
}

func mapGeminiBatch(op *GeminiBatchOperation, model string) *ai.BatchJob {
	var raw json.RawMessage
	if b, err := json.Marshal(op); err == nil {
		raw = b
	}

	job := &ai.BatchJob{
		ID:               op.Name,
		Provider:         "gemini",
		Model:            model,
		Status:           op.Metadata.State,
		Done:             op.Done,
		OutputFileID:     op.Response.ResponsesFile,
		ResultInline:     len(op.Response.InlinedResponses) > 0,
		ProviderResponse: raw,
	}
	now := time.Now().UTC()
	job.CreatedAt = &now
	if op.Done && op.Metadata.State == "JOB_STATE_SUCCEEDED" {
		job.CompletedAt = &now
	}
	if op.Done && op.Metadata.State == "JOB_STATE_FAILED" {
		job.FailedAt = &now
	}
	if op.Done && op.Metadata.State == "JOB_STATE_CANCELLED" {
		job.CancelledAt = &now
	}
	return job
}

func parseGeminiBatchFile(data []byte) ([]ai.BatchResult, error) {
	lines := bytes.Split(data, []byte("\n"))
	out := make([]ai.BatchResult, 0, len(lines))
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var env GeminiBatchResultEnvelope
		if err := json.Unmarshal(line, &env); err != nil {
			return nil, fmt.Errorf("gemini: parse batch result line: %w", err)
		}
		result, err := mapGeminiBatchEnvelope(env, line)
		if err != nil {
			return nil, err
		}
		out = append(out, result)
	}
	return out, nil
}

func parseGeminiBatchEnvelopes(items []GeminiBatchResultEnvelope) ([]ai.BatchResult, error) {
	out := make([]ai.BatchResult, 0, len(items))
	for _, item := range items {
		raw, _ := json.Marshal(item)
		result, err := mapGeminiBatchEnvelope(item, raw)
		if err != nil {
			return nil, err
		}
		out = append(out, result)
	}
	return out, nil
}

func mapGeminiBatchEnvelope(env GeminiBatchResultEnvelope, raw []byte) (ai.BatchResult, error) {
	customID := env.Key
	if customID == "" && env.Metadata != nil {
		customID = env.Metadata.Key
	}
	result := ai.BatchResult{
		CustomID:         customID,
		ProviderResponse: append(json.RawMessage(nil), raw...),
	}
	if env.Response != nil {
		if len(env.Response.Candidates) > 0 {
			result.Output = strings.TrimSpace(extractCandidateText(Candidate{
				Content: env.Response.Candidates[0].Content,
			}))
		}
		result.Usage = &ai.CompletionUsage{
			PromptTokens:     env.Response.UsageMetadata.PromptTokenCount,
			CompletionTokens: env.Response.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      env.Response.UsageMetadata.TotalTokenCount,
			CachedTokens:     env.Response.UsageMetadata.CachedContentTokenCount,
			Cost:             estimateCost(defaultModel, env.Response.UsageMetadata.PromptTokenCount, env.Response.UsageMetadata.CandidatesTokenCount, env.Response.UsageMetadata.CachedContentTokenCount),
		}
	}
	if env.Error != nil {
		result.Error = env.Error.Message
	}
	return result, nil
}

func costPerMTokens(model string) (inputPerM, cachedPerM, outputPerM float64) {
	switch model {
	case defaultModel:
		// Preview model pricing may change. Use 0 until we intentionally publish a cost table.
		return 0, 0, 0
	default:
		return 0, 0, 0
	}
}
