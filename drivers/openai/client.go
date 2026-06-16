package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"time"

	"github.com/spectrum-labs-tech/ai/internal/httpretry"
)

// APIError represents an HTTP error response from the OpenAI API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("OpenAI API error (status %d): %s", e.StatusCode, e.Body)
}

// isNotFound reports whether err is an HTTP 404 from the OpenAI API.
func isNotFound(err error) bool {
	var apiErr *APIError
	return errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound
}

// Client is an HTTP client for the OpenAI API.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new OpenAI API client.
func NewClient(apiKey, baseURL string, maxRetries int) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 6 * time.Minute,
			Transport: &httpretry.Transport{
				MaxRetries: maxRetries,
			},
		},
	}
}

// ChatCompletionRequest is the request structure for chat completions.
type ChatCompletionRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Temperature    float64         `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_completion_tokens,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     string          `json:"tool_choice,omitempty"` // "auto", "none", "required"
}

// Tool defines a callable function the model may invoke.
type Tool struct {
	Type     string   `json:"type"` // always "function"
	Function Function `json:"function"`
}

// Function is the function definition inside a Tool.
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      bool            `json:"strict,omitempty"`
}

// ToolCall is one tool invocation requested by the model.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // always "function"
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the name and JSON-encoded arguments of a tool call.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// Message represents a chat message.
// Content is a plain string, []MessageContentPart (vision), or nil (tool-call-only turns).
// ToolCalls is set on assistant turns that invoke tools.
// ToolCallID is set on tool result turns (role == "tool").
type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`                // string, []MessageContentPart, or nil
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`   // set when role == "assistant" with tool calls
	ToolCallID string     `json:"tool_call_id,omitempty"` // set when role == "tool"
}

// MessageContentPart is one element in a multimodal message content array.
type MessageContentPart struct {
	Type     string              `json:"type"`                // "text" or "image_url"
	Text     string              `json:"text,omitempty"`      // set when Type == "text"
	ImageURL *MessageImageURLRef `json:"image_url,omitempty"` // set when Type == "image_url"
}

// MessageImageURLRef holds the URL for an image_url content part.
type MessageImageURLRef struct {
	URL string `json:"url"`
}

// ResponseFormat specifies the format of the model response.
type ResponseFormat struct {
	Type       string                    `json:"type"`                 // "text", "json_object", or "json_schema"
	JSONSchema *ResponseFormatJSONSchema `json:"json_schema,omitempty"` // set when Type == "json_schema"
}

// ResponseFormatJSONSchema configures structured output enforcement.
type ResponseFormatJSONSchema struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema,omitempty"`
}

// ChatCompletionResponse is the response structure from OpenAI.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage tracks token usage.
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	CachedTokens            int                      `json:"cached_tokens,omitempty"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// FileObject describes an uploaded OpenAI file.
type FileObject struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int    `json:"bytes"`
	CreatedAt int64  `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
	Status    string `json:"status,omitempty"`
}

// BatchRequestCounts reports batch item counts.
type BatchRequestCounts struct {
	Total     int `json:"total"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

// BatchObject is the OpenAI batch API object.
type BatchObject struct {
	ID               string             `json:"id"`
	Object           string             `json:"object"`
	Endpoint         string             `json:"endpoint"`
	InputFileID      string             `json:"input_file_id"`
	CompletionWindow string             `json:"completion_window"`
	Status           string             `json:"status"`
	OutputFileID     string             `json:"output_file_id,omitempty"`
	ErrorFileID      string             `json:"error_file_id,omitempty"`
	CreatedAt        int64              `json:"created_at"`
	InProgressAt     *int64             `json:"in_progress_at,omitempty"`
	CompletedAt      *int64             `json:"completed_at,omitempty"`
	FailedAt         *int64             `json:"failed_at,omitempty"`
	CancelledAt      *int64             `json:"cancelled_at,omitempty"`
	RequestCounts    BatchRequestCounts `json:"request_counts"`
	Metadata         map[string]string  `json:"metadata,omitempty"`
}

// CreateBatchRequest submits a new batch job.
type CreateBatchRequest struct {
	InputFileID      string            `json:"input_file_id"`
	Endpoint         string            `json:"endpoint"`
	CompletionWindow string            `json:"completion_window"`
	Metadata         map[string]string `json:"metadata,omitempty"`
}

// OpenAIBatchRequestLine is one JSONL line in the batch input file.
type OpenAIBatchRequestLine struct {
	CustomID string                 `json:"custom_id"`
	Method   string                 `json:"method"`
	URL      string                 `json:"url"`
	Body     map[string]interface{} `json:"body"`
}

// OpenAIBatchResultLine is one JSONL line in the batch output or error file.
type OpenAIBatchResultLine struct {
	ID       string                 `json:"id"`
	CustomID string                 `json:"custom_id"`
	Response *OpenAIBatchResponse   `json:"response,omitempty"`
	Error    map[string]interface{} `json:"error,omitempty"`
}

// OpenAIBatchResponse wraps one completed request inside a batch output line.
type OpenAIBatchResponse struct {
	StatusCode int             `json:"status_code"`
	RequestID  string          `json:"request_id"`
	Body       json.RawMessage `json:"body"`
}

// PromptTokensDetails provides a breakdown of prompt token usage.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// CompletionTokensDetails provides a breakdown of completion token usage.
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// CreateChatCompletion calls the OpenAI chat completions API.
func (c *Client) CreateChatCompletion(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	return &chatResp, nil
}

// UploadFile uploads a file for batch processing.
func (c *Client) UploadFile(ctx context.Context, filename string, content []byte, purpose string) (*FileObject, error) {
	url := fmt.Sprintf("%s/files", c.baseURL)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)

	if err := writer.WriteField("purpose", purpose); err != nil {
		return nil, fmt.Errorf("failed to write purpose field: %w", err)
	}
	part, err := writer.CreateFormFile("file", filepath.Base(filename))
	if err != nil {
		return nil, fmt.Errorf("failed to create multipart file: %w", err)
	}
	if _, err := part.Write(content); err != nil {
		return nil, fmt.Errorf("failed to write multipart file: %w", err)
	}
	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to finalize multipart body: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}

	var out FileObject
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	return &out, nil
}

// DownloadFileContent downloads the raw bytes for a file.
func (c *Client) DownloadFileContent(ctx context.Context, fileID string) ([]byte, error) {
	url := fmt.Sprintf("%s/files/%s/content", c.baseURL, fileID)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}
	return respBody, nil
}

// CreateBatch creates a batch job.
func (c *Client) CreateBatch(ctx context.Context, req *CreateBatchRequest) (*BatchObject, error) {
	return c.doBatchJSON(ctx, http.MethodPost, fmt.Sprintf("%s/batches", c.baseURL), req)
}

// GetBatch fetches one batch job.
func (c *Client) GetBatch(ctx context.Context, batchID string) (*BatchObject, error) {
	return c.doBatchJSON(ctx, http.MethodGet, fmt.Sprintf("%s/batches/%s", c.baseURL, batchID), nil)
}

// CancelBatch cancels one batch job.
func (c *Client) CancelBatch(ctx context.Context, batchID string) (*BatchObject, error) {
	return c.doBatchJSON(ctx, http.MethodPost, fmt.Sprintf("%s/batches/%s/cancel", c.baseURL, batchID), nil)
}

func (c *Client) doBatchJSON(ctx context.Context, method, url string, payload interface{}) (*BatchObject, error) {
	var body io.Reader
	if payload != nil {
		bodyBytes, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}
		body = bytes.NewReader(bodyBytes)
	}

	httpReq, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	if payload != nil {
		httpReq.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}

	var out BatchObject
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	return &out, nil
}
