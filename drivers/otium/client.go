package otium

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

// Otium exposes an OpenAI-compatible HTTP API (chat completions, files, batches), so this
// client speaks that wire format against a caller-supplied base URL. It is a thin HTTP client
// per the repo's driver conventions — no vendor SDK.

// APIError represents a non-2xx HTTP response from the Otium API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("otium API error (status %d): %s", e.StatusCode, e.Body)
}

// isNotFound reports whether err is an HTTP 404 from the Otium API.
func isNotFound(err error) bool {
	var apiErr *APIError
	return errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusNotFound
}

// Client is an HTTP client for the Otium OpenAI-compatible API.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new Otium API client against baseURL (e.g. "https://…/v1").
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

// ChatCompletionRequest is the request body for chat completions.
type ChatCompletionRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Temperature    float64         `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_completion_tokens,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// Message represents a chat message. Content is a plain string or []MessageContentPart
// (multimodal, when an image is attached).
type Message struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
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
	Type       string                    `json:"type"`                  // "json_object" or "json_schema"
	JSONSchema *ResponseFormatJSONSchema `json:"json_schema,omitempty"` // set when Type == "json_schema"
}

// ResponseFormatJSONSchema configures structured-output enforcement.
type ResponseFormatJSONSchema struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema,omitempty"`
}

// ChatCompletionResponse is the chat completions response body.
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
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	CachedTokens        int                  `json:"cached_tokens,omitempty"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

// PromptTokensDetails provides a breakdown of prompt token usage.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// FileObject describes an uploaded file.
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

// BatchObject is the batch API object.
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

// BatchRequestLine is one JSONL line in the batch input file.
type BatchRequestLine struct {
	CustomID string                 `json:"custom_id"`
	Method   string                 `json:"method"`
	URL      string                 `json:"url"`
	Body     map[string]interface{} `json:"body"`
}

// BatchResultLine is one JSONL line in the batch output or error file.
type BatchResultLine struct {
	ID       string                 `json:"id"`
	CustomID string                 `json:"custom_id"`
	Response *BatchResponse         `json:"response,omitempty"`
	Error    map[string]interface{} `json:"error,omitempty"`
}

// BatchResponse wraps one completed request inside a batch output line.
type BatchResponse struct {
	StatusCode int             `json:"status_code"`
	RequestID  string          `json:"request_id"`
	Body       json.RawMessage `json:"body"`
}

// CreateChatCompletion calls the chat completions API.
func (c *Client) CreateChatCompletion(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	var out ChatCompletionResponse
	if err := c.doJSON(ctx, http.MethodPost, c.baseURL+"/chat/completions", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// UploadFile uploads a file (multipart) for batch processing.
func (c *Client) UploadFile(ctx context.Context, filename string, content []byte, purpose string) (*FileObject, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("purpose", purpose); err != nil {
		return nil, fmt.Errorf("write purpose field: %w", err)
	}
	part, err := writer.CreateFormFile("file", filepath.Base(filename))
	if err != nil {
		return nil, fmt.Errorf("create multipart file: %w", err)
	}
	if _, err := part.Write(content); err != nil {
		return nil, fmt.Errorf("write multipart file: %w", err)
	}
	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("finalize multipart body: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/files", &body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())

	respBody, err := c.do(httpReq)
	if err != nil {
		return nil, err
	}
	var out FileObject
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	return &out, nil
}

// DownloadFileContent downloads the raw bytes for a file.
func (c *Client) DownloadFileContent(ctx context.Context, fileID string) ([]byte, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/files/"+fileID+"/content", nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	return c.do(httpReq)
}

// CreateBatch creates a batch job.
func (c *Client) CreateBatch(ctx context.Context, req *CreateBatchRequest) (*BatchObject, error) {
	var out BatchObject
	if err := c.doJSON(ctx, http.MethodPost, c.baseURL+"/batches", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetBatch fetches one batch job.
func (c *Client) GetBatch(ctx context.Context, batchID string) (*BatchObject, error) {
	var out BatchObject
	if err := c.doJSON(ctx, http.MethodGet, c.baseURL+"/batches/"+batchID, nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// CancelBatch cancels one batch job.
func (c *Client) CancelBatch(ctx context.Context, batchID string) (*BatchObject, error) {
	var out BatchObject
	if err := c.doJSON(ctx, http.MethodPost, c.baseURL+"/batches/"+batchID+"/cancel", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// doJSON performs a JSON request/response round-trip, decoding a 2xx body into out. A nil
// payload sends no body (GET / parameterless POST).
func (c *Client) doJSON(ctx context.Context, method, url string, payload, out interface{}) error {
	var body io.Reader
	if payload != nil {
		bodyBytes, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}
		body = bytes.NewReader(bodyBytes)
	}
	httpReq, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	if payload != nil {
		httpReq.Header.Set("Content-Type", "application/json")
	}
	respBody, err := c.do(httpReq)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("parse response: %w", err)
	}
	return nil
}

// do sends the request and returns the body for a 2xx, or an *APIError otherwise.
func (c *Client) do(httpReq *http.Request) ([]byte, error) {
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}
	return respBody, nil
}
