package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client is an HTTP client for the Anthropic API.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	// batchClient has a longer timeout for batch result downloads.
	batchClient *http.Client
}

// NewClient creates a new Anthropic API client.
func NewClient(apiKey, baseURL string) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
		batchClient: &http.Client{
			Timeout: 6 * time.Minute,
		},
	}
}

// messageRequest is the Anthropic Messages API request body.
type messageRequest struct {
	Model     string        `json:"model"`
	MaxTokens int           `json:"max_tokens"`
	System    []systemBlock `json:"system,omitempty"`
	Messages  []messageItem `json:"messages"`
}

// messageItem is one turn in the messages array.
type messageItem struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// systemBlock is one element in the system content array.
// Set CacheControl to mark this block for Anthropic prompt caching.
type systemBlock struct {
	Type         string        `json:"type"` // always "text"
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

// cacheControl marks a content block for prompt caching.
type cacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// messageResponse is the Anthropic Messages API response.
type messageResponse struct {
	ID      string         `json:"id"`
	Type    string         `json:"type"`
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
	Model   string         `json:"model"`
	Usage   anthropicUsage `json:"usage"`
}

// contentBlock is one element of the content array in a message response.
type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// anthropicUsage holds token usage from an Anthropic response.
type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// batchRequestLine is one item in the Anthropic batch request array.
type batchRequestLine struct {
	CustomID string         `json:"custom_id"`
	Params   messageRequest `json:"params"`
}

// batchSubmitRequest is the body sent to POST /v1/messages/batches.
type batchSubmitRequest struct {
	Requests []batchRequestLine `json:"requests"`
}

// batchStatusResponse is the Anthropic batch status response.
type batchStatusResponse struct {
	ID               string             `json:"id"`
	Type             string             `json:"type"`
	ProcessingStatus string             `json:"processing_status"`
	RequestCounts    batchRequestCounts `json:"request_counts"`
	CreatedAt        string             `json:"created_at"`
	ExpiresAt        string             `json:"expires_at"`
	EndedAt          *string            `json:"ended_at,omitempty"`
	ResultsURL       *string            `json:"results_url,omitempty"`
}

// batchRequestCounts holds the request counts from a batch status response.
type batchRequestCounts struct {
	Processing int `json:"processing"`
	Succeeded  int `json:"succeeded"`
	Errored    int `json:"errored"`
	Canceled   int `json:"canceled"`
	Expired    int `json:"expired"`
}

// batchResultLine is one JSONL line in the batch results stream.
type batchResultLine struct {
	CustomID string            `json:"custom_id"`
	Result   batchResultDetail `json:"result"`
}

// batchResultDetail holds the outcome of one batch request.
type batchResultDetail struct {
	Type    string            `json:"type"`
	Message *messageResponse  `json:"message,omitempty"`
	Error   *batchResultError `json:"error,omitempty"`
}

// batchResultError is the error shape in a batch result.
type batchResultError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// CreateMessage sends a single message request to the Anthropic API.
func (c *Client) CreateMessage(ctx context.Context, req *messageRequest) (*messageResponse, error) {
	return c.doMessageJSON(ctx, req)
}

// SubmitBatch submits a batch of message requests.
func (c *Client) SubmitBatch(ctx context.Context, req *batchSubmitRequest) (*batchStatusResponse, error) {
	return c.doBatchJSON(ctx, http.MethodPost, c.baseURL+"/messages/batches", req, false)
}

// GetBatch fetches the current status of a batch job.
func (c *Client) GetBatch(ctx context.Context, batchID string) (*batchStatusResponse, error) {
	return c.doBatchJSON(ctx, http.MethodGet, fmt.Sprintf("%s/messages/batches/%s", c.baseURL, batchID), nil, false)
}

// CancelBatch attempts to cancel a batch job.
func (c *Client) CancelBatch(ctx context.Context, batchID string) (*batchStatusResponse, error) {
	return c.doBatchJSON(ctx, http.MethodPost, fmt.Sprintf("%s/messages/batches/%s/cancel", c.baseURL, batchID), nil, false)
}

// GetBatchResults downloads the JSONL results stream for a completed batch.
func (c *Client) GetBatchResults(ctx context.Context, batchID string) ([]byte, error) {
	url := fmt.Sprintf("%s/messages/batches/%s/results", c.baseURL, batchID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	c.setCommonHeaders(req, true)

	resp, err := c.batchClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anthropic API error (status %d): %s", resp.StatusCode, string(body))
	}
	return body, nil
}

func (c *Client) doMessageJSON(ctx context.Context, payload *messageRequest) (*messageResponse, error) {
	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/messages", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	c.setCommonHeaders(req, false)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anthropic API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var out messageResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	return &out, nil
}

func (c *Client) doBatchJSON(ctx context.Context, method, url string, payload interface{}, isBatch bool) (*batchStatusResponse, error) {
	var body io.Reader
	if payload != nil {
		bodyBytes, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}
		body = bytes.NewReader(bodyBytes)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	c.setCommonHeaders(req, true)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anthropic API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var out batchStatusResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	return &out, nil
}

func (c *Client) setCommonHeaders(req *http.Request, isBatch bool) {
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")
	if isBatch {
		req.Header.Set("anthropic-beta", "message-batches-2024-09-24")
	}
}

// CostPerMTokens returns input, cache-read, and output cost per million tokens
// for a model (batch pricing — 50% off standard rates).
// Cache-write cost is inputPerM * 1.25; callers compute it inline.
func CostPerMTokens(model string) (inputPerM, cacheReadPerM, outputPerM float64) {
	switch model {
	case "claude-sonnet-4-6":
		return 1.50, 0.15, 7.50
	case "claude-opus-4-7":
		return 7.50, 0.75, 37.50
	case "claude-haiku-4-5-20251001":
		return 0.40, 0.04, 2.00
	default:
		return 1.50, 0.15, 7.50
	}
}
