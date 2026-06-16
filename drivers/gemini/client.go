package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/spectrum-labs-tech/ai/internal/httpretry"
)

// Client is an HTTP client for the Gemini API.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new Gemini API client.
func NewClient(apiKey, baseURL string, maxRetries int) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 6 * time.Minute,
			Transport: &httpretry.Transport{
				MaxRetries: maxRetries,
			},
		},
	}
}

// GenerateContentRequest is the request structure for Gemini content generation.
type GenerateContentRequest struct {
	SystemInstruction *Content          `json:"system_instruction,omitempty"`
	Contents          []Content         `json:"contents"`
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
}

// Content represents one conversational turn.
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

// Part represents one content part.
type Part struct {
	Text string `json:"text,omitempty"`
}

// GenerationConfig controls model generation behavior.
type GenerationConfig struct {
	Temperature        float64         `json:"temperature,omitempty"`
	MaxOutputTokens    int             `json:"maxOutputTokens,omitempty"`
	ResponseMIMEType   string          `json:"responseMimeType,omitempty"`
	ResponseJSONSchema json.RawMessage `json:"responseJsonSchema,omitempty"`
}

// GenerateContentResponse is the response structure from Gemini.
type GenerateContentResponse struct {
	Candidates     []Candidate    `json:"candidates"`
	UsageMetadata  UsageMetadata  `json:"usageMetadata"`
	PromptFeedback PromptFeedback `json:"promptFeedback"`
}

// BatchGenerateContentRequest creates a Gemini batch job.
type BatchGenerateContentRequest struct {
	Batch GeminiBatchConfig `json:"batch"`
}

// GeminiBatchConfig configures one batch job.
type GeminiBatchConfig struct {
	DisplayName string            `json:"display_name,omitempty"`
	InputConfig GeminiInputConfig `json:"input_config"`
}

// GeminiInputConfig holds either inline requests or an uploaded file reference.
type GeminiInputConfig struct {
	Requests *GeminiInlineRequests `json:"requests,omitempty"`
	FileName string                `json:"file_name,omitempty"`
}

// GeminiInlineRequests wraps inline batch requests.
type GeminiInlineRequests struct {
	Requests []GeminiBatchRequestItem `json:"requests"`
}

// GeminiBatchRequestItem is one batch request envelope.
type GeminiBatchRequestItem struct {
	Request  *GenerateContentRequest `json:"request,omitempty"`
	Metadata *GeminiBatchMetadata    `json:"metadata,omitempty"`
}

// GeminiBatchMetadata preserves the request key for reconciliation.
type GeminiBatchMetadata struct {
	Key string `json:"key,omitempty"`
}

// GeminiBatchOperation is the long-running operation returned by the batch API.
type GeminiBatchOperation struct {
	Name     string                 `json:"name"`
	Done     bool                   `json:"done"`
	Metadata GeminiBatchMetadataObj `json:"metadata"`
	Response GeminiBatchResponseObj `json:"response"`
	Error    *GeminiAPIError        `json:"error,omitempty"`
}

// GeminiBatchMetadataObj describes batch job state.
type GeminiBatchMetadataObj struct {
	State string `json:"state"`
}

// GeminiBatchResponseObj holds either inline responses or a file name.
type GeminiBatchResponseObj struct {
	InlinedResponses []GeminiBatchResultEnvelope `json:"inlinedResponses,omitempty"`
	ResponsesFile    string                      `json:"responsesFile,omitempty"`
}

// GeminiBatchResultEnvelope is one output item from a Gemini batch.
type GeminiBatchResultEnvelope struct {
	Key      string                   `json:"key,omitempty"`
	Metadata *GeminiBatchMetadata     `json:"metadata,omitempty"`
	Response *GenerateContentResponse `json:"response,omitempty"`
	Error    *GeminiAPIError          `json:"error,omitempty"`
}

// GeminiAPIError is the Gemini error shape.
type GeminiAPIError struct {
	Code    int    `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Status  string `json:"status,omitempty"`
}

// GeminiFileUploadResponse captures file upload completion.
type GeminiFileUploadResponse struct {
	File struct {
		Name string `json:"name"`
		URI  string `json:"uri"`
	} `json:"file"`
}

// Candidate represents one model response candidate.
type Candidate struct {
	Content      Content `json:"content"`
	FinishReason string  `json:"finishReason"`
}

// UsageMetadata tracks token usage.
type UsageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
}

// PromptFeedback carries block reasons when no candidates are returned.
type PromptFeedback struct {
	BlockReason string `json:"blockReason"`
}

// APIErrorResponse is the standard Gemini error envelope.
type APIErrorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

// GenerateContent calls the Gemini generateContent API.
func (c *Client) GenerateContent(ctx context.Context, model string, req *GenerateContentRequest) (*GenerateContentResponse, error) {
	endpoint := fmt.Sprintf("%s/models/%s:generateContent", c.baseURL, model)

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.apiKey)

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
		var apiErr APIErrorResponse
		if err := json.Unmarshal(respBody, &apiErr); err == nil && apiErr.Error.Message != "" {
			return nil, fmt.Errorf("gemini API error (status %d): %s", resp.StatusCode, apiErr.Error.Message)
		}
		return nil, fmt.Errorf("gemini API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var out GenerateContentResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	return &out, nil
}

// CreateBatch creates a Gemini batch job using inline requests or an uploaded file.
func (c *Client) CreateBatch(ctx context.Context, model string, req *BatchGenerateContentRequest) (*GeminiBatchOperation, error) {
	endpoint := fmt.Sprintf("%s/models/%s:batchGenerateContent", c.baseURL, model)
	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.apiKey)
	return c.doBatchRequest(httpReq)
}

// GetBatch retrieves one Gemini batch job.
func (c *Client) GetBatch(ctx context.Context, batchName string) (*GeminiBatchOperation, error) {
	endpoint := fmt.Sprintf("%s/%s", c.baseURL, strings.TrimPrefix(batchName, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("x-goog-api-key", c.apiKey)
	return c.doBatchRequest(httpReq)
}

// CancelBatch cancels one Gemini batch job.
func (c *Client) CancelBatch(ctx context.Context, batchName string) (*GeminiBatchOperation, error) {
	endpoint := fmt.Sprintf("%s/%s:cancel", c.baseURL, strings.TrimPrefix(batchName, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("x-goog-api-key", c.apiKey)
	return c.doBatchRequest(httpReq)
}

// UploadJSONLFile uploads a JSONL batch input file using Gemini's resumable File API.
func (c *Client) UploadJSONLFile(ctx context.Context, displayName string, content []byte) (string, error) {
	startReqBody := []byte(fmt.Sprintf(`{"file":{"display_name":%q}}`, displayName))
	startReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://generativelanguage.googleapis.com/upload/v1beta/files", bytes.NewReader(startReqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create upload start request: %w", err)
	}
	startReq.Header.Set("x-goog-api-key", c.apiKey)
	startReq.Header.Set("X-Goog-Upload-Protocol", "resumable")
	startReq.Header.Set("X-Goog-Upload-Command", "start")
	startReq.Header.Set("X-Goog-Upload-Header-Content-Length", fmt.Sprintf("%d", len(content)))
	startReq.Header.Set("X-Goog-Upload-Header-Content-Type", "application/jsonl")
	startReq.Header.Set("Content-Type", "application/json")

	startResp, err := c.httpClient.Do(startReq)
	if err != nil {
		return "", fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = startResp.Body.Close() }()

	if startResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(startResp.Body)
		return "", fmt.Errorf("gemini upload start error (status %d): %s", startResp.StatusCode, string(body))
	}
	uploadURL := startResp.Header.Get("X-Goog-Upload-URL")
	if uploadURL == "" {
		return "", fmt.Errorf("gemini upload start response missing X-Goog-Upload-URL")
	}

	uploadReq, err := http.NewRequestWithContext(ctx, http.MethodPost, uploadURL, bytes.NewReader(content))
	if err != nil {
		return "", fmt.Errorf("failed to create upload request: %w", err)
	}
	uploadReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(content)))
	uploadReq.Header.Set("X-Goog-Upload-Offset", "0")
	uploadReq.Header.Set("X-Goog-Upload-Command", "upload, finalize")
	uploadResp, err := c.httpClient.Do(uploadReq)
	if err != nil {
		return "", fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = uploadResp.Body.Close() }()

	respBody, err := io.ReadAll(uploadResp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}
	if uploadResp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("gemini upload finalize error (status %d): %s", uploadResp.StatusCode, string(respBody))
	}

	var out GeminiFileUploadResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return "", fmt.Errorf("failed to parse upload response: %w", err)
	}
	if out.File.Name == "" {
		return "", fmt.Errorf("gemini upload response missing file name")
	}
	return out.File.Name, nil
}

// DownloadFile downloads a Gemini batch result file.
func (c *Client) DownloadFile(ctx context.Context, fileName string) ([]byte, error) {
	endpoint := fmt.Sprintf("https://generativelanguage.googleapis.com/download/v1beta/%s:download?alt=media", strings.TrimPrefix(fileName, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("x-goog-api-key", c.apiKey)

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
		return nil, fmt.Errorf("gemini download error (status %d): %s", resp.StatusCode, string(respBody))
	}
	return respBody, nil
}

func (c *Client) doBatchRequest(httpReq *http.Request) (*GeminiBatchOperation, error) {
	resp, err := c.httpClient.Do(httpReq) //nolint:gosec // URL is constructed from validated baseURL and API-defined paths by callers.
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		var apiErr APIErrorResponse
		if err := json.Unmarshal(respBody, &apiErr); err == nil && apiErr.Error.Message != "" {
			return nil, fmt.Errorf("gemini API error (status %d): %s", resp.StatusCode, apiErr.Error.Message)
		}
		return nil, fmt.Errorf("gemini API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var out GeminiBatchOperation
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	return &out, nil
}
