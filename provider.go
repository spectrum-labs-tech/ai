package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrBatchOutputExpired is returned by GetBatchResults when the provider's
// output file is no longer available (deleted after retention window).
// Callers should treat this the same as an expired batch and reset entities.
var ErrBatchOutputExpired = errors.New("batch output file expired or unavailable")

// Provider is the generic interface for AI completions with structured output.
// Implementations must be safe for concurrent use.
type Provider interface {
	// Complete sends a prompt and returns the raw JSON response string.
	// jsonSchema constrains the response format (passed to the provider's
	// structured output / function calling API).
	Complete(ctx context.Context, systemPrompt, userPrompt, jsonSchema string, opts Options) (string, error)

	// ProviderName returns the name of the AI provider (e.g., "openai").
	ProviderName() string

	// ModelName returns the specific model being used (e.g., "gpt-4o-mini").
	ModelName() string

	// Close releases any resources held by the provider.
	Close() error
}

// BatchProvider extends Provider with asynchronous batch submission support.
// Implementations should translate each BatchRequest into the provider's native
// request shape and preserve CustomID in batch results for reconciliation.
type BatchProvider interface {
	Provider

	// SubmitBatch submits a provider-managed asynchronous batch job.
	SubmitBatch(ctx context.Context, requests []BatchRequest, opts BatchOptions) (*BatchJob, error)

	// GetBatch retrieves the current status for a submitted batch.
	GetBatch(ctx context.Context, batchID string) (*BatchJob, error)

	// CancelBatch attempts to cancel an in-flight batch.
	CancelBatch(ctx context.Context, batchID string) (*BatchJob, error)

	// GetBatchResults retrieves all currently available batch results. Providers
	// may return partial results for completed, cancelled, or expired batches.
	GetBatchResults(ctx context.Context, batchID string) ([]BatchResult, error)
}

// Options contains per-request tuning knobs.
type Options struct {
	// Temperature controls randomness (nil = use provider default).
	Temperature *float64

	// MaxTokens limits the response length (0 = use provider default).
	MaxTokens int

	// ImageURL is a publicly accessible URL of an image to include in the user
	// message. When set, providers that support vision will send the image
	// alongside the text prompt. Empty string means no image.
	ImageURL string
}

// BatchRequest is one structured completion request inside an asynchronous batch.
type BatchRequest struct {
	CustomID     string
	SystemPrompt string
	UserPrompt   string
	JSONSchema   string
	Options      Options
}

// BatchOptions controls provider batch submission behavior.
type BatchOptions struct {
	// CompletionWindow is the requested provider turnaround window, if supported.
	CompletionWindow string

	// DisplayName is an optional human-readable label for the batch.
	DisplayName string

	// Metadata carries provider-supported batch metadata.
	Metadata map[string]string

	// ForceFile prefers file-backed submission when the provider supports both
	// inline and file input modes.
	ForceFile bool
}

// BatchRequestCounts summarizes request completion status inside a batch.
type BatchRequestCounts struct {
	Total     int `json:"total"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

// BatchJob describes a provider batch at a normalized level.
type BatchJob struct {
	ID               string             `json:"id"`
	Provider         string             `json:"provider"`
	Model            string             `json:"model,omitempty"`
	Status           string             `json:"status"`
	InputFileID      string             `json:"input_file_id,omitempty"`
	OutputFileID     string             `json:"output_file_id,omitempty"`
	ErrorFileID      string             `json:"error_file_id,omitempty"`
	RequestCounts    BatchRequestCounts `json:"request_counts"`
	CreatedAt        *time.Time         `json:"created_at,omitempty"`
	StartedAt        *time.Time         `json:"started_at,omitempty"`
	CompletedAt      *time.Time         `json:"completed_at,omitempty"`
	FailedAt         *time.Time         `json:"failed_at,omitempty"`
	CancelledAt      *time.Time         `json:"cancelled_at,omitempty"`
	Metadata         map[string]string  `json:"metadata,omitempty"`
	Done             bool               `json:"done"`
	ResultInline     bool               `json:"result_inline,omitempty"`
	ProviderResponse json.RawMessage    `json:"provider_response,omitempty"`
}

// BatchResult is one normalized result row from a batch output.
type BatchResult struct {
	CustomID         string           `json:"custom_id"`
	Output           string           `json:"output,omitempty"`
	Error            string           `json:"error,omitempty"`
	StatusCode       int              `json:"status_code,omitempty"`
	RequestID        string           `json:"request_id,omitempty"`
	Usage            *CompletionUsage `json:"usage,omitempty"`
	ProviderResponse json.RawMessage  `json:"provider_response,omitempty"`
}

// Config holds configuration for creating a Provider.
type Config struct {
	// Provider is the AI provider name (e.g., "openai", "anthropic").
	Provider string

	// Model is the specific model to use (e.g., "gpt-4o-mini").
	Model string

	// APIKey is the API key for cloud providers.
	APIKey string

	// BaseURL is the base URL for the API (optional, provider-specific).
	BaseURL string

	// Options allows provider-specific configuration.
	Options map[string]interface{}
}

// Schema defines the prompt template and JSON schema for a completion task.
type Schema struct {
	// Name is a unique identifier for this schema.
	Name string

	// SystemPrompt is the system message with instructions.
	SystemPrompt string

	// UserPromptTemplate is a Go text/template for the user message.
	UserPromptTemplate string

	// JSONSchema is the JSON Schema for structured output.
	JSONSchema string
}

// ProviderFactory creates a BatchProvider for a given model on demand.
// Calling the factory at task run time (rather than at startup) means
// infrequent tasks don't hold a live connection for their entire idle period.
type ProviderFactory func(model string) (BatchProvider, error)

// NewProviderFactory returns a ProviderFactory that constructs BatchProviders
// using the given provider name and API key, with the model supplied per-call.
func NewProviderFactory(providerName, apiKey string) ProviderFactory {
	return func(model string) (BatchProvider, error) {
		raw, err := New(&Config{Provider: providerName, APIKey: apiKey, Model: model})
		if err != nil {
			return nil, fmt.Errorf("create %s provider (model=%s): %w", providerName, model, err)
		}
		p, ok := raw.(BatchProvider)
		if !ok {
			return nil, fmt.Errorf("%s provider (model=%s) does not implement BatchProvider", providerName, model)
		}
		return p, nil
	}
}

// ToolDefinition describes a callable function for providers that support tool use.
type ToolDefinition struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object describing the function arguments.
	Parameters json.RawMessage
}

// ToolCallRequest is one tool invocation requested by the AI model.
type ToolCallRequest struct {
	// ID is the provider-assigned call identifier; must be echoed back in ToolCallResult.
	ID string
	// Name is the tool name, matching a ToolDefinition.Name.
	Name string
	// ArgsJSON is the JSON-encoded arguments matching the tool's parameter schema.
	ArgsJSON string
}

// ToolCallResult is the response to one ToolCallRequest.
type ToolCallResult struct {
	ID      string // matches ToolCallRequest.ID
	Content string // tool output, typically JSON
}

// AgentProvider extends Provider with an agentic tool-calling loop. The model
// may call tools zero or more times before producing its final JSON response.
// Usage across all iterations is recorded via the context UsageRecorder.
type AgentProvider interface {
	Provider
	// CompleteWithTools runs an agentic loop: the model receives tool definitions
	// and may call them via execTools before returning a final JSON response.
	CompleteWithTools(
		ctx context.Context,
		systemPrompt, userPrompt, jsonSchema string,
		tools []ToolDefinition,
		opts Options,
		execTools func(context.Context, []ToolCallRequest) ([]ToolCallResult, error),
	) (string, error)
}

// DriverFunc is a function that creates a new Provider from a Config.
type DriverFunc func(*Config) (Provider, error)

var (
	driversMu sync.RWMutex
	drivers   = make(map[string]DriverFunc)
)

// Register makes a driver available by the provided name.
// If Register is called twice with the same name or if driver is nil, it panics.
func Register(name string, driver DriverFunc) {
	driversMu.Lock()
	defer driversMu.Unlock()
	if driver == nil {
		panic("ai: Register driver is nil")
	}
	if _, dup := drivers[name]; dup {
		panic("ai: Register called twice for driver " + name)
	}
	drivers[name] = driver
}

// New creates a new Provider based on the provided config.
// The provider must be registered via Register.
func New(cfg *Config) (Provider, error) {
	driversMu.RLock()
	driverFunc, ok := drivers[cfg.Provider]
	driversMu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("ai: unknown provider %q (forgot to import driver?)", cfg.Provider)
	}

	return driverFunc(cfg)
}

// Drivers returns a list of the names of the registered drivers.
func Drivers() []string {
	driversMu.RLock()
	defer driversMu.RUnlock()
	list := make([]string, 0, len(drivers))
	for name := range drivers {
		list = append(list, name)
	}
	return list
}
