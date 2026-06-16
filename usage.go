package ai

import (
	"context"
	"sync"
)

type usageKey struct{}

// CompletionUsage holds token usage and cost from a provider completion call.
type CompletionUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CachedTokens     int
	Cost             float64 // estimated USD; computed by the provider
}

// UsageRecorder captures token usage from a Provider.Complete() call via context.
// Safe for concurrent use.
type UsageRecorder struct {
	mu    sync.Mutex
	usage *CompletionUsage
}

// Record is called by Provider implementations to report usage after a completion.
func (r *UsageRecorder) Record(u CompletionUsage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.usage = &u
}

// Usage returns the recorded usage, or nil if Complete() has not been called.
func (r *UsageRecorder) Usage() *CompletionUsage {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.usage == nil {
		return nil
	}
	cp := *r.usage
	return &cp
}

// WithUsageRecorder returns a child context carrying r.
// The Provider will call r.Record() after a successful completion.
func WithUsageRecorder(ctx context.Context, r *UsageRecorder) context.Context {
	return context.WithValue(ctx, usageKey{}, r)
}

// UsageRecorderFromContext retrieves the UsageRecorder from ctx.
// Returns nil if none was set. Intended for Provider implementations.
func UsageRecorderFromContext(ctx context.Context) *UsageRecorder {
	r, _ := ctx.Value(usageKey{}).(*UsageRecorder)
	return r
}
