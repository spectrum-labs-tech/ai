package ai

import (
	"testing"
)

func TestUsageRecorder(t *testing.T) {
	t.Parallel()

	var r UsageRecorder

	if got := r.Usage(); got != nil {
		t.Fatalf("Usage() before Record = %v, want nil", got)
	}

	r.Record(CompletionUsage{
		PromptTokens:     10,
		CompletionTokens: 5,
		TotalTokens:      15,
		CachedTokens:     2,
		Cost:             0.12,
	})

	got := r.Usage()
	if got == nil {
		t.Fatal("Usage() after Record = nil, want usage")
	}
	if got.TotalTokens != 15 || got.Cost != 0.12 {
		t.Fatalf("Usage() = %+v", got)
	}

	got.TotalTokens = 99
	if r.Usage().TotalTokens != 15 {
		t.Fatal("Usage() should return a copy, but original usage was mutated")
	}
}

func TestUsageRecorderContext(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	if got := UsageRecorderFromContext(ctx); got != nil {
		t.Fatalf("UsageRecorderFromContext(background) = %v, want nil", got)
	}

	recorder := &UsageRecorder{}
	ctx = WithUsageRecorder(ctx, recorder)
	if got := UsageRecorderFromContext(ctx); got != recorder {
		t.Fatalf("UsageRecorderFromContext() = %p, want %p", got, recorder)
	}
}
