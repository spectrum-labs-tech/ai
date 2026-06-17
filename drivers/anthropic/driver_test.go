package anthropic

import (
	"testing"

	"github.com/spectrum-labs-tech/ai"
)

func TestNew_MissingAPIKey(t *testing.T) {
	t.Parallel()

	_, err := New(&ai.Config{APIKey: ""})
	if err == nil {
		t.Fatal("expected error for missing API key, got nil")
	}
}

func TestNew_DefaultModel(t *testing.T) {
	t.Parallel()

	p, err := New(&ai.Config{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	d, ok := p.(*Driver)
	if !ok {
		t.Fatal("expected *Driver type")
	}
	if d.model != "claude-sonnet-4-6" {
		t.Errorf("model = %q, want %q", d.model, "claude-sonnet-4-6")
	}
}

func TestDriverName(t *testing.T) {
	t.Parallel()

	p, err := New(&ai.Config{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := p.ProviderName(); got != "anthropic" {
		t.Errorf("ProviderName() = %q, want %q", got, "anthropic")
	}
	if got := p.ModelName(); got != "claude-sonnet-4-6" {
		t.Errorf("ModelName() = %q, want %q", got, "claude-sonnet-4-6")
	}
}

func TestMapBatchStatus(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		processingStatus string
		wantStatus       string
		wantDone         bool
	}{
		{
			name:             "in_progress maps to in_progress",
			processingStatus: "in_progress",
			wantStatus:       "in_progress",
			wantDone:         false,
		},
		{
			name:             "ended maps to completed",
			processingStatus: "ended",
			wantStatus:       "completed",
			wantDone:         true,
		},
		{
			name:             "unknown passes through",
			processingStatus: "canceling",
			wantStatus:       "canceling",
			wantDone:         false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			resp := &batchStatusResponse{
				ID:               "batch-123",
				ProcessingStatus: tc.processingStatus,
				CreatedAt:        "2026-01-01T00:00:00Z",
			}
			job := mapBatchStatus(resp, "claude-sonnet-4-6")
			if job.Status != tc.wantStatus {
				t.Errorf("Status = %q, want %q", job.Status, tc.wantStatus)
			}
			if job.Done != tc.wantDone {
				t.Errorf("Done = %v, want %v", job.Done, tc.wantDone)
			}
		})
	}
}

func TestCostPerMTokens(t *testing.T) {
	t.Parallel()

	tests := []struct {
		model             string
		wantInputPerM     float64
		wantCacheReadPerM float64
		wantOutputPerM    float64
	}{
		{ModelClaudeFable5, 10.00, 1.00, 50.00},
		{ModelClaudeOpus48, 5.00, 0.50, 25.00},
		{ModelClaudeOpus47, 5.00, 0.50, 25.00},
		{ModelClaudeOpus46, 5.00, 0.50, 25.00},
		{ModelClaudeOpus45, 5.00, 0.50, 25.00},
		{ModelClaudeOpus41, 15.00, 1.50, 75.00},
		{ModelClaudeSonnet46, 3.00, 0.30, 15.00},
		{ModelClaudeSonnet45, 3.00, 0.30, 15.00},
		{ModelClaudeHaiku45, 1.00, 0.10, 5.00},
		{"unknown-model", 0, 0, 0},
	}

	for _, tc := range tests {
		t.Run(tc.model, func(t *testing.T) {
			t.Parallel()
			inputPerM, cacheReadPerM, outputPerM := CostPerMTokens(tc.model)
			if inputPerM != tc.wantInputPerM {
				t.Errorf("inputPerM = %v, want %v", inputPerM, tc.wantInputPerM)
			}
			if cacheReadPerM != tc.wantCacheReadPerM {
				t.Errorf("cacheReadPerM = %v, want %v", cacheReadPerM, tc.wantCacheReadPerM)
			}
			if outputPerM != tc.wantOutputPerM {
				t.Errorf("outputPerM = %v, want %v", outputPerM, tc.wantOutputPerM)
			}
		})
	}
}

func TestExtractTextContent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		blocks []contentBlock
		want   string
	}{
		{
			name:   "single text block",
			blocks: []contentBlock{{Type: "text", Text: "hello"}},
			want:   "hello",
		},
		{
			name:   "first text block returned",
			blocks: []contentBlock{{Type: "text", Text: "first"}, {Type: "text", Text: "second"}},
			want:   "first",
		},
		{
			name:   "no text blocks returns empty",
			blocks: []contentBlock{{Type: "tool_use", Text: ""}},
			want:   "",
		},
		{
			name:   "empty blocks returns empty",
			blocks: []contentBlock{},
			want:   "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := extractTextContent(tc.blocks)
			if got != tc.want {
				t.Errorf("extractTextContent() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestUnwrapMarkdownJSONFence(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "json fenced with language",
			in:   "```json\n{\n  \"ok\": true\n}\n```",
			want: "{\n  \"ok\": true\n}",
		},
		{
			name: "json fenced without language",
			in:   "```\n{\"ok\":true}\n```",
			want: "{\"ok\":true}",
		},
		{
			name: "invalid json not unwrapped",
			in:   "```json\nnot-json\n```",
			want: "```json\nnot-json\n```",
		},
		{
			name: "non-json language not unwrapped",
			in:   "```text\n{\"ok\":true}\n```",
			want: "```text\n{\"ok\":true}\n```",
		},
		{
			name: "plain text unchanged",
			in:   "hello",
			want: "hello",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := unwrapMarkdownJSONFence(tc.in)
			if got != tc.want {
				t.Errorf("unwrapMarkdownJSONFence() = %q, want %q", got, tc.want)
			}
		})
	}
}
