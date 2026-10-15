package openai

import (
	"testing"

	"github.com/spectrum-labs-tech/ai"
)

func TestTemperatureFor(t *testing.T) {
	t.Parallel()
	zero := 0.0
	tests := []struct {
		model string
		temp  *float64
		sent  bool
	}{
		{"gpt-4.1-mini", &zero, true}, // 0 is sent, no longer dropped by omitempty
		{"gpt-4.1-mini", nil, false},
		// These accept only their default and reject any temperature with a 400.
		{ModelGPT54Nano, &zero, false},
		{ModelGPT5, &zero, false},
		{"o3-mini", &zero, false},
		{"O4-mini", &zero, false},
	}
	for _, tt := range tests {
		got := temperatureFor(tt.model, ai.Options{Temperature: tt.temp})
		if (got != nil) != tt.sent || (got != nil && *got != *tt.temp) {
			t.Errorf("temperatureFor(%q, %v) = %v, sent %v", tt.model, tt.temp, got, tt.sent)
		}
		if got != nil && got == tt.temp {
			t.Errorf("temperatureFor(%q) returned the caller's pointer", tt.model)
		}
	}
}
