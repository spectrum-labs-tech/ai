package otium

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/spectrum-labs-tech/ai"
)

// A temperature the caller sets is sent, 0 included; none set leaves the model's default.
// Before, the field was a float64 with omitempty, so an explicit 0 was dropped.
func TestBuildChatCompletionRequest_Temperature(t *testing.T) {
	t.Parallel()
	zero, low := 0.0, 0.2
	tests := []struct {
		name string
		temp *float64
		want string // the temperature JSON, or "" for absent
	}{
		{"unset: the model's default", nil, ""},
		{"zero is sent", &zero, `"temperature":0`},
		{"a value is sent", &low, `"temperature":0.2`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			d := newTestDriver(t, "otium-small", nil)
			body, err := json.Marshal(d.buildChatCompletionRequest("sys", "user", "", ai.Options{Temperature: tt.temp}))
			if err != nil {
				t.Fatal(err)
			}
			got := strings.Contains(string(body), `"temperature"`)
			if tt.want == "" && got {
				t.Errorf("body sends a temperature: %s", body)
			}
			if tt.want != "" && !strings.Contains(string(body), tt.want) {
				t.Errorf("body = %s, want %s", body, tt.want)
			}
		})
	}
}
