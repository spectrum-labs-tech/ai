package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/spectrum-labs-tech/ai"
)

func TestBuildGenerateContentRequest_Temperature(t *testing.T) {
	t.Parallel()
	p, err := New(&ai.Config{APIKey: "test-key"})
	if err != nil {
		t.Fatal(err)
	}
	d := p.(*Driver)
	zero := 0.0
	for _, tt := range []struct {
		temp *float64
		want string
	}{
		{nil, ""},
		{&zero, `"temperature":0`},
	} {
		body, err := json.Marshal(d.buildGenerateContentRequest("sys", "user", "", ai.Options{Temperature: tt.temp}))
		if err != nil {
			t.Fatal(err)
		}
		has := strings.Contains(string(body), `"temperature"`)
		if (tt.want == "") == has || (tt.want != "" && !strings.Contains(string(body), tt.want)) {
			t.Errorf("temperature %v: body = %s", tt.temp, body)
		}
	}
}
