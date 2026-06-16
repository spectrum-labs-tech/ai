package httpretry

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func noDelay(_ int, _ *http.Response) time.Duration { return 0 }

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func okResponse() *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("")),
		Header:     http.Header{},
	}
}

func statusResponse(code int) *http.Response {
	return &http.Response{
		StatusCode: code,
		Body:       io.NopCloser(strings.NewReader("")),
		Header:     http.Header{},
	}
}

func TestTransport_SuccessOnFirstAttempt(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	tr := &Transport{
		MaxRetries: 3,
		backoffFn:  noDelay,
		Base:       roundTripFunc(func(_ *http.Request) (*http.Response, error) { calls.Add(1); return okResponse(), nil }),
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://example.com", nil)
	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	if calls.Load() != 1 {
		t.Errorf("calls = %d, want 1", calls.Load())
	}
}

func TestTransport_RetriesOnTransientStatus(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	tr := &Transport{
		MaxRetries: 3,
		backoffFn:  noDelay,
		Base: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			if calls.Add(1) < 3 {
				return statusResponse(http.StatusServiceUnavailable), nil
			}
			return okResponse(), nil
		}),
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://example.com", nil)
	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	if calls.Load() != 3 {
		t.Errorf("calls = %d, want 3", calls.Load())
	}
}

func TestTransport_ExhaustsRetries(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	tr := &Transport{
		MaxRetries: 2,
		backoffFn:  noDelay,
		Base: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			calls.Add(1)
			return statusResponse(http.StatusServiceUnavailable), nil
		}),
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://example.com", nil)
	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
	if calls.Load() != 3 { // initial + 2 retries
		t.Errorf("calls = %d, want 3 (initial + 2 retries)", calls.Load())
	}
}

func TestTransport_NoRetryOnClientError(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	tr := &Transport{
		MaxRetries: 3,
		backoffFn:  noDelay,
		Base: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			calls.Add(1)
			return statusResponse(http.StatusBadRequest), nil
		}),
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://example.com", nil)
	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if calls.Load() != 1 {
		t.Errorf("calls = %d, want 1 (no retry on 400)", calls.Load())
	}
}

func TestTransport_NegativeMaxRetriesDisablesRetry(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	tr := &Transport{
		MaxRetries: -1,
		backoffFn:  noDelay,
		Base: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			calls.Add(1)
			return statusResponse(http.StatusServiceUnavailable), nil
		}),
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://example.com", nil)
	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if calls.Load() != 1 {
		t.Errorf("calls = %d, want 1 (retries disabled)", calls.Load())
	}
}

func TestTransport_RespectsRetryAfterHeader(t *testing.T) {
	t.Parallel()
	var delays []time.Duration
	var calls atomic.Int32
	tr := &Transport{
		MaxRetries: 2,
		backoffFn: func(_ int, resp *http.Response) time.Duration {
			d := backoffDelay(1, resp)
			delays = append(delays, d)
			return 0 // don't actually wait
		},
		Base: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			if calls.Add(1) < 3 {
				r := statusResponse(http.StatusTooManyRequests)
				r.Header.Set("Retry-After", "42")
				return r, nil
			}
			return okResponse(), nil
		}),
	}
	req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://example.com", nil)
	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	for _, d := range delays {
		if d != 42*time.Second {
			t.Errorf("delay = %v, want 42s (from Retry-After header)", d)
		}
	}
}

func TestTransport_ContextCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(t.Context())
	var calls atomic.Int32
	tr := &Transport{
		MaxRetries: 3,
		backoffFn: func(_ int, _ *http.Response) time.Duration {
			cancel() // cancel during backoff
			return 10 * time.Millisecond
		},
		Base: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
			calls.Add(1)
			return statusResponse(http.StatusServiceUnavailable), nil
		}),
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "http://example.com", nil)
	resp, err := tr.RoundTrip(req)
	if resp != nil {
		defer func() { _ = resp.Body.Close() }()
	}
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
	if calls.Load() != 1 {
		t.Errorf("calls = %d, want 1 (cancelled before first retry)", calls.Load())
	}
}
