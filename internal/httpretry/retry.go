package httpretry

import (
	"bytes"
	"io"
	"math/rand"
	"net/http"
	"strconv"
	"time"
)

// DefaultMaxRetries is the default number of retry attempts.
const DefaultMaxRetries = 3

// Transport is an http.RoundTripper that retries on transient errors with
// exponential backoff and jitter. It retries on network errors and on
// 429, 500, 502, 503, and 504 responses. Retry-After response headers are
// honored when present.
type Transport struct {
	// Base is the underlying RoundTripper. Defaults to http.DefaultTransport.
	Base http.RoundTripper

	// MaxRetries is the number of retry attempts after the initial request.
	// Zero uses DefaultMaxRetries. Negative values disable retries entirely.
	MaxRetries int

	// backoffFn overrides the delay function. Used in tests to eliminate waits.
	backoffFn func(attempt int, resp *http.Response) time.Duration
}

// RoundTrip implements http.RoundTripper.
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Buffer the request body so it can be replayed on each retry.
	var bodyBytes []byte
	if req.Body != nil && req.Body != http.NoBody {
		var err error
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		_ = req.Body.Close()
	}

	max := t.resolvedMax()
	var lastResp *http.Response
	var lastErr error

	for attempt := 0; attempt <= max; attempt++ {
		if attempt > 0 {
			delay := t.delay(attempt, lastResp)
			if lastResp != nil {
				_, _ = io.Copy(io.Discard, lastResp.Body)
				_ = lastResp.Body.Close()
			}
			timer := time.NewTimer(delay)
			select {
			case <-req.Context().Done():
				timer.Stop()
				return nil, req.Context().Err()
			case <-timer.C:
			}
		}

		clone := req.Clone(req.Context())
		if bodyBytes != nil {
			clone.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			clone.ContentLength = int64(len(bodyBytes))
		}

		lastResp, lastErr = t.base().RoundTrip(clone)
		if lastErr != nil {
			continue
		}

		if !isRetryable(lastResp.StatusCode) || attempt == max {
			return lastResp, nil
		}
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return lastResp, nil
}

func (t *Transport) base() http.RoundTripper {
	if t.Base != nil {
		return t.Base
	}
	return http.DefaultTransport
}

func (t *Transport) resolvedMax() int {
	if t.MaxRetries < 0 {
		return 0
	}
	if t.MaxRetries == 0 {
		return DefaultMaxRetries
	}
	return t.MaxRetries
}

func (t *Transport) delay(attempt int, resp *http.Response) time.Duration {
	if t.backoffFn != nil {
		return t.backoffFn(attempt, resp)
	}
	return backoffDelay(attempt, resp)
}

func isRetryable(status int) bool {
	switch status {
	case http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	}
	return false
}

// backoffDelay returns the delay before the given retry attempt (1-based).
// It honors the Retry-After response header when present, otherwise uses
// exponential backoff (1s, 2s, 4s, … capped at 30s) with up to 1s of jitter.
func backoffDelay(attempt int, resp *http.Response) time.Duration {
	if resp != nil {
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			if secs, err := strconv.Atoi(ra); err == nil && secs > 0 {
				return time.Duration(secs) * time.Second
			}
		}
	}
	steps := []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 16 * time.Second, 30 * time.Second}
	idx := attempt - 1
	if idx >= len(steps) {
		idx = len(steps) - 1
	}
	jitter := time.Duration(rand.Int63n(int64(time.Second))) //nolint:gosec
	return steps[idx] + jitter
}
