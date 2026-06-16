package ai

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
)

// mockProvider is a simple mock implementation of Provider for testing.
type mockProvider struct {
	provider string
	model    string
}

func (m *mockProvider) Complete(_ context.Context, _, _, _ string, _ Options) (string, error) {
	return "{}", nil
}
func (m *mockProvider) ProviderName() string { return m.provider }
func (m *mockProvider) ModelName() string    { return m.model }
func (m *mockProvider) Close() error         { return nil }

func mockDriverFunc(provider, model string) DriverFunc {
	return func(cfg *Config) (Provider, error) {
		return &mockProvider{provider: provider, model: model}, nil
	}
}

func errorDriverFunc(err error) DriverFunc {
	return func(cfg *Config) (Provider, error) {
		return nil, err
	}
}

func resetDrivers() {
	driversMu.Lock()
	defer driversMu.Unlock()
	drivers = make(map[string]DriverFunc)
}

func TestRegister_NilDriver(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Register(nil) should panic, but did not")
		}
	}()
	Register("test-nil", nil)
}

func TestRegister_DuplicateName(t *testing.T) {
	resetDrivers()
	Register("test-duplicate", mockDriverFunc("test", "v1"))
	defer func() {
		if r := recover(); r == nil {
			t.Error("Register with duplicate name should panic, but did not")
		}
		resetDrivers()
	}()
	Register("test-duplicate", mockDriverFunc("test", "v2"))
}

func TestRegister_Success(t *testing.T) {
	resetDrivers()
	defer resetDrivers()
	Register("test-provider", mockDriverFunc("test", "v1"))
	driversMu.RLock()
	_, exists := drivers["test-provider"]
	driversMu.RUnlock()
	if !exists {
		t.Error("driver was not registered")
	}
}

func TestNew_UnknownProvider(t *testing.T) {
	resetDrivers()
	defer resetDrivers()
	cfg := &Config{Provider: "unknown-provider-12345", Model: "test-model"}
	provider, err := New(cfg)
	if err == nil {
		t.Error("New() with unknown provider should return error, got nil")
	}
	if provider != nil {
		t.Error("New() should return nil provider on error")
	}
	if !strings.Contains(err.Error(), "unknown-provider-12345") {
		t.Errorf("error message should mention provider name, got: %v", err)
	}
	if !strings.Contains(err.Error(), "unknown provider") {
		t.Errorf("error message should mention 'unknown provider', got: %v", err)
	}
}

func TestNew_Success(t *testing.T) {
	resetDrivers()
	defer resetDrivers()
	Register("test-openai", mockDriverFunc("openai", "gpt-4"))
	cfg := &Config{Provider: "test-openai", Model: "gpt-4"}
	provider, err := New(cfg)
	if err != nil {
		t.Errorf("New() unexpected error: %v", err)
	}
	if provider == nil {
		t.Fatal("New() returned nil provider")
	}
	if provider.ProviderName() != "openai" {
		t.Errorf("ProviderName() = %q, want %q", provider.ProviderName(), "openai")
	}
	if provider.ModelName() != "gpt-4" {
		t.Errorf("ModelName() = %q, want %q", provider.ModelName(), "gpt-4")
	}
}

func TestNew_DriverError(t *testing.T) {
	resetDrivers()
	defer resetDrivers()
	expectedErr := fmt.Errorf("driver initialization failed")
	Register("test-error-driver", errorDriverFunc(expectedErr))
	cfg := &Config{Provider: "test-error-driver", Model: "test-model"}
	provider, err := New(cfg)
	if err == nil {
		t.Error("New() should return error from driver, got nil")
	}
	if provider != nil {
		t.Error("New() should return nil provider when driver returns error")
	}
	if !errors.Is(err, expectedErr) {
		t.Errorf("New() error = %v, want %v", err, expectedErr)
	}
}

func TestDrivers_Empty(t *testing.T) {
	resetDrivers()
	defer resetDrivers()
	driverList := Drivers()
	if len(driverList) != 0 {
		t.Errorf("Drivers() = %v, want empty list", driverList)
	}
}

func TestDrivers_Multiple(t *testing.T) {
	resetDrivers()
	defer resetDrivers()
	Register("openai", mockDriverFunc("openai", "gpt-4"))
	Register("anthropic", mockDriverFunc("anthropic", "claude-3"))
	Register("ollama", mockDriverFunc("ollama", "llama2"))
	driverList := Drivers()
	if len(driverList) != 3 {
		t.Errorf("Drivers() returned %d drivers, want 3", len(driverList))
	}
	sort.Strings(driverList)
	expected := []string{"anthropic", "ollama", "openai"}
	for i, name := range expected {
		if driverList[i] != name {
			t.Errorf("Drivers()[%d] = %q, want %q", i, driverList[i], name)
		}
	}
}

func TestConcurrentDriverAccess(t *testing.T) {
	resetDrivers()
	defer resetDrivers()
	Register("provider1", mockDriverFunc("provider1", "model1"))
	Register("provider2", mockDriverFunc("provider2", "model2"))
	const numGoroutines = 100
	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			switch id % 3 {
			case 0:
				driverList := Drivers()
				if len(driverList) < 2 {
					t.Errorf("concurrent Drivers() got %d drivers, want at least 2", len(driverList))
				}
			case 1:
				cfg := &Config{Provider: "provider1", Model: "model1"}
				p, err := New(cfg)
				if err != nil {
					t.Errorf("concurrent New(provider1) error: %v", err)
				}
				if p == nil {
					t.Error("concurrent New(provider1) returned nil")
				}
			case 2:
				cfg := &Config{Provider: "provider2", Model: "model2"}
				p, err := New(cfg)
				if err != nil {
					t.Errorf("concurrent New(provider2) error: %v", err)
				}
				if p == nil {
					t.Error("concurrent New(provider2) returned nil")
				}
			}
		}(i)
	}
	wg.Wait()
}

func TestRegister_MultipleProviders(t *testing.T) {
	resetDrivers()
	defer resetDrivers()
	providers := []struct {
		name     string
		provider string
		model    string
	}{
		{"openai", "openai", "gpt-4o-mini"},
		{"anthropic", "anthropic", "claude-3-5-haiku"},
		{"ollama", "ollama", "llama2"},
	}
	for _, p := range providers {
		Register(p.name, mockDriverFunc(p.provider, p.model))
	}
	for _, p := range providers {
		cfg := &Config{Provider: p.name, Model: p.model}
		provider, err := New(cfg)
		if err != nil {
			t.Errorf("New(%s) error: %v", p.name, err)
			continue
		}
		if provider == nil {
			t.Errorf("New(%s) returned nil", p.name)
			continue
		}
		if provider.ProviderName() != p.provider {
			t.Errorf("New(%s).ProviderName() = %q, want %q", p.name, provider.ProviderName(), p.provider)
		}
		if provider.ModelName() != p.model {
			t.Errorf("New(%s).ModelName() = %q, want %q", p.name, provider.ModelName(), p.model)
		}
		if err := provider.Close(); err != nil {
			t.Errorf("Close() error: %v", err)
		}
	}
	driverList := Drivers()
	if len(driverList) != len(providers) {
		t.Errorf("Drivers() returned %d drivers, want %d", len(driverList), len(providers))
	}
}

func TestProviderInterface(t *testing.T) {
	t.Parallel()
	var _ Provider = (*mockProvider)(nil)
	mock := &mockProvider{provider: "test-provider", model: "test-model"}
	if mock.ProviderName() != "test-provider" {
		t.Errorf("ProviderName() = %q, want %q", mock.ProviderName(), "test-provider")
	}
	if mock.ModelName() != "test-model" {
		t.Errorf("ModelName() = %q, want %q", mock.ModelName(), "test-model")
	}
	ctx := t.Context()
	content, err := mock.Complete(ctx, "system", "user", "{}", Options{})
	if err != nil {
		t.Errorf("Complete() error: %v", err)
	}
	if content == "" {
		t.Error("Complete() returned empty content")
	}
	if err := mock.Close(); err != nil {
		t.Errorf("Close() error: %v", err)
	}
}

func TestConfig_AllFields(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Provider: "openai",
		Model:    "gpt-4o-mini",
		APIKey:   "test-api-key",
		BaseURL:  "https://api.test.com",
		Options: map[string]interface{}{
			"temperature": 0.7,
			"max_tokens":  1000,
		},
	}
	if cfg.Provider != "openai" {
		t.Errorf("Config.Provider = %q, want %q", cfg.Provider, "openai")
	}
	if cfg.Model != "gpt-4o-mini" {
		t.Errorf("Config.Model = %q, want %q", cfg.Model, "gpt-4o-mini")
	}
	if cfg.APIKey != "test-api-key" {
		t.Errorf("Config.APIKey = %q, want %q", cfg.APIKey, "test-api-key")
	}
	if cfg.BaseURL != "https://api.test.com" {
		t.Errorf("Config.BaseURL = %q, want %q", cfg.BaseURL, "https://api.test.com")
	}
	if temp, ok := cfg.Options["temperature"].(float64); !ok || temp != 0.7 {
		t.Errorf("Config.Options[temperature] = %v, want 0.7", cfg.Options["temperature"])
	}
}
