package openai

import (
	"context"
	stderrors "errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestNewCompatible(t *testing.T) {
	// Note: Not using t.Parallel() here because child test uses t.Setenv.

	t.Run("creates provider with valid config", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://localhost:8080/v1",
			DefaultAPIKey:  "test-key",
			RequireAPIKey:  false,
			Capabilities: providers.Capabilities{
				Completion: true,
			},
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, "test-provider", provider.Name())
	})

	t.Run("returns error when name is missing", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			DefaultBaseURL: "http://localhost:8080/v1",
		}

		provider, err := NewCompatible(baseCfg)
		require.Error(t, err)
		require.Nil(t, provider)
		require.Contains(t, err.Error(), "provider name is required")
	})

	t.Run("returns error when API key required but missing", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:          "test-provider",
			APIKeyEnvVar:  "TEST_API_KEY",
			RequireAPIKey: true,
		}

		provider, err := NewCompatible(baseCfg)
		require.Error(t, err)
		require.Nil(t, provider)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
	})

	t.Run("uses default API key when not required", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:          "test-provider",
			DefaultAPIKey: "default-key",
			RequireAPIKey: false,
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("uses config base URL over default", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://default:8080/v1",
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg, config.WithBaseURL("http://custom:9090/v1"))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("uses environment variable for base URL", func(t *testing.T) {
		t.Setenv("TEST_BASE_URL", "http://env:8080/v1")

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			BaseURLEnvVar:  "TEST_BASE_URL",
			DefaultBaseURL: "http://default:8080/v1",
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestNewCompatibleRequireBaseURL(t *testing.T) {
	// Note: Not using t.Parallel() because subtests use t.Setenv.

	const (
		envVar       = "TEST_COMPATIBLE_REQUIRE_BASEURL"
		providerName = "test-provider"
	)

	tests := []struct {
		name           string
		baseURLEnvVar  string
		defaultBaseURL string
		envValue       string
		withBaseURL    string
		requireBaseURL bool
		wantErr        string // empty means no error expected.
	}{
		{
			name:           "errors when required and no env var configured",
			requireBaseURL: true,
			wantErr:        providerName + " base URL is required (set via WithBaseURL option)",
		},
		{
			name:           "errors when required and env var name set but unset",
			baseURLEnvVar:  envVar,
			requireBaseURL: true,
			wantErr: providerName + ` base URL is required (set via WithBaseURL option or "` +
				envVar + `" env var)`,
		},
		{
			name:           "succeeds when required and WithBaseURL is provided",
			requireBaseURL: true,
			withBaseURL:    "http://custom:9090/v1",
		},
		{
			name:           "succeeds when required and env var resolves",
			baseURLEnvVar:  envVar,
			envValue:       "http://env:8080/v1",
			requireBaseURL: true,
		},
		{
			name:           "succeeds when required and DefaultBaseURL is set",
			defaultBaseURL: "http://default:8080/v1",
			requireBaseURL: true,
		},
		{
			name:           "does not error when not required and no URL resolves",
			requireBaseURL: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(envVar, tc.envValue)

			baseCfg := CompatibleConfig{
				Name:           providerName,
				BaseURLEnvVar:  tc.baseURLEnvVar,
				DefaultAPIKey:  "test-key",
				DefaultBaseURL: tc.defaultBaseURL,
				RequireBaseURL: tc.requireBaseURL,
			}

			var opts []config.Option
			if tc.withBaseURL != "" {
				opts = append(opts, config.WithBaseURL(tc.withBaseURL))
			}

			provider, err := NewCompatible(baseCfg, opts...)

			if tc.wantErr != "" {
				require.EqualError(t, err, tc.wantErr)
				require.Nil(t, provider)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, provider)
		})
	}
}

func TestCompatibleProviderCapabilities(t *testing.T) {
	t.Parallel()

	expectedCaps := providers.Capabilities{
		Completion:          true,
		CompletionStreaming: true,
		Embedding:           true,
	}

	baseCfg := CompatibleConfig{
		Name:         "test-provider",
		Capabilities: expectedCaps,
	}

	provider, err := NewCompatible(baseCfg)
	require.NoError(t, err)

	caps := provider.Capabilities()
	require.Equal(t, expectedCaps, caps)
}

func TestValidateCompletionParams(t *testing.T) {
	t.Parallel()

	t.Run("returns error when model is empty", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}

		err := validateCompletionParams(params)
		require.Error(t, err)
		require.Contains(t, err.Error(), "model is required")
	})

	t.Run("returns error when messages is empty", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: []providers.Message{},
		}

		err := validateCompletionParams(params)
		require.Error(t, err)
		require.Contains(t, err.Error(), "at least one message is required")
	})

	t.Run("returns error for unknown message role", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "gpt-4",
			Messages: []providers.Message{
				{Role: "unknown_role", Content: "Hello"},
			},
		}

		err := validateCompletionParams(params)
		require.Error(t, err)
		require.Contains(t, err.Error(), "unknown message role")
	})

	t.Run("accepts valid params", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "gpt-4",
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Hello"},
			},
		}

		err := validateCompletionParams(params)
		require.NoError(t, err)
	})
}

func TestConvertResponseFormat(t *testing.T) {
	t.Parallel()

	t.Run("handles nil format", func(t *testing.T) {
		t.Parallel()

		result := convertResponseFormat(nil)
		require.NotNil(t, result)
	})

	t.Run("converts json_object format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: responseFormatJSONObject}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfJSONObject)
	})

	t.Run("converts json_schema format", func(t *testing.T) {
		t.Parallel()

		strict := true
		format := &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:        "test_schema",
				Description: "Test schema",
				Schema:      map[string]any{"type": "object"},
				Strict:      &strict,
			},
		}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfJSONSchema)
	})

	t.Run("defaults to text format for unknown type", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: "unknown"}
		result := convertResponseFormat(format)
		require.NotNil(t, result.OfText)
	})
}

func TestConvertEmbeddingParams(t *testing.T) {
	t.Parallel()

	t.Run("converts string input", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: "Hello, world!",
		}

		result := convertEmbeddingParams(params)
		require.NotNil(t, result.Input.OfString)
	})

	t.Run("converts string array input", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: []string{"Hello", "World"},
		}

		result := convertEmbeddingParams(params)
		require.NotNil(t, result.Input.OfArrayOfStrings)
	})

	t.Run("handles unknown input type", func(t *testing.T) {
		t.Parallel()

		params := providers.EmbeddingParams{
			Model: "text-embedding-3-small",
			Input: 12345, // Unsupported type.
		}

		result := convertEmbeddingParams(params)
		// Should convert to string representation.
		require.NotNil(t, result.Input.OfString)
	})

	t.Run("includes optional parameters", func(t *testing.T) {
		t.Parallel()

		dims := 256
		params := providers.EmbeddingParams{
			Model:          "text-embedding-3-small",
			Input:          "Hello",
			EncodingFormat: "float",
			Dimensions:     &dims,
			User:           "test-user",
		}

		result := convertEmbeddingParams(params)
		require.Equal(t, int64(256), result.Dimensions.Value)
		require.Equal(t, "test-user", result.User.Value)
	})
}

func TestStreamingContextCancellation(t *testing.T) {
	t.Parallel()

	t.Run("respects context cancellation", func(t *testing.T) {
		t.Parallel()

		baseCfg := CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: "http://localhost:9999/v1", // Non-existent server.
			DefaultAPIKey:  "test-key",
		}

		provider, err := NewCompatible(baseCfg)
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately.

		params := providers.CompletionParams{
			Model:    "test-model",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}

		chunks, errs := provider.CompletionStream(ctx, params)

		// Drain channels.
		for range chunks {
		}
		<-errs

		// Test passes if it doesn't hang.
	})

	// Regression for #85: when the caller cancels the context, the consumer
	// reading from `errs` should receive `context.Canceled` (not a closed
	// channel with no value) so it can distinguish "stream finished cleanly"
	// from "I cancelled the request".
	t.Run("surfaces ctx.Err on cancellation", func(t *testing.T) {
		t.Parallel()

		// Slow upstream: holds the connection open until the test cancels.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
		}))
		t.Cleanup(srv.Close)

		provider, err := NewCompatible(CompatibleConfig{
			Name:           "test-provider",
			DefaultBaseURL: srv.URL + "/v1",
			DefaultAPIKey:  "test-key",
		})
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(context.Background())
		params := providers.CompletionParams{
			Model:    "test-model",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
		}

		chunks, errs := provider.CompletionStream(ctx, params)

		go func() {
			time.Sleep(50 * time.Millisecond)
			cancel()
		}()

		// Drain chunks until the channel closes.
		for range chunks {
		}

		select {
		case got, ok := <-errs:
			require.True(t, ok, "errs should yield a value before close")
			require.ErrorIs(t, got, context.Canceled)
		case <-time.After(2 * time.Second):
			t.Fatal("expected an error on errs after cancellation, got nothing")
		}
	})
}

// TestNewCompatibleIgnoresSDKCustomHeadersEnv is a regression test: openai-go
// v3 reads OPENAI_CUSTOM_HEADERS and adds its entries to every request, which
// would forward them to non-OpenAI hosts. Headers set explicitly via
// config.WithHeader must still be sent.
func TestNewCompatibleIgnoresSDKCustomHeadersEnv(t *testing.T) {
	// Note: Not using t.Parallel() because of t.Setenv.
	t.Setenv(envSDKCustomHeaders, "X-Leak: secret\nX-Other: 2\nmalformed-line")

	var (
		mu       sync.Mutex
		captured []http.Header
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		captured = append(captured, r.Header.Clone())
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"stop","type":"invalid_request_error","code":"test"}}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := NewCompatible(
		CompatibleConfig{
			Name:                             "test-provider",
			DefaultAPIKey:                    "test-key",
			DefaultBaseURL:                   srv.URL + "/v1",
			UseResponsesAPIForReasoningTools: true,
		},
		config.WithHeader("X-Other", "explicit"),
	)
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    "test-model",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
	}
	responsesParams := params
	responsesParams.ReasoningEffort = providers.ReasoningEffortLow
	responsesParams.Tools = []providers.Tool{{Type: "function", Function: providers.Function{Name: "noop"}}}

	// Errors are expected: the server rejects every request after capturing it.
	_, _ = provider.Completion(ctx, params)
	_, _ = provider.Completion(ctx, responsesParams)
	_, _ = provider.Embedding(ctx, providers.EmbeddingParams{Model: "test-embed", Input: "hi"})
	_, _ = provider.ListModels(ctx)
	chunks, errs := provider.CompletionStream(ctx, params)
	for range chunks {
	}
	<-errs

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, captured, 5)
	for _, h := range captured {
		require.Empty(t, h.Values("X-Leak"))
		require.Equal(t, []string{"explicit"}, h.Values("X-Other"))
	}
}

// TestConvertErrorIncludesServerDetail is a regression test: openai-go v3's
// Error() no longer includes the response body, so the server's message and
// code must be surfaced by ConvertError while keeping the SDK error reachable.
func TestConvertErrorIncludesServerDetail(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(
			[]byte(
				`{"error":{"message":"Failed to call a function","type":"invalid_request_error","code":"tool_use_failed"}}`,
			),
		)
	}))
	t.Cleanup(srv.Close)

	provider, err := NewCompatible(CompatibleConfig{
		Name:           "test-provider",
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: srv.URL + "/v1",
	})
	require.NoError(t, err)

	_, err = provider.Completion(context.Background(), providers.CompletionParams{
		Model:    "test-model",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "Hello"}},
	})
	require.Error(t, err)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
	require.Contains(t, err.Error(), "Failed to call a function")
	require.Contains(t, err.Error(), "code=tool_use_failed")
	require.Contains(t, err.Error(), "type=invalid_request_error")
	require.NotContains(t, err.Error(), srv.URL)

	var apiErr *openai.Error
	require.True(t, stderrors.As(err, &apiErr))
	require.Equal(t, "tool_use_failed", apiErr.Code)
}

func TestWithAPIErrorDetail(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		apiErr *openai.Error
		want   string
	}{
		{
			name:   "no detail returns original error text",
			apiErr: &openai.Error{StatusCode: http.StatusBadRequest},
			want:   "OpenAI API error: 400 Bad Request",
		},
		{
			name:   "message only",
			apiErr: &openai.Error{StatusCode: http.StatusBadRequest, Message: "bad"},
			want:   "OpenAI API error: 400 Bad Request: bad",
		},
		{
			name:   "code only",
			apiErr: &openai.Error{StatusCode: http.StatusBadRequest, Code: "c"},
			want:   "OpenAI API error: 400 Bad Request: (code=c)",
		},
		{
			name:   "all fields",
			apiErr: &openai.Error{StatusCode: http.StatusBadRequest, Code: "c", Message: "bad", Type: "t"},
			want:   "OpenAI API error: 400 Bad Request: bad (code=c, type=t)",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := withAPIErrorDetail(tc.apiErr, tc.apiErr)
			require.EqualError(t, got, tc.want)
			require.ErrorIs(t, got, tc.apiErr)
		})
	}
}
