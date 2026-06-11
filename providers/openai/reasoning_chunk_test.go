package openai

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
)

// TestConvertChunkSurfacesReasoningContent verifies the fork patch: a streamed
// delta carrying the non-standard "reasoning_content" field (DeepSeek-R1 /
// Moonshot Kimi / Cloudflare Workers AI) is surfaced on ChunkDelta.Reasoning.
// It exercises the full mechanism — openai-go capturing the unknown field into
// Delta.JSON.ExtraFields, then convertChunk copying it across.
func TestConvertChunkSurfacesReasoningContent(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "reasoning_content string",
			raw:  `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":"let me think"}}]}`,
			want: "let me think",
		},
		{
			name: "reasoning string fallback",
			raw:  `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"alt field"}}]}`,
			want: "alt field",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var chunk openai.ChatCompletionChunk
			if err := json.Unmarshal([]byte(tc.raw), &chunk); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			got := convertChunk(&chunk)
			if len(got.Choices) != 1 {
				t.Fatalf("want 1 choice, got %d", len(got.Choices))
			}
			r := got.Choices[0].Delta.Reasoning
			if r == nil {
				t.Fatalf("reasoning was dropped (nil)")
			}
			if r.Content != tc.want {
				t.Fatalf("reasoning content = %q, want %q", r.Content, tc.want)
			}
		})
	}
}

// TestConvertChunkNoReasoningStaysNil verifies a normal content delta does not
// fabricate a Reasoning value (and that a "reasoning" OBJECT — not a string —
// is skipped rather than panicking).
func TestConvertChunkNoReasoningStaysNil(t *testing.T) {
	cases := []string{
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning":{"effort":"high"}}}]}`,
	}
	for _, raw := range cases {
		var chunk openai.ChatCompletionChunk
		if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		got := convertChunk(&chunk)
		if got.Choices[0].Delta.Reasoning != nil {
			t.Fatalf("expected nil reasoning, got %+v", got.Choices[0].Delta.Reasoning)
		}
	}
}
