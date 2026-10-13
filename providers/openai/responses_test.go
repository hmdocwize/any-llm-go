package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func float64Ptr(v float64) *float64 { return &v }
func intPtr(v int) *int             { return &v }

// TestShouldUseResponsesAPI verifies the routing gate: only the OpenAI
// platform (config flag), only with tools, and only with a non-none
// reasoning effort.
func TestShouldUseResponsesAPI(t *testing.T) {
	tools := []providers.Tool{{Type: "function", Function: providers.Function{Name: "noop"}}}
	cases := []struct {
		name    string
		enabled bool
		params  providers.CompletionParams
		want    bool
	}{
		{
			"tools + high effort",
			true,
			providers.CompletionParams{Tools: tools, ReasoningEffort: providers.ReasoningEffortHigh},
			true,
		},
		{
			"tools + medium effort",
			true,
			providers.CompletionParams{Tools: tools, ReasoningEffort: providers.ReasoningEffortMedium},
			true,
		},
		{
			"tools + none effort",
			true,
			providers.CompletionParams{Tools: tools, ReasoningEffort: providers.ReasoningEffortNone},
			false,
		},
		{"tools + no effort", true, providers.CompletionParams{Tools: tools}, false},
		{
			"no tools + high effort",
			true,
			providers.CompletionParams{ReasoningEffort: providers.ReasoningEffortHigh},
			false,
		},
		{
			"disabled (compatible endpoint)",
			false,
			providers.CompletionParams{Tools: tools, ReasoningEffort: providers.ReasoningEffortHigh},
			false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := &CompatibleProvider{compatibleConfig: CompatibleConfig{
				Name:                             "openai",
				UseResponsesAPIForReasoningTools: tc.enabled,
			}}
			if got := p.shouldUseResponsesAPI(tc.params); got != tc.want {
				t.Fatalf("shouldUseResponsesAPI = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestConvertResponsesParamsWire verifies the request marshals to the exact
// wire fields the Responses API requires: store=false, reasoning effort +
// summary, max_output_tokens, non-strict function tools, and the full
// multi-turn input item sequence (system → user → function_call →
// function_call_output).
func TestConvertResponsesParamsWire(t *testing.T) {
	params := providers.CompletionParams{
		Model: "gpt-5.6-luna",
		Messages: []providers.Message{
			{Role: providers.RoleSystem, Content: "be helpful"},
			{Role: providers.RoleUser, Content: "weather in Paris?"},
			{Role: providers.RoleAssistant, ToolCalls: []providers.ToolCall{{
				ID: "call_1", Type: "function",
				Function: providers.FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`},
			}}},
			{Role: providers.RoleTool, ToolCallID: "call_1", Content: "22C"},
		},
		MaxTokens:       intPtr(128000),
		ReasoningEffort: providers.ReasoningEffortHigh,
		// Opt in to the reasoning summary (now off by default) so this wire test
		// still exercises the summary=auto field. See the summary-gate test below.
		Extra: map[string]any{"reasoning_summary": "auto"},
		Tools: []providers.Tool{{Type: "function", Function: providers.Function{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  map[string]any{"type": "object"},
		}}},
	}

	raw, err := json.Marshal(convertResponsesParams(params))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if wire["store"] != false {
		t.Errorf("store = %v, want false", wire["store"])
	}
	if wire["max_output_tokens"] != float64(128000) {
		t.Errorf("max_output_tokens = %v", wire["max_output_tokens"])
	}
	reasoning, _ := wire["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" || reasoning["summary"] != "auto" {
		t.Errorf("reasoning = %v, want effort=high summary=auto", reasoning)
	}

	tools, _ := wire["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v, want 1 entry", wire["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "get_weather" || tool["strict"] != false {
		t.Errorf("tool = %v, want flat function shape with strict=false", tool)
	}

	input, _ := wire["input"].([]any)
	if len(input) != 4 {
		t.Fatalf("input has %d items, want 4: %s", len(input), raw)
	}
	wantTypes := []struct{ itemType, roleOrCallID string }{
		{"message", "system"},
		{"message", "user"},
		{"function_call", "call_1"},
		{"function_call_output", "call_1"},
	}
	for i, want := range wantTypes {
		item, _ := input[i].(map[string]any)
		itemType, _ := item["type"].(string)
		if itemType == "" {
			itemType = "message" // EasyInputMessage omits type
		}
		if itemType != want.itemType {
			t.Errorf("input[%d].type = %q, want %q", i, itemType, want.itemType)
		}
		got := item["role"]
		if got == nil {
			got = item["call_id"]
		}
		if got != want.roleOrCallID {
			t.Errorf("input[%d] role/call_id = %v, want %q", i, got, want.roleOrCallID)
		}
	}
}

// TestConvertResponsesParamsAutoEffortOmitted verifies "auto" effort omits
// the reasoning block entirely (not a valid Responses API level).
func TestConvertResponsesParamsAutoEffortOmitted(t *testing.T) {
	raw, err := json.Marshal(convertResponsesParams(providers.CompletionParams{
		Model:           "gpt-5.6-luna",
		Messages:        []providers.Message{{Role: providers.RoleUser, Content: "hi"}},
		ReasoningEffort: providers.ReasoningEffortAuto,
	}))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, present := wire["reasoning"]; present {
		t.Errorf("reasoning block present for auto effort: %s", raw)
	}
}

// TestConvertResponsesParamsReasoningSummaryGate verifies the reasoning summary
// is opt-in: absent from the request unless Extra["reasoning_summary"]="auto",
// while the reasoning effort is applied either way (the model still reasons).
func TestConvertResponsesParamsReasoningSummaryGate(t *testing.T) {
	reasoningOf := func(extra map[string]any) map[string]any {
		raw, err := json.Marshal(convertResponsesParams(providers.CompletionParams{
			Model:           "gpt-5.6-luna",
			Messages:        []providers.Message{{Role: providers.RoleUser, Content: "hi"}},
			ReasoningEffort: providers.ReasoningEffortHigh,
			Extra:           extra,
		}))
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var wire map[string]any
		if err := json.Unmarshal(raw, &wire); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		reasoning, _ := wire["reasoning"].(map[string]any)
		return reasoning
	}

	// No opt-in → effort present, summary omitted.
	off := reasoningOf(nil)
	if off["effort"] != "high" {
		t.Errorf("default: effort = %v, want high", off["effort"])
	}
	if _, present := off["summary"]; present {
		t.Errorf("default: summary present, want omitted (opt-in only): %v", off)
	}

	// Opt-in → summary=auto alongside the effort.
	on := reasoningOf(map[string]any{"reasoning_summary": "auto"})
	if on["effort"] != "high" || on["summary"] != "auto" {
		t.Errorf("opted-in: reasoning = %v, want effort=high summary=auto", on)
	}
}

// TestConvertResponsesResponse verifies output items map back to
// chat-completion shape: message text concatenated, function calls to
// ToolCalls with finish_reason tool_calls, and usage carried across.
func TestConvertResponsesResponse(t *testing.T) {
	raw := `{
		"id": "resp_1", "created_at": 1770000000, "model": "gpt-5.6-luna", "status": "completed",
		"output": [
			{"type": "reasoning", "id": "rs_1", "summary": []},
			{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
			 "content": [{"type": "output_text", "text": "Checking "}, {"type": "output_text", "text": "now."}]},
			{"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
		],
		"usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
			"input_tokens_details": {"cached_tokens": 60}, "output_tokens_details": {"reasoning_tokens": 30}}
	}`
	var resp responses.Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got := convertResponsesResponse(&resp)
	if len(got.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(got.Choices))
	}
	choice := got.Choices[0]
	if choice.Message.Content != "Checking now." {
		t.Errorf("content = %q", choice.Message.Content)
	}
	if choice.FinishReason != providers.FinishReasonToolCalls {
		t.Errorf("finish_reason = %q, want tool_calls", choice.FinishReason)
	}
	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(choice.Message.ToolCalls))
	}
	toolCall := choice.Message.ToolCalls[0]
	if toolCall.ID != "call_9" || toolCall.Function.Name != "get_weather" ||
		toolCall.Function.Arguments != `{"city":"Paris"}` {
		t.Errorf("tool call = %+v", toolCall)
	}
	if got.Usage == nil {
		t.Fatal("usage dropped")
	}
	want := providers.Usage{
		PromptTokens:       100,
		CompletionTokens:   50,
		TotalTokens:        150,
		ReasoningTokens:    30,
		CachedPromptTokens: 60,
	}
	if *got.Usage != want {
		t.Errorf("usage = %+v, want %+v", *got.Usage, want)
	}
}

// TestConvertResponsesInputMultiModal verifies image parts survive as
// input_image entries.
func TestConvertResponsesInputMultiModal(t *testing.T) {
	items := convertResponsesInput([]providers.Message{{
		Role: providers.RoleUser,
		Content: []providers.ContentPart{
			{Type: contentTypeText, Text: "what is this?"},
			{Type: contentTypeImageURL, ImageURL: &providers.ImageURL{URL: "data:image/png;base64,AAAA"}},
		},
	}})
	raw, err := json.Marshal(items)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var wire []map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(wire) != 1 {
		t.Fatalf("items = %d, want 1", len(wire))
	}
	content, _ := wire[0]["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content parts = %d, want 2: %s", len(content), raw)
	}
	imagePart, _ := content[1].(map[string]any)
	if imagePart["type"] != "input_image" || imagePart["image_url"] != "data:image/png;base64,AAAA" {
		t.Errorf("image part = %v", imagePart)
	}
}

// TestConvertResponsesUsageTemperatureOmitted ensures an unset temperature is
// absent from the wire (reasoning models reject the parameter).
func TestConvertResponsesParamsTemperature(t *testing.T) {
	base := providers.CompletionParams{
		Model:    "gpt-5.6-luna",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hi"}},
	}

	raw, _ := json.Marshal(convertResponsesParams(base))
	var wire map[string]any
	_ = json.Unmarshal(raw, &wire)
	if _, present := wire["temperature"]; present {
		t.Errorf("unset temperature reached the wire: %s", raw)
	}

	base.Temperature = float64Ptr(0.3)
	raw, _ = json.Marshal(convertResponsesParams(base))
	_ = json.Unmarshal(raw, &wire)
	if wire["temperature"] != 0.3 {
		t.Errorf("temperature = %v, want 0.3", wire["temperature"])
	}
}

// TestConvertMessagesMultiModalToolResult (chat path) verifies a tool message
// carrying image parts expands to a text tool message plus a follow-up user
// message with the images — Chat Completions tool messages are text-only.
func TestConvertMessagesMultiModalToolResult(t *testing.T) {
	messages, err := convertMessages([]providers.Message{
		{Role: providers.RoleAssistant, ToolCalls: []providers.ToolCall{{
			ID: "call_1", Type: "function",
			Function: providers.FunctionCall{Name: "read_cf_image", Arguments: "{}"},
		}}},
		{Role: providers.RoleTool, ToolCallID: "call_1", Content: []providers.ContentPart{
			{Type: contentTypeText, Text: "page 3 rendered"},
			{Type: contentTypeImageURL, ImageURL: &providers.ImageURL{URL: "data:image/png;base64,AAAA"}},
		}},
	})
	if err != nil {
		t.Fatalf("convertMessages: %v", err)
	}
	if len(messages) != 3 {
		t.Fatalf("messages = %d, want 3 (assistant, tool, user-with-image)", len(messages))
	}

	raw, _ := json.Marshal(messages)
	var wire []map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	toolMsg := wire[1]
	if toolMsg["role"] != "tool" || toolMsg["content"] != "page 3 rendered" || toolMsg["tool_call_id"] != "call_1" {
		t.Errorf("tool message = %v", toolMsg)
	}
	userMsg := wire[2]
	if userMsg["role"] != "user" {
		t.Fatalf("follow-up role = %v, want user", userMsg["role"])
	}
	content, _ := userMsg["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("user content parts = %d, want 2 (header + image): %s", len(content), raw)
	}
	imagePart, _ := content[1].(map[string]any)
	imageURL, _ := imagePart["image_url"].(map[string]any)
	if imagePart["type"] != "image_url" || imageURL["url"] != "data:image/png;base64,AAAA" {
		t.Errorf("image part = %v", imagePart)
	}
}

// TestConvertMessagesTextPartsToolResult verifies a tool result with only
// text parts joins them into the tool message (previously flattened to "").
func TestConvertMessagesTextPartsToolResult(t *testing.T) {
	messages, err := convertMessages([]providers.Message{
		{Role: providers.RoleTool, ToolCallID: "call_1", Content: []providers.ContentPart{
			{Type: contentTypeText, Text: "line one"},
			{Type: contentTypeText, Text: "line two"},
		}},
	})
	if err != nil {
		t.Fatalf("convertMessages: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(messages))
	}
	raw, _ := json.Marshal(messages[0])
	var wire map[string]any
	_ = json.Unmarshal(raw, &wire)
	if wire["content"] != "line one\nline two" {
		t.Errorf("content = %q, want joined text", wire["content"])
	}
}

// TestConvertResponsesInputMultiModalToolResult (responses path) verifies the
// same expansion: function_call_output text plus a user message with images.
func TestConvertResponsesInputMultiModalToolResult(t *testing.T) {
	items := convertResponsesInput([]providers.Message{
		{Role: providers.RoleTool, ToolCallID: "call_1", Content: []providers.ContentPart{
			{Type: contentTypeImageURL, ImageURL: &providers.ImageURL{URL: "data:image/png;base64,BBBB"}},
		}},
	})
	raw, _ := json.Marshal(items)
	var wire []map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(wire) != 2 {
		t.Fatalf("items = %d, want 2 (function_call_output + user image message): %s", len(wire), raw)
	}
	if wire[0]["type"] != "function_call_output" || wire[0]["call_id"] != "call_1" {
		t.Errorf("output item = %v", wire[0])
	}
	if output, _ := wire[0]["output"].(string); output == "" {
		t.Errorf("image-only tool result produced empty output text")
	}
	content, _ := wire[1]["content"].([]any)
	if wire[1]["role"] != "user" || len(content) != 2 {
		t.Fatalf("follow-up = %v, want user message with 2 parts", wire[1])
	}
	imagePart, _ := content[1].(map[string]any)
	if imagePart["type"] != "input_image" || imagePart["image_url"] != "data:image/png;base64,BBBB" {
		t.Errorf("image part = %v", imagePart)
	}
}

// testResponsesParams returns params that route to the Responses API
// (tools + a non-none reasoning effort).
func testResponsesParams() providers.CompletionParams {
	return providers.CompletionParams{
		Model:           "gpt-5.6-luna",
		Messages:        []providers.Message{{Role: providers.RoleUser, Content: "hi"}},
		ReasoningEffort: providers.ReasoningEffortHigh,
		Tools:           []providers.Tool{{Type: "function", Function: providers.Function{Name: "noop"}}},
	}
}

// newTestResponsesProvider returns a provider with Responses routing enabled
// pointed at a test server running handler.
func newTestResponsesProvider(t *testing.T, handler http.HandlerFunc) *CompatibleProvider {
	t.Helper()

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	provider, err := NewCompatible(CompatibleConfig{
		Name:                             "test-provider",
		DefaultAPIKey:                    "test-key",
		DefaultBaseURL:                   srv.URL + "/v1",
		UseResponsesAPIForReasoningTools: true,
	})
	require.NoError(t, err)
	return provider
}

// TestCompletionViaResponsesFailedStatus is a regression test: an HTTP-200
// Responses body with status=failed must surface as an error, not as an empty
// successful completion.
func TestCompletionViaResponsesFailedStatus(t *testing.T) {
	t.Parallel()

	provider := newTestResponsesProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","created_at":1770000000,"model":"gpt-5.6-luna",` +
			`"status":"failed","error":{"code":"server_error","message":"boom"},"output":[]}`))
	})

	got, err := provider.Completion(context.Background(), testResponsesParams())
	require.Error(t, err)
	require.Nil(t, got)
	require.Contains(t, err.Error(), "server_error: boom")
}

// TestResponsesFinishReason verifies the Responses termination mapping,
// including content_filter and unknown incomplete reasons (regression: both
// previously reported a clean stop).
func TestResponsesFinishReason(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name             string
		incompleteReason string
		sawToolCall      bool
		want             string
	}{
		{"completed", "", false, providers.FinishReasonStop},
		{"completed with tool call", "", true, providers.FinishReasonToolCalls},
		{"max output tokens", incompleteReasonMaxOutputTokens, true, providers.FinishReasonLength},
		{"content filter", incompleteReasonContentFilter, false, providers.FinishReasonContentFilter},
		{"content filter with tool call", incompleteReasonContentFilter, true, providers.FinishReasonContentFilter},
		{"unknown reason surfaced verbatim", "max_messages", true, "max_messages"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tc.want, responsesFinishReason(tc.incompleteReason, tc.sawToolCall))
		})
	}
}

// TestConvertResponsesResponseContentFilter verifies the non-streaming
// converter reports a content-filtered incomplete response as content_filter.
func TestConvertResponsesResponseContentFilter(t *testing.T) {
	t.Parallel()

	var resp responses.Response
	require.NoError(t, json.Unmarshal([]byte(`{"id":"resp_1","model":"gpt-5.6-luna","status":"incomplete",`+
		`"incomplete_details":{"reason":"content_filter"},"output":[]}`), &resp))

	got := convertResponsesResponse(&resp)
	require.Len(t, got.Choices, 1)
	require.Equal(t, providers.FinishReasonContentFilter, got.Choices[0].FinishReason)
}

// TestStreamCompletionViaResponses verifies the streaming Responses path:
// chunks report the server-resolved model from response.created, reasoning
// deltas carry the assistant role, and a content-filtered response.incomplete
// ends with finish_reason content_filter.
func TestStreamCompletionViaResponses(t *testing.T) {
	t.Parallel()

	const resolvedModel = "gpt-5.6-luna-2026-01-01"
	responseJSON := func(status, extra string) string {
		return fmt.Sprintf(`{"id":"resp_1","created_at":1770000000,"model":%q,"status":%q,"output":[]%s}`,
			resolvedModel, status, extra)
	}
	events := []struct{ eventType, data string }{
		{streamEventCreated, `{"type":"response.created","sequence_number":0,"response":` +
			responseJSON("in_progress", "") + `}`},
		{streamEventReasoningSummaryDelta, `{"type":"response.reasoning_summary_text.delta","sequence_number":1,` +
			`"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"thinking"}`},
		{streamEventIncomplete, `{"type":"response.incomplete","sequence_number":2,"response":` +
			responseJSON("incomplete", `,"incomplete_details":{"reason":"content_filter"}`) + `}`},
	}

	provider := newTestResponsesProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.eventType, event.data)
		}
	})

	chunks, errs := provider.CompletionStream(context.Background(), testResponsesParams())
	var got []providers.ChatCompletionChunk
	for chunk := range chunks {
		got = append(got, chunk)
	}
	require.NoError(t, <-errs)
	require.Len(t, got, 2)

	for _, chunk := range got {
		require.Equal(t, resolvedModel, chunk.Model)
		require.Len(t, chunk.Choices, 1)
	}

	reasoning := got[0].Choices[0]
	require.Equal(t, providers.RoleAssistant, reasoning.Delta.Role)
	require.NotNil(t, reasoning.Delta.Reasoning)
	require.Equal(t, "thinking", reasoning.Delta.Reasoning.Content)

	require.Equal(t, providers.FinishReasonContentFilter, got[1].Choices[0].FinishReason)
}

// TestResponsesRouteRejectsUnsupportedParams is a regression test: Stop and
// Seed have no Responses API equivalent, so a routed request carrying either
// must fail with ErrUnsupportedParam before any HTTP call instead of silently
// dropping them.
func TestResponsesRouteRejectsUnsupportedParams(t *testing.T) {
	t.Parallel()

	seed := 42
	cases := []struct {
		name   string
		mutate func(*providers.CompletionParams)
	}{
		{"stop", func(p *providers.CompletionParams) { p.Stop = []string{"END"} }},
		{"seed", func(p *providers.CompletionParams) { p.Seed = &seed }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var calls atomic.Int32
			provider := newTestResponsesProvider(t, func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			})
			params := testResponsesParams()
			tc.mutate(&params)

			got, err := provider.Completion(context.Background(), params)
			require.ErrorIs(t, err, errors.ErrUnsupportedParam)
			require.Nil(t, got)

			chunks, errs := provider.CompletionStream(context.Background(), params)
			for range chunks {
				t.Fatal("unexpected chunk")
			}
			require.ErrorIs(t, <-errs, errors.ErrUnsupportedParam)

			require.Zero(t, calls.Load())
		})
	}
}

// TestConvertMessagesParallelToolResultsDeferImages is a regression test:
// with parallel tool calls, images re-attached from a non-final tool result
// must follow the whole tool-result group, not split it (Chat Completions
// rejects a user message before every tool_call_id is answered).
func TestConvertMessagesParallelToolResultsDeferImages(t *testing.T) {
	t.Parallel()

	messages, err := convertMessages(testParallelToolResultMessages())
	require.NoError(t, err)

	raw, err := json.Marshal(messages)
	require.NoError(t, err)
	var wire []map[string]any
	require.NoError(t, json.Unmarshal(raw, &wire))

	want := []struct{ role, toolCallID string }{
		{providers.RoleAssistant, ""},
		{providers.RoleTool, "call_a"},
		{providers.RoleTool, "call_b"},
		{providers.RoleUser, ""},
		{providers.RoleUser, ""},
	}
	require.Len(t, wire, len(want), "%s", raw)
	for i, w := range want {
		require.Equal(t, w.role, wire[i]["role"], "message %d: %s", i, raw)
		if w.toolCallID != "" {
			require.Equal(t, w.toolCallID, wire[i]["tool_call_id"], "message %d", i)
		}
	}
	require.Equal(t, "next question", wire[4]["content"])
}

// TestConvertResponsesInputParallelToolResultsDeferImages is the Responses
// path counterpart: all function_call_output items precede re-attached images.
func TestConvertResponsesInputParallelToolResultsDeferImages(t *testing.T) {
	t.Parallel()

	raw, err := json.Marshal(convertResponsesInput(testParallelToolResultMessages()))
	require.NoError(t, err)
	var wire []map[string]any
	require.NoError(t, json.Unmarshal(raw, &wire))

	want := []struct{ itemType, roleOrCallID string }{
		{outputItemTypeFunctionCall, "call_a"},
		{outputItemTypeFunctionCall, "call_b"},
		{inputItemTypeFunctionCallOutput, "call_a"},
		{inputItemTypeFunctionCallOutput, "call_b"},
		{outputItemTypeMessage, providers.RoleUser}, // images re-attached from call_a
		{outputItemTypeMessage, providers.RoleUser}, // next user turn
	}
	require.Len(t, wire, len(want), "%s", raw)
	for i, w := range want {
		itemType, _ := wire[i]["type"].(string)
		if itemType == "" {
			itemType = outputItemTypeMessage // EasyInputMessage omits type
		}
		require.Equal(t, w.itemType, itemType, "item %d: %s", i, raw)
		got := wire[i]["role"]
		if got == nil {
			got = wire[i]["call_id"]
		}
		require.Equal(t, w.roleOrCallID, got, "item %d: %s", i, raw)
	}
	images, _ := wire[4]["content"].([]any)
	require.Len(t, images, 2, "header + image")
}

// testParallelToolResultMessages returns an assistant turn with two parallel
// tool calls whose first (non-final) result carries an image, followed by a
// new user turn.
func testParallelToolResultMessages() []providers.Message {
	return []providers.Message{
		{Role: providers.RoleAssistant, ToolCalls: []providers.ToolCall{
			{ID: "call_a", Type: "function", Function: providers.FunctionCall{Name: "screenshot", Arguments: "{}"}},
			{ID: "call_b", Type: "function", Function: providers.FunctionCall{Name: "lookup", Arguments: "{}"}},
		}},
		{Role: providers.RoleTool, ToolCallID: "call_a", Content: []providers.ContentPart{
			{Type: contentTypeText, Text: "rendered"},
			{Type: contentTypeImageURL, ImageURL: &providers.ImageURL{URL: "data:image/png;base64,AAAA"}},
		}},
		{Role: providers.RoleTool, ToolCallID: "call_b", Content: "found"},
		{Role: providers.RoleUser, Content: "next question"},
	}
}
