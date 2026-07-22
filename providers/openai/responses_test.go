package openai

import (
	"encoding/json"
	"testing"

	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/openai/openai-go/responses"
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
		{"tools + high effort", true, providers.CompletionParams{Tools: tools, ReasoningEffort: providers.ReasoningEffortHigh}, true},
		{"tools + medium effort", true, providers.CompletionParams{Tools: tools, ReasoningEffort: providers.ReasoningEffortMedium}, true},
		{"tools + none effort", true, providers.CompletionParams{Tools: tools, ReasoningEffort: providers.ReasoningEffortNone}, false},
		{"tools + no effort", true, providers.CompletionParams{Tools: tools}, false},
		{"no tools + high effort", true, providers.CompletionParams{ReasoningEffort: providers.ReasoningEffortHigh}, false},
		{"disabled (compatible endpoint)", false, providers.CompletionParams{Tools: tools, ReasoningEffort: providers.ReasoningEffortHigh}, false},
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
	if toolCall.ID != "call_9" || toolCall.Function.Name != "get_weather" || toolCall.Function.Arguments != `{"city":"Paris"}` {
		t.Errorf("tool call = %+v", toolCall)
	}
	if got.Usage == nil {
		t.Fatal("usage dropped")
	}
	want := providers.Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150, ReasoningTokens: 30, CachedPromptTokens: 60}
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
