package openai

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Responses API output item and content types.
const (
	outputItemTypeFunctionCall = "function_call"
	outputItemTypeMessage      = "message"
	outputContentTypeText      = "output_text"
)

// Responses API stream event types (subset this provider consumes).
const (
	streamEventCompleted             = "response.completed"
	streamEventCreated               = "response.created"
	streamEventError                 = "error"
	streamEventFailed                = "response.failed"
	streamEventIncomplete            = "response.incomplete"
	streamEventOutputItemDone        = "response.output_item.done"
	streamEventOutputTextDelta       = "response.output_text.delta"
	streamEventReasoningSummaryDelta = "response.reasoning_summary_text.delta"
)

const incompleteReasonMaxOutputTokens = "max_output_tokens"

// shouldUseResponsesAPI reports whether a request must be served by
// POST /v1/responses instead of /v1/chat/completions. OpenAI's reasoning
// models (gpt-5.6 family) reject function tools combined with reasoning on
// the Chat Completions endpoint ("Function tools with reasoning_effort are
// not supported ... use /v1/responses or set reasoning_effort to 'none'"),
// so tools + a non-none reasoning effort forces the Responses API. Gated on
// config because embedders of CompatibleProvider (DeepSeek, Groq, gateways,
// Cloudflare Workers AI, ...) are chat-completions-compatible only and do
// not serve /v1/responses.
func (p *CompatibleProvider) shouldUseResponsesAPI(params providers.CompletionParams) bool {
	return p.compatibleConfig.UseResponsesAPIForReasoningTools &&
		len(params.Tools) > 0 &&
		params.ReasoningEffort != "" &&
		params.ReasoningEffort != providers.ReasoningEffortNone
}

// completionViaResponses performs a non-streaming completion through the
// Responses API and converts the result back to chat-completion shape.
func (p *CompatibleProvider) completionViaResponses(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	resp, err := p.client.Responses.New(ctx, convertResponsesParams(params))
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertResponsesResponse(resp), nil
}

// streamCompletionViaResponses streams a completion through the Responses API,
// converting its event stream into chat-completion chunks so callers see the
// exact same wire shapes as the Chat Completions path:
//
//   - output_text deltas   → ChunkDelta.Content fragments
//   - reasoning summaries  → ChunkDelta.Reasoning fragments
//   - function calls       → ONE ToolCall chunk per call, emitted complete at
//     output_item.done (ID + Name + full Arguments). Argument fragments are
//     not re-streamed incrementally; consumers accumulate by ID exactly as
//     they do for chat-completions first-chunks.
//   - completed/incomplete → a final chunk carrying Usage and FinishReason
//
// The caller owns chunks/errs channel lifecycle (close), matching how
// CompletionStream's goroutine wraps the chat-completions path.
func (p *CompatibleProvider) streamCompletionViaResponses(
	ctx context.Context,
	params providers.CompletionParams,
	chunks chan<- providers.ChatCompletionChunk,
	errs chan<- error,
) {
	stream := p.client.Responses.NewStreaming(ctx, convertResponsesParams(params))

	var responseID string
	var created int64
	sawToolCall := false

	emit := func(delta providers.ChunkDelta, finishReason string, usage *providers.Usage) bool {
		chunk := providers.ChatCompletionChunk{
			ID:      responseID,
			Object:  objectChatCompletionChunk,
			Created: created,
			Model:   params.Model,
			Choices: []providers.ChunkChoice{{Delta: delta, FinishReason: finishReason}},
			Usage:   usage,
		}
		select {
		case chunks <- chunk:
			return true
		case <-ctx.Done():
			errs <- ctx.Err()
			return false
		}
	}

	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case streamEventCreated:
			responseID = event.Response.ID
			created = int64(event.Response.CreatedAt)

		case streamEventOutputTextDelta:
			if event.Delta.OfString == "" {
				continue
			}
			if !emit(providers.ChunkDelta{Role: providers.RoleAssistant, Content: event.Delta.OfString}, "", nil) {
				return
			}

		case streamEventReasoningSummaryDelta:
			if event.Delta.OfString == "" {
				continue
			}
			if !emit(providers.ChunkDelta{Reasoning: &providers.Reasoning{Content: event.Delta.OfString}}, "", nil) {
				return
			}

		case streamEventOutputItemDone:
			if event.Item.Type != outputItemTypeFunctionCall {
				continue
			}
			sawToolCall = true
			toolCall := providers.ToolCall{
				ID:   event.Item.CallID,
				Type: "function",
				Function: providers.FunctionCall{
					Name:      event.Item.Name,
					Arguments: event.Item.Arguments,
				},
			}
			if !emit(providers.ChunkDelta{Role: providers.RoleAssistant, ToolCalls: []providers.ToolCall{toolCall}}, "", nil) {
				return
			}

		case streamEventCompleted, streamEventIncomplete:
			finishReason := providers.FinishReasonStop
			if sawToolCall {
				finishReason = providers.FinishReasonToolCalls
			}
			if event.Type == streamEventIncomplete &&
				event.Response.IncompleteDetails.Reason == incompleteReasonMaxOutputTokens {
				finishReason = providers.FinishReasonLength
			}
			if !emit(providers.ChunkDelta{}, finishReason, convertResponsesUsage(event.Response.Usage)) {
				return
			}

		case streamEventFailed:
			errs <- errors.NewProviderError(p.compatibleConfig.Name, fmt.Errorf(
				"responses API failure: %s: %s", event.Response.Error.Code, event.Response.Error.Message))
			return

		case streamEventError:
			errs <- errors.NewProviderError(p.compatibleConfig.Name, fmt.Errorf(
				"responses API stream error: %s: %s", event.Code, event.Message))
			return
		}
	}

	if err := stream.Err(); err != nil {
		errs <- p.ConvertError(err)
	}
}

// convertResponsesParams converts providers.CompletionParams to Responses API
// request parameters.
func convertResponsesParams(params providers.CompletionParams) responses.ResponseNewParams {
	req := responses.ResponseNewParams{
		Model: shared.ResponsesModel(params.Model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: convertResponsesInput(params.Messages)},
		// Callers replay their own conversation history on every request;
		// never persist turns server-side (the Responses API defaults to
		// store=true, unlike Chat Completions).
		Store: openai.Bool(false),
	}

	if params.MaxTokens != nil {
		req.MaxOutputTokens = openai.Int(int64(*params.MaxTokens))
	}
	if params.Temperature != nil {
		req.Temperature = openai.Float(*params.Temperature)
	}
	if params.TopP != nil {
		req.TopP = openai.Float(*params.TopP)
	}
	if params.ParallelToolCalls != nil {
		req.ParallelToolCalls = openai.Bool(*params.ParallelToolCalls)
	}
	if params.User != "" {
		req.User = openai.String(params.User)
	}

	// prompt_cache_key: routing hint for OpenAI automatic prompt caching, same as
	// the Chat Completions path (convertParams). ResponseNewParams.PromptCacheKey
	// is also param.Opt[string]. Only emitted when a caller supplied one via Extra.
	if key := promptCacheKeyFromExtra(params.Extra); key != "" {
		req.PromptCacheKey = openai.String(key)
	}

	// "auto" is not a Responses API effort level — omit it and let the
	// server apply the model's default. "none"/"" never reach this path
	// (shouldUseResponsesAPI), but tolerate them the same way.
	switch params.ReasoningEffort {
	case "", providers.ReasoningEffortAuto, providers.ReasoningEffortNone:
	default:
		req.Reasoning = shared.ReasoningParam{
			Effort: shared.ReasoningEffort(params.ReasoningEffort),
			// Reasoning summaries are the only reasoning trace the Responses
			// API exposes; requesting them feeds ChunkDelta.Reasoning.
			Summary: shared.ReasoningSummaryAuto,
		}
	}

	if len(params.Tools) > 0 {
		req.Tools = convertResponsesTools(params.Tools)
	}
	if params.ToolChoice != nil {
		req.ToolChoice = convertResponsesToolChoice(params.ToolChoice)
	}
	if params.ResponseFormat != nil {
		req.Text = convertResponsesTextConfig(params.ResponseFormat)
	}

	return req
}

// convertResponsesInput converts chat messages to Responses API input items.
// Assistant tool calls become function_call items and tool-role messages
// become function_call_output items, preserving conversation order.
func convertResponsesInput(messages []providers.Message) responses.ResponseInputParam {
	items := make(responses.ResponseInputParam, 0, len(messages))
	for _, msg := range messages {
		switch msg.Role {
		case providers.RoleSystem:
			items = append(items, responses.ResponseInputItemParamOfMessage(
				msg.ContentString(), responses.EasyInputMessageRoleSystem))

		case providers.RoleUser:
			if msg.IsMultiModal() {
				items = append(items, responses.ResponseInputItemParamOfInputMessage(
					convertResponsesContentParts(msg.ContentParts()), string(responses.EasyInputMessageRoleUser)))
			} else {
				items = append(items, responses.ResponseInputItemParamOfMessage(
					msg.ContentString(), responses.EasyInputMessageRoleUser))
			}

		case providers.RoleAssistant:
			if text := msg.ContentString(); text != "" {
				items = append(items, responses.ResponseInputItemParamOfMessage(
					text, responses.EasyInputMessageRoleAssistant))
			}
			for _, toolCall := range msg.ToolCalls {
				items = append(items, responses.ResponseInputItemParamOfFunctionCall(
					toolCall.Function.Arguments, toolCall.ID, toolCall.Function.Name))
			}

		case providers.RoleTool:
			if msg.IsMultiModal() {
				items = append(items, convertResponsesMultiModalToolResult(msg)...)
			} else {
				items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(
					msg.ToolCallID, msg.ContentString()))
			}
		}
	}
	return items
}

// convertResponsesMultiModalToolResult converts a tool message whose content
// is a []ContentPart. Responses API function_call_output accepts only string
// output, so text parts form the output and image parts are re-attached as an
// immediately following user message referencing the tool call — the same
// strategy as convertMultiModalToolMessage on the Chat Completions path.
func convertResponsesMultiModalToolResult(msg providers.Message) responses.ResponseInputParam {
	text, images := splitToolResultParts(msg)
	items := responses.ResponseInputParam{
		responses.ResponseInputItemParamOfFunctionCallOutput(msg.ToolCallID, text),
	}
	if len(images) == 0 {
		return items
	}

	content := responses.ResponseInputMessageContentListParam{
		responses.ResponseInputContentUnionParam{
			OfInputText: &responses.ResponseInputTextParam{Text: toolImageHeader(msg.ToolCallID, len(images))},
		},
	}
	for _, image := range images {
		content = append(content, responses.ResponseInputContentUnionParam{
			OfInputImage: &responses.ResponseInputImageParam{
				ImageURL: openai.String(image.ImageURL.URL),
				Detail:   responses.ResponseInputImageDetailAuto,
			},
		})
	}
	return append(items, responses.ResponseInputItemParamOfInputMessage(
		content, string(responses.EasyInputMessageRoleUser)))
}

// convertResponsesContentParts converts multi-modal content parts to
// Responses API input content.
func convertResponsesContentParts(parts []providers.ContentPart) responses.ResponseInputMessageContentListParam {
	list := make(responses.ResponseInputMessageContentListParam, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case contentTypeText:
			list = append(list, responses.ResponseInputContentUnionParam{
				OfInputText: &responses.ResponseInputTextParam{Text: part.Text},
			})
		case contentTypeImageURL:
			if part.ImageURL != nil {
				list = append(list, responses.ResponseInputContentUnionParam{
					OfInputImage: &responses.ResponseInputImageParam{
						ImageURL: openai.String(part.ImageURL.URL),
						Detail:   responses.ResponseInputImageDetailAuto,
					},
				})
			}
		}
	}
	return list
}

// convertResponsesTools converts provider tools to Responses API tool params.
func convertResponsesTools(tools []providers.Tool) []responses.ToolUnionParam {
	result := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		result = append(result, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.Function.Name,
				Description: openai.String(tool.Function.Description),
				Parameters:  tool.Function.Parameters,
				// Caller schemas are plain JSON Schema, not the strict-mode
				// subset (which requires additionalProperties:false and every
				// property listed in required); strict=true would 400.
				Strict: openai.Bool(false),
			},
		})
	}
	return result
}

// convertResponsesToolChoice converts provider tool choice to Responses API format.
func convertResponsesToolChoice(choice any) responses.ResponseNewParamsToolChoiceUnion {
	switch v := choice.(type) {
	case string:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions(v)),
		}
	case providers.ToolChoice:
		if v.Function != nil {
			return responses.ResponseNewParamsToolChoiceUnion{
				OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: v.Function.Name},
			}
		}
	}
	return responses.ResponseNewParamsToolChoiceUnion{
		OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
	}
}

// convertResponsesTextConfig converts provider response format to the
// Responses API text.format config.
func convertResponsesTextConfig(format *providers.ResponseFormat) responses.ResponseTextConfigParam {
	switch format.Type {
	case responseFormatJSONObject:
		return responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
			},
		}
	case responseFormatJSONSchema:
		if format.JSONSchema != nil {
			strict := format.JSONSchema.Strict != nil && *format.JSONSchema.Strict
			return responses.ResponseTextConfigParam{
				Format: responses.ResponseFormatTextConfigUnionParam{
					OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
						Name:        format.JSONSchema.Name,
						Description: openai.String(format.JSONSchema.Description),
						Schema:      format.JSONSchema.Schema,
						Strict:      openai.Bool(strict),
					},
				},
			}
		}
	}
	return responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfText: &shared.ResponseFormatTextParam{},
		},
	}
}

// convertResponsesResponse converts a Responses API response to
// chat-completion shape.
func convertResponsesResponse(resp *responses.Response) *providers.ChatCompletion {
	message := providers.Message{Role: providers.RoleAssistant}
	var textBuilder strings.Builder

	for _, item := range resp.Output {
		switch item.Type {
		case outputItemTypeMessage:
			for _, content := range item.Content {
				if content.Type == outputContentTypeText {
					textBuilder.WriteString(content.Text)
				}
			}
		case outputItemTypeFunctionCall:
			message.ToolCalls = append(message.ToolCalls, providers.ToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: providers.FunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		}
	}
	message.Content = textBuilder.String()

	finishReason := providers.FinishReasonStop
	if len(message.ToolCalls) > 0 {
		finishReason = providers.FinishReasonToolCalls
	}
	if resp.IncompleteDetails.Reason == incompleteReasonMaxOutputTokens {
		finishReason = providers.FinishReasonLength
	}

	return &providers.ChatCompletion{
		ID:      resp.ID,
		Object:  objectChatCompletion,
		Created: int64(resp.CreatedAt),
		Model:   resp.Model,
		Choices: []providers.Choice{{Message: message, FinishReason: finishReason}},
		Usage:   convertResponsesUsage(resp.Usage),
	}
}

// convertResponsesUsage converts Responses API usage to chat-completion usage.
// OutputTokens already includes reasoning tokens, matching how Chat
// Completions reports completion_tokens for reasoning models. Returns nil
// when the response carried no usage.
func convertResponsesUsage(usage responses.ResponseUsage) *providers.Usage {
	if usage.InputTokens == 0 && usage.OutputTokens == 0 {
		return nil
	}
	return &providers.Usage{
		PromptTokens:       int(usage.InputTokens),
		CompletionTokens:   int(usage.OutputTokens),
		TotalTokens:        int(usage.TotalTokens),
		ReasoningTokens:    int(usage.OutputTokensDetails.ReasoningTokens),
		CachedPromptTokens: int(usage.InputTokensDetails.CachedTokens),
	}
}
