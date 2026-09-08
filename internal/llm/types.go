package llm

import (
	"context"
	"encoding/json"
)

type Provider interface {
	Complete(ctx context.Context, req Request) (Response, error)
}

type ModelLister interface {
	Models(ctx context.Context) ([]ModelInfo, error)
}

type ModelInfo struct {
	ID      string          `json:"id"`
	OwnedBy string          `json:"owned_by,omitempty"`
	Raw     json.RawMessage `json:"raw,omitempty"`
}

type Request struct {
	Model          string     `json:"model,omitempty"`
	SystemPrompt   string     `json:"system_prompt,omitempty"`
	Messages       []Message  `json:"messages,omitempty"`
	Tools          []ToolSpec `json:"tools,omitempty"`
	MaxTokens      int        `json:"max_tokens,omitempty"`
	Temperature    float64    `json:"temperature,omitempty"`
	Seed           int64      `json:"seed,omitempty"`
	ResponseFormat string     `json:"response_format,omitempty"`
}

type Response struct {
	Message      string          `json:"message,omitempty"`
	ToolCalls    []ToolCall      `json:"tool_calls,omitempty"`
	Usage        Usage           `json:"usage,omitempty"`
	FinishReason string          `json:"finish_reason,omitempty"`
	Model        string          `json:"model,omitempty"`
	Raw          json.RawMessage `json:"raw,omitempty"`
}

func (r Response) TokenCount() int {
	return r.Usage.TotalTokens
}

type Message struct {
	Role    string `json:"role"`
	Name    string `json:"name,omitempty"`
	Content string `json:"content"`
}

type ToolSpec struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ToolCall struct {
	ID            string `json:"id,omitempty"`
	Type          string `json:"type,omitempty"`
	Name          string `json:"name,omitempty"`
	ArgumentsJSON string `json:"arguments_json,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

type ProviderConfig struct {
	BaseURL      string
	APIKeyEnv    string
	Model        string
	TimeoutMS    int
	MaxTokens    int
	Temperature  float64
	Seed         int64
	Capabilities Capabilities
}

type Capabilities struct {
	ChatCompletions bool
	Streaming       bool
	StreamingUsage  bool
	StreamingTools  bool
	JSONMode        bool
	Tools           bool
	Seed            bool
	UsageTokens     bool
}

func cloneRequest(req Request) Request {
	req.Messages = append([]Message(nil), req.Messages...)
	req.Tools = cloneTools(req.Tools)
	return req
}

func cloneResponse(res Response) Response {
	res.ToolCalls = append([]ToolCall(nil), res.ToolCalls...)
	res.Raw = append(json.RawMessage(nil), res.Raw...)
	return res
}

func cloneTools(tools []ToolSpec) []ToolSpec {
	if len(tools) == 0 {
		return nil
	}
	out := make([]ToolSpec, len(tools))
	copy(out, tools)
	for i := range out {
		out[i].Function.Parameters = append(json.RawMessage(nil), out[i].Function.Parameters...)
	}
	return out
}
