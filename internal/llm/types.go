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
	ID      string
	OwnedBy string
	Raw     json.RawMessage
}

type Request struct {
	Model          string
	SystemPrompt   string
	Messages       []Message
	Tools          []ToolSpec
	MaxTokens      int
	Temperature    float64
	Seed           int64
	ResponseFormat string
}

type Response struct {
	Message      string
	ToolCalls    []ToolCall
	Usage        Usage
	FinishReason string
	Model        string
	Raw          json.RawMessage
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
	ID            string
	Type          string
	Name          string
	ArgumentsJSON string
}

type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
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
