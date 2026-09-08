package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const defaultMaxResponseBytes int64 = 8 << 20

var (
	ErrBaseURLRequired = errors.New("llm base url required")
	ErrModelRequired   = errors.New("llm model required")
)

type OpenAICompatibleProvider struct {
	cfg    ProviderConfig
	client *http.Client
}

func NewOpenAICompatibleProvider(cfg ProviderConfig) (*OpenAICompatibleProvider, error) {
	return NewOpenAICompatibleProviderWithClient(cfg, nil)
}

func NewOpenAICompatibleProviderWithClient(cfg ProviderConfig, client *http.Client) (*OpenAICompatibleProvider, error) {
	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		return nil, ErrBaseURLRequired
	}
	if _, err := url.ParseRequestURI(baseURL); err != nil {
		return nil, fmt.Errorf("invalid llm base url: %w", err)
	}
	cfg.BaseURL = strings.TrimRight(baseURL, "/")
	if cfg.TimeoutMS <= 0 {
		cfg.TimeoutMS = 30000
	}
	if client == nil {
		client = &http.Client{Timeout: time.Duration(cfg.TimeoutMS) * time.Millisecond}
	}
	return &OpenAICompatibleProvider{cfg: cfg, client: client}, nil
}

func (p *OpenAICompatibleProvider) Complete(ctx context.Context, req Request) (Response, error) {
	if p == nil {
		return Response{}, ErrBaseURLRequired
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = strings.TrimSpace(p.cfg.Model)
	}
	if model == "" {
		return Response{}, ErrModelRequired
	}

	body, err := json.Marshal(p.buildChatRequest(model, req))
	if err != nil {
		return Response{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Response{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if key := p.apiKey(); key != "" {
		httpReq.Header.Set("Authorization", "Bearer "+key)
	}

	httpRes, err := p.client.Do(httpReq)
	if err != nil {
		return Response{}, err
	}
	defer httpRes.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(httpRes.Body, defaultMaxResponseBytes))
	if err != nil {
		return Response{}, err
	}
	if httpRes.StatusCode < http.StatusOK || httpRes.StatusCode >= http.StatusMultipleChoices {
		return Response{}, fmt.Errorf("llm chat completion failed: status=%d body=%s", httpRes.StatusCode, strings.TrimSpace(string(raw)))
	}

	return parseChatResponse(raw)
}

func (p *OpenAICompatibleProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	if p == nil {
		return nil, ErrBaseURLRequired
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, p.cfg.BaseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	if key := p.apiKey(); key != "" {
		httpReq.Header.Set("Authorization", "Bearer "+key)
	}

	httpRes, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpRes.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(httpRes.Body, defaultMaxResponseBytes))
	if err != nil {
		return nil, err
	}
	if httpRes.StatusCode < http.StatusOK || httpRes.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("llm models request failed: status=%d body=%s", httpRes.StatusCode, strings.TrimSpace(string(raw)))
	}

	return parseModelsResponse(raw)
}

func (p *OpenAICompatibleProvider) buildChatRequest(model string, req Request) chatCompletionRequest {
	messages := make([]Message, 0, len(req.Messages)+1)
	if strings.TrimSpace(req.SystemPrompt) != "" {
		messages = append(messages, Message{Role: "system", Content: req.SystemPrompt})
	}
	messages = append(messages, req.Messages...)

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = p.cfg.MaxTokens
	}
	temperature := req.Temperature
	if temperature == 0 {
		temperature = p.cfg.Temperature
	}
	seed := req.Seed
	if seed == 0 {
		seed = p.cfg.Seed
	}

	chatReq := chatCompletionRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	}
	if seed != 0 && p.cfg.Capabilities.Seed {
		chatReq.Seed = &seed
	}
	if len(req.Tools) > 0 && p.cfg.Capabilities.Tools {
		chatReq.Tools = cloneTools(req.Tools)
	}
	if req.ResponseFormat != "" && p.cfg.Capabilities.JSONMode {
		chatReq.ResponseFormat = map[string]string{"type": req.ResponseFormat}
	}
	return chatReq
}

func (p *OpenAICompatibleProvider) apiKey() string {
	env := strings.TrimSpace(p.cfg.APIKeyEnv)
	if env == "" {
		return ""
	}
	return strings.TrimSpace(os.Getenv(env))
}

type chatCompletionRequest struct {
	Model          string            `json:"model"`
	Messages       []Message         `json:"messages"`
	MaxTokens      int               `json:"max_tokens,omitempty"`
	Temperature    float64           `json:"temperature,omitempty"`
	Seed           *int64            `json:"seed,omitempty"`
	ResponseFormat map[string]string `json:"response_format,omitempty"`
	Tools          []ToolSpec        `json:"tools,omitempty"`
}

type chatCompletionResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func parseChatResponse(raw []byte) (Response, error) {
	var parsed chatCompletionResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return Response{}, err
	}
	res := Response{
		Model: parsed.Model,
		Usage: Usage{
			PromptTokens:     parsed.Usage.PromptTokens,
			CompletionTokens: parsed.Usage.CompletionTokens,
			TotalTokens:      parsed.Usage.TotalTokens,
		},
		Raw: append([]byte(nil), raw...),
	}
	if len(parsed.Choices) == 0 {
		return res, nil
	}
	choice := parsed.Choices[0]
	res.Message = choice.Message.Content
	res.FinishReason = choice.FinishReason
	for _, call := range choice.Message.ToolCalls {
		res.ToolCalls = append(res.ToolCalls, ToolCall{
			ID:            call.ID,
			Type:          call.Type,
			Name:          call.Function.Name,
			ArgumentsJSON: call.Function.Arguments,
		})
	}
	return res, nil
}

func parseModelsResponse(raw []byte) ([]ModelInfo, error) {
	var envelope struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, err
	}
	models := make([]ModelInfo, 0, len(envelope.Data))
	for _, item := range envelope.Data {
		var parsed struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		}
		if err := json.Unmarshal(item, &parsed); err != nil {
			return nil, err
		}
		models = append(models, ModelInfo{
			ID:      parsed.ID,
			OwnedBy: parsed.OwnedBy,
			Raw:     append(json.RawMessage(nil), item...),
		})
	}
	return models, nil
}
