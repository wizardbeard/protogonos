# LLM Scapes

This document sketches a possible `v0.2.x` feature line: scapes where some actors use LLMs to communicate, plan, judge, or choose bounded actions.

This is a concept note, not an implementation contract.

## Position

LLM-backed scapes are larger than a patch release.

They add:

- external model calls,
- token and latency costs,
- prompt and response artifacts,
- new IO surfaces,
- stronger replay requirements,
- new failure modes,
- possible provider configuration.

That fits a `v0.2.x` line better than `v0.1.x`.

## Core Rule

The scape still owns fitness.

TWEANN and genetic algorithm runs need selection pressure. In this project that pressure comes from numeric fitness. LLMs can help agents decide, communicate, summarize, or judge language, but the evolution loop still needs a score it can rank.

Best practice:

- use deterministic numeric scoring for task outcomes,
- charge explicit token and latency costs,
- treat invalid LLM output as a bounded action failure,
- store prompts and responses in traces or artifacts,
- keep LLM judge use narrow and auditable.

## Runtime Shape

A minimal LLM actor can use the same process style as sensors, actuators, and substrate CEP actors.

```go
type LLMActor interface {
	ID() string
	Respond(ctx context.Context, req LLMRequest) (LLMResponse, error)
}

type LLMRequest struct {
	SystemPrompt string
	Messages     []LanguageMessage
	Tools        []ToolSpec
	MaxTokens    int
	Temperature  float64
}

type LLMResponse struct {
	Message      string
	ToolCalls    []ToolCall
	TokenCount   int
	FinishReason string
}
```

The first implementation should hide provider details behind an adapter:

```go
type LLMProvider interface {
	Complete(ctx context.Context, req LLMRequest) (LLMResponse, error)
}
```

This keeps scapes independent from any single model provider.

## Provider Strategy

Prefer an OpenAI-compatible HTTP surface, but do not bind core code to OpenAI, LM Studio, Ollama, or any other vendor.

The first provider should use raw HTTP and JSON from the Go standard library:

- `net/http`,
- `encoding/json`,
- `context`,
- `time`.

Do not add SDK dependencies for the core path.

Core provider types should stay generic:

```go
type LLMProvider interface {
	Complete(ctx context.Context, req LLMRequest) (LLMResponse, error)
}

type LLMProviderConfig struct {
	BaseURL      string
	APIKeyEnv    string
	Model        string
	TimeoutMS    int
	MaxTokens    int
	Temperature  float64
	Seed         int64
	Capabilities LLMCapabilities
}

type LLMCapabilities struct {
	ChatCompletions bool
	JSONMode        bool
	Tools           bool
	Seed            bool
	UsageTokens     bool
}
```

The first real adapter should target the common subset:

- `GET /v1/models`,
- `POST /v1/chat/completions`.

Use only common request fields at first:

- `model`,
- `messages`,
- `temperature`,
- `max_tokens`,
- `seed` when the provider supports it,
- `response_format` when JSON mode is enabled,
- `tools` only after basic structured output works.

Use only common response fields at first:

- `choices[0].message.content`,
- `choices[0].message.tool_calls`,
- `choices[0].finish_reason`,
- `usage.prompt_tokens`,
- `usage.completion_tokens`,
- `usage.total_tokens`.

Provider-specific fields can be stored in raw trace metadata, but scapes should not depend on them.

## Provider Config

Config should allow local or remote OpenAI-compatible endpoints.

Example:

```json
{
  "llm": {
    "provider": "openai-compatible",
    "base_url": "http://192.168.1.50:1234/v1",
    "api_key_env": "PROTOGONOS_LLM_API_KEY",
    "model": "local-model",
    "timeout_ms": 30000,
    "max_tokens": 256,
    "temperature": 0.2,
    "seed": 1
  }
}
```

For LM Studio on a LAN host, `base_url` can point at that host. The API key can be a placeholder if the server does not require one, but the config should still support real keys.

For hosted services, read the key from the named environment variable. Do not store keys in run artifacts.

## Provider Risks

OpenAI-compatible does not mean identical behavior.

Expected differences:

- tool calling quality varies by model,
- JSON mode support varies,
- `seed` support varies,
- token usage fields may be absent or approximate,
- local servers may return model-loading errors,
- LAN calls can fail or time out,
- model IDs and loaded-model behavior differ by server.

The scape must handle these cases:

- timeout,
- connection error,
- malformed JSON,
- missing usage data,
- unsupported tool calls,
- invalid action output.

Each failure should map to a bounded evaluation result. The run should not panic.

## IO Surfaces

LLM scapes need language-aware sensors and actuators.

Possible sensors:

- `language_inbox`: recent messages addressed to the agent,
- `public_chat`: shared message history,
- `task_brief`: current task text,
- `llm_summary`: compressed state summary,
- `claim_score`: numeric trust or contradiction score.

Possible actuators:

- `language_send`: send a bounded text message,
- `language_vote`: choose from fixed options,
- `language_offer`: emit a structured negotiation offer,
- `ask_planner`: request a bounded LLM plan,
- `tool_choice`: choose one allowed tool or action.

For neural agents, language often needs numeric encoding:

- message count,
- speaker ID,
- target ID,
- intent class,
- offer value,
- trust score,
- contradiction flag,
- embedding projection.

For LLM agents, the scape can pass text directly and decode the response into bounded actions.

## Prototype: Comm Grid

`comm-grid` is a good first prototype.

World:

- small grid,
- two to four agents,
- partial observation,
- one hidden objective,
- optional hazards,
- bounded episode length.

Each agent can:

- move north, south, east, west, or stay,
- pick up or drop an item,
- send one short message per tick,
- ask an optional LLM planner for advice.

Observation:

- local tiles,
- inventory state,
- last reward,
- recent messages,
- optional planner response.

Fitness:

- goal completed,
- useful delivery progress,
- invalid move penalty,
- collision penalty,
- message cost,
- token cost,
- timeout penalty,
- hidden-test generalization score.

Interesting variants:

- one LLM teammate,
- one LLM adversary,
- shared LLM planner,
- neural agents only with language encoded as numeric features,
- mixed teams with neural and LLM actors.

## Other Training Scenarios

Cooperative logistics:

- agents move resources through a graph,
- each agent sees only part of the state,
- fitness rewards delivery, low travel cost, and concise messages.

Deception and verification:

- helper and adversary actors make claims about hidden state,
- fitness rewards correct decisions, useful questions, and false-claim detection.

Auction or market:

- agents bid for scarce resources,
- fitness rewards profit, completed contracts, and low message cost.

Scientific team:

- actors choose experiments to identify a hidden rule,
- fitness rewards correct hypothesis, fewer experiments, and clear evidence.

Incident response:

- actors inspect logs and metrics in a simulated outage,
- fitness rewards correct mitigation, low false-action count, and short time to recovery.

Social deduction:

- agents infer roles from statements and actions,
- fitness rewards correct role inference and team objective success.

Code repair:

- LLM actors propose patches,
- evolved agents choose which patch to test, reject, or combine,
- fitness rewards passing tests and small safe diffs.

Tutor scape:

- one actor teaches a hidden rule,
- another actor solves held-out cases,
- fitness rewards learner accuracy and short instruction length.

## Fitness Notes

Do not score only with an LLM judge.

A safer scoring shape is:

```go
fitness := 0.0
fitness += taskScore
fitness += cooperationBonus
fitness -= invalidActionPenalty
fitness -= tokenCost
fitness -= latencyCost
fitness -= ruleViolationPenalty
fitness = clamp(fitness, minFitness, maxFitness)
```

If an LLM judge is needed:

- keep its prompt fixed,
- store the prompt and response,
- score only narrow language properties,
- combine it with deterministic task metrics,
- sample more than once if judge variance affects selection.

## Replay Requirements

LLM scapes must be replayable enough for debugging.

Store:

- model/provider ID,
- prompt templates,
- request parameters,
- messages,
- raw responses,
- parsed actions,
- token counts,
- latency,
- errors and retries,
- random seeds,
- final fitness components.

If exact provider replay is not possible, support fixture replay from stored responses.

## First Implementation Slice

A small first slice should avoid provider lock-in:

- add language message structs,
- add the `LLMProvider` interface,
- add `LLMProviderConfig` and `LLMCapabilities`,
- add a deterministic fixture provider for tests,
- add an OpenAI-compatible provider with raw `net/http`,
- add fake HTTP server tests for provider behavior,
- add `comm-grid` as an experimental scape,
- keep all actions bounded and parseable,
- add trace artifacts for messages and parsed actions,
- add tests for deterministic replay, invalid output handling, timeout handling, and token-cost fitness.

This gives the system a useful LLM integration path without making evolution depend on unbounded free-form text.
