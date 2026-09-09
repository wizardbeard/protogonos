# LLM Scapes

This document sketches a possible `v0.2.x` feature line: scapes where some actors use LLMs to communicate, plan, judge, or choose bounded actions.

This is a concept note, not an implementation contract.

See [LLM Scape Spec](llm-scape-spec.md) for the first bounded training target.

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
	Streaming       bool
	StreamingUsage  bool
	StreamingTools  bool
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

## Streaming Policy

Do not require streaming in the first implementation.

The first code path should use non-streaming `POST /v1/chat/completions`. This is simpler to test, easier to replay, and enough for `comm-grid` turn decisions.

Streaming should be optional through a second interface:

```go
type LLMStreamingProvider interface {
	Stream(ctx context.Context, req LLMRequest) (LLMStream, error)
}

type LLMStream interface {
	Next(ctx context.Context) (LLMStreamEvent, error)
	Close() error
}

type LLMStreamEvent struct {
	Type         LLMStreamEventType
	Delta        string
	ToolCallID   string
	ToolName     string
	ToolArgsJSON string
	FinishReason string
	Usage        LLMUsage
	Raw          map[string]any
}
```

OpenAI-compatible streaming should use server-sent events from `/v1/chat/completions` with `stream: true`.

The stream parser should:

- read `data:` lines from the response body,
- emit `delta` events for partial content,
- emit `done` on `data: [DONE]`,
- collect tool-call argument fragments before execution,
- close the response body on `Close`,
- honor context cancellation,
- enforce max byte and max event limits.

For now, scapes should not depend on streaming. Add it only when a scenario needs partial output timing or mid-message reaction.

## Conversation State

The provider should not own conversation state.

The scape should:

- build the message list,
- call the provider,
- append assistant output,
- execute allowed tool calls if needed,
- append tool results,
- call the provider again if the scenario allows another turn.

This keeps replay simple. The run artifact can store the full request and response sequence.

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
- streaming chunks may have provider-specific gaps,
- local servers may return model-loading errors,
- LAN calls can fail or time out,
- model IDs and loaded-model behavior differ by server.

The scape must handle these cases:

- timeout,
- connection error,
- malformed JSON,
- malformed stream chunks,
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

The first implementation slice is deterministic and does not call an LLM. It defines the world, actions, messages, bounded scoring, and trace shape first. LLM actors can be added after this core loop is stable.

The second slice adds language IO names and structured language-action decoding. The decoder maps JSON fields such as `action`, `message`, `to`, and `tokens` into bounded simulator inputs. This keeps provider output parseable before any real provider is wired into the scape.

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

## Fixture Demo

The current command-line demo runs `comm-grid` with a fixture-backed LLM actor. It does not call a real provider.

```bash
protogonosctl comm-grid-llm --plan solve
```

The command writes `benchmarks/<run-id>/comm_grid_llm.json` by default:

```bash
protogonosctl comm-grid-llm \
  --run-id comm-grid-fixture-001 \
  --plan solve
```

It also writes `benchmarks/<run-id>/comm_grid_llm_transcript.md` for quick inspection.
Each artifact write appends one summary line to `benchmarks/comm_grid_llm_runs.jsonl`.

List indexed runs:

```bash
protogonosctl comm-grid-llm-runs
```

Filter or emit JSON:

```bash
protogonosctl comm-grid-llm-runs --completed false
protogonosctl comm-grid-llm-runs --provider fixture --plan solve --json
```

Emit CSV for spreadsheet analysis:

```bash
protogonosctl comm-grid-llm-runs --csv
```

Print the latest matching transcript:

```bash
protogonosctl comm-grid-llm-runs --plan multi-solve --transcript
```

Compare indexed runs by task shape, provider, and plan:

```bash
protogonosctl comm-grid-llm-runs --compare
protogonosctl comm-grid-llm-runs --compare --json
protogonosctl comm-grid-llm-runs --compare --csv
```

Run a small suite of plans or prompt variants:

```bash
protogonosctl comm-grid-llm-suite \
  --suite-id comm-grid-suite-001 \
  --plans solve,tool,invalid \
  --prompt 'strict=Return JSON only.' \
  --repeats 2
```

Run the same suite from a JSON manifest:

```json
{
  "suite_id": "comm-grid-suite-001",
  "plans": ["solve", "tool", "invalid"],
  "repeats": 2,
  "prompts": [
    {
      "name": "strict",
      "system_prompt": "Return JSON only."
    }
  ]
}
```

```bash
protogonosctl comm-grid-llm-suite --manifest suite.json
```

Manifest files are validated before any suite row runs. Unknown fields fail. Invalid ranges fail with field names.

Run the checked-in fixture manifest:

```bash
protogonosctl comm-grid-llm-suite \
  --manifest testdata/fixtures/comm_grid_llm_suite_manifest.json
```

Print a starter manifest from suite flags:

```bash
protogonosctl comm-grid-llm-suite \
  --suite-id comm-grid-suite-002 \
  --plans solve,tool \
  --prompt 'strict=Return JSON only.' \
  --emit-manifest
```

Preview a suite without provider calls or artifacts:

```bash
protogonosctl comm-grid-llm-suite \
  --manifest testdata/fixtures/comm_grid_llm_suite_manifest.json \
  --dry-run
```

Continue a suite after row-level errors:

```bash
protogonosctl comm-grid-llm-suite \
  --manifest testdata/fixtures/comm_grid_llm_suite_manifest.json \
  --fail-fast=false
```

Emit suite rows as CSV:

```bash
protogonosctl comm-grid-llm-suite \
  --manifest testdata/fixtures/comm_grid_llm_suite_manifest.json \
  --csv
```

Emit compact suite aggregates:

```bash
protogonosctl comm-grid-llm-suite \
  --manifest testdata/fixtures/comm_grid_llm_suite_manifest.json \
  --summary
protogonosctl comm-grid-llm-suite \
  --manifest testdata/fixtures/comm_grid_llm_suite_manifest.json \
  --summary --csv
```

Disable artifact writes for quick console checks:

```bash
protogonosctl comm-grid-llm --plan solve --artifacts=false
```

Run the first fixture-only mentor scape:

```bash
protogonosctl comm-grid-mentor --plan solve
protogonosctl comm-grid-mentor --plan solve --json
protogonosctl comm-grid-mentor --plan solve --compare-baseline
protogonosctl comm-grid-mentor --replay-run-id comm-grid-mentor-fixture-001
```

Run fixture-backed mentor evolution through the normal scape registry:

```bash
protogonosctl run --scape comm-grid-mentor --pop 4 --gens 1 --seed 91
protogonosctl run --scape comm-grid-mentor --comm-grid-mentor-plan silent --pop 4 --gens 1 --seed 91
protogonosctl run --scape comm-grid-mentor --comm-grid-mentor-compare-baseline --pop 4 --gens 1 --seed 91
```

Replay a stored artifact through the fixture provider and compare the final trace:

```bash
protogonosctl comm-grid-llm --replay-run-id comm-grid-fixture-001
```

JSON output is available for trace inspection:

```bash
protogonosctl comm-grid-llm --plan tool --json
```

Set the task geometry and message bound when a run needs a different grid:

```bash
protogonosctl comm-grid-llm \
  --run-id comm-grid-custom-001 \
  --width 4 \
  --height 2 \
  --key 1,0 \
  --goal 2,0 \
  --agent worker-a \
  --agent-pos 0,0 \
  --message-limit 40
```

Run a simple multi-agent fixture with fixed sequential turns:

```bash
protogonosctl comm-grid-llm \
  --run-id comm-grid-multi-001 \
  --plan multi-solve \
  --agents 'agent-a@0,0:agent-b@0,1' \
  --turn-order agent-a,agent-b
```

Give agents different roles or exact system prompts:

```bash
protogonosctl comm-grid-llm \
  --run-id comm-grid-roles-001 \
  --plan multi-solve \
  --agents 'agent-a@0,0:agent-b@0,1' \
  --turn-order agent-a,agent-b \
  --system-prompt 'Return JSON only.' \
  --agent-roles 'agent-a=carrier:agent-b=observer' \
  --agent-prompts 'agent-b=Return JSON only. Wait unless asked.'
```

Fixture plans:

- `solve`: uses assistant message JSON and completes the key-delivery task,
- `tool`: uses tool-call argument JSON and completes the same task,
- `invalid`: starts with one invalid wall move, then recovers,
- `malformed`: returns unsupported JSON actions and records bounded failure steps,
- `provider-error`: simulates provider errors or timeouts and records bounded failure steps,
- `multi-solve`: alternates two agents over shared message history and completes the key-delivery task.

The same command can use a live OpenAI-compatible endpoint when requested:

```bash
protogonosctl comm-grid-llm \
  --provider openai-compatible \
  --base-url http://192.168.1.50:1234/v1 \
  --model local-model \
  --api-key-env PROTOGONOS_LLM_API_KEY \
  --json
```

Add retry controls for local or LAN inference servers:

```bash
protogonosctl comm-grid-llm \
  --provider openai-compatible \
  --base-url http://192.168.1.50:1234/v1 \
  --model local-model \
  --provider-retries 2 \
  --retry-backoff-ms 250
```

Fixture mode remains the default. Live provider mode is explicit so local tests and examples do not call external services by accident.

The artifact file stores:

- provider mode and run ID,
- task geometry, key, goal, agents, roles, prompts, turn order, and message limit,
- prompt request,
- provider response,
- parsed bounded action,
- bounded failure text and kind,
- provider retry attempts,
- message history,
- token counts,
- final fitness and trace.

The transcript file stores:

- replay command,
- task summary,
- each actor prompt,
- provider response payload,
- provider retry attempts,
- parsed action,
- step fitness,
- final trace.

The run index stores:

- run ID,
- provider and plan,
- task config,
- completion status,
- fitness,
- total tokens,
- average tokens per step,
- run duration in milliseconds,
- failure count,
- retry count,
- artifact paths.

## First Implementation Slice

A small first slice should avoid provider lock-in:

- add language message structs,
- add the `LLMProvider` interface,
- add `LLMProviderConfig` and `LLMCapabilities`,
- add a deterministic fixture provider for tests,
- add an OpenAI-compatible provider with raw `net/http`,
- add fake HTTP server tests for provider behavior,
- define optional streaming interfaces, but do not wire scapes to streaming yet,
- add `comm-grid` as an experimental scape,
- keep all actions bounded and parseable,
- add trace artifacts for messages and parsed actions,
- add a fixture-backed LLM actor adapter for `comm-grid` that requests structured JSON from `internal/llm.Provider`,
- prefer tool-call argument JSON when available, with plain message JSON as the fallback,
- expose `protogonosctl comm-grid-llm` for deterministic fixture demos without external provider calls,
- add explicit `openai-compatible` provider flags to `protogonosctl comm-grid-llm`, covered by fake-server tests,
- write `comm_grid_llm.json` artifacts with request, response, parsed action, messages, token counts, and final trace,
- replay `comm_grid_llm.json` artifacts with stored provider responses and report final-trace match status,
- convert malformed output, provider errors, and provider timeouts into bounded failed steps that remain artifact-backed and replayable,
- add configurable `comm-grid-llm` task geometry, key, goal, agent ID, agent start, and message limit, persisted in artifacts and reused on replay,
- add multi-agent `comm-grid-llm` fixture runs with fixed sequential turns, shared message history, per-step actor IDs, artifact persistence, and replay reuse,
- add per-agent role and system-prompt controls for `comm-grid-llm`, persisted in artifacts and reused on replay,
- write `comm_grid_llm_transcript.md` beside the JSON artifact for quick provider-turn inspection,
- add provider retry/backoff controls with stored attempts in JSON artifacts and transcripts,
- append `benchmarks/comm_grid_llm_runs.jsonl` summary records for easier run comparison,
- add read-only `protogonosctl comm-grid-llm-runs` table and JSON views over the JSONL run index,
- add `protogonosctl comm-grid-llm-runs --transcript` to print the latest matching transcript,
- add `protogonosctl comm-grid-llm-runs --compare` to group runs by task shape, provider, and plan,
- add CSV output for `protogonosctl comm-grid-llm-runs` and `protogonosctl comm-grid-llm-runs --compare`,
- add token and duration summaries to the run index and compare output,
- add `protogonosctl comm-grid-llm-suite` to run plan and prompt-variant batches through the normal artifact/index path,
- add JSON manifest input for `protogonosctl comm-grid-llm-suite`, with explicit CLI flags taking precedence,
- validate `protogonosctl comm-grid-llm-suite` manifests before execution, including unknown-field rejection and field-specific errors,
- add `testdata/fixtures/comm_grid_llm_suite_manifest.json` as a known-good fixture suite,
- add `protogonosctl comm-grid-llm-suite --emit-manifest` to print the effective suite manifest without running the suite,
- add `protogonosctl comm-grid-llm-suite --dry-run` to preview the run matrix without provider calls or artifacts,
- add `protogonosctl comm-grid-llm-suite --fail-fast=false` to continue after row-level suite errors,
- add CSV output for `protogonosctl comm-grid-llm-suite` and `protogonosctl comm-grid-llm-suite --dry-run`,
- add `protogonosctl comm-grid-llm-suite --summary` to aggregate a suite by plan and prompt,
- add `docs/llm-scape-spec.md` to bound the first real LLM scape around a mentor actor, fixed `comm-grid` task, and deterministic fitness,
- add tests for deterministic replay, invalid output handling, timeout handling, and token-cost fitness.
- add `protogonosctl comm-grid-mentor` as a fixture-only console runner for the first mentor scape slice.
- add file-backed `comm_grid_mentor.json` artifacts and replay with final trace matching for `protogonosctl comm-grid-mentor`.
- add `protogonosctl comm-grid-mentor --compare-baseline` to compare the mentor path with the same learner policy and no hints.
- add a focused `Polis.RunEvolution` test that compares fixture-mentor and no-hint baseline scapes end to end.
- add fixture-backed `protogonosctl run --scape comm-grid-mentor` support through the normal scape registry.
- add `--comm-grid-mentor-plan solve|silent` so normal runs can execute mentor and no-hint baseline variants.
- add `--comm-grid-mentor-compare-baseline` so normal runs can write paired solve/silent artifacts and print the fitness delta.

This gives the system a useful LLM integration path without making evolution depend on unbounded free-form text.
