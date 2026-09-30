# Investigator

[![CI](https://github.com/bronto-community/bronto-alert-investigation-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/bronto-community/bronto-alert-investigation-agent/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/bronto-community/bronto-alert-investigation-agent/badge)](https://scorecard.dev/viewer/?uri=github.com/bronto-community/bronto-alert-investigation-agent)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://github.com/bronto-community/bronto-alert-investigation-agent)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A logs-based alert investigation agent for [Bronto](https://bronto.io), built with
[LangGraph](https://langchain-ai.github.io/langgraph/). Describe an alert in plain English
(even a historical one - from days or weeks ago). The agent finds the triggered monitor, investigates your logs
around it, and writes a human-readable incident report.

## Quick start

You need:

- Python **3.11+** and [`uv`](https://docs.astral.sh/uv/)
- A **Bronto API key** and region (`eu` or `us`)
- An API key for one **LLM provider**: Anthropic, OpenAI, or AWS Bedrock

```bash
uv sync
export BRONTO_API_KEY=…  BRONTO_REGION=eu
uv run investigator --provider bedrock --model-id eu.anthropic.claude-sonnet-5
```

Configuration is entirely through environment variables. [`.env.example`](.env.example) lists
every one of them.

| Provider    | Required environment                                                          |
| ----------- | ----------------------------------------------------------------------------- |
| `anthropic` | `ANTHROPIC_API_KEY`                                                           |
| `openai`    | `OPENAI_API_KEY`                                                              |
| `bedrock`   | `AWS_REGION` (or `AWS_DEFAULT_REGION`) + `AWS_PROFILE`, or a key pair         |

A Bedrock key pair for an IAM user (`AKIA…`) is `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`.
A temporary one from SSO or assume-role (`ASIA…`) also needs `AWS_SESSION_TOKEN`. If all three
are set and Bedrock still refuses, the session has probably expired: run `aws sso login`.

## What happens in a session

1. **Describe the alert** in plain English, e.g. _"elevated 500s on my api gateway"_. It can be
   a historical event.
2. **Confirm the event.** The agent searches your Bronto audit log for the triggered alert that
   matches your description and asks you to confirm it. You can pick another candidate, or reply
   with more detail to send it back to search.
3. **Add a playbook or context (optional).** This could be a playbook or other useful context,
   such as how your services depend on each other. If the monitor already has a playbook in
   Bronto, the agent offers it to you.
4. **Investigation.** The agent explores the monitor and your logs to gather system context
   around the time the alert triggered. Then it drills down into your data to find a root cause.
5. **Report.** You get a write-up that answers:
   - What does this triggered alert mean, in simple terms?
   - What are its implications and impact?
   - What are its likely root causes, and what is the evidence?
   - What are the suggested next steps and remediations?

An investigation takes around a minute with Claude Sonnet 5 on the default settings. To let the
agent think harder and run for longer, raise `--max-model-calls`.

| Flag                | Default | Description                                            |
| ------------------- | ------- | ------------------------------------------------------ |
| `--provider`        | —       | Required. One of `anthropic`, `bedrock`, `openai`      |
| `--model-id`        | —       | Required. The provider's model identifier              |
| `--log-level`       | `INFO`  | Python log level (`DEBUG` for full tracing to stderr)  |
| `--max-model-calls` | `5`     | Model-call budget for the investigation loop only      |
| `--event`           | —       | Investigate an already-identified event from a JSON file (`-` for stdin), skipping identification |

### Investigating a known event

With `--event`, the alert is already identified. The run skips straight to gathering context:
no description, no confirmation prompt, and no model call spent working out what you meant.

```bash
uv run investigator --provider anthropic --model-id claude-sonnet-4-5 \
  --event ./tests/fixtures/monitor_event.json
```

#### The event payload

The payload is a JSON object. Two fields are required. Everything else in the body is kept and
passed to the investigator. This is `tests/fixtures/monitor_event.json` verbatim. The test suite
validates it with the same function a run uses, so it cannot drift from what the code accepts:

```json
{
  "monitor_id": "mon-1",
  "monitor_name": "OOM killer invoked",
  "triggered_at": "2026-07-24T14:19:09Z",
  "evaluation_timestamp": 1784901720000,
  "new_status": "ALERT",
  "previous_status": "OK"
}
```

| Field | Type | | What it is |
| ----- | ---- | --- | ---------- |
| `monitor_id` | string | **required** | Which monitor fired. Every context lookup keys on it. |
| `evaluation_window_start_ts` | integer | **required** | Unix-epoch **milliseconds**, and the **start** of the monitored window. The older name, `evaluation_timestamp`, is still accepted; when both are sent, the new name wins. |
| `monitor_name` | string | optional | The name the report uses. Falls back to `monitor_id`. |
| `triggered_at` | string | optional | ISO-8601, when the alert fired. For display only. |
| anything else | any | optional | Kept verbatim. |

Get `evaluation_window_start_ts` right, because a wrong value fails **silently**. It must be in
milliseconds, not seconds, and it must be the start of the window, not the time the alert fired.
Otherwise the run searches the wrong time, finds nothing, and the report reads like a quiet
system. A payload the agent cannot work from is rejected up front: the CLI exits `1` and the
webhook answers `400`.

### Tracing

To send traces to an OpenTelemetry collector, including one that forwards them into Bronto, set
`OTEL_EXPORTER_OTLP_ENDPOINT` (and `OTEL_EXPORTER_OTLP_HEADERS` if needed). Every model call and
tool call becomes a span, with its inputs and outputs. When these are unset, tracing stays off.

## What this is (and isn't)

This is a standalone tool you can run against real alerts. It is also a **teaching resource**:
it shows the design patterns for building an agent with LangGraph on top of Bronto, at a size you
can read end to end in one sitting.

The feature set is **deliberately small, and frozen**. Every feature it doesn't have is one less
concept between a reader and the pattern being shown. It will not grow more providers, surfaces
or options.

To go further, **fork it, or copy the parts you need into your own project**. That is the
intended use. The phases under `src/investigator/phases/` each work on their own. Bug fixes and readability improvements are
very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for what is accepted and what isn't.

## How it works

The agent is a LangGraph state graph of four phases: **identification**, **context
generation**, **investigation**, and **report generation**. The two agentic phases are built with
LangChain's `create_agent` and sit behind wrapper nodes. Each one keeps its Reason/Act transcript
in its own checkpoint namespace, so the investigation state carries only what crosses between
phases.

![Agent graph](graph.png)

The picture is generated from the compiled graph. Solid edges are unconditional. Dashed edges are
conditional routes, labelled with the routing decision. `uv run python visualize_graph.py --xray 1`
expands the agents' internals.

### Phase 1: Identification

A Reason/Act agent that turns your description into a confirmed alert. It binds two Bronto
audit-search tools plus two *human-in-the-loop* tools, and forces a tool call every turn: it
either searches more or asks you something. It can't answer in prose, so a vague description
costs no wasted turns.

The two questions are modelled as tools rather than as bare interrupts, because
`HumanInTheLoopMiddleware` only interrupts on a tool call. The middleware raises the interrupt,
renders the prompt, and applies your decision. The tool bodies run only once you have chosen.

- **`confirm_alert`** asks which alert you mean. Accepting takes the proposal, picking an
  option chooses another, and free text sends the agent back to search with your correction.
  If your answer is unusable, the surface you are on asks again without resuming the graph, so
  it costs no model call. A confirmed candidate must resolve back to a retrieved audit record.
  An ungrounded (hallucinated) pick is rejected rather than passed on.
- **`confirm_playbook`** asks for an optional playbook, offering any pre-configured one. It is
  the phase's exit, so the agent ends as soon as you answer. Middleware that runs ahead of the
  interrupt fetches the monitor's configured playbook just before you are asked. That costs no
  extra model call, and the monitor id comes from the grounded audit row rather than from the
  model.

When no alert is confirmed, the phase routes to **`give_up`**, which is a node, not a tool. The
agent is not capped on model calls, because searching is productive work. What is capped is
floundering. A turn that calls no tool has achieved nothing, so it is retried, and after four
such turns the phase gives up. An agent that searches forever is stopped by the loop's own bound
(twelve turns) and reported the same way, rather than crashing the run.

### Phase 2: Context generation

A single node that assembles the **alert context** through deterministic REST lookups
(via `bronto-sdk`) plus one optional model call:

- fetch and trim the monitor, with the queries from its backing metric definition,
- resolve each monitor query to the datasets it runs against,
- (if a playbook was given) ask the model which datasets it implies,
- compute a **timeseries** of the monitor's state over the pre-trigger window,
- compute an absolute **investigation window** (epoch-ms) the later searches are scoped to.

### Phase 3: Investigation

A second Reason/Act agent, kept deliberately small. It defaults to 5 model calls, and it is the
only phase `--max-model-calls` governs. Context and report generation each make at most one call
outside that budget, and identification has its own mechanism.

- The **investigator** works from the alert context and a subset of the MCP tools: `get_keys`,
  `get_key_values`, `search_logs`, `timeseries`. Requested calls run **concurrently**.
- Its **findings** (root cause, evidence, remediation, confidence) are a structured output
  bound as a tool alongside the search tools. Every turn is therefore a tool call: batch more
  searches, or conclude.
- Middleware scopes each time-aware tool call to the alert context's investigation window. It
  strips `time_range` from the schema the model sees and defaults the absolute epoch range from
  the alert context. Failed calls are retried with backoff. Tool results are **not** capped, so a very wide
  log search can grow the transcript quickly.
- The call budget spends its **last** turn on the conclusion rather than cutting the loop off.
  On that turn the search tools are unbound, leaving the findings tool as the only thing
  callable, so a long investigation still ends with findings. If a provider ignores that and
  answers in prose anyway, report generation emits an inconclusive notice instead.

### Phase 4: Report generation

A single plain model call turns the alert context and findings into the human-readable report. The report
covers what the event means, its implications, the root cause and evidence, and remediation and
next steps. If there are no findings, because the investigation's call budget ran out mid-search,
the report says so and spends no model call at all.

## Development

```bash
uv sync --all-extras                # project, both extras, and dev tools
just prepare                        # format, lint, typecheck, tests — green before every commit
just graph                          # regenerate graph.png
```

This needs [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/). See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full list of recipes and what a change needs to be
accepted.

The suite mocks every model and Bronto call. `tests/conftest.py` provides `ScriptedChatModel`,
a real `BaseChatModel` that pops pre-scripted responses and records the prompts and per-call
kwargs it was given. A duck-typed double will not do, because `create_agent` drives the model
through Runnable machinery and calls `bind_tools` on every turn.

`graph.open_app`, which opens the MCP session, is the one function with no unit tests. It needs
a live server, so it is verified by running the CLI.

## Deploying it

The [`Dockerfile`](Dockerfile) builds one image for every surface. The command the container runs
picks the surface. Every surface needs `--provider`, `--model-id`, and the Bronto and provider
environment described above.

### On every alert: the webhook

```bash
uv sync --extra webhook
export BRONTO_WEBHOOK_SECRET=…    # a secret token of your choosing
uv run investigator-webhook --provider anthropic --model-id claude-sonnet-4-5
```

In Bronto, give your monitors a webhook that points at `POST https://<your-host>/events`. Set the
`X-Bronto-Webhook-Secret` header to your secret, and use this payload:

```json
{
  "monitor_id": "$monitor_id",
  "evaluation_window_start_ts": "$start_time_window"
}
```

`--report logs` (the default) writes the report to stdout. `--report slack --slack-channel C…`
posts it to Slack instead, which needs `SLACK_BOT_TOKEN` and the `slack` extra. The webhook
answers with these status codes:

- `202`: investigation started
- `200`: duplicate of an alert already started, keyed on `monitor_id` + the window start
- `401`: bad secret
- `400`: bad payload
- `503`: all `--max-live` slots busy, so retry

`GET /health` needs no secret. In-flight runs and de-duplication live in memory, so run **one
instance** and don't let the host stop it between requests.

To try it locally before wiring up Bronto:

```bash
curl -i -X POST http://localhost:8080/events \
  -H "X-Bronto-Webhook-Secret: $BRONTO_WEBHOOK_SECRET" \
  -H 'Content-Type: application/json' \
  --data @tests/fixtures/monitor_event.json
```

### Interactively: the Slack bot

```bash
uv sync --extra slack
export SLACK_BOT_TOKEN=xoxb-…  SLACK_APP_TOKEN=xapp-…
uv run investigator-slack --provider anthropic --model-id claude-sonnet-4-5
```

`/investigate OOM on the checkout hosts` opens a thread, and the bot proposes matching alerts as
buttons. Click one, or reply in the thread with more detail. You can also `@mention` the bot in
an existing thread. The report is posted into the thread. The bot uses Socket Mode, so it needs
no public URL, just a container that stays up. In-flight runs don't survive a restart.

| Slack app setting | Value |
| ------- | ----- |
| Socket Mode | enabled, with an app-level token that has `connections:write` |
| Bot token scopes | `commands`, `chat:write`, `app_mentions:read`, `channels:history`, `groups:history` |
| Event subscriptions | `app_mention`, `message.channels`, `message.groups` |
| Slash command | `/investigate` |

**Reinstall the app after changing any of these settings.** Until you do, Slack silently drops
mention and message events. Invite the bot to a channel (`/invite @your-bot`) before using it
there.

### As a job: one container per event

```bash
investigator --provider anthropic --model-id claude-sonnet-4-5 --event - < event.json
```

This form reads the payload, prints the report to stdout, and exits. Exit code `0` means the
report was written, `1` means the run failed, and `2` means a usage error. It suits any
job-style host that collects stdout. To post event reports to Slack instead, run the webhook
surface with `--report slack --slack-channel C…`. On a host with no stdin, pass the payload in
`EVENT_JSON` and run the image's `investigate-event` helper.

### On Fly.io

[`deploy/fly.example.toml`](deploy/fly.example.toml) runs the Slack bot, and
[`deploy/fly.webhook.example.toml`](deploy/fly.webhook.example.toml) runs the webhook. Copy one to
the repo root without `.example` in its name, where it is gitignored. Set your app name,
`fly secrets set` the environment, then `fly deploy` (for the webhook config, add
`-c fly.webhook.toml`).

## License

MIT — see [LICENSE](LICENSE).
