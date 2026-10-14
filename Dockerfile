# syntax=docker/dockerfile:1
#
# One image, every deployment mode. The CLI, the Slack bot and the webhook listener are
# the same install differing only in which console script the host is told to
# run.
#
# Neither extra is optional here. `slack` is needed by the Slack bot and by the
# webhook's `--report slack`; `webhook` is needed by the listener.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependencies resolve in their own layer, so editing a phase does not reinstall
# langchain. --frozen fails the build if uv.lock has drifted from pyproject.toml
# rather than silently resolving something CI never saw.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra slack --extra webhook --no-install-project

# The package metadata names both files, so the project build needs them too.
COPY README.md LICENSE ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra slack --extra webhook --no-editable


FROM python:3.14-slim-bookworm

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# A job host that starts a container per event often gives it no stdin, so the
# payload arrives in the environment instead and this script does the one
# translation.
COPY <<'SH' /usr/local/bin/investigate-event
#!/bin/sh
set -eu
: "${EVENT_JSON:?EVENT_JSON is not set}"
printf '%s' "$EVENT_JSON" | investigator --event - "$@"
SH
RUN chmod +x /usr/local/bin/investigate-event

RUN useradd --create-home --uid 10001 investigator
USER investigator

# The Slack bot is the default; the webhook listener and the event path each
# override it. All three need --provider and --model-id, so none has a useful
# default here.
CMD ["investigator-slack", "--help"]
