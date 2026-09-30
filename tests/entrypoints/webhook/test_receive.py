"""Tests for the receiver: what it accepts, what it refuses, and what it starts.

Every test drives the real Starlette app through its test client. Nothing here
opens an MCP session or calls a model — `start` is the seam, and it is injected.
"""

import json

import pytest
from starlette.testclient import TestClient

from investigator.entrypoints.webhook.receive import SECRET_HEADER, build_receiver
from investigator.entrypoints.webhook.runs import AtCapacity

EVENT = {
    "monitor_id": "mon-1",
    "monitor_name": "foo bar baz",
    "triggered_at": "2026-07-24T14:19:09Z",
    "evaluation_timestamp": 1784901720000,
}

SECRET = "shared-secret"


@pytest.fixture
def started() -> list[dict]:
    return []


@pytest.fixture
def client(started) -> TestClient:
    def start(payload: dict) -> str:
        started.append(payload)
        return f"webhook:run-{len(started)}"

    return TestClient(build_receiver(start, secret=SECRET))


def _post(client: TestClient, payload, *, secret: str | None = SECRET):
    headers = {} if secret is None else {SECRET_HEADER: secret}
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return client.post("/events", content=body, headers=headers)


# --- who is allowed to ask ----------------------------------------------------


def test_a_good_secret_starts_a_run(client, started):
    response = _post(client, EVENT)

    assert response.status_code == 202
    assert response.json()["run_id"] == "webhook:run-1"
    assert started[0]["monitor_id"] == "mon-1"


@pytest.mark.parametrize("secret", [None, "", "wrong-secret", SECRET + "x"])
def test_anything_but_the_secret_is_refused_and_starts_nothing(client, started, secret):
    """The URL is public; the model budget behind it is not."""
    response = _post(client, EVENT, secret=secret)

    assert response.status_code == 401
    assert started == []


def test_the_secret_is_checked_before_the_payload_is_read(client, started):
    """An unauthenticated caller learns nothing about what we would have parsed."""
    response = _post(client, "not json at all", secret="wrong-secret")

    assert response.status_code == 401
    assert started == []


# --- what counts as an event --------------------------------------------------


def test_a_payload_the_agent_could_not_work_from_is_refused_at_the_edge(
    client, started
):
    """`state_for_event` is the same validator the CLI runs, one hop earlier —
    so a malformed webhook costs a 400, not a run and a model call."""
    response = _post(client, {"monitor_name": "no id, no timestamp"})

    assert response.status_code == 400
    assert started == []


def test_unparseable_json_is_refused(client, started):
    response = _post(client, "{not json")

    assert response.status_code == 400
    assert started == []


def test_a_json_array_is_refused(client, started):
    """The CLI's `_read_event` makes the same demand for the same reason."""
    response = _post(client, [EVENT])

    assert response.status_code == 400
    assert started == []


# --- retries ------------------------------------------------------------------


def test_the_same_alert_twice_starts_one_run(client, started):
    """Webhooks retry. Two POSTs for one alert must not mean two reports."""
    first = _post(client, EVENT)
    second = _post(client, EVENT)

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert len(started) == 1


def test_a_second_firing_of_the_same_monitor_is_not_a_duplicate(client, started):
    """A monitor is a class; an alert is an occurrence. The window tells them
    apart, which is why the key is the pair and not the monitor id."""
    _post(client, EVENT)
    _post(client, {**EVENT, "evaluation_timestamp": 1784901780000})

    assert len(started) == 2


def test_the_renamed_window_start_is_the_same_alert(client, started):
    """Bronto renamed `evaluation_timestamp` to `evaluation_window_start_ts`.
    The dedup key reads the raw body, so it must read either name, and one alert
    sent in both formats is still one alert."""
    renamed = {k: v for k, v in EVENT.items() if k != "evaluation_timestamp"} | {
        "evaluation_window_start_ts": EVENT["evaluation_timestamp"]
    }

    first = _post(client, renamed)
    second = _post(client, EVENT)

    assert first.status_code == 202
    assert first.json()["alert"] == "mon-1@1784901720000"
    assert second.json()["duplicate"] is True
    assert len(started) == 1


# --- when there is no room ----------------------------------------------------


def test_a_full_registry_refuses_rather_than_queues():
    """503 is the sender's cue to retry. A queue here would be a queue to lose."""

    def start(_payload: dict) -> str:
        raise AtCapacity("already running 3 investigations")

    client = TestClient(build_receiver(start, secret=SECRET))
    response = _post(client, EVENT)

    assert response.status_code == 503


def test_a_refused_alert_is_not_remembered():
    """Otherwise the retry — the one thing that could still save it — comes
    back as 'duplicate' and the alert is lost for good."""
    attempts = []

    def start(payload: dict) -> str:
        attempts.append(payload)
        if len(attempts) == 1:
            raise AtCapacity("no room")
        return "webhook:run-2"

    client = TestClient(build_receiver(start, secret=SECRET))
    assert _post(client, EVENT).status_code == 503
    assert _post(client, EVENT).status_code == 202
    assert len(attempts) == 2


# --- staying up ---------------------------------------------------------------


def test_the_health_check_needs_no_secret(client):
    """The host polls this; it has no secret and should not need one."""
    assert client.get("/health").status_code == 200
