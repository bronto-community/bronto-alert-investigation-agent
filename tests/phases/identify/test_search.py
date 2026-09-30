"""Tests for the identification phase's audit-trail search tools."""

import json
from datetime import datetime

import pytest
from langchain_core.messages import ToolMessage

from investigator.phases.identify.search import (
    AlertTimeseries,
    find_audit_row,
    make_alert_timeseries_tool,
    make_audit_events_tool,
)


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


@pytest.mark.parametrize(
    "make_tool", [make_audit_events_tool, make_alert_timeseries_tool]
)
def test_each_tool_tells_the_model_start_and_end_are_iso(make_tool):
    """The tools convert ISO-8601 to epoch-ms, so the model must send ISO."""
    assert "iso-8601" in make_tool(object()).description.lower()


# --- audit events -------------------------------------------------------------


def test_audit_search_tool_builds_request(mocker):
    """The tool converts ISO-8601 start/end to epoch-ms from_ts/to_ts."""
    client = mocker.Mock()
    client.post.return_value = {"result": []}
    tool = make_audit_events_tool(client)

    tool.invoke(
        {
            "start": "2026-07-15T00:00:00Z",
            "end": "2026-07-16T00:00:00Z",
            "pattern": "signup%",
        }
    )

    client.post.assert_called_once_with(
        "/search",
        json={
            "async_enabled": False,
            "from": [".audit-trail.events"],
            "select": ["*"],
            "most_recent_first": True,
            "from_ts": _ms("2026-07-15T00:00:00Z"),
            "to_ts": _ms("2026-07-16T00:00:00Z"),
            "where": (
                "\"event_info.type\"='MONITOR_STATE_CHANGE' "
                "AND \"service_payload.new_status\"='ALERT' "
                "AND \"service_payload.previous_status\" IN ('OK', 'WARN') "
                "AND 'signup%'"
            ),
            "limit": 50,
        },
    )


def test_audit_search_escapes_quote_in_pattern(mocker):
    """A model-supplied pattern containing `'` is escaped, not spliced raw."""
    client = mocker.Mock()
    client.post.return_value = {"result": []}
    tool = make_audit_events_tool(client)

    tool.invoke(
        {
            "start": "2026-07-15T00:00:00Z",
            "end": "2026-07-16T00:00:00Z",
            "pattern": "O'Brien%",
        }
    )

    where = client.post.call_args.kwargs["json"]["where"]
    assert where.endswith(" AND 'O''Brien%'")


def test_audit_search_response_adapted(mocker):
    """Adapts result[] into a ToolMessage: lean content, full payload artifact."""
    client = mocker.Mock()
    client.post.return_value = {
        "result": [
            {
                "@time": "2026-07-24 14:19:09.453 UTC",
                "event_id": "evt-1",
                "event_info.type": "MONITOR_STATE_CHANGE",
                "timestamp": "2026-07-24T14:19:09.453550458Z",
                "service_payload.monitor_id": "m-1",
                "service_payload.monitor_name": "foo bar baz",
                "service_payload.evaluation_timestamp": "1784901720000",
                "service_payload.new_status": "ALERT",
                "service_payload.threshold": "50",
            },
            {
                "@time": "2026-07-24 13:10:15.566 UTC",
                "event_id": "evt-2",
                "event_info.type": "MONITOR_STATE_CHANGE",
                "timestamp": "2026-07-24T13:10:15.566770440Z",
                "service_payload.monitor_id": "m-2",
                "service_payload.monitor_name": "Load Balancer Healthcheck disabled",
                "service_payload.evaluation_timestamp": "1784894880000",
                "service_payload.new_status": "ALERT",
                "service_payload.observed_value": "1",
            },
        ],
    }
    tool = make_audit_events_tool(client)

    msg = tool.invoke(
        {
            "name": "search_audit_events",
            "args": {"start": "2026-07-15T00:00:00Z", "end": "2026-07-16T00:00:00Z"},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert isinstance(msg, ToolMessage)

    # Artifact: every service_payload field (prefix stripped) with the raw
    # evaluation_timestamp (window start) and triggered_at = the fire-time
    # `timestamp` field, formatted to whole seconds.
    assert msg.artifact == [
        {
            "monitor_id": "m-1",
            "monitor_name": "foo bar baz",
            "evaluation_timestamp": 1784901720000,
            "new_status": "ALERT",
            "threshold": "50",
            "triggered_at": "2026-07-24T14:19:09Z",
        },
        {
            "monitor_id": "m-2",
            "monitor_name": "Load Balancer Healthcheck disabled",
            "evaluation_timestamp": 1784894880000,
            "new_status": "ALERT",
            "observed_value": "1",
            "triggered_at": "2026-07-24T13:10:15Z",
        },
    ]

    # Content: only triggered_at (fire time), monitor name, monitor id.
    assert isinstance(msg.content, str)
    assert json.loads(msg.content) == [
        {
            "triggered_at": "2026-07-24T14:19:09Z",
            "monitor_name": "foo bar baz",
            "monitor_id": "m-1",
        },
        {
            "triggered_at": "2026-07-24T13:10:15Z",
            "monitor_name": "Load Balancer Healthcheck disabled",
            "monitor_id": "m-2",
        },
    ]


@pytest.mark.parametrize(
    "names",
    [
        {"evaluation_window_start_ts": "1784901720000"},
        {
            "evaluation_window_start_ts": "1784901720000",
            "evaluation_timestamp": "1784894880000",
        },
    ],
)
def test_an_audit_row_with_the_renamed_window_start_is_adapted(mocker, names):
    """Bronto renamed `evaluation_timestamp` to `evaluation_window_start_ts`.
    The row keeps the name the context phase reads, and the new name wins."""
    client = mocker.Mock()
    client.post.return_value = {
        "result": [
            {
                "timestamp": "2026-07-24T14:19:09.453550458Z",
                "service_payload.monitor_id": "m-1",
                **{f"service_payload.{k}": v for k, v in names.items()},
            }
        ]
    }

    msg = make_audit_events_tool(client).invoke(
        {
            "name": "search_audit_events",
            "args": {"start": "2026-07-24T00:00:00Z", "end": "2026-07-25T00:00:00Z"},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert msg.artifact == [
        {
            "monitor_id": "m-1",
            "evaluation_timestamp": 1784901720000,
            "triggered_at": "2026-07-24T14:19:09Z",
        }
    ]


# --- reading an audit row back ------------------------------------------------


_ROW = {
    "monitor_id": "mon-1",
    "monitor_name": "OOM killer invoked",
    "triggered_at": "2026-07-24T14:19:09Z",
    "evaluation_timestamp": 1784901720000,
}


def _found(*rows):
    return ToolMessage(content="[]", tool_call_id="t1", artifact=list(rows))


def test_a_pick_from_the_content_finds_its_full_row(mocker):
    """The round trip the confirm tool relies on: the model copies an alert out
    of `content`, and that is enough to recover the whole row from `artifact`."""
    client = mocker.Mock()
    client.post.return_value = {
        "result": [
            {
                "timestamp": "2026-07-24T14:19:09.453550458Z",
                "service_payload.monitor_id": "mon-1",
                "service_payload.monitor_name": "OOM killer invoked",
                "service_payload.evaluation_timestamp": "1784901720000",
                "service_payload.threshold": "1",
            }
        ]
    }
    msg = make_audit_events_tool(client).invoke(
        {
            "name": "search_audit_events",
            "args": {"start": "2026-07-24T00:00:00Z", "end": "2026-07-25T00:00:00Z"},
            "id": "call-1",
            "type": "tool_call",
        }
    )
    seen_by_model = json.loads(msg.content)[0]

    row = find_audit_row(
        [msg], seen_by_model["monitor_id"], seen_by_model["triggered_at"]
    )

    assert row is msg.artifact[0]
    assert row["threshold"] == "1"  # a field the model never saw


def test_find_audit_row_matches_on_both_monitor_and_time():
    """A monitor can fire more than once, so the same monitor at another time
    is a different alert."""
    earlier = {**_ROW, "triggered_at": "2026-07-24T00:00:00Z"}
    other = {**_ROW, "monitor_id": "mon-2"}

    assert (
        find_audit_row([_found(earlier, other, _ROW)], "mon-1", _ROW["triggered_at"])
        is _ROW
    )


def test_find_audit_row_returns_none_for_an_invented_alert():
    assert find_audit_row([_found(_ROW)], "mon-9", _ROW["triggered_at"]) is None


# --- alert timeseries ---------------------------------------------------------


def _series(name, counts):
    """Build a minimal groups_series entry like Bronto's /search response."""
    return {
        "key": "service_payload.monitor_name",
        "name": name,
        "count": sum(counts),
        "stat": "count(*)",
        "value": float(sum(counts)),
        "timeseries": [
            {"@timestamp": 1000 + i, "count": c, "value": float(c)}
            for i, c in enumerate(counts)
        ],
    }


def test_timeseries_tool_builds_request(mocker):
    """The tool converts ISO-8601 start/end to epoch-ms from_ts/to_ts."""
    client = mocker.Mock()
    client.post.return_value = {}
    tool = make_alert_timeseries_tool(client)

    tool.invoke(
        {
            "start": "2026-07-15T00:00:00Z",
            "end": "2026-07-16T00:00:00Z",
            "pattern": "checkout%",
        }
    )

    client.post.assert_called_once_with(
        "/search",
        json={
            "async_enabled": False,
            "from": [".audit-trail.events"],
            "groups": ['"service_payload.monitor_name"'],
            "num_of_slices": 20,
            "limit": 10,
            "select": ["count(*)"],
            "from_ts": _ms("2026-07-15T00:00:00Z"),
            "to_ts": _ms("2026-07-16T00:00:00Z"),
            "where": (
                "\"event_info.type\"='MONITOR_STATE_CHANGE' "
                "AND \"service_payload.new_status\"='ALERT' "
                "AND \"service_payload.previous_status\" IN ('OK', 'WARN') "
                "AND 'checkout%'"
            ),
        },
    )


def test_timeseries_response_adapted(mocker):
    """groups_series is adapted to per-monitor series, ordered by total desc."""
    client = mocker.Mock()
    client.post.return_value = {
        "groups_series": [
            _series("foo", [0, 2, 3, 0]),
            _series("bar", [0, 0, 6, 0]),
        ],
        "totals": {"count": 11},  # ignored by the adapter
    }
    tool = make_alert_timeseries_tool(client)

    result = tool.invoke(
        {"start": "2026-07-15T00:00:00Z", "end": "2026-07-16T00:00:00Z"}
    )

    assert result == [
        AlertTimeseries(monitor_name="bar", total=6, counts=[0, 0, 6, 0]),
        AlertTimeseries(monitor_name="foo", total=5, counts=[0, 2, 3, 0]),
    ]
