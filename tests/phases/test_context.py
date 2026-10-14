"""Tests for context generation: the node, and each step it composes."""

import logging

from bronto_sdk.models import (
    DatasetsResponse,
    SearchRequest,
    SearchResponse,
)
from langchain_core.exceptions import OutputParserException

from investigator.phases.context import (
    _TRIGGER_BUFFER_MS,
    LOOKBACK_WINDOWS,
    TIMESERIES_SLICES,
    DeriveDatasets,
    assemble_alert_context,
    build_context_node,
    derive_related_datasets,
    fetch_monitor,
    list_log_datasets,
    monitor_state_timeseries,
    resolve_queries,
    resolve_related_datasets,
    trim_monitor,
)
from investigator.settings import Deps
from investigator.state import TriggeredAlert, Window, state_for_event
from tests.conftest import a_monitor

_WINDOW: Window = {"start_ms": 100, "end_ms": 200}


def _datasets(*rows) -> DatasetsResponse:
    """A /datasets response in the REST shape: `id` and `dataset`, not MCP's."""
    return DatasetsResponse.model_validate(
        {
            "datasets": [
                {"id": log_id, "dataset": name, "collection": collection}
                for log_id, name, collection in rows
            ]
        }
    )


# --- the node -----------------------------------------------------------------


def _deps(client, model) -> Deps:
    return Deps(client=client, model=model, investigation_tools=[])


def _triggered_alert(row: dict) -> dict:
    """The triggered alert as the identify phase leaves it on the state."""
    return TriggeredAlert(
        id=row["monitor_id"],
        label=row["monitor_name"],
        occurred_at=row["triggered_at"],
        data=row,
    ).model_dump(mode="json")


_TRIGGERED = {
    "monitor_id": "mon-1",
    "monitor_name": "foo bar baz",
    "triggered_at": "2026-07-24T14:19:09Z",
    "evaluation_timestamp": 1784901720000,
}

# The monitor query as it appears in the bundle: datasets folded in, with
# from/from_expr and aggregation dropped.
_RESOLVED_QUERY = {
    "select": ["COUNT(*)"],
    "where": "'foo bar'",
    "groups": [],
    "datasets": [{"log_id": "aaa", "name": "syslog", "collection": "my-service"}],
}


async def test_context_node_without_a_playbook_never_calls_the_model(
    mocker, fake_bronto
):
    """Nothing to infer related datasets from, so no model call and no full
    dataset listing; the rest of the bundle is built exactly as with a playbook."""
    model = mocker.Mock()
    state = {"triggered_alert": _triggered_alert(_TRIGGERED), "playbook": None}

    delta = await build_context_node(_deps(fake_bronto, model))(state)

    model.with_structured_output.assert_not_called()
    # The only listing is the query's own datasets, selected by id.
    fake_bronto.datasets.list.assert_called_once_with(from_=["aaa"])
    assert delta["alert_context"]["playbook"] is None
    assert delta["alert_context"]["detail"]["related_datasets"] == []
    # window: [end - N*window, end], end = eval + window + buffer, in ms
    window_ms = 30 * 60_000  # monitor window "last 30 minutes"
    end = 1784901720000 + window_ms + _TRIGGER_BUFFER_MS
    assert delta["alert_context"]["window"] == {
        "start_ms": end - LOOKBACK_WINDOWS * window_ms,
        "end_ms": end,
    }


async def test_context_node_end_to_end_with_playbook(mocker, fake_bronto):
    """Full flow with a playbook: the model derives related datasets, which are
    resolved and folded into the complete assembled bundle."""
    query_datasets = _datasets(("aaa", "syslog", "my-service"))
    every_dataset = _datasets(
        ("aaa", "syslog", "my-service"), ("app-1", "app.log", "my-app")
    )
    fake_bronto.datasets.list.side_effect = [query_datasets, every_dataset]
    model = mocker.Mock(supports_tool_choice_values=("auto", "any", "tool"))
    model.with_structured_output.return_value.ainvoke = mocker.AsyncMock(
        return_value=DeriveDatasets(datasets=["app.log"])
    )
    state = {"triggered_alert": _triggered_alert(_TRIGGERED), "playbook": "check redis"}

    delta = await build_context_node(_deps(fake_bronto, model))(state)

    fake_bronto.monitors.retrieve.assert_called_once_with("mon-1")
    fake_bronto.metrics.retrieve_definition.assert_called_once_with("md-1")
    model.with_structured_output.assert_called_once_with(
        DeriveDatasets, method="function_calling"
    )
    assert delta["alert_context"] == {
        "window": {"start_ms": 1784885580000, "end_ms": 1784903580000},
        "playbook": "check redis",
        "detail": {
            "monitor": {
                "name": "foo bar baz",
                "monitor_type": "PATTERN",
                "window": "last 30 minutes",
                "queries": [_RESOLVED_QUERY],
            },
            "triggered_event": {"evaluation_timestamp": 1784901720000},
            "related_datasets": [
                {"log_id": "app-1", "name": "app.log", "collection": "my-app"}
            ],
            "monitor_state_timeseries": [
                {"group": None, "counts": [5], "values": [5.0]}
            ],
        },
    }


EVENT = {
    "monitor_id": "mon-1",
    "monitor_name": "foo bar baz",
    "triggered_at": "2026-07-24T14:19:09Z",
    "evaluation_timestamp": 1784901720000,
    "new_status": "ALERT",
}


async def test_an_event_and_an_identified_alert_ground_the_same_context(
    mocker, make_model, fake_bronto
):
    """The invariant that makes the webhook entry trustworthy: a payload-derived
    triggered alert and an identified one produce the same alert context, so
    the two entry paths cannot drift apart."""
    node = build_context_node(_deps(fake_bronto, make_model([])))

    from_event = await node(
        {"triggered_alert": state_for_event(EVENT)["triggered_alert"], "playbook": None}
    )
    from_identification = await node(
        {
            "triggered_alert": TriggeredAlert(
                id="mon-1",
                label="foo bar baz",
                occurred_at="2026-07-24T14:19:09Z",
                data=EVENT,
            ).model_dump(mode="json"),
            "playbook": None,
        }
    )

    assert from_event == from_identification


# --- the monitor and its queries ----------------------------------------------


def test_the_queries_come_from_the_metric_definition(fake_bronto):
    """`GET /monitors/{id}` carries no queries; its `metric_id` names the
    definition that does. They are dumped by alias, keeping the wire key `from`."""
    monitor = fetch_monitor(fake_bronto, "mon-1")

    fake_bronto.metrics.retrieve_definition.assert_called_once_with("md-1")
    assert monitor["name"] == "foo bar baz"
    assert monitor["queries"][0]["from"] == ["aaa"]
    assert "from_" not in monitor["queries"][0]


def test_a_monitor_without_a_metric_definition_keeps_what_it_has(fake_bronto):
    fake_bronto.monitors.retrieve.return_value = a_monitor(metric_id=None)

    monitor = fetch_monitor(fake_bronto, "mon-1")

    fake_bronto.metrics.retrieve_definition.assert_not_called()
    assert "queries" not in monitor


_FULL_MONITOR = {
    "id": "b415076f-692e-4683-ae7e-28bb73ebf454",
    "name": "foo bar baz",
    "description": "A box went foo bar",
    "metric_id": "7559517c-6137-4cb3-a787-951f7e9c146d",
    "monitor_type": "PATTERN",
    "threshold": 1.0,
    "comparison_operator": "ABOVE_OR_EQUAL",
    "no_data_status": "OK",
    "window": "last 30 minutes",
    "actions": [{"type": "EMAIL", "email": "somebody@example.com"}],
    "notify_once": False,
    "ai_report_enabled": False,
    "status": "OK",
    "last_trigger_ts": 1755364510857,
    "metadata": {"created_at": 1738857414077},
    "aux": {"query_mode": "raw"},
    "queries": [{"select": ["COUNT(*)"], "from": ["078a"], "where": "'foo bar'"}],
    "formulas": [],
}

_KEPT = {
    "name",
    "description",
    "monitor_type",
    "threshold",
    "comparison_operator",
    "no_data_status",
    "window",
    "status",
    "queries",
    "formulas",
}


def test_trim_monitor_retains_whitelisted_fields():
    """Only the fields the investigation agent needs survive: ids, actions, and
    metadata are dropped."""
    trimmed = trim_monitor(_FULL_MONITOR)

    assert set(trimmed) == _KEPT
    assert trimmed["queries"] == _FULL_MONITOR["queries"]


def test_trim_monitor_retains_change_detection_fields():
    """A CHANGE_DETECTION monitor also keeps compare_to and change_type."""
    monitor = {
        **_FULL_MONITOR,
        "monitor_type": "CHANGE_DETECTION",
        "compare_to": "2 hours ago",
        "change_type": "PERCENTAGE",
    }

    trimmed = trim_monitor(monitor)

    assert set(trimmed) == _KEPT | {"compare_to", "change_type"}
    assert trimmed["compare_to"] == "2 hours ago"


async def test_resolve_queries_folds_in_the_monitor_own_datasets(mocker):
    """A monitor query's `from` dataset ids are resolved to {log_id, name,
    collection} and folded into the (trimmed) query, replacing `from`."""
    client = mocker.Mock()
    client.datasets.list.return_value = _datasets(("078a81b8", "syslog", "my-service"))
    monitor = {
        "queries": [
            {"select": ["COUNT(*)"], "from": ["078a81b8"], "where": "'foo bar'"}
        ]
    }

    result = await resolve_queries(client, monitor)

    client.datasets.list.assert_called_once_with(from_=["078a81b8"])
    assert result == [
        {
            "select": ["COUNT(*)"],
            "where": "'foo bar'",
            "datasets": [
                {"log_id": "078a81b8", "name": "syslog", "collection": "my-service"}
            ],
        }
    ]


async def test_resolve_queries_counts_a_pattern_monitor_select_star(mocker):
    """A pattern monitor selects `*` and counts matches through its
    `aggregation`; Bronto rejects `*` beside `groups`, so the resolved query
    selects `count(*)` instead."""
    client = mocker.Mock()
    client.datasets.list.return_value = _datasets(("6c62", "s3-access", "store"))
    groups = ['"RequestURI_operation"', '"Requester.resource_id.role_name"']
    monitor = {
        "queries": [
            {
                "select": ["*"],
                "where": "\"Requester.resource_id.role_name\" NOT IN ('S3UploaderRole')",
                "groups": groups,
                "aggregation": [{"time": "count", "reduce_to": "avg"}],
                "from": ["6c62"],
            }
        ]
    }

    [query] = await resolve_queries(client, monitor)

    assert query["select"] == ["count(*)"]
    assert query["groups"] == groups
    assert "aggregation" not in query


async def test_resolve_queries_routes_from_expr_query(mocker):
    """A query selecting its datasets by `from_expr` (an empty `from` is unset)
    passes the expression through instead."""
    client = mocker.Mock()
    client.datasets.list.return_value = _datasets(("8f7e471d", "app.log", "my-app"))
    expr = "\"collection\" IN ('org/ap/app.log')"
    monitor = {"queries": [{"from": [], "from_expr": expr, "where": "'foo bar'"}]}

    result = await resolve_queries(client, monitor)

    client.datasets.list.assert_called_once_with(from_expr=expr)
    assert result == [
        {
            "where": "'foo bar'",
            "datasets": [
                {"log_id": "8f7e471d", "name": "app.log", "collection": "my-app"}
            ],
        }
    ]


async def test_resolve_queries_one_call_per_query(mocker):
    """A monitor with several queries makes one listing per query, keeping each
    query's datasets on that query."""
    client = mocker.Mock()
    client.datasets.list.side_effect = [
        _datasets(("aaa", "syslog", "yebo")),
        _datasets(("bbb", "app.log", "nobo")),
    ]
    expr = "\"collection\" IN ('org/ap/nobo')"
    monitor = {
        "queries": [
            {"from": ["aaa"], "where": "'x'"},
            {"from_expr": expr, "where": "'y'"},
        ]
    }

    result = await resolve_queries(client, monitor)

    assert client.datasets.list.call_count == 2
    assert [q["datasets"] for q in result] == [
        [{"log_id": "aaa", "name": "syslog", "collection": "yebo"}],
        [{"log_id": "bbb", "name": "app.log", "collection": "nobo"}],
    ]


# --- related datasets ---------------------------------------------------------


def test_the_log_datasets_leave_out_metrics_and_traces(mocker):
    """A full listing returns the metric and trace collections too, and no log
    search can use them."""
    client = mocker.Mock()
    client.datasets.list.return_value = _datasets(
        ("aaa", "syslog", "my-service"),
        ("m-1", "cpu", ".metrics"),
        ("t-1", "spans", "default_traces"),
    )

    assert list_log_datasets(client) == [
        {"log_id": "aaa", "name": "syslog", "collection": "my-service"}
    ]


_LOG_DATASETS = [
    {"log_id": "8f7e471d", "name": "app.log", "collection": "nobo"},
    {"log_id": "9a0b1c2d", "name": "app.log", "collection": "yebo"},
    {"log_id": "abc123", "name": "reader", "collection": "reader"},
    {"log_id": "zzz", "name": "syslog", "collection": "yebo"},
]


def test_resolve_related_datasets_matches_names_across_collections():
    """A playbook names a dataset, not its collection, so every match counts."""
    assert resolve_related_datasets(_LOG_DATASETS, ["app.log", "reader"]) == [
        {"log_id": "8f7e471d", "name": "app.log", "collection": "nobo"},
        {"log_id": "9a0b1c2d", "name": "app.log", "collection": "yebo"},
        {"log_id": "abc123", "name": "reader", "collection": "reader"},
    ]


def test_resolve_related_datasets_warns_on_unresolved_name(caplog):
    """A name that matches no dataset is logged, rather than failing the phase."""
    with caplog.at_level(logging.WARNING):
        result = resolve_related_datasets(_LOG_DATASETS, ["reader", "ghost-log"])

    assert result == [{"log_id": "abc123", "name": "reader", "collection": "reader"}]
    assert "ghost-log" in caplog.text


async def test_derive_related_datasets_grounds_prompt_with_valid_names(mocker):
    """With a playbook, the step feeds the model both the playbook and the list
    of valid dataset names (to pick from), and returns the chosen names."""
    model = mocker.Mock(supports_tool_choice_values=("auto", "any", "tool"))
    structured = model.with_structured_output.return_value
    structured.ainvoke = mocker.AsyncMock()
    structured.ainvoke.return_value = DeriveDatasets(datasets=["app.log", "reader"])
    playbook = "If dopamine drops, inspect the reader and app logs for restarts."
    valid = ["app.log", "reader", "syslog"]

    result = await derive_related_datasets(model, playbook, valid)

    model.with_structured_output.assert_called_once_with(
        DeriveDatasets, method="function_calling"
    )
    sent = str(structured.ainvoke.call_args.args[0])
    assert playbook in sent
    assert "syslog" in sent  # a valid name the playbook never mentions
    assert result == ["app.log", "reader"]


async def test_derive_related_datasets_uses_native_schema_when_tool_cannot_be_forced(
    mocker,
):
    """A model that rejects forced tool use answers in the provider's native
    JSON schema instead of through a forced tool call."""
    model = mocker.Mock(supports_tool_choice_values=("auto",))
    structured = model.with_structured_output.return_value
    structured.ainvoke = mocker.AsyncMock(return_value=DeriveDatasets(datasets=[]))

    await derive_related_datasets(model, "check redis", ["app.log"])

    model.with_structured_output.assert_called_once_with(
        DeriveDatasets, method="json_schema"
    )


async def test_derive_related_datasets_survives_a_malformed_answer(mocker, caplog):
    """A weak model can wrap its JSON in markdown; related datasets are an
    enrichment, so the step logs and continues without them."""
    model = mocker.Mock(supports_tool_choice_values=("auto",))
    model.with_structured_output.return_value.ainvoke = mocker.AsyncMock(
        side_effect=OutputParserException('Invalid json output: **{"datasets": []}')
    )

    result = await derive_related_datasets(model, "check redis", ["app.log"])

    assert result == []
    assert "related datasets" in caplog.text


async def test_derive_related_datasets_survives_no_answer(mocker):
    """An unforced tool call the model never makes parses to None."""
    model = mocker.Mock(supports_tool_choice_values=("auto",))
    model.with_structured_output.return_value.ainvoke = mocker.AsyncMock(
        return_value=None
    )

    assert await derive_related_datasets(model, "check redis", ["app.log"]) == []


async def test_derive_related_datasets_skips_model_without_playbook(mocker):
    """With no playbook there is nothing to infer, so the model is never
    called and no related datasets are returned."""
    model = mocker.Mock()

    result = await derive_related_datasets(model, None, ["my.app", "syslog"])

    assert result == []
    model.with_structured_output.assert_not_called()


# --- the monitor state timeseries ---------------------------------------------


def _query(group_by=None):
    return {
        "select": ["COUNT(*)"],
        "where": "'foo bar'",
        "groups": group_by or [],
        "datasets": [
            {"log_id": "aaa", "name": "syslog", "collection": "yebo"},
            {"log_id": "bbb", "name": "app.log", "collection": "nobo"},
        ],
    }


def _answering(mocker, body: dict):
    client = mocker.Mock()
    client.search.run.return_value = SearchResponse.model_validate(body)
    return client


async def test_monitor_state_runs_the_query_over_the_search_window(mocker):
    """The exact request is the contract: the monitor's own query as an
    aggregate over the window it is given, scoped to the query's datasets,
    preserving group-by keys."""
    client = _answering(mocker, {})

    await monitor_state_timeseries(client, _query(["host"]), _WINDOW)

    client.search.run.assert_called_once_with(
        SearchRequest(
            select=["COUNT(*)"],
            where="'foo bar'",
            groups=["host"],
            from_=["aaa", "bbb"],
            from_ts=_WINDOW["start_ms"],
            to_ts=_WINDOW["end_ms"],
            num_of_slices=TIMESERIES_SLICES,
        )
    )


async def test_monitor_state_adapts_a_grouped_response(mocker):
    """A grouped response becomes one compact {group, counts, values} entry per
    series, dropping timestamps, totals, and quantiles."""
    client = _answering(
        mocker,
        {
            "groups_series": [
                {
                    "name": "i-075",
                    "timeseries": [
                        {"@timestamp": 1, "count": 863, "value": 863.0},
                        {"@timestamp": 2, "count": 864, "value": 864.0},
                    ],
                },
                {
                    "name": "i-085",
                    "timeseries": [
                        {"@timestamp": 1, "count": 864, "value": 864.0},
                        {"@timestamp": 2, "count": 863, "value": 863.0},
                    ],
                },
            ],
            "totals": {"count": 3454},  # ignored
        },
    )

    result = await monitor_state_timeseries(
        client, _query(["$ec2_instance_id"]), _WINDOW
    )

    assert result == [
        {"group": "i-075", "counts": [863, 864], "values": [863.0, 864.0]},
        {"group": "i-085", "counts": [864, 863], "values": [864.0, 863.0]},
    ]


async def test_monitor_state_adapts_an_ungrouped_response_from_its_totals(mocker):
    """Ungrouped, /search returns no groups_series at all, only totals — which
    become the one series with no group."""
    client = _answering(
        mocker,
        {
            "groups_series": [],
            "totals": {
                "count": 3409,
                "timeseries": [
                    {"@timestamp": 1, "count": 1727, "value": 1727.0},
                    {"@timestamp": 2, "count": 1682, "value": 1682.0},
                ],
            },
        },
    )

    result = await monitor_state_timeseries(client, _query(), _WINDOW)

    assert result == [
        {"group": None, "counts": [1727, 1682], "values": [1727.0, 1682.0]},
    ]


async def test_monitor_state_with_no_matches_is_no_series(mocker):
    """Neither groups nor a total means nothing matched, not a series of zeros."""
    result = await monitor_state_timeseries(_answering(mocker, {}), _query(), _WINDOW)

    assert result == []


# --- assembling the alert context ---------------------------------------------


def test_assemble_prunes_monitor_redundant_triggered_event_fields():
    """Fields duplicated in the monitor object (ids, names, pattern, query) are
    dropped from the triggered_event; the rest is carried whole."""
    triggered_event = {
        "alert_id": "a-1",
        "monitor_id": "mon-1",
        "monitor_name": "foo bar baz",
        "monitor_type": "PATTERN",
        "monitor_pattern": "'foo bar'",
        "window": "last 30 minutes",
        "org_id": "org-9",
        "query": "SELECT ...",
        "group_key.host": "i-123",
        "group_key_0": "us-east-1",
        "new_status": "ALERT",
        "previous_status": "OK",
        "evaluation_timestamp": 1784901720000,
        "triggered_at": "2026-07-24T14:02:00Z",
        "threshold": "1",
    }

    bundle = assemble_alert_context(
        monitor={"name": "foo bar baz", "monitor_type": "PATTERN"},
        triggered_event=triggered_event,
        related_datasets=[],
        playbook=None,
        monitor_state_timeseries=[],
        window=_WINDOW,
    )

    assert bundle["detail"]["triggered_event"] == {
        "alert_id": "a-1",  # kept for debugging
        "new_status": "ALERT",
        "previous_status": "OK",
        "evaluation_timestamp": 1784901720000,
        "threshold": "1",
        # triggered_at (fire time) dropped — it's identification-only
    }


def test_assemble_resolves_compare_to_ms_for_change_detection():
    """For a CHANGE_DETECTION monitor the compare_to offset is resolved to ms
    and carried in the bundle so the investigation phase needn't re-parse it."""
    monitor = {
        "name": "dopamine drop",
        "monitor_type": "CHANGE_DETECTION",
        "compare_to": "2 hours ago",
    }

    bundle = assemble_alert_context(
        monitor=monitor,
        triggered_event={"triggered_at": "2026-07-24T14:03:00Z"},
        related_datasets=[],
        playbook=None,
        monitor_state_timeseries=[],
        window=_WINDOW,
    )

    assert bundle["detail"]["monitor"]["compare_to_ms"] == 2 * 60 * 60 * 1000
    assert bundle["detail"]["monitor"]["compare_to"] == "2 hours ago"
