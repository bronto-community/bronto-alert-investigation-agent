"""Phase 2: gather context around the time when the alert triggered."""

import asyncio
import logging
from typing import Any, cast

from bronto_sdk import BrontoClient
from bronto_sdk.domain import parse_window, project_dataset
from bronto_sdk.models import DatasetsResponse, SearchRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ..settings import Deps
from ..state import AlertContext, InvestigationState, Window

logger = logging.getLogger(__name__)


def build_context_node(deps: Deps):
    """Build the single context-generation node.

    Composes deterministic REST lookups with one (conditional) model call:
    - fetch the monitor, and fold each query's datasets into it
    - derive & resolve playbook-implied datasets
    - run the monitor's own query as a timeseries over the search window
    - assemble the alert context.
    """
    client, model = deps.client, deps.model

    async def context(state: InvestigationState) -> dict[str, Any]:
        triggered_alert = state["triggered_alert"]
        assert triggered_alert is not None, (
            "context runs only once a triggered alert is grounded"
        )
        triggered_event: dict[str, Any] = triggered_alert["data"]
        playbook = state.get("playbook")

        monitor: dict[str, Any] = trim_monitor(
            await asyncio.to_thread(
                fetch_monitor, client, triggered_event["monitor_id"]
            )
        )
        monitor = {**monitor, "queries": await resolve_queries(client, monitor)}
        related: list[dict[str, Any]] = []
        if playbook:
            # One listing both grounds the model's pick and resolves it.
            log_datasets = await asyncio.to_thread(list_log_datasets, client)
            names = sorted({dataset["name"] for dataset in log_datasets})
            picked = await derive_related_datasets(model, playbook, names)
            related = resolve_related_datasets(log_datasets, picked)
        window = search_window(
            triggered_event["evaluation_timestamp"], parse_window(monitor["window"])
        )
        monitor_state = await monitor_state_timeseries(
            client, monitor["queries"][0], window
        )
        alert_context = assemble_alert_context(
            monitor=monitor,
            triggered_event=triggered_event,
            related_datasets=related,
            playbook=playbook,
            monitor_state_timeseries=monitor_state,
            window=window,
        )
        logger.info(
            "context generated: %d quer(y/ies), %d related dataset(s)",
            len(monitor["queries"]),
            len(related),
        )
        return {"alert_context": alert_context}

    return context


def fetch_monitor(client: BrontoClient, monitor_id: str) -> dict[str, Any]:
    """The monitor, with the queries and formulas of its metric definition.

    A monitor names its backing metric by ``metric_id``, and that definition is
    where its queries live.
    """
    monitor = client.monitors.retrieve(monitor_id)
    dumped = monitor.model_dump(mode="json", by_alias=True, exclude_none=True)
    if monitor.metric_id:
        definition = client.metrics.retrieve_definition(monitor.metric_id)
        dumped |= definition.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
            include={"queries", "formulas"},
        )
    return dumped


def trim_monitor(monitor: dict[str, Any]) -> dict[str, Any]:
    """Reduce a monitor to the whitelisted fields."""
    return {key: monitor[key] for key in _MONITOR_WHITELIST if key in monitor}


# Monitor fields forwarded to the investigation sub-agent. `compare_to` and
# `change_type` only appear on CHANGE_DETECTION monitors; whichever are
# present are kept.
_MONITOR_WHITELIST = (
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
    "compare_to",
    "change_type",
)


async def resolve_queries(
    client: BrontoClient, monitor: dict[str, Any]
) -> list[dict[str, Any]]:
    """Trim each monitor query and fold in the datasets it applies to.

    ``from``/``from_expr`` name the datasets by id or expression; they are
    replaced by the resolved datasets, so the investigation agent can re-run the
    same query against the logs it applies to. The queries are independent, so
    their lookups overlap rather than running one round-trip at a time.
    """
    queries = monitor.get("queries", [])
    return list(await asyncio.gather(*(_resolve_query(client, q) for q in queries)))


async def _resolve_query(client: BrontoClient, query: dict[str, Any]) -> dict[str, Any]:
    if query.get("from"):
        selector: dict[str, Any] = {"from_": query["from"]}
    else:
        selector = {"from_expr": query["from_expr"]}
    response = await asyncio.to_thread(client.datasets.list, **selector)
    return {**_trim_query(query), "datasets": _project_datasets(response)}


# Query fields the investigation agent needs to re-run the query. `from`/
# `from_expr` are dropped — the resolved datasets replace them.
_QUERY_WHITELIST = ("select", "where", "groups")


def _trim_query(query: dict[str, Any]) -> dict[str, Any]:
    """Reduce a monitor query to the fields needed to re-run it."""
    return {key: query[key] for key in _QUERY_WHITELIST if key in query}


# --- related datasets ---------------------------------------------------------


async def derive_related_datasets(
    model: BaseChatModel, playbook: str | None, valid_datasets: list[str]
) -> list[str]:
    """Ask the model which datasets the playbook implies are relevant.

    The model is grounded with the valid dataset names so it picks real logs
    instead of hallucinating. With no playbook there is nothing to infer, so the
    model is not called.
    """
    if not playbook:
        return []
    prompt = [
        HumanMessage(
            f"Playbook:\n{playbook}\n\n"
            f"Valid dataset (log) names:\n{', '.join(valid_datasets)}\n\n"
            "List the datasets this playbook implies are relevant. Choose only "
            "from the valid names above; omit anything not in that list."
        )
    ]
    result = cast(
        DeriveDatasets,
        await model.with_structured_output(DeriveDatasets).ainvoke(prompt),
    )
    return result.datasets


class DeriveDatasets(BaseModel):
    """Structured-output schema: dataset names implied by the playbook."""

    datasets: list[str]


def list_log_datasets(client: BrontoClient) -> list[dict[str, Any]]:
    """Every log dataset, as {log_id, name, collection}."""
    datasets = _project_datasets(client.datasets.list())
    return [d for d in datasets if d["collection"] not in _NON_LOG_COLLECTIONS]


# Collections that hold metrics and traces rather than logs: a full listing
# returns them too, and they are not datasets a log search can use.
_NON_LOG_COLLECTIONS = frozenset({".metrics", ".traces", "metrics", "default_traces"})


def resolve_related_datasets(
    log_datasets: list[dict[str, Any]], names: list[str]
) -> list[dict[str, Any]]:
    """The log datasets the model named, across every collection.

    Names that match no dataset are dropped and logged.
    """
    wanted = set(names)
    related = [dataset for dataset in log_datasets if dataset["name"] in wanted]
    for name in sorted(wanted - {dataset["name"] for dataset in related}):
        logger.warning("Dropping unresolved related dataset %r", name)
    return related


def _project_datasets(response: DatasetsResponse) -> list[dict[str, Any]]:
    """Reduce a /datasets response to each dataset's id, name and collection."""
    return [project_dataset(d.model_dump()) for d in response.datasets or []]


# --- the window and the monitor's state ---------------------------------------


# The search window covers LOOKBACK_WINDOWS monitor windows before the trigger,
# and the monitor state timeseries splits it into TIMESERIES_SLICES buckets (one
# per window). It ends one minute past the monitored window's close
# (evaluation_timestamp is the window START, so the alert fired around
# evaluation_timestamp + window).
LOOKBACK_WINDOWS = 10
TIMESERIES_SLICES = 10
_TRIGGER_BUFFER_MS = 60_000


def search_window(evaluation_timestamp: int, window_ms: int) -> Window:
    """The absolute epoch-ms window the investigation searches over.

    ``evaluation_timestamp`` is the monitored window's START, so the window ends
    one minute past its close (evaluation_timestamp + window_ms) and spans
    LOOKBACK_WINDOWS monitor windows back from there.
    """
    end = int(evaluation_timestamp) + window_ms + _TRIGGER_BUFFER_MS
    return {"start_ms": end - LOOKBACK_WINDOWS * window_ms, "end_ms": end}


async def monitor_state_timeseries(
    client: BrontoClient, query: dict[str, Any], window: Window
) -> list[dict[str, Any]]:
    """Run a resolved monitor query as a timeseries over the search window.

    Scoped to the query's datasets, so it shows the monitor's own signal leading
    up to the trigger. Returns the compact per-series state timeseries.
    """
    request = SearchRequest(
        select=query["select"],
        where=query.get("where"),
        groups=query.get("groups") or [],
        from_=[dataset["log_id"] for dataset in query["datasets"]],
        from_ts=window["start_ms"],
        to_ts=window["end_ms"],
        num_of_slices=TIMESERIES_SLICES,
    )
    response = await asyncio.to_thread(client.search.run, request)
    return adapt_series(response.model_dump(by_alias=True, exclude_none=True))


def adapt_series(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt a /search aggregate into compact per-series records."""
    groups: list[dict[str, Any]] = response.get("groups_series") or []
    totals: dict[str, Any] = response.get("totals") or {}
    if not groups and totals.get("timeseries"):
        groups = [{"name": None, "timeseries": totals["timeseries"]}]
    series: list[dict[str, Any]] = []
    for group in groups:
        buckets: list[dict[str, Any]] = group.get("timeseries") or []
        series.append(
            {
                "group": group.get("name") or None,
                "counts": [bucket["count"] for bucket in buckets],
                "values": [bucket["value"] for bucket in buckets],
            }
        )
    return series


# --- the alert context --------------------------------------------------------


def assemble_alert_context(
    *,
    monitor: dict[str, Any],
    triggered_event: dict[str, Any],
    related_datasets: list[dict[str, Any]],
    playbook: str | None,
    monitor_state_timeseries: list[dict[str, Any]],
    window: Window,
) -> AlertContext:
    """Assemble the investigation context handed to the shared phases.

    ``window`` and ``playbook`` are promoted to the top level because they are
    the parts other phases actually consume — the window in code, the playbook
    in the report prompt. Everything else goes under ``detail``, which no code
    reads: it is JSON-dumped into the model's prompt.
    """
    return {
        "window": window,
        "playbook": playbook,
        "detail": {
            "monitor": _monitor_with_compare_to_ms(monitor),
            "triggered_event": _prune_triggered_event(triggered_event),
            "related_datasets": related_datasets,
            "monitor_state_timeseries": monitor_state_timeseries,
        },
    }


def _monitor_with_compare_to_ms(monitor: dict[str, Any]) -> dict[str, Any]:
    """Enrich the monitor with compare_to_ms beside compare_to, when set."""
    compare_to = monitor.get("compare_to")
    if not compare_to:
        return monitor
    return {**monitor, "compare_to_ms": parse_window(compare_to)}


# Triggered-event fields dropped from the bundle: redundant with the monitor
# object (which already carries name, type, window, and queries), plus
# triggered_at (the fire time is only for the identification confirm prompt).
# alert_id is kept for debugging. group_key* prefixes are per-group breakdowns.
_TRIGGERED_EVENT_DROP = frozenset(
    {
        "monitor_id",
        "monitor_name",
        "monitor_type",
        "monitor_pattern",
        "window",
        "org_id",
        "query",
        "triggered_at",
    }
)
_TRIGGERED_EVENT_DROP_PREFIXES = ("group_key",)


def _prune_triggered_event(event: dict[str, Any]) -> dict[str, Any]:
    """Carry the triggered event whole, minus fields redundant with the monitor."""
    return {
        key: value
        for key, value in event.items()
        if key not in _TRIGGERED_EVENT_DROP
        and not key.startswith(_TRIGGERED_EVENT_DROP_PREFIXES)
    }
