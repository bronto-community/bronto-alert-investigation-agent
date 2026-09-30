"""One investigation, from its starting state to its report.

The interrupt loop is the same whatever surface is answering.
Nothing in this module prints or reads stdin.

``run_to_completion`` is named for the assumption it makes: one process owns a
run from start to finish, and a paused run is an awaited coroutine in that
process's memory.

A host that cannot keep a process alive across a pause would need this split
into start and resume halves over a durable checkpointer instead.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from opentelemetry import trace

from .human import Event, Human, Question, question_of, to_decision
from .state import InvestigationState

logger = logging.getLogger(__name__)


DESCRIBE_REQUEST = Question(
    action="describe_request",
    prompt="Describe what you'd like investigated:",
    free_text="respond",
)


def run_config(thread_id: str) -> RunnableConfig:
    """The invoke config for one run. ``thread_id`` scopes its checkpoints."""
    return {"configurable": {"thread_id": thread_id}}


async def run_to_completion(
    app: CompiledStateGraph[InvestigationState],
    initial_state: InvestigationState,
    human: Human,
    *,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Invoke the graph, answer its questions, emit its report, return its state.

    The whole run sits under one root span, so the graph runs either side of
    every interrupt share a single trace.
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("investigation"):
        result = await app.ainvoke(initial_state, config)

        while "__interrupt__" in result:
            decisions = await _answer_all(result["__interrupt__"][0].value, human)
            result = await app.ainvoke(Command(resume={"decisions": decisions}), config)

        if report := result.get("report"):
            await human.tell(Event(type="report", text=report))

    logger.info("investigation finished")
    return result


async def _answer_all(payload: dict[str, Any], human: Human) -> list[dict[str, Any]]:
    """One decision per pending action request, in order."""
    decisions: list[dict[str, Any]] = []
    for request in payload["action_requests"]:
        question = question_of(request)
        answer = await human.ask(question)
        decisions.append(to_decision(question, answer))
    return decisions


__all__ = ["DESCRIBE_REQUEST", "run_config", "run_to_completion"]
