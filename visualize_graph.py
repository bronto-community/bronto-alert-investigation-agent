#!/usr/bin/env python
"""Render a Graphviz visualisation of the agent's LangGraph.

The graph is a structural artifact: ``build_app`` only wires nodes together at
construction (no network, no model calls), so we build it with placeholder
dependencies purely to read its shape via ``app.get_graph()``.

Both agentic phases are prebuilt agents behind wrapper nodes. They are drawn as
single nodes by default, which is the investigation state view worth looking
at: expanding them (``--xray 1``) inlines each agent's own ``model``/``tools``
nodes without their namespace, so the two phases become indistinguishable.

Usage:
    uv run python visualize_graph.py                 # -> graph.png (or graph.dot)
    uv run python visualize_graph.py -o agent.svg    # pick format by extension
    uv run python visualize_graph.py --format dot     # DOT source only

Rendering to png/svg needs the Graphviz ``dot`` binary (``brew install graphviz``);
without it the script writes the ``.dot`` source you can render elsewhere.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.runnables.graph import Graph

from investigator.graph import build_app
from investigator.settings import Deps

# START/END get a distinct look; everything else is a plain rounded box.
_TERMINALS = {"__start__", "__end__"}


def build_graph(xray: int = 0) -> Graph:
    """Build the orchestrator with placeholder deps and return its drawable graph.

    ``create_agent`` introspects the model it is given, so this needs a real
    BaseChatModel rather than a bare mock — it is never called. The premise is
    that ``build_app`` does no network or model work at construction, so the
    shape it draws is a purely structural artifact.
    """
    app = build_app(
        Deps(
            client=MagicMock(),
            model=FakeMessagesListChatModel(responses=[]),
            investigation_tools=[],
        ),
        max_model_calls=5,
    )
    return app.get_graph(xray=xray)


def to_dot(graph: Graph) -> str:
    """Convert a LangGraph drawable graph into Graphviz DOT source.

    Conditional edges (branches) are drawn dashed and labelled with the routing
    key; unconditional edges are solid.
    """
    lines = [
        "digraph agent {",
        "  rankdir=TB;",
        '  node [shape=box, style=rounded, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]
    for node_id, node in graph.nodes.items():
        label = getattr(node, "name", None) or node_id
        if node_id in _TERMINALS:
            lines.append(
                f'  "{node_id}" [label="{label}", shape=ellipse,'
                ' style=filled, fillcolor="#d9d9d9"];'
            )
        else:
            lines.append(f'  "{node_id}" [label="{label}"];')
    for edge in graph.edges:
        attrs: list[str] = []
        if getattr(edge, "conditional", False):
            attrs.append("style=dashed")
        if getattr(edge, "data", None):
            attrs.append(f'label="{edge.data}"')
        suffix = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f'  "{edge.source}" -> "{edge.target}"{suffix};')
    lines.append("}")
    return "\n".join(lines) + "\n"


def render(dot: str, output: Path) -> None:
    """Write the graph to ``output``; render via ``dot`` unless the format is dot."""
    fmt = output.suffix.lstrip(".").lower() or "dot"
    if fmt == "dot":
        output.write_text(dot)
        print(f"wrote {output}")
        return
    if shutil.which("dot") is None:
        fallback = output.with_suffix(".dot")
        fallback.write_text(dot)
        print(
            f"Graphviz 'dot' not found; wrote {fallback} instead.\n"
            "Install Graphviz (e.g. 'brew install graphviz') to render images.",
            file=sys.stderr,
        )
        return
    subprocess.run(
        ["dot", f"-T{fmt}", "-o", str(output)], input=dot, text=True, check=True
    )
    print(f"wrote {output}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the agent's LangGraph with Graphviz."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="graph.png",
        help="output path; the extension picks the format (default: graph.png)",
    )
    parser.add_argument(
        "--format",
        dest="format",
        help="override the output format (dot, png, svg, ...) regardless of extension",
    )
    parser.add_argument(
        "--xray",
        type=int,
        default=0,
        help="depth to expand nested agent subgraphs (default 0: one node each)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Render the compiled graph to the format and path given on the command line."""
    args = _parse_args(argv)
    output = Path(args.output)
    if args.format:
        output = output.with_suffix(f".{args.format}")
    render(to_dot(build_graph(args.xray)), output)


if __name__ == "__main__":
    main()
