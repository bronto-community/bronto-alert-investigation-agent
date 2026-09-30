"""The four phases of an investigation, one module each, in the order they run.

    identify/       which triggered monitor the user means — a Reason/Act agent
                    with a human in the loop
    context.py      what is already known about it — deterministic lookups
    investigate.py  why it fired — a Reason/Act agent over real Bronto tools
    report.py       the write-up — one model call

`graph.py` wires them together and is the place to start reading.
"""
