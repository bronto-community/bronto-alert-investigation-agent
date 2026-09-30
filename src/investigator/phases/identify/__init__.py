"""Phase 1: identify the alert which matches user's natural language description.

Four files for one phase, because identifying something with a human in the
loop has four distinct parts:

    identify.py   the Reason/Act agent, and the bound that stops it searching
                  forever
    search.py     what it searches — the audit trail of triggered monitors
    questions.py  what the user is shown when it proposes a match
    confirm.py    the tools those questions interrupt on, and what runs once
                  someone has decided
"""
