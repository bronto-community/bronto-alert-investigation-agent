# Contributing

Thanks for your interest. Before you write any code, please read the first
section: this project takes fewer kinds of contribution than most.

## What this project accepts

This repository is mainly a **teaching resource**. It shows how to build an
agent with LangGraph on top of Bronto, and it is kept small on purpose so a
first-time reader can follow the whole thing. It also works as a real tool, but
the feature set is **frozen**. Every addition is one more concept for the next
reader to learn, and that cost usually outweighs what the feature adds.

**Welcome:**

- bug fixes and correctness fixes
- changes that make the code easier to read, such as clearer names, simpler
  control flow, or fewer concepts for the same behaviour
- documentation fixes and improvements
- tests that pin down behaviour that isn't covered yet
- keeping dependencies working as they release new versions

**Not accepted, however useful:**

- new features, providers, surfaces, flags or configuration options
- extra abstraction layers or generalisation "for later"
- anything that makes the code harder to follow in exchange for capability

**Want more than it does? Fork it, or copy the parts you need.** That is the
intended way to use this project. The pieces are built to be lifted out on
their own. The phase modules under `src/investigator/phases/` each work
without the others.

For anything bigger than a typo, please open an issue first, so we can agree it
fits before you spend time on it.

## 5-minute path

```sh
git clone https://github.com/bronto-community/bronto-alert-investigation-agent
cd bronto-alert-investigation-agent
uv sync --all-extras   # .venv with the project, both extras, and dev tools
just prepare           # format, lint, typecheck, tests
```

You don't need network access, a Bronto account or model credentials to run
the suite. Every model call and every Bronto call is mocked.

## Development environment

- **[`uv`](https://docs.astral.sh/uv/)** manages the virtual environment and
  dependencies. Commit `uv.lock` whenever dependencies change.
- **[`just`](https://just.systems/)** runs every project command through
  `uv run`. Run `just --list` to see all the recipes.
- The minimum supported Python is **3.11**. `ruff` and `pyright` both target
  3.11, so syntax that needs a newer Python can't slip in.

| Task | Command |
| --- | --- |
| Everything, before every commit | `just prepare` |
| Format | `just format` / `just format-check` |
| Lint (ruff) | `just lint` |
| Typecheck (pyright strict on `src/`) | `just typecheck` |
| Tests, with coverage | `just test` / `just test-one <name>` |
| Shellcheck `scripts/` | `just lint-shell` |
| Build sdist + wheel | `just build` |
| Verify the live ruleset matches the recorded checks | `just check-ruleset` |
| Regenerate `graph.png` | `just graph` |

## Tests

- Prefer test-driven development.
- Use `pytest` and `pytest-mock`. Async tests need no marker, because
  `asyncio_mode = "auto"`.
- Test tools, graph routing and node deltas without an LLM.
  `tests/conftest.py` provides `ScriptedChatModel`, a real `BaseChatModel` that
  replays scripted responses.
- The suite never touches the network. Mock Bronto and model calls.
- Warnings are errors, and coverage has a floor. Both are set in
  `pyproject.toml`.

## Code style

The linters enforce the mechanical rules. These are the ones they can't
enforce:

- **The reader wins.** When a feature and a first-time reader's understanding
  compete, prefer the version with fewer concepts, even if the richer one is
  more capable.
- **Keep data and behaviour apart.** Put data in `TypedDict`, `dataclass` or
  Pydantic models, and behaviour in plain functions. Avoid base classes and
  inheritance hierarchies.
- **Nodes return deltas.** A LangGraph node takes the state and returns a dict
  of changes. It never mutates the state it was given.
- **Not every phase is an agent.** Context generation is deterministic on
  purpose. Only reach for a model call where judgement is needed.
- **Findings stay grounded.** Findings must point back to retrieved logs,
  metrics or playbook steps, never to speculation.
- **Verify a rename at both ends.** Renaming a keyword argument, a dict key or a
  LangGraph state channel breaks a contract that the type checker can't see.
  Grep the other end, and add a test that would have caught it.

## Pull requests

Keep each PR to one change, with a description that says why. Run
`just prepare` before you push.

## What the repository enforces

- `main` takes changes only through a pull request. It needs linear history,
  one approval, all review threads resolved, and every `ci.yml` job green.
  Force-pushes and deletion are blocked, and nobody can bypass these rules.
- Actions must be pinned to a commit SHA. GitHub rejects a tag reference
  outright, and `zizmor` enforces the same rule in CI.
- Security vulnerabilities are reported privately. See [SECURITY.md](SECURITY.md).

### Required checks

The required check names are mirrored in `.github/required-status-checks.txt`,
and two gates keep that mirror honest. `tests/test_required_checks.py` fails
when the list stops matching the names `ci.yml` generates. `just check-ruleset`,
which `ruleset-drift.yml` runs weekly, fails when the list stops matching the
live ruleset.

**A pull request that renames or removes a required check blocks itself.** The
old check stops reporting on that pull request, and GitHub shows it as pending
rather than failed. With no bypass actors, nobody can merge it, admins included.
Adding the new name to the ruleset first doesn't help either: a required check
that no job produces yet blocks every open pull request. The order that never
blocks anything is to remove first:

1. Remove the outgoing check from the `protect-main` ruleset
   (Settings → Rules, or `gh api`).
2. Merge the pull request that renames the job or changes the matrix. It carries
   the updated `.github/required-status-checks.txt`.
3. Add the new check to the ruleset.

## Reporting security issues

Please don't open a public issue. See [SECURITY.md](SECURITY.md).
