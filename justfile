set quiet

# The recipes below all run through `uv run`, which warns when an unrelated
# VIRTUAL_ENV is active (a global default venv on the developer's machine) and
# then ignores it. Clearing it for the duration of a recipe keeps the output
# readable and makes the project's own .venv unambiguously the one in use.
export VIRTUAL_ENV := ""

_default:
    just --list --unsorted

# ⭐ run format, lint, typecheck, the shell gates and tests — green before every commit
prepare: format lint lint-shell typecheck test

# ⭐ run all unit tests
[positional-arguments]
test *args:
    # configured in pyproject.toml
    uv run pytest "$@"

# run a single test by name (coverage off — the report is noise for one test)
test-one test_name:
    uv run pytest -k "{{ test_name }}" --no-cov

# ⭐ check for potential mistakes
lint:
    uv run ruff check .

# verify types; src/ is held to strict (see pyproject.toml)
typecheck:
    # suppress pyright's version-update chatter
    PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright

# ⭐ format all code
format:
    uv run ruff format .

# verify formatting, but don't modify files
format-check:
    uv run ruff format . --check

# check the shell scripts in scripts/ — they are untyped and untested otherwise
lint-shell:
    #!/usr/bin/env bash
    set -euo pipefail
    # Preinstalled on GitHub's ubuntu runners; shellcheck-py ships the same
    # binary for everyone else, so this needs no brew install to run locally.
    if command -v shellcheck >/dev/null 2>&1; then
      shellcheck scripts/*.sh
    else
      uvx --from shellcheck-py shellcheck scripts/*.sh
    fi
    echo "shellcheck: ok ($(ls scripts/*.sh | wc -l | tr -d ' ') scripts)"

# verify the live ruleset still requires exactly the recorded checks — needs network
check-ruleset:
    scripts/ruleset-drift.sh

# build the sdist and wheel, and list what went into the sdist
build:
    rm -rf dist
    uv build
    tar tzf dist/*.tar.gz

# regenerate graph.png from the compiled graph (needs graphviz for png)
graph:
    uv run python visualize_graph.py

# remove build artifacts and tool caches
clean:
    rm -rf dist build .pytest_cache .ruff_cache .coverage htmlcov
    find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
