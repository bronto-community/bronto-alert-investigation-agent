"""The dependency rules for slack sdk extra"""

import importlib
import sys

import pytest

PURE = ("blocks", "thread", "runs")
BLOCKED = ("slack_sdk", "slack_bolt")


class _Blocked:
    """A meta_path finder that refuses the Slack SDK to anything under test."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is not reachable from the pure Slack modules")
        return None


@pytest.mark.parametrize("module", PURE)
def test_the_pure_slack_modules_cannot_reach_the_slack_sdk(module, monkeypatch):
    package = "investigator.entrypoints.slack"
    full = f"{package}.{module}"
    for name in list(sys.modules):
        root = name.split(".")[0]
        if name == full or name == package or root in BLOCKED:
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocked(), *sys.meta_path])

    importlib.import_module(full)  # must not raise


def test_the_core_package_imports_without_the_slack_sdk(monkeypatch):
    """`uv sync` without the slack extra must still give a working agent."""
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED or name.startswith("investigator"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Blocked(), *sys.meta_path])

    importlib.import_module("investigator.graph")
    importlib.import_module("investigator.run")
