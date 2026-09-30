"""Fakes for the Slack human, so none of its tests need a workspace."""

import pytest


class FakeSlackApi:
    """Stands in for chat.postMessage / chat.update. Imports no Slack SDK."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.updates: list[dict] = []

    async def post(
        self, *, blocks: list[dict], text: str, thread_ts: str | None = None
    ) -> str:
        self.posts.append({"thread_ts": thread_ts, "blocks": blocks, "text": text})
        return f"1700000000.{len(self.posts):06d}"

    async def update(self, *, ts: str, blocks: list[dict], text: str) -> None:
        self.updates.append({"ts": ts, "blocks": blocks, "text": text})


@pytest.fixture
def slack_api() -> FakeSlackApi:
    return FakeSlackApi()
