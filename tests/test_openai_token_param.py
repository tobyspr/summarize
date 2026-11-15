"""OpenAI GPT-5+ models need max_completion_tokens; other providers keep max_tokens."""

import asyncio
from unittest.mock import patch

import pytest

from summarizer import api


class _FakeResponse:
    status = 200

    async def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, sent):
        self.sent = sent

    def post(self, url, **kwargs):
        self.sent.append(kwargs["json"])
        return _FakeResponse()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "base_url, expected, absent",
    [
        ("https://api.openai.com/v1", "max_completion_tokens", "max_tokens"),
        ("https://api.groq.com/openai/v1", "max_tokens", "max_completion_tokens"),
    ],
)
def test_token_param_matches_provider(base_url, expected, absent):
    sent = []
    config = {"api_key": "k", "model": "m", "base_url": base_url, "max_output_tokens": 123}
    with patch.object(api.aiohttp, "ClientSession", lambda: _FakeSession(sent)):
        asyncio.run(api.process_chunk("some text", "Summarize: {text}", config))
    assert sent[0][expected] == 123
    assert absent not in sent[0]
