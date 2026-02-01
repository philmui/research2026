"""
Pytest module for asdrp.agent.base.BaseAgent

These tests focus on the base agent behavior: reply normalization and
graceful error handling.
"""
import pytest
from llama_index.core.llms import ChatMessage, TextBlock

from asdrp.agent.base import BaseAgent, AgentReply


class _FakeRunner:
    def __init__(self, return_value=None, raise_exc: Exception | None = None):
        self._return_value = return_value
        self._raise_exc = raise_exc

    async def run(self, *args, **kwargs):
        if self._raise_exc:
            raise self._raise_exc
        return self._return_value


class _DummyLLM:
    """Minimal LLM stub for BaseAgent construction."""


class _TestAgent(BaseAgent):
    def __init__(self, runner: _FakeRunner):
        self._runner = runner
        super().__init__(llm=_DummyLLM(), memory=None, tools=None)

    async def achat(self, user_msg: str) -> AgentReply:
        return await self._run(user_msg=user_msg)

    def _create_agent(self, memory=None, tools=None):
        return self._runner


@pytest.mark.asyncio
async def test_to_agent_reply_from_chat_message():
    msg = ChatMessage(blocks=[TextBlock(text="Hello!")], additional_kwargs={})
    agent = _TestAgent(_FakeRunner(return_value=msg))
    reply = await agent.achat("Hi")
    assert isinstance(reply, AgentReply)
    assert reply.response_str == "Hello!"


@pytest.mark.asyncio
async def test_to_agent_reply_from_string():
    agent = _TestAgent(_FakeRunner(return_value="plain response"))
    reply = await agent.achat("Hi")
    assert isinstance(reply, AgentReply)
    assert reply.response_str == "plain response"


@pytest.mark.asyncio
async def test_error_handling_returns_fallback():
    agent = _TestAgent(_FakeRunner(raise_exc=RuntimeError("boom")))
    reply = await agent.achat("Hi")
    assert isinstance(reply, AgentReply)
    assert "trouble processing" in reply.response_str.lower()
