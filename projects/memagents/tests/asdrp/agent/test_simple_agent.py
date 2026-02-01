"""
Pytest module for asdrp.agent.simple_agent.SimpleAgent

These tests verify default memory/tools and basic chat behavior.
"""
import pytest

from asdrp.agent.base import BaseAgent, AgentReply
from asdrp.agent.simple_agent import SimpleAgent, get_current_time, get_current_weather


class _FakeRunner:
    def __init__(self, return_value=None):
        self._return_value = return_value

    async def run(self, *args, **kwargs):
        return self._return_value


class _DummyLLM:
    """Minimal LLM stub for SimpleAgent construction."""


@pytest.mark.asyncio
async def test_simple_agent_defaults(monkeypatch):
    """
    Ensure SimpleAgent builds default memory and tools when none provided.
    """
    runner = _FakeRunner(return_value="ok")

    def _fake_create_agent(self, memory=None, tools=None):
        return runner

    monkeypatch.setattr(BaseAgent, "_create_agent", _fake_create_agent, raising=True)

    agent = SimpleAgent(llm=_DummyLLM())
    assert agent.memory is not None
    assert agent.tools is not None
    assert len(agent.tools) == 2
    assert get_current_time in agent.tools
    assert get_current_weather in agent.tools

    reply = await agent.achat("hello")
    assert isinstance(reply, AgentReply)
    assert reply.response_str == "ok"


@pytest.mark.asyncio
async def test_simple_agent_custom_memory_and_tools(monkeypatch):
    """
    Ensure custom memory/tools are preserved when provided.
    """
    runner = _FakeRunner(return_value="custom")

    def _fake_create_agent(self, memory=None, tools=None):
        return runner

    monkeypatch.setattr(BaseAgent, "_create_agent", _fake_create_agent, raising=True)

    custom_tools = [lambda: "tool"]
    agent = SimpleAgent(llm=_DummyLLM(), memory=object(), tools=custom_tools)
    assert agent.memory is not None
    assert agent.tools is custom_tools

    reply = await agent.achat("hello")
    assert reply.response_str == "custom"
