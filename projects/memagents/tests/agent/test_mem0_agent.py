"""
Pytest module for asdrp.agent.mem0_agent.Mem0Agent

These tests validate Mem0 memory initialization and proxy methods without
requiring a live Mem0 backend.
"""
import sys
import types

import pytest

from asdrp.agent.base import BaseAgent
from asdrp.agent.mem0_agent import Mem0Agent


class _FakeRunner:
    def __init__(self, return_value=None):
        self._return_value = return_value

    async def run(self, *args, **kwargs):
        return self._return_value


class _DummyLLM:
    """Minimal LLM stub for Mem0Agent construction."""


class _FakeMem0Memory:
    def __init__(self, context=None, config=None, search_msg_limit=5, api_key=None):
        self.context = context or {}
        self.config = config
        self.search_msg_limit = search_msg_limit
        self.api_key = api_key

    @classmethod
    def from_client(cls, context, search_msg_limit=5, api_key=None):
        return cls(context=context, search_msg_limit=search_msg_limit, api_key=api_key)

    @classmethod
    def from_config(cls, context, config, search_msg_limit=5):
        return cls(context=context, config=config, search_msg_limit=search_msg_limit)

    def search(self, **kwargs):
        return {"search": kwargs}

    def add(self, **kwargs):
        return {"add": kwargs}


@pytest.fixture(autouse=True)
def _patch_mem0_module(monkeypatch):
    module = types.ModuleType("llama_index.memory.mem0")
    module.Mem0Memory = _FakeMem0Memory
    sys.modules["llama_index.memory.mem0"] = module
    yield
    sys.modules.pop("llama_index.memory.mem0", None)


@pytest.fixture
def _patch_base_agent(monkeypatch):
    def _fake_create_agent(self, memory=None, tools=None):
        return _FakeRunner(return_value="ok")

    monkeypatch.setattr(BaseAgent, "_create_agent", _fake_create_agent, raising=True)


def test_platform_mem0_initialization(_patch_base_agent):
    agent = Mem0Agent(
        llm=_DummyLLM(),
        context={"user_id": "alice"},
        mem0_api_key="test-key",
    )
    assert isinstance(agent.mem0_memory, _FakeMem0Memory)
    assert agent.mem0_memory.context["user_id"] == "alice"
    assert agent.mem0_memory.api_key == "test-key"


def test_oss_mem0_requires_config(_patch_base_agent):
    with pytest.raises(ValueError):
        Mem0Agent(llm=_DummyLLM(), use_platform=False)


def test_oss_mem0_initialization(_patch_base_agent):
    agent = Mem0Agent(
        llm=_DummyLLM(),
        use_platform=False,
        mem0_config={"vector_store": {"provider": "qdrant", "config": {}}},
    )
    assert isinstance(agent.mem0_memory, _FakeMem0Memory)
    assert agent.mem0_memory.config is not None


def test_search_and_add_proxy(_patch_base_agent):
    agent = Mem0Agent(llm=_DummyLLM(), context={"user_id": "bob"})
    search = agent.search_memories("test query", limit=2)
    assert search["search"]["query"] == "test query"
    assert search["search"]["limit"] == 2
    assert search["search"]["user_id"] == "bob"

    added = agent.add_memories(messages=[{"role": "user", "content": "hi"}])
    assert added["add"]["user_id"] == "bob"
