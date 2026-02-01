#############################################################################
# base.py
#
# base class for agent replies and agent scaffolding
#
# @author Theodore Mui
# @email  theodoremui@gmail.com
# Fri Jul 04 11:30:53 PDT 2025
#############################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from llama_index.core.agent.workflow import FunctionAgent, AgentOutput
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool


@dataclass
class AgentReply:
    """Normalized agent reply returned by all agents in this package."""

    response_str: str


class BaseAgent(ABC):
    """
    Minimal base class for agents in this package.

    Design goals:
    - Keep the interface tiny: only `achat()` is required.
    - Normalize replies for downstream callers.
    - Centralize error handling so child agents stay focused.
    """

    def __init__(
        self,
        llm: LLM,
        memory: Optional[Memory] = None,
        tools: Optional[List[FunctionTool]] = None,
    ):
        self.llm = llm
        self.memory = memory
        self.tools = tools or []
        self.agent = self._create_agent(memory=self.memory, tools=self.tools)

    @abstractmethod
    async def achat(self, user_msg: str) -> AgentReply:
        """Async chat interface. Child classes define their prompting logic."""

    def _create_agent(
        self, memory: Optional[Memory], tools: List[FunctionTool]
    ) -> FunctionAgent:
        """Create the underlying LlamaIndex FunctionAgent."""
        return FunctionAgent(llm=self.llm, memory=memory, tools=tools)

    async def _run(
        self, user_msg: str, memory: Optional[Memory] = None
    ) -> AgentReply:
        """
        Execute the underlying agent and normalize the response.

        This is the shared happy-path; subclasses can preprocess `user_msg`
        and then call this method.
        """
        try:
            response = await self.agent.run(
                user_msg=user_msg, memory=memory or self.memory
            )
            return self._to_agent_reply(response)
        except Exception as exc:
            print(f"Error in {self.__class__.__name__}: {exc}")
            return AgentReply(
                response_str=(
                    "I'm sorry, I'm having trouble processing your request. "
                    "Please try again."
                )
            )

    @staticmethod
    def _to_agent_reply(response: object) -> AgentReply:
        """Normalize LlamaIndex outputs into a stable `AgentReply`."""
        if isinstance(response, AgentOutput):
            return AgentReply(response_str=response.response.content)
        if isinstance(response, ChatMessage):
            return AgentReply(response_str=response.content)
        return AgentReply(response_str=str(response))
