from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import asyncio
from datetime import datetime
from typing import List, Optional

from llama_index.core.llms import LLM
from llama_index.core.memory import Memory, InsertMethod
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from asdrp.agent.base import AgentReply, BaseAgent


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


class SimpleAgent(BaseAgent):
    """
    A minimal agent with short-term working memory.

    This is intentionally small: a practical baseline for memory experiments.
    """

    def __init__(
        self,
        llm: LLM = OpenAI(model="gpt-4.1-mini"),
        memory: Optional[Memory] = None,
        tools: Optional[List[FunctionTool]] = None,
    ):
        if memory is None:
            memory = Memory.from_defaults(
                session_id="simple_agent",
                token_limit=50,
                chat_history_token_ratio=0.7,
                token_flush_size=10,
                insert_method=InsertMethod.SYSTEM,
            )
        if tools is None:
            tools = [get_current_time, get_current_weather]
        super().__init__(llm=llm, memory=memory, tools=tools)

    async def achat(self, user_msg: str) -> AgentReply:
        return await self._run(user_msg=user_msg)


async def process(user_input: str) -> AgentReply:
    """Small CLI helper for manual testing."""
    agent = SimpleAgent()
    return await agent.achat(user_input)


if __name__ == "__main__":
    user_input = input("Enter your input: ")
    while user_input.strip() != "":
        reply = asyncio.run(process(user_input))
        print(f"Response: {reply.response_str}")
        user_input = input("Enter your input: ")

    print("Thank you for chatting with me!")
