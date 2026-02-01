#############################################################################
# mem0_agent.py
#
# agent with Mem0 memory (via LlamaIndex Mem0Memory wrapper)
#
# Usage:
#   Run demo (clean output): ./run_mem0_demo.sh
#   Run demo (direct):       uv run python asdrp/agent/mem0_agent.py
#   In code:                 from asdrp.agent.mem0_agent import Mem0Agent
#
# @author Theodore Mui
# @email  theodoremui@gmail.com
# Sat Jan 31 12:00:00 PDT 2026
#############################################################################

from __future__ import annotations

# Suppress ALL dependency deprecation warnings at import time
import warnings
import os

# Also use warnings module filters as backup
warnings.filterwarnings("ignore")  # Catch-all
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from typing import Any, Dict, Iterable, List, Optional

from llama_index.core.llms import LLM
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.memory.mem0 import Mem0Memory

from asdrp.agent.base import BaseAgent, AgentReply


class Mem0Agent(BaseAgent):
    """
    An agent wired to Mem0 via the LlamaIndex Mem0Memory wrapper.

    This class aims to be a compact, extensible example:
    - Uses Mem0Memory as the memory backend (platform or OSS).
    - Exposes simple convenience methods for search/add.
    - Stays compatible with the BaseAgent interface.

    **Current Limitations**:
    The LlamaIndex Mem0Memory integration may fail during chat operations
    with "filters required" errors due to upstream integration issues.
    Direct memory operations (search_memories, add_memories) work correctly.

    **Usage**: See `notebooks/Agentic Memory.ipynb` for complete working examples
    with proper error handling and workarounds.
    """

    def __init__(
        self,
        llm: LLM = OpenAI(model="gpt-4.1-mini"),
        memory: Optional[Memory] = None,
        tools: Optional[List[FunctionTool]] = None,
        *,
        context: Optional[Dict[str, str]] = None,
        mem0_config: Optional[Dict[str, Any]] = None,
        mem0_api_key: Optional[str] = None,
        search_msg_limit: int = 5,
        use_platform: bool = True,
    ):
        self.context = context or {"user_id": "default_user"}
        self.mem0_config = mem0_config
        self.mem0_api_key = mem0_api_key
        self.search_msg_limit = search_msg_limit
        self.use_platform = use_platform

        if memory is None:
            memory = self._create_mem0_memory()

        # Store a direct handle to the memory backend for convenience methods.
        self.mem0_memory = memory

        super().__init__(llm=llm, memory=memory, tools=tools)

    async def achat(self, user_msg: str) -> AgentReply:
        return await self._run(user_msg=user_msg)

    def _create_mem0_memory(self) -> Memory:
        """
        Create a Mem0Memory instance using either platform or OSS config.

        Platform mode uses Mem0 API keys, while OSS mode uses a config dict.
        """

        if self.use_platform:
            kwargs = {"context": self.context, "search_msg_limit": self.search_msg_limit}
            if self.mem0_api_key:
                kwargs["api_key"] = self.mem0_api_key
            return Mem0Memory.from_client(**kwargs)

        if not self.mem0_config:
            raise ValueError("mem0_config is required when use_platform=False.")

        return Mem0Memory.from_config(
            context=self.context,
            config=self.mem0_config,
            search_msg_limit=self.search_msg_limit,
        )

    def search_memories(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Search for relevant memories stored in Mem0.

        This method proxies to the Mem0Memory `search` API when available.
        """
        if not hasattr(self.mem0_memory, "search"):
            raise AttributeError("Mem0Memory backend does not expose `search`.")
        if user_id is None:
            user_id = self.context.get("user_id")
        if not filters:
            filters = {"user_id": user_id}
        return self.mem0_memory.search(
            query=query, user_id=user_id, limit=limit, filters=filters, **kwargs
        )

    def add_memories(
        self,
        messages: Iterable[Dict[str, str]],
        *,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Add messages to Mem0 directly (outside of LlamaIndex chat flow).
        """
        if not hasattr(self.mem0_memory, "add"):
            raise AttributeError("Mem0Memory backend does not expose `add`.")
        if user_id is None:
            user_id = self.context.get("user_id")
        return self.mem0_memory.add(messages=messages, user_id=user_id, **kwargs)

#----------------------------
# Example usage
#----------------------------

async def main():
    """
    Demonstrates Mem0Agent usage with direct memory operations.
    
    Note: Chat operations (achat) may fail due to LlamaIndex Mem0Memory
    integration issues. This example focuses on the working features:
    direct memory add/search operations.
    """
    print("=" * 60)
    print("Mem0Agent Example: Direct Memory Operations")
    print("=" * 60)
    
    # Initialize agent with context
    agent = Mem0Agent(context={"user_id": "alice", "agent_id": "demo_agent"})
    print(f"\n✓ Created Mem0Agent with context: {agent.context}")
    
    # Example 1: Add memories directly
    print("\n--- Example 1: Adding Memories ---")
    agent.add_memories(
        messages=[
            {"role": "user", "content": "My name is Alice and I live in Seattle."},
            {"role": "assistant", "content": "Nice to meet you, Alice from Seattle!"},
            {"role": "user", "content": "I prefer decaf coffee and email updates."},
            {"role": "assistant", "content": "Got it, I'll remember your preferences."},
        ],
        user_id="alice"
    )
    print("✓ Added 4 messages to memory")
    
    # Example 2: Search memories with filters
    print("\n--- Example 2: Searching Memories ---")
    results = agent.search_memories(
        "coffee preferences",
        filters={"user_id": "alice"},
        limit=3
    )
    print(f"Search results for 'coffee preferences':")
    if results and "results" in results:
        for i, mem in enumerate(results["results"], 1):
            print(f"  {i}. {mem.get('memory', mem)}")
    else:
        print(f"  {results}")
    
    # Example 3: Search for location info
    print("\n--- Example 3: Location Search ---")
    results = agent.search_memories(
        "where does the user live",
        filters={"user_id": "alice"},
        limit=2
    )
    print(f"Search results for 'location':")
    if results and "results" in results:
        for i, mem in enumerate(results["results"], 1):
            print(f"  {i}. {mem.get('memory', mem)}")
    else:
        print(f"  {results}")
    
    # Example 4: Demonstrate chat (with error handling)
    print("\n--- Example 4: Chat (may have limitations) ---")
    try:
        reply = await agent.achat("What are my coffee preferences?")
        print(f"Agent: {reply.response_str}")
    except Exception as e:
        print(f"⚠ Chat failed (known limitation): {str(e)[:100]}...")
        print("  → Use direct memory operations instead (Examples 1-3)")
    
    print("\n" + "=" * 60)
    print("For more examples, see: notebooks/Agentic Memory.ipynb")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
