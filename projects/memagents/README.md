# MemAgent Project

MemAgent explores agentic memory from the simplest working buffers to the newest research on self-organizing, reflective, and policy-driven memory systems. The project is a lab bench: a place to compare memory styles, measure their effects, and tell a clearer story about what an agent should remember and why.

If you are new, start with the notebooks and the example agents in `asdrp/agent`. They are intentionally minimal, each one spotlighting a different memory style so you can feel the tradeoffs in your hands.

## Table of Contents

- [Example Usage: Three Agents, Three Memory Styles](#example-usage-three-agents-three-memory-styles)
  - [1) The Simple Agent: Short-Term Working Memory](#1-the-simple-agent-short-term-working-memory)
  - [2) The Summary Agent: Condensed Memory for Continuity](#2-the-summary-agent-condensed-memory-for-continuity)
  - [3) The Reductive Agent: Proposition Memory for Precision](#3-the-reductive-agent-proposition-memory-for-precision)
- [How to Read Memory Like a Researcher](#how-to-read-memory-like-a-researcher)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [A Narrative Arc of Agentic Memory Research](#a-narrative-arc-of-agentic-memory-research)
- [Where We Are Going Next](#where-we-are-going-next)

## Example Usage: Three Agents, Three Memory Styles

Each agent below is a living example of a memory philosophy. Run these snippets as written, then alter the prompts to stress each memory type.

### 1) The Simple Agent: Short-Term Working Memory

Use this when the task is short, or when you want a "goldfish test" baseline. The memory is an in-context buffer: cheap, fast, and honest about forgetting.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory, InsertMethod
from llama_index.llms.openai import OpenAI

def get_current_time():
    return "2026-01-31 12:00:00"

def get_current_weather(city: str):
    return f"The weather in {city} is sunny."

tools = [get_current_time, get_current_weather]
llm = OpenAI(model="gpt-4.1-mini")

memory = Memory.from_defaults(
    session_id="simple_agent",
    token_limit=50,
    chat_history_token_ratio=0.7,
    token_flush_size=10,
    insert_method=InsertMethod.SYSTEM,
)

agent = FunctionAgent(llm=llm, tools=tools)
```

**Why this memory exists:** It reveals the core limit of LLMs. When the buffer fills, details vanish. This is the baseline you compare everything else against.

**Direct example (what the memory forgets):**

1. User: "My dog's name is Luma. I live in Seattle. I prefer decaf."
2. After enough extra turns to overflow the buffer:
3. User: "What coffee should I order?"

The simple memory might answer without recalling "decaf," because that detail fell out of the working buffer.

### 2) The Summary Agent: Condensed Memory for Continuity

This agent compresses the conversation into a smaller, persistent representation. It keeps a story thread even when the full transcript no longer fits.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory, InsertMethod
from llama_index.llms.openai import OpenAI

from asdrp.memory.condensed_memory import CondensedMemoryBlock

llm = OpenAI(model="gpt-4.1-mini")
condensed_memory = CondensedMemoryBlock(name="summary_agent", token_limit=200)

memory = Memory.from_defaults(
    session_id="summary_agent",
    token_limit=200,
    chat_history_token_ratio=0.7,
    token_flush_size=20,
    insert_method=InsertMethod.SYSTEM,
    memory_blocks=[condensed_memory],
)

agent = FunctionAgent(llm=llm, memory=memory)
```

**Why this memory exists:** Summaries trade raw detail for narrative continuity. Use it when you care about the "gist" across long sessions: constraints, commitments, and trajectory.

**Direct example (what the memory keeps):**

1. User: "I'm planning a 3-day hike. I have knee pain and can only carry 20 lbs."
2. Later: "Suggest a packing list."

The summary memory should preserve the constraints (knee pain, 20 lb limit) even if the original lines are gone.

### 3) The Reductive Agent: Proposition Memory for Precision

This agent extracts discrete propositions (facts, beliefs, preferences, goals) and uses them as an explicit knowledge layer.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory, InsertMethod
from llama_index.llms.openai import OpenAI

from asdrp.memory.proposition_extraction_memory import PropositionExtractionMemoryBlock

llm = OpenAI(model="gpt-4.1-mini")
propositions = PropositionExtractionMemoryBlock(
    name="proposition_extraction_memory",
    llm=llm,
    max_propositions=50,
)

memory = Memory.from_defaults(
    session_id="reductive_agent",
    token_limit=200,
    insert_method=InsertMethod.SYSTEM,
    memory_blocks=[propositions],
)

agent = FunctionAgent(llm=llm, memory=memory)
```

**Why this memory exists:** Sometimes you need the exact belief, not a summary. If the user says, "I am allergic to penicillin," the proposition memory should preserve that literally, not poetically.

**Direct example (what the memory preserves verbatim):**

1. User: "I am allergic to penicillin."
2. Later: "What antibiotic should I avoid?"

The proposition memory should return the exact fact without rephrasing it into a vague caution.

## How to Read Memory Like a Researcher

Think of memory as the agent's internal map of a shared world. Each memory type is a lens:

- **Working memory** is a short, honest window. Use it to measure baseline task success.
- **Condensed memory** is narrative continuity. Use it for long-horizon planning and stable persona.
- **Proposition memory** is a sparse knowledge graph. Use it for safety-critical details, preferences, and evolving facts.

If you are deciding which memory to use, ask: *Do I need detail, continuity, or truth-preserving facts?* You can even mix them, using the condensed memory for narrative flow and proposition memory for strict constraints.

## Evaluation and Benchmarking

This project uses the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark as a north star for long-term memory evaluation. It tests five core abilities: information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. The full benchmark is described in the paper [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813).

Here is the simplest evaluation loop to start with:

1. **Build long sessions** with deliberate facts, preferences, and updates.
2. **Ask delayed questions** across sessions (not immediately).
3. **Measure accuracy and abstention** (did the agent answer only when it had evidence?).
4. **Track memory cost** (token usage, latency, and retrieval footprint).

You can inspect the dataset format here:

- `longmemeval_s.json`: 500 evaluation instances, ~115k tokens history.
- `longmemeval_m.json`: 500 instances, ~500 sessions per history.
- `longmemeval_oracle.json`: history includes only evidence sessions.

Use `notebooks/Agentic Memory.ipynb` as a starting point for reproducing the evaluation pipeline and visualizing memory behaviors over time.

## A Narrative Arc of Agentic Memory Research

If you want to understand where this project is heading, read the field through a simple story: *memory began as a buffer, evolved into a journal, learned to reflect, and now tries to organize itself.*

1. **Memory as a buffer**: early systems relied on short windows. This is the simplest baseline and still a useful control case.
2. **Memory as a journal**: systems like **MemoryBank** introduced long-term storage with selective forgetting inspired by human memory dynamics ([arXiv:2305.10250](https://arxiv.org/abs/2305.10250)).
3. **Memory with reflection**: **Generative Agents** used memory streams and reflective summaries to plan future actions ([arXiv:2304.03442](https://arxiv.org/abs/2304.03442)). **Reflexion** added verbal feedback loops as an episodic memory for self-improvement ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)).
4. **Memory as a managed hierarchy**: **MemGPT** treats memory like an operating system, paging between tiers to handle long contexts ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)).
5. **Memory that organizes itself**: **A-MEM** proposes dynamic, Zettelkasten-inspired linking that evolves as new memories arrive ([arXiv:2502.12110](https://arxiv.org/abs/2502.12110)).
6. **Memory as a policy**: **AgeMem** teaches agents to decide when to store, retrieve, summarize, or discard memory as part of their action policy ([arXiv:2601.01885](https://arxiv.org/abs/2601.01885)).
7. **Memory as efficient lifelong compression**: **SimpleMem** focuses on semantic, lossless compression and intent-aware retrieval planning to reduce token cost while preserving long-term utility ([arXiv:2601.02553](https://arxiv.org/abs/2601.02553)).
8. **Memory retrieval under explicit control**: **MemR^3** builds a retrieval router with a closed-loop evidence tracker to decide when to retrieve, reflect, or answer ([arXiv:2512.20237](https://arxiv.org/abs/2512.20237)).

This project is deliberately positioned along that arc. We begin with a clean buffer, add condensation, then enforce proposition extraction, and ultimately aim to explore agentic memory that is self-organizing and policy-driven.

## Where We Are Going Next

Planned and experimental directions for this repository:

- **Hybrid memories**: combine condensed summaries with proposition extraction so the agent can both narrate and obey strict facts.
- **Retrieval-augmented memory**: vector-store backed recall for large memory corpora.
- **Temporal reasoning**: time-aware indexing and question answering inspired by LongMemEval.
- **Reflective memory**: periodic reflection to convert episodic experiences into stable beliefs.
- **Policy-driven memory actions**: expose memory actions to the agent as tools (store, retrieve, prune, summarize).

If you want to contribute, start by adding a new memory block in `asdrp/memory`, or by extending the evaluation notebook with your own benchmarks and probes.