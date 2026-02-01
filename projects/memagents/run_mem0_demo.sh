#!/bin/bash
# Wrapper script to run mem0_agent demo with warnings suppressed

# Get the directory where this script lives (repo root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to repo root so imports work
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run the demo and filter out deprecation warnings from stderr
cd "${SCRIPT_DIR}"
uv run python -W ignore asdrp/agent/mem0_agent.py 2>&1 | \
  grep -v "PydanticDeprecated" | \
  grep -v "DeprecationWarning" | \
  grep -v "inspect.py:602" | \
  grep -v "Deprecated in Pydantic" | \
  grep -v "value = getter(object, key)" | \
  grep -v "See Pydantic V2 Migration Guide"
