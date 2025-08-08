"""LangGraph-ready tool implementations and a helper to build a ToolNode.

This module provides a minimal, production-friendly pattern to define
structured tools that work seamlessly with LangGraph. Tools are defined
using Pydantic schemas and exported as a LangGraph `ToolNode`.

Usage example:

    from typing import Annotated, TypedDict
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import AnyMessage
    from langchain_openai import ChatOpenAI

    from src.framework.tools.langgraph_tools import (
        build_tool_node,
        TOOLS,
    )

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], "Conversation messages"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Bind tools to model (so it can emit tool_calls)
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: AgentState):
        response = llm_with_tools.invoke(state["messages"]) 
        return {"messages": state["messages"] + [response]}

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)

    # Tool execution node (routes tool_calls to actual tools)
    tool_node: ToolNode = build_tool_node()
    graph.add_node("tools", tool_node)

    # Router: if the last model message contains tool_calls, go to tools
    from langchain_core.messages import ToolMessage
    def route_on_tools(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "__end__"

    graph.add_conditional_edges("model", route_on_tools, {"tools": "tools", "__end__": "__end__"})
    graph.add_edge("tools", "model")
    graph.set_entry_point("model")

    app = graph.compile()

Notes:
- Tools here use Google-style docstrings for clarity.
- Inline comments are concise and explain non-obvious logic only.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


# Import the tool decorator in a version-tolerant way.
try:  # langchain >= 0.2
    from langchain_core.tools import tool
except Exception:  # langchain < 0.2 fallback
    from langchain.tools import tool  # type: ignore

try:
    from langgraph.prebuilt import ToolNode
except Exception as exc:  # pragma: no cover - import-time failure hint
    raise ImportError(
        "langgraph is required to build a ToolNode. Install via `pip install langgraph`."
    ) from exc


class SumNumbersInput(BaseModel):
    """Input schema for the sum_numbers tool."""

    numbers: List[float] = Field(
        ..., description="List of float numbers to summarize."
    )


@tool("sum_numbers", args_schema=SumNumbersInput)
def sum_numbers(numbers: List[float]) -> Dict[str, Any]:
    """Compute simple statistics over a list of numbers.

    Args:
        numbers: List of numeric values (float-compatible).

    Returns:
        A dictionary with the following keys:
        - sum: Sum of the provided numbers.
        - average: Arithmetic mean (None if list is empty).
        - count: Number of items provided.

    Example:
        >>> sum_numbers([1, 2, 3])
        {'sum': 6.0, 'average': 2.0, 'count': 3}
    """

    # Defensive handling for empty input.
    count = len(numbers)
    if count == 0:
        return {"sum": 0.0, "average": None, "count": 0}

    total = float(sum(numbers))
    average = total / count
    return {"sum": total, "average": average, "count": count}


class CurrentTimeInput(BaseModel):
    """Input schema for the get_current_time tool."""

    timezone: Optional[str] = Field(
        default=None,
        description=(
            "IANA timezone (e.g., 'UTC', 'Asia/Seoul'). Defaults to UTC when omitted "
            "or invalid."
        ),
    )


@tool("get_current_time", args_schema=CurrentTimeInput)
def get_current_time(timezone: Optional[str] = None) -> str:
    """Return the current time in ISO 8601 format for a given timezone.

    Args:
        timezone: IANA timezone name (e.g., 'UTC', 'Asia/Seoul'). If omitted or
            invalid, UTC will be used.

    Returns:
        ISO 8601 timestamp string, e.g., '2024-01-01T12:00:00+09:00'.

    Example:
        >>> get_current_time("Asia/Seoul")
        '2024-01-01T09:00:00+09:00'
    """

    from datetime import datetime, timezone as dt_timezone

    # Use stdlib zoneinfo when available; fall back to UTC on error.
    tzinfo = dt_timezone.utc
    if timezone:
        try:
            from zoneinfo import ZoneInfo

            tzinfo = ZoneInfo(timezone)  # may raise if invalid
        except Exception:
            tzinfo = dt_timezone.utc

    now = datetime.now(tzinfo)
    return now.isoformat()


# Exported tool list that can be bound to an LLM or to a ToolNode.
TOOLS = [sum_numbers, get_current_time]


def build_tool_node(custom_tools: Optional[List[Any]] = None) -> ToolNode:
    """Build and return a LangGraph ToolNode from tools.

    Args:
        custom_tools: Optional list of tool callables. When omitted, uses
            the default tools defined in this module.

    Returns:
        A `ToolNode` that can be inserted into a LangGraph.

    Example:
        >>> node = build_tool_node()  # uses default tools
        >>> node = build_tool_node([sum_numbers])  # custom subset
    """

    tools_to_use = custom_tools if custom_tools is not None else TOOLS
    # ToolNode accepts a list of LangChain tools (decorated callables).
    return ToolNode(tools_to_use)


__all__ = [
    "sum_numbers",
    "get_current_time",
    "TOOLS",
    "build_tool_node",
]


