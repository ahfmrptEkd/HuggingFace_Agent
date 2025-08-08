"""Smoke tests for tools modules.

Tests are network-free by default. Network-dependent tests can be enabled
with RUN_NETWORK_TESTS=true in the environment.
"""

from __future__ import annotations

import os
from typing import Any

# Auto-load environment variables from .env if available
try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass


def test_langgraph_tools_sum() -> None:
    """Verify `sum_numbers` tool returns expected simple stats."""

    from src.framework.tools.langgraph_tools import sum_numbers

    result: dict[str, Any] = sum_numbers.invoke({"numbers": [1, 2, 3]})
    assert result["sum"] == 6.0
    assert result["average"] == 2.0
    assert result["count"] == 3


def test_langgraph_tools_time() -> None:
    """Verify `get_current_time` returns an ISO-8601 string."""

    from src.framework.tools.langgraph_tools import get_current_time

    ts = get_current_time.invoke({"timezone": "UTC"})
    assert "T" in ts and ts.endswith("Z") is False  # timezone offset included


def test_smola_tools_local_tools() -> None:
    """Verify smola tools (requires network) only when enabled."""

    if os.environ.get("RUN_NETWORK_TESTS", "false").lower() != "true":
        return
    from src.framework.tools.smola_tools import run_catering_tool, run_theme_tool

    best = run_catering_tool("any query")
    assert "Gotham Catering Co." in best

    theme = run_theme_tool("villain masquerade")
    assert "Gotham Rogues' Ball" in theme


def test_llama_tools_functiontool() -> None:
    """Verify the FunctionTool demo returns expected text."""

    from src.framework.tools.llama_tools import run_function_tool_demo

    out = run_function_tool_demo("Seoul")
    assert "The weather in Seoul is sunny" in out


if __name__ == "__main__":
    # Simple manual runner
    tests = [
        test_langgraph_tools_sum,
        test_langgraph_tools_time,
        test_smola_tools_local_tools,
        test_llama_tools_functiontool,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[OK] {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {t.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)


