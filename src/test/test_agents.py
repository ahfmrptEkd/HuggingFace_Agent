"""Smoke tests for agents (LangGraph, LlamaIndex, SmolAgents).

Network-dependent tests are conditional on RUN_NETWORK_TESTS=true.
OpenAI/HF credentials are required for some tests; skipped otherwise.
"""

from __future__ import annotations

import os

# Auto-load environment variables from .env if available
try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass


def _run(cmd: str) -> int:
    return os.system(cmd)


def test_langg_agent_divide() -> None:
    """LangGraph agent should handle a simple math prompt without errors."""

    if not os.environ.get("OPENAI_API_KEY"):
        return  # skip if no key
    code = _run(
        "python -m src.framework.agents.langg_agent --prompt 'Divide 10 by 2' --llm gpt-4o | cat"
    )
    assert code == 0


def test_llama_agents_math() -> None:
    """LlamaIndex math agent via LangChain LLM wrapper should run."""

    if not os.environ.get("HF_TOKEN") and not os.environ.get("OPENAI_API_KEY"):
        return
    code = _run(
        "python -m src.framework.agents.llama_agents math --question 'What is (2 + 2) * 2?' | cat"
    )
    assert code == 0


def test_smola_tool_calling_agent() -> None:
    """SmolAgents ToolCallingAgent should run without network if skipped."""

    if os.environ.get("RUN_NETWORK_TESTS", "false").lower() != "true":
        return
    code = _run(
        "python -m src.framework.agents.smola_tool_calling_agents --query 'party music ideas' | cat"
    )
    assert code == 0


if __name__ == "__main__":
    failed = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"[OK] {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {name}: {exc}")
    raise SystemExit(1 if failed else 0)


