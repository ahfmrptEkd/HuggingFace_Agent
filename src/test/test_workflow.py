"""Smoke tests for workflow scripts (LangGraph, LlamaIndex, SmolAgents).

Network-dependent tests require RUN_NETWORK_TESTS=true and relevant tokens.
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


def test_mail_sorting_local() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        return
    code = _run(
        "python -m src.framework.workflow.langg_mail_sorting run-sample --kind spam --llm gpt-4o | cat"
    )
    assert code == 0


def test_llama_workflows_basic() -> None:
    code = _run("python -m src.framework.workflow.llama_workflows basic | cat")
    assert code == 0


def test_smola_multiagent_single() -> None:
    # Requires network; keep under flag
    if os.environ.get("RUN_NETWORK_TESTS", "false").lower() != "true":
        return
    code = _run(
        "python -m src.framework.workflow.smola_multiagent_notebook single --task 'Find a few points' | cat"
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


