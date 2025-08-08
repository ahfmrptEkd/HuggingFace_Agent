"""Smoke tests for RAG scripts.

By default, only local, non-network parts are checked.
Set RUN_NETWORK_TESTS=true to exercise full flows that require tokens.
"""

from __future__ import annotations

import os
from pathlib import Path

# Auto-load environment variables from .env if available
try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass


def _run(cmd: str) -> int:
    return os.system(cmd)


def test_llama_components_prepare_and_ingest() -> None:
    """Prepare a tiny dataset and ingest a few docs (requires datasets + chroma)."""

    if os.environ.get("RUN_NETWORK_TESTS", "false").lower() != "true":
        return
    data_dir = Path("./data_test")
    chroma_dir = Path("./alfred_chroma_db_test")
    _run(f"python -m src.framework.rag.llama_components prepare-data --output {data_dir} --limit 3 | cat")
    code = _run(
        f"python -m src.framework.rag.llama_components ingest --data-dir {data_dir} --chroma {chroma_dir} --max-docs 3 | cat"
    )
    assert code == 0


def test_smola_retrieval_kb() -> None:
    """Run KB retrieval (no external network needed)."""

    code = _run(
        "python -m src.framework.rag.smola_retrieval_agents kb --query 'party catering ideas with superhero theme' | cat"
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


