"""CLI-friendly utilities to demonstrate LlamaIndex tools usage.

This module converts the original Jupyter notebook examples into a
non-interactive, scriptable form with clear entrypoints. It covers:

- FunctionTool basics
- QueryEngineTool over a Chroma vector store with HF LLM + embeddings
- ToolSpec (GmailToolSpec) inspection

Environment variables:
- HF_TOKEN: Hugging Face API token for serverless Inference (optional but recommended)

Example:
    python -m src.framework.tools.llama_tools function-tool \
        --location "New York"

    python -m src.framework.tools.llama_tools query-engine-tool \
        --query "Impact of AI on future of work?" \
        --llm-model meta-llama/Llama-3.2-3B-Instruct \
        --embed-model BAAI/bge-small-en-v1.5 \
        --chroma-path ./alfred_chroma_db

    python -m src.framework.tools.llama_tools gmail-toolspec
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def login_hf_if_token_present() -> None:
    """Login to Hugging Face using HF_TOKEN env var if available.

    This avoids interactive prompts. If the token is missing, the function is
    a no-op.
    """

    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login  # lazy import

        # Non-interactive login using the provided token
        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - best-effort login
        print(f"[WARN] HF login skipped: {exc}")


# ---------- FunctionTool demo ----------


def run_function_tool_demo(location: str) -> str:
    """Run a minimal FunctionTool example.

    Args:
        location: Target location for weather query.

    Returns:
        The tool's return string.
    """

    try:
        from llama_index.core.tools import FunctionTool
    except Exception as exc:
        raise RuntimeError(
            "llama-index is required. Install via `pip install llama-index`."
        ) from exc

    def get_weather(target_location: str) -> str:
        """Useful for getting the weather for a given location.

        Args:
            target_location: City or place name.

        Returns:
            A short, human-readable weather description.
        """

        # Demo behavior only; no external API calls.
        print(f"Getting weather for {target_location}")
        return f"The weather in {target_location} is sunny"

    tool = FunctionTool.from_defaults(
        get_weather,
        name="my_weather_tool",
        description="Useful for getting the weather for a given location.",
    )

    # Execute the tool
    result = tool.call(location)
    return str(result)


# ---------- QueryEngineTool demo ----------


async def run_query_engine_tool_demo(
    query: str,
    chroma_path: str,
    embed_model: str,
    llm_model: str,
) -> str:
    """Run a QueryEngineTool backed by Chroma + HF embeddings + HF LLM.

    Args:
        query: Natural-language question to ask the index.
        chroma_path: Filesystem path for the Chroma persistent client.
        embed_model: Embedding model id on Hugging Face (e.g., 'BAAI/bge-small-en-v1.5').
        llm_model: Inference API model id (e.g., 'meta-llama/Llama-3.2-3B-Instruct').

    Returns:
        The text answer produced by the query engine tool.
    """

    # Lazy imports to keep the CLI responsive and optional.
    try:
        import chromadb  # type: ignore
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface_api import (
            HuggingFaceInferenceAPI,
        )
        from llama_index.core.tools import QueryEngineTool
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except Exception as exc:
        raise RuntimeError(
            "Required packages missing. Install: chromadb, llama-index, "
            "llama-index-embeddings-huggingface, llama-index-llms-huggingface-api, "
            "llama-index-vector-stores-chroma"
        ) from exc

    login_hf_if_token_present()

    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed = HuggingFaceEmbedding(model_name=embed_model)
    llm = HuggingFaceInferenceAPI(model_name=llm_model)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed
    )
    query_engine = index.as_query_engine(llm=llm)

    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="qa_index",
        description="QA over Chroma store with HF LLM.",
    )

    # Use async interface mirroring the notebook
    result = await tool.acall(query)
    return str(result)


# ---------- Gmail ToolSpec demo ----------


def run_gmail_toolspec_demo() -> List[str]:
    """List Gmail ToolSpec tool names and descriptions.

    Returns:
        A list of strings, each describing one tool's name and description.
    """

    try:
        from llama_index.tools.google import GmailToolSpec
    except Exception as exc:
        raise RuntimeError(
            "`llama-index-tools-google` is required. Install via: "
            "pip install llama-index-tools-google"
        ) from exc

    spec = GmailToolSpec()
    tools = spec.to_tool_list()
    lines = [f"{t.metadata.name}: {t.metadata.description}" for t in tools]
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured `ArgumentParser` instance.
    """

    parser = argparse.ArgumentParser(
        description="LlamaIndex tools demos (script version of the notebooks)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_func = sub.add_parser("function-tool", help="Run FunctionTool demo")
    p_func.add_argument("--location", required=True, help="Location name")

    p_qe = sub.add_parser("query-engine-tool", help="Run QueryEngineTool demo")
    p_qe.add_argument("--query", required=True, help="Natural-language query")
    p_qe.add_argument(
        "--chroma-path",
        default="./alfred_chroma_db",
        help="Path for Chroma persistent client",
    )
    p_qe.add_argument(
        "--embed-model",
        default="BAAI/bge-small-en-v1.5",
        help="HF embedding model id",
    )
    p_qe.add_argument(
        "--llm-model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HF Inference API LLM id",
    )

    sub.add_parser("gmail-toolspec", help="List Gmail ToolSpec tools")

    return parser


def main() -> None:
    """Entry point for command-line usage."""

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "function-tool":
        output = run_function_tool_demo(args.location)
        print(output)
        return

    if args.command == "query-engine-tool":
        text = asyncio.run(
            run_query_engine_tool_demo(
                query=args.query,
                chroma_path=args.chroma_path,
                embed_model=args.embed_model,
                llm_model=args.llm_model,
            )
        )
        print(text)
        return

    if args.command == "gmail-toolspec":
        lines = run_gmail_toolspec_demo()
        for line in lines:
            print(line)
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()


