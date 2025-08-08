"""LlamaIndex agent examples as a runnable CLI script.

This module mirrors the notebook examples:
- Basic math agent using AgentWorkflow.from_tools_or_functions
- RAG agent using QueryEngineTool backed by Chroma + HF embeddings/LLM
- Simple multi-agent workflow (calculator + query agent)

Environment variables:
- HF_TOKEN: Optional, used for HF Inference API

Usage:
    python -m src.framework.agents.llama_agents math --question "What is (2 + 2) * 2?"
    python -m src.framework.agents.llama_agents rag --query "find sci-fi personas"
    python -m src.framework.agents.llama_agents multi --question "add 5 and 3"
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def login_hf_if_token_present() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] HF login skipped: {exc}")


def build_llm(model_name: str):
    """Build a LlamaIndex LLM that routes through LangChain for LangSmith tracing.

    Tries OpenAI Chat if model looks like an OpenAI model id, otherwise tries
    Hugging Face Inference API via ChatHuggingFace.
    """

    login_hf_if_token_present()

    # Import here to keep optional deps lazy
    from llama_index.llms.langchain import LangChainLLM
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        ChatOpenAI = None  # type: ignore
    try:
        from langchain_huggingface import ChatHuggingFace
    except Exception:
        ChatHuggingFace = None  # type: ignore

    lc_model = None
    model_lower = (model_name or "").lower()
    if ChatOpenAI and (model_lower.startswith("gpt-") or model_lower.startswith("o")):
        lc_model = ChatOpenAI(model=model_name)
    elif ChatHuggingFace:
        # Uses HF Inference API with HF_TOKEN
        lc_model = ChatHuggingFace(repo_id=model_name)

    if lc_model is None:
        raise RuntimeError(
            "Cannot build LangChain LLM. Install langchain_openai or langchain_huggingface, "
            "and set a valid model id (OpenAI 'gpt-*' or HF repo id)."
        )

    return LangChainLLM(lc_model)


def run_math_agent(question: str, model_name: str) -> str:
    try:
        from llama_index.core.agent.workflow import AgentWorkflow
    except Exception as exc:
        raise RuntimeError("llama-index is required. Install via pip.") from exc

    def add(a: int, b: int) -> int:
        """Add two numbers."""

        return a + b

    def subtract(a: int, b: int) -> int:
        """Subtract two numbers."""

        return a - b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""

        return a * b

    def divide(a: int, b: int) -> float:
        """Divide two numbers."""

        return a / b

    llm = build_llm(model_name)
    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[subtract, multiply, divide, add],
        llm=llm,
        system_prompt=(
            "You are a math agent that can add, subtract, multiply, and divide numbers using provided tools."
        ),
    )

    async def _run() -> str:
        handler = agent.run(question)
        # Drain stream to stdout (optional); here we just wait for the final result
        await handler.stream_events().__anext__() if False else None  # no-op
        resp = await handler
        return str(resp)

    return asyncio.run(_run())


def run_rag_agent(query: str, chroma_path: str, embed_model: str, model_name: str) -> str:
    try:
        import chromadb
        from llama_index.core import VectorStoreIndex
        from llama_index.core.tools import QueryEngineTool
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except Exception as exc:
        raise RuntimeError(
            "Install dependencies: chromadb, llama-index, llama-index-vector-stores-chroma, "
            "llama-index-embeddings-huggingface"
        ) from exc

    llm = build_llm(model_name)
    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed = HuggingFaceEmbedding(model_name=embed_model)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed)
    query_engine = index.as_query_engine(llm=llm)
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="personas",
        description="descriptions for various types of personas",
        return_direct=False,
    )

    try:
        from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("llama-index is required. Install via pip.") from exc

    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[query_engine_tool],
        llm=llm,
        system_prompt=(
            "You are a helpful assistant that has access to a database containing persona descriptions."
        ),
    )

    async def _run() -> str:
        handler = agent.run(query)
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                print(f"\nCalled tool: {ev.tool_name} {ev.tool_kwargs} => {ev.tool_output}")
            elif isinstance(ev, AgentStream):
                print(getattr(ev, "delta", ""), end="", flush=True)
        resp = await handler
        return str(resp)

    return asyncio.run(_run())


def run_multi_agent(question: str, chroma_path: str, embed_model: str, model_name: str) -> str:
    # Shared resources
    result_rag = run_rag_agent  # reuse builder and flow for simplicity

    try:
        from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
        from llama_index.core.tools import QueryEngineTool
        import chromadb
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except Exception as exc:
        raise RuntimeError("Missing llama-index or chromadb.") from exc

    llm = build_llm(model_name)

    # Basic tools for calculator
    def add(a: int, b: int) -> int:
        return a + b

    def subtract(a: int, b: int) -> int:
        return a - b

    calculator_agent = ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant. Use your tools for any math operation.",
        tools=[add, subtract],
        llm=llm,
    )

    # Build query tool
    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed = HuggingFaceEmbedding(model_name=embed_model)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed)
    query_engine = index.as_query_engine(llm=llm)
    from llama_index.core.tools import QueryEngineTool

    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="personas",
        description="descriptions for various types of personas",
        return_direct=False,
    )

    query_agent = ReActAgent(
        name="info_lookup",
        description="Looks up information about XYZ",
        system_prompt="Use your tool to query a RAG system to answer information about XYZ",
        tools=[query_engine_tool],
        llm=llm,
    )

    agent = AgentWorkflow(agents=[calculator_agent, query_agent], root_agent="calculator")

    async def _run() -> str:
        handler = agent.run(user_msg=question)
        resp = await handler
        return str(resp)

    return asyncio.run(_run())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LlamaIndex agent demos")
    sub = parser.add_subparsers(dest="command", required=True)

    p_math = sub.add_parser("math", help="Run basic math agent")
    p_math.add_argument("--question", required=True)
    p_math.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="HF Inference model id"
    )

    p_rag = sub.add_parser("rag", help="Run RAG agent (Chroma + HF)")
    p_rag.add_argument("--query", required=True)
    p_rag.add_argument("--chroma-path", default="./alfred_chroma_db")
    p_rag.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    p_rag.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="HF Inference model id"
    )

    p_multi = sub.add_parser("multi", help="Run multi-agent workflow")
    p_multi.add_argument("--question", required=True)
    p_multi.add_argument("--chroma-path", default="./alfred_chroma_db")
    p_multi.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    p_multi.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="HF Inference model id"
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "math":
        print(run_math_agent(args.question, args.model))
        return
    if args.command == "rag":
        print(run_rag_agent(args.query, args.chroma_path, args.embed_model, args.model))
        return
    if args.command == "multi":
        print(run_multi_agent(args.question, args.chroma_path, args.embed_model, args.model))
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()


