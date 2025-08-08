"""SmolAgents Retrieval (Agentic RAG) examples as a CLI script.

This script mirrors the notebook and provides:
- web: CodeAgent with DuckDuckGoSearchTool for web retrieval
- kb: Custom retrieval Tool (BM25Retriever) over in-memory docs

Environment variables:
- HF_TOKEN: Optional for HF Inference API

Examples:
    python -m src.framework.rag.smola_retrieval_agents web \
        --query "luxury superhero-themed party ideas"

    python -m src.framework.rag.smola_retrieval_agents kb \
        --query "party catering ideas with superhero theme"
"""

from __future__ import annotations

import argparse
import os
from typing import List

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


def run_web(query: str, model_id: str) -> str:
    try:
        from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via pip.") from exc

    login_hf_if_token_present()
    agent = CodeAgent(model=InferenceClientModel(model_id), tools=[DuckDuckGoSearchTool()])
    return str(agent.run(query))


def run_kb(query: str, model_id: str) -> str:
    try:
        from langchain.docstore.document import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.retrievers import BM25Retriever
        from smolagents import CodeAgent, InferenceClientModel, Tool
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install: smolagents, langchain-community, rank_bm25"
        ) from exc

    # Simulated knowledge base
    party_ideas = [
        {
            "text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.",
            "source": "Party Ideas 1",
        },
        {
            "text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.",
            "source": "Entertainment Ideas",
        },
        {
            "text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'",
            "source": "Catering Ideas",
        },
        {
            "text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.",
            "source": "Decoration Ideas",
        },
        {
            "text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.",
            "source": "Entertainment Ideas",
        },
    ]

    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in party_ideas
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, add_start_index=True, strip_whitespace=True
    )
    docs_processed = splitter.split_documents(source_docs)

    class PartyPlanningRetrieverTool(Tool):
        name = "party_planning_retriever"
        description = (
            "Uses BM25 semantic search to retrieve relevant party planning ideas."
        )
        inputs = {
            "query": {
                "type": "string",
                "description": "Query related to party planning or superhero themes.",
            }
        }
        output_type = "string"

        def __init__(self, docs, **kwargs):
            super().__init__(**kwargs)
            self.retriever = BM25Retriever.from_documents(docs, k=5)

        def forward(self, query: str) -> str:  # type: ignore[override]
            assert isinstance(query, str), "Your search query must be a string"
            docs = self.retriever.invoke(query)
            return "\nRetrieved ideas:\n" + "".join(
                [
                    f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                    for i, doc in enumerate(docs)
                ]
            )

    tool = PartyPlanningRetrieverTool(docs_processed)
    agent = CodeAgent(tools=[tool], model=InferenceClientModel(model_id))
    return str(
        agent.run(
            "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolAgents Retrieval demos")
    sub = parser.add_subparsers(dest="command", required=True)

    p_web = sub.add_parser("web", help="Web retrieval via DuckDuckGo")
    p_web.add_argument("--query", required=True)
    p_web.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct", required=False
    )

    p_kb = sub.add_parser("kb", help="Local KB retrieval via BM25")
    p_kb.add_argument("--query", required=True)
    p_kb.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct", required=False
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "web":
        print(run_web(args.query, args.model_id))
        return
    if args.command == "kb":
        print(run_kb(args.query, args.model_id))
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()


