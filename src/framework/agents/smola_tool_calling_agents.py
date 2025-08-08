"""SmolAgents ToolCallingAgent example (script version).

This script mirrors the notebook that demonstrates ToolCallingAgent
using DuckDuckGoSearchTool and the Hugging Face Inference API model.

Usage:
    python -m src.framework.agents.smola_tool_calling_agents \
        --query "best music recommendations for a party at Wayne's mansion"
"""

from __future__ import annotations

import argparse
import os

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


def run(query: str, model_id: str) -> str:
    try:
        from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, InferenceClientModel
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via pip.") from exc

    login_hf_if_token_present()

    agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=InferenceClientModel(model_id))
    return str(agent.run(query))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolAgents ToolCallingAgent demo")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--model-id",
        required=False,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HF Inference API model id",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(run(args.query, args.model_id))


if __name__ == "__main__":
    main()


