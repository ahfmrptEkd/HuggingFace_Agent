"""CLI-friendly utilities to demonstrate SmolAgents tools usage.

This module converts the original Jupyter notebook examples into a
non-interactive, scriptable form with clear entrypoints. It covers:

- `@tool` decorated function tool
- `Tool` subclass with `forward`
- Loading tool from the Hub and from a Space
- Loading a LangChain tool via `Tool.from_langchain`

Environment variables:
- HF_TOKEN: Hugging Face token for API access (optional for public models)
- SERPAPI_API_KEY: Required for the SerpAPI-based tool example

Examples:
    python -m src.framework.tools.smola_tools catering --query "bestu catering in Gotham"

    python -m src.framework.tools.smola_tools theme --category "villain masquerade"

    python -m src.framework.tools.smola_tools hub-image \
        --repo m-ric/text-to-image --prompt "A superhero party at Wayne Manor"

    python -m src.framework.tools.smola_tools space-image \
        --space black-forest-labs/FLUX.1-schnell \
        --prompt "A grand superhero-themed party at Wayne Manor"

    python -m src.framework.tools.smola_tools serpapi-search \
        --query "luxury entertainment ideas for superhero-themed event"
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def login_hf_if_token_present() -> None:
    """Login to Hugging Face using HF_TOKEN env var if available (best effort)."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login  # lazy import

        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] HF login skipped: {exc}")


def run_catering_tool(query: str) -> str:
    """Run a simple @tool-decorated function using SmolAgents.

    Args:
        query: Search term, used here only to mirror the notebook interface.

    Returns:
        The highest-rated catering service name (demo values).
    """

    try:
        from smolagents import CodeAgent, InferenceClientModel, tool
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via `pip install smolagents`.") from exc

    @tool
    def catering_service_tool(user_query: str) -> str:
        """Return the highest-rated catering service in Gotham City.

        Args:
            user_query: Search phrase for filtering (demo only).
        """

        services = {
            "Gotham Catering Co.": 4.9,
            "Wayne Manor Catering": 4.8,
            "Gotham City Events": 4.7,
        }
        best_service = max(services, key=services.get)
        return best_service

    agent = CodeAgent(tools=[catering_service_tool], model=InferenceClientModel())
    result = agent.run(
        "Can you give me the name of the highest-rated catering service in Gotham City?"
    )
    return str(result)


def run_theme_tool(category: str) -> str:
    """Run a Tool subclass that returns a themed party idea."""

    try:
        from smolagents import Tool, CodeAgent, InferenceClientModel
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via `pip install smolagents`.") from exc

    class SuperheroPartyThemeTool(Tool):
        name = "superhero_party_theme_generator"
        description = (
            "This tool suggests creative superhero-themed party ideas based on a category."
        )
        inputs = {
            "category": {
                "type": "string",
                "description": (
                    "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham')."
                ),
            }
        }
        output_type = "string"

        def forward(self, category: str) -> str:  # type: ignore[override]
            themes = {
                "classic heroes": (
                    "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'."
                ),
                "villain masquerade": (
                    "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains."
                ),
                "futuristic gotham": (
                    "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
                ),
            }
            return themes.get(
                category.lower(),
                "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.",
            )

    tool = SuperheroPartyThemeTool()
    agent = CodeAgent(tools=[tool], model=InferenceClientModel())
    result = agent.run(
        f"What would be a good superhero party idea for a '{category}' theme?"
    )
    return str(result)


def run_hub_tool(repo_id: str, prompt: str) -> str:
    """Load a tool from the Hub and run it with a prompt (text-to-image demo)."""

    try:
        from smolagents import load_tool, CodeAgent, InferenceClientModel
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via `pip install smolagents`.") from exc

    login_hf_if_token_present()

    image_generation_tool = load_tool(repo_id, trust_remote_code=True)
    agent = CodeAgent(tools=[image_generation_tool], model=InferenceClientModel())
    result = agent.run(prompt)
    return str(result)


def run_space_tool(space_id: str, prompt: str, model_id: str) -> str:
    """Load a Space as a tool and run it with a prompt."""

    try:
        from smolagents import CodeAgent, InferenceClientModel, Tool
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via `pip install smolagents`.") from exc

    login_hf_if_token_present()

    image_generation_tool = Tool.from_space(
        space_id,
        name="image_generator",
        description="Generate an image from a prompt",
    )

    model = InferenceClientModel(model_id)
    agent = CodeAgent(tools=[image_generation_tool], model=model)
    result = agent.run(
        "Improve this prompt, then generate an image of it.",
        additional_args={
            "user_prompt": prompt,
        },
    )
    return str(result)


def run_serpapi_search(query: str, model_id: str) -> str:
    """Load a LangChain SerpAPI tool and run a search via SmolAgents."""

    if not os.environ.get("SERPAPI_API_KEY"):
        raise RuntimeError(
            "SERPAPI_API_KEY is required in the environment for the serpapi-search command."
        )

    try:
        from langchain.agents import load_tools
        from smolagents import CodeAgent, InferenceClientModel, Tool
    except Exception as exc:
        raise RuntimeError(
            "Required dependencies missing. Install langchain-community and smolagents."
        ) from exc

    search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])
    agent = CodeAgent(tools=[search_tool], model=InferenceClientModel(model_id))
    result = agent.run(query)
    return str(result)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="SmolAgents tools demos (script version of the notebooks)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cat = sub.add_parser("catering", help="Run @tool catering demo")
    p_cat.add_argument("--query", required=False, default="best catering in Gotham")

    p_theme = sub.add_parser("theme", help="Run Tool subclass demo")
    p_theme.add_argument("--category", required=True)

    p_hub = sub.add_parser("hub-image", help="Run Hub tool (text-to-image) demo")
    p_hub.add_argument("--repo", required=True, help="Tool repo id, e.g., m-ric/text-to-image")
    p_hub.add_argument("--prompt", required=True)

    p_space = sub.add_parser("space-image", help="Run Space tool demo")
    p_space.add_argument("--space", required=True, help="Space id, e.g., black-forest-labs/FLUX.1-schnell")
    p_space.add_argument("--prompt", required=True)
    p_space.add_argument(
        "--model-id",
        required=False,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HF Inference model id used to orchestrate the agent",
    )

    p_serp = sub.add_parser("serpapi-search", help="Run SerpAPI search via LangChain tool")
    p_serp.add_argument("--query", required=True)
    p_serp.add_argument(
        "--model-id",
        required=False,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HF Inference model id used to orchestrate the agent",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "catering":
        print(run_catering_tool(args.query))
        return

    if args.command == "theme":
        print(run_theme_tool(args.category))
        return

    if args.command == "hub-image":
        print(run_hub_tool(args.repo, args.prompt))
        return

    if args.command == "space-image":
        print(run_space_tool(args.space, args.prompt, args.model_id))
        return

    if args.command == "serpapi-search":
        print(run_serpapi_search(args.query, args.model_id))
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()


