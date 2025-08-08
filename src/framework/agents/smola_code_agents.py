"""SmolAgents CodeAgent examples as runnable CLI commands.

This consolidates multiple notebook sections into a single script:
- Web search with DuckDuckGoSearchTool
- Custom @tool for menu suggestion
- Allowing extra imports for code execution (datetime)
- Push/load agent to/from the Hub (optional)
- Telemetry (OpenTelemetry + Langfuse) setup (optional)

Environment variables:
- HF_TOKEN: Optional for HF Inference API
- LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY: For telemetry (optional)
"""

from __future__ import annotations

import argparse
import base64
import os
from typing import Optional

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


def run_search(query: str, model_id: str) -> str:
    try:
        from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via pip.") from exc

    login_hf_if_token_present()
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=InferenceClientModel(model_id))
    return str(agent.run(query))


def run_menu_tool(occasion: str, model_id: str) -> str:
    try:
        from smolagents import CodeAgent, InferenceClientModel, tool
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via pip.") from exc

    @tool
    def suggest_menu(occasion: str) -> str:
        """Suggest a menu based on the occasion."""

        if occasion == "casual":
            return "Pizza, snacks, and drinks."
        if occasion == "formal":
            return "3-course dinner with wine and dessert."
        if occasion == "superhero":
            return "Buffet with high-energy and healthy food."
        return "Custom menu for the butler."

    agent = CodeAgent(tools=[suggest_menu], model=InferenceClientModel(model_id))
    return str(agent.run("Prepare a formal menu for the party."))


def run_time_planning(model_id: str) -> str:
    """Demonstrate additional_authorized_imports (datetime)."""

    try:
        from smolagents import CodeAgent, InferenceClientModel
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via pip.") from exc

    agent = CodeAgent(
        tools=[],
        model=InferenceClientModel(model_id),
        additional_authorized_imports=["datetime"],
    )
    prompt = (
        """
        Alfred needs to prepare for the party. Here are the tasks:
        1. Prepare the drinks - 30 minutes
        2. Decorate the mansion - 60 minutes
        3. Set up the menu - 45 minutes
        3. Prepare the music and playlist - 45 minutes

        If we start right now, at what time will the party be ready?
        """
    )
    return str(agent.run(prompt))


def push_agent_to_hub(repo_id: str, model_id: str) -> str:
    try:
        from smolagents import (
            CodeAgent,
            DuckDuckGoSearchTool,
            InferenceClientModel,
            VisitWebpageTool,
            FinalAnswerTool,
            Tool,
            tool,
        )
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via pip.") from exc

    login_hf_if_token_present()

    @tool
    def suggest_menu(occasion: str) -> str:
        """Suggest a menu based on the occasion."""

        if occasion == "casual":
            return "Pizza, snacks, and drinks."
        if occasion == "formal":
            return "3-course dinner with wine and dessert."
        if occasion == "superhero":
            return "Buffet with high-energy and healthy food."
        return "Custom menu for the butler."

    @tool
    def catering_service_tool(query: str) -> str:
        """Return the highest-rated catering service in Gotham City."""

        services = {
            "Gotham Catering Co.": 4.9,
            "Wayne Manor Catering": 4.8,
            "Gotham City Events": 4.7,
        }
        return max(services, key=services.get)

    class SuperheroPartyThemeTool(Tool):
        name = "superhero_party_theme_generator"
        description = (
            "This tool suggests creative superhero-themed party ideas based on a category."
        )
        inputs = {
            "category": {
                "type": "string",
                "description": (
                    "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic gotham')."
                ),
            }
        }
        output_type = "string"

        def forward(self, category: str) -> str:  # type: ignore[override]
            themes = {
                "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
                "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
                "futuristic gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets.",
            }
            return themes.get(
                category.lower(),
                "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic gotham'.",
            )

    agent = CodeAgent(
        tools=[
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
            suggest_menu,
            catering_service_tool,
            SuperheroPartyThemeTool(),
        ],
        model=InferenceClientModel(model_id),
        max_steps=10,
        verbosity_level=2,
    )

    agent.push_to_hub(repo_id)
    return f"Pushed agent to {repo_id}"


def load_agent_from_hub(repo_id: str, model_id: str) -> str:
    try:
        from smolagents import CodeAgent, InferenceClientModel
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via pip.") from exc

    login_hf_if_token_present()
    tmp = CodeAgent(tools=[], model=InferenceClientModel(model_id))
    agent = tmp.from_hub(repo_id, trust_remote_code=True)
    return str(
        agent.run(
            "Give me best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme"
        )
    )


def enable_telemetry(public_key: str, secret_key: str, region: str = "eu") -> None:
    """Enable OpenTelemetry + Langfuse export (best-effort)."""

    try:
        from opentelemetry.sdk.trace import TracerProvider
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except Exception as exc:  # pragma: no cover - optional path
        print(f"[WARN] Telemetry not enabled: {exc}")
        return

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    if region.lower() == "us":
        endpoint = "https://us.cloud.langfuse.com/api/public/otel"
    else:
        endpoint = "https://cloud.langfuse.com/api/public/otel"

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    SmolagentsInstrumentor().instrument(tracer_provider=provider)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolAgents CodeAgent demos")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="DuckDuckGo search via CodeAgent")
    p_search.add_argument("--query", required=True)
    p_search.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct", required=False
    )

    p_menu = sub.add_parser("menu", help="Run custom @tool to suggest menu")
    p_menu.add_argument("--occasion", default="formal")
    p_menu.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct", required=False
    )

    p_time = sub.add_parser("time", help="Demonstrate additional_authorized_imports")
    p_time.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct", required=False
    )

    p_push = sub.add_parser("push", help="Push composed agent to the Hub")
    p_push.add_argument("--repo", required=True)
    p_push.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct", required=False
    )

    p_load = sub.add_parser("load", help="Load agent from the Hub and run it")
    p_load.add_argument("--repo", required=True)
    p_load.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct", required=False
    )

    p_tel = sub.add_parser("telemetry", help="Enable telemetry (Langfuse)")
    p_tel.add_argument("--public-key", required=True)
    p_tel.add_argument("--secret-key", required=True)
    p_tel.add_argument("--region", choices=["eu", "us"], default="eu")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "search":
        print(run_search(args.query, args.model_id))
        return
    if args.command == "menu":
        print(run_menu_tool(args.occasion, args.model_id))
        return
    if args.command == "time":
        print(run_time_planning(args.model_id))
        return
    if args.command == "push":
        print(push_agent_to_hub(args.repo, args.model_id))
        return
    if args.command == "load":
        print(load_agent_from_hub(args.repo, args.model_id))
        return
    if args.command == "telemetry":
        enable_telemetry(args.public_key, args.secret_key, args.region)
        print("Telemetry enabled (best effort).")
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()


