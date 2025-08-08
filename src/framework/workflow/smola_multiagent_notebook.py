"""SmolAgents multi-agent workflow as a runnable CLI.

This script mirrors the complex notebook that builds a two-agent hierarchy
to solve a geospatial analysis task and plot a map.

Commands:
- single: Single CodeAgent baseline (search + webpage + cargo time tool)
- team: Manager CodeAgent + web CodeAgent with planning and plot validation

Environment variables:
- HF_TOKEN: Optional for HF Inference API
- SERPAPI_API_KEY or SERPER_API_KEY: for GoogleSearchTool provider
- OPENAI_API_KEY: for `OpenAIServerModel` in final plot validation step (optional)

Note: This script expects `plotly`, `geopandas`, `shapely`, `kaleido` installed
for plotting and saving images in the `team` command.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional, Tuple
from smolagents import tool

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


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,
) -> float:
    """Estimate cargo plane travel time (hours) using great-circle distance.

    Args:
        origin_coords: (lat, lon)
        destination_coords: (lat, lon)
        cruising_speed_kmh: Average speed in km/h

    Returns:
        Travel time in hours (rounded to 2 decimals)
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)
    earth_radius_km = 6371.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = earth_radius_km * c
    actual_distance = distance * 1.1
    flight_time = (actual_distance / float(cruising_speed_kmh or 750.0)) + 1.0
    return round(flight_time, 2)


def run_single(task: str, provider: Optional[str], model_id: str, search_provider: Optional[str]) -> str:
    try:
        from smolagents import (
            CodeAgent,
            GoogleSearchTool,
            InferenceClientModel,
            VisitWebpageTool,
        )
    except Exception as exc:
        raise RuntimeError("smolagents is required. Install via pip.") from exc

    login_hf_if_token_present()
    agent = CodeAgent(
        model=InferenceClientModel(model_id=model_id, provider=provider or "together"),
        tools=[
            GoogleSearchTool(provider=search_provider or "duckduckgo"),
            VisitWebpageTool(),
            calculate_cargo_travel_time,
        ],
        additional_authorized_imports=["pandas"],
        max_steps=20,
    )
    return str(agent.run(task))


def run_team(task: str, provider: Optional[str], manager_model: str, web_model: str, search_provider: Optional[str]) -> str:
    try:
        from smolagents import (
            CodeAgent,
            GoogleSearchTool,
            InferenceClientModel,
            VisitWebpageTool,
            OpenAIServerModel,
        )
        from smolagents.utils import encode_image_base64, make_image_url
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "smolagents and Pillow are required. Install via `pip install smolagents pillow`."
        ) from exc

    login_hf_if_token_present()

    web_agent = CodeAgent(
        model=InferenceClientModel(web_model, provider=provider or "together", max_tokens=8096),
        tools=[
            GoogleSearchTool(provider=search_provider or "duckduckgo"),
            VisitWebpageTool(),
            calculate_cargo_travel_time,
        ],
        name="web_agent",
        description="Browses the web to find information",
        verbosity_level=0,
        max_steps=10,
    )

    def check_reasoning_and_plot(final_answer, agent_memory):
        multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
        filepath = "saved_map.png"
        assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
        image = Image.open(filepath)
        prompt = (
            f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
            "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
            "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
            "Don't be harsh: if the plot mostly solves the task, it should pass."
            "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": make_image_url(encode_image_base64(image))},
                    },
                ],
            }
        ]
        output = multimodal_model(messages).content
        print("Feedback: ", output)
        if "FAIL" in output:
            raise Exception(output)
        return True

    manager_agent = CodeAgent(
        model=InferenceClientModel(manager_model, provider=provider or "together", max_tokens=8096),
        tools=[calculate_cargo_travel_time],
        managed_agents=[web_agent],
        additional_authorized_imports=[
            "geopandas",
            "plotly",
            "shapely",
            "json",
            "pandas",
            "numpy",
        ],
        planning_interval=5,
        verbosity_level=2,
        final_answer_checks=[check_reasoning_and_plot],
        max_steps=15,
    )

    return str(manager_agent.run(task))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolAgents multi-agent workflow demos")
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", help="Single CodeAgent baseline")
    p_single.add_argument("--task", required=True)
    p_single.add_argument("--provider", required=False, default="together", help="Model provider (e.g., together)")
    p_single.add_argument("--search-provider", required=False, default="duckduckgo", help="Search provider: duckduckgo | serpapi | serper")
    p_single.add_argument("--model", required=False, default="Qwen/Qwen2.5-Coder-32B-Instruct")

    p_team = sub.add_parser("team", help="Manager + Web agent setup with plotting")
    p_team.add_argument("--task", required=True)
    p_team.add_argument("--provider", required=False, default="together", help="Model provider (e.g., together)")
    p_team.add_argument("--search-provider", required=False, default="duckduckgo", help="Search provider: duckduckgo | serpapi | serper")
    p_team.add_argument("--manager-model", required=False, default="deepseek-ai/DeepSeek-R1")
    p_team.add_argument("--web-model", required=False, default="Qwen/Qwen2.5-Coder-32B-Instruct")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "single":
        print(run_single(args.task, args.provider, args.model, args.search_provider))
        return
    if args.command == "team":
        print(run_team(args.task, args.provider, args.manager_model, args.web_model, args.search_provider))
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()

