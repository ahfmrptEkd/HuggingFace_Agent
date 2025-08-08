"""LangGraph agent script (notebook parity) with tool-calling loop.

This converts the original Jupyter notebook to a runnable CLI script
that builds a small ReAct-style loop in LangGraph. The agent can:

- Call a math tool `divide(a, b)`
- Call a vision tool `extract_text(img_path)` using a multimodal LLM

Environment:
- OPENAI_API_KEY: required for OpenAI models

Usage:
    python -m src.framework.agents.langg_agent \
        --prompt "Divide 6790 by 5" \
        --llm gpt-4o

    python -m src.framework.agents.langg_agent \
        --prompt "Extract the shopping list from the uploaded note" \
        --image Batman_training_and_meals.png \
        --llm gpt-4o
"""

from __future__ import annotations

import argparse
import base64
from typing import Annotated, Optional, TypedDict
import os

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

def extract_text(img_path: str) -> str:
    """Extract text from a PNG/JPG image using a multimodal model.

    Args:
        img_path: Local path to the image file.

    Returns:
        Extracted textual content.
    """

    # Read and base64-encode the image
    with open(img_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Prepare a vision-style content payload
    message = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Extract all the text from this image. Return only the extracted text, no explanations."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        )
    ]

    response = vision_llm.invoke(message)
    return str(response.content).strip()


def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: Dividend
        b: Divisor (non-zero)

    Returns:
        Floating division result.
    """

    return a / b


class AgentState(TypedDict):
    """State structure for the LangGraph agent."""

    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]


def build_assistant(llm_with_tools: ChatOpenAI):
    """Create the assistant node callable bound to tools.

    Args:
        llm_with_tools: ChatOpenAI with tools bound via `bind_tools`.

    Returns:
        A function(state) -> dict to be used as a graph node.
    """

    def assistant(state: AgentState):
        # Short system description of available tools
        textual_description_of_tool = (
            "extract_text(img_path: str) -> str: Extract text from an image.\n"
            "divide(a: int, b: int) -> float: Divide a and b."
        )

        image = state.get("input_file")
        sys_msg = SystemMessage(
            content=(
                "You are a helpful agent that can analyze images and perform simple math.\n"
                f"Available tools:\n{textual_description_of_tool}\n"
                f"Current optional image: {image}"
            )
        )

        result = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [result], "input_file": state.get("input_file")}

    return assistant


def build_graph(llm_with_tools: ChatOpenAI):
    """Assemble the LangGraph state machine with assistant and tools nodes."""

    builder = StateGraph(AgentState)
    assistant = build_assistant(llm_with_tools)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode([divide, extract_text]))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a LangGraph agent with tool-calling loop (LLM + tools)."
    )
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument(
        "--image",
        required=False,
        default=None,
        help="Optional image path for OCR extraction",
    )
    parser.add_argument(
        "--llm", required=False, default="gpt-4o", help="OpenAI chat model id"
    )
    parser.add_argument(
        "--temperature", required=False, type=float, default=0.0, help="LLM temperature"
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    global vision_llm  # used inside extract_text
    vision_llm = ChatOpenAI(model=args.llm, temperature=args.temperature)

    llm = ChatOpenAI(model=args.llm, temperature=args.temperature)
    llm_with_tools = llm.bind_tools([divide, extract_text], parallel_tool_calls=False)

    app = build_graph(llm_with_tools)
    messages = [HumanMessage(content=args.prompt)]

    state = {"messages": messages, "input_file": args.image}
    final_state = app.invoke(state)

    for m in final_state["messages"]:
        role = getattr(m, "type", getattr(m, "name", "assistant"))
        print(f"[{role}] {getattr(m, 'content', m)}")


if __name__ == "__main__":
    main()


