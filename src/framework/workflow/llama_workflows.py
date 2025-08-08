"""LlamaIndex workflow examples as a runnable CLI.

This script mirrors the notebook and provides commands to:
- basic: Single-step workflow
- multi: Multi-step workflow passing events
- loop: Workflow with branches/loops
- state: Workflow using Context store
- multiagent: Multi-agent workflow (ReActAgent) example

Environment variables:
- HF_TOKEN: Optional for HF Inference API
"""

from __future__ import annotations

import argparse
import asyncio
import os

# Import workflow types at module scope so decorators can resolve annotations
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)

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


async def run_basic() -> str:
    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            return StopEvent(result="Hello, world!")

    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    return str(result)


async def run_multi() -> str:
    class ProcessingEvent(Event):
        intermediate_result: str

    class MultiStepWorkflow(Workflow):
        @step
        async def step_one(self, ev: StartEvent) -> ProcessingEvent:
            return ProcessingEvent(intermediate_result="Step 1 complete")

        @step
        async def step_two(self, ev: ProcessingEvent) -> StopEvent:
            return StopEvent(result=f"Finished processing: {ev.intermediate_result}")

    w = MultiStepWorkflow(timeout=10, verbose=False)
    result = await w.run()
    return str(result)


async def run_loop() -> str:
    import random

    class ProcessingEvent(Event):
        intermediate_result: str

    class LoopEvent(Event):
        loop_output: str

    class MultiStepWorkflow(Workflow):
        @step
        async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
            if random.randint(0, 1) == 0:
                return LoopEvent(loop_output="Back to step one.")
            return ProcessingEvent(intermediate_result="First step complete.")

        @step
        async def step_two(self, ev: ProcessingEvent) -> StopEvent:
            return StopEvent(result=f"Finished processing: {ev.intermediate_result}")

    w = MultiStepWorkflow(verbose=False)
    result = await w.run()
    return str(result)


async def run_state() -> str:
    class ProcessingEvent(Event):
        intermediate_result: str

    class MultiStepWorkflow(Workflow):
        @step
        async def step_one(self, ev: StartEvent, ctx: Context) -> ProcessingEvent:
            await ctx.store.set("query", "What is the capital of France?")
            return ProcessingEvent(intermediate_result="Step 1 complete")

        @step
        async def step_two(self, ev: ProcessingEvent, ctx: Context) -> StopEvent:
            query = await ctx.store.get("query")
            print(f"Query: {query}")
            return StopEvent(result=f"Finished processing: {ev.intermediate_result}")

    w = MultiStepWorkflow(timeout=10, verbose=False)
    result = await w.run()
    return str(result)


async def run_multiagent(model_id: str) -> str:
    login_hf_if_token_present()
    try:
        from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    except Exception as exc:
        raise RuntimeError("llama-index (agent + hf api) is required.") from exc

    def add(a: int, b: int) -> int:
        return a + b

    def multiply(a: int, b: int) -> int:
        return a * b

    llm = HuggingFaceInferenceAPI(model_name=model_id)
    multiply_agent = ReActAgent(
        name="multiply_agent",
        description="Is able to multiply two integers",
        system_prompt="A helpful assistant that can use a tool to multiply numbers.",
        tools=[multiply],
        llm=llm,
    )
    addition_agent = ReActAgent(
        name="add_agent",
        description="Is able to add two integers",
        system_prompt="A helpful assistant that can use a tool to add numbers.",
        tools=[add],
        llm=llm,
    )

    workflow = AgentWorkflow(agents=[multiply_agent, addition_agent], root_agent="multiply_agent")
    resp = await workflow.run(user_msg="Can you add 5 and 3?")
    return str(resp)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LlamaIndex workflow demos")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("basic")
    sub.add_parser("multi")
    sub.add_parser("loop")
    sub.add_parser("state")
    p_ma = sub.add_parser("multiagent")
    p_ma.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "basic":
        print(asyncio.run(run_basic()))
        return
    if args.command == "multi":
        print(asyncio.run(run_multi()))
        return
    if args.command == "loop":
        print(asyncio.run(run_loop()))
        return
    if args.command == "state":
        print(asyncio.run(run_state()))
        return
    if args.command == "multiagent":
        print(asyncio.run(run_multiagent(args.model)))
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()


