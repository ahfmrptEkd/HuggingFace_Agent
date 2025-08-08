"""LangGraph email processing workflow (mail sorting) as a CLI script.

This script mirrors the notebook example and provides a small workflow to:
- Read an email
- Classify as SPAM/HAM via LLM
- Handle spam or draft a response
- Notify Mr. Wayne

Environment variables:
- OPENAI_API_KEY: required for OpenAI models

Usage:
    python -m src.framework.workflow.langg_mail_sorting run-sample --kind legitimate --llm gpt-4o
    python -m src.framework.workflow.langg_mail_sorting run \
        --sender "Crypto bro" --subject "The best investment of 2025" \
        --body "Mr Wayne, I just launched an ALT coin and want you to buy some !" \
        --llm gpt-4o
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


class EmailState(TypedDict):
    """State carried across the workflow nodes."""

    email: Dict[str, Any]
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    email_category: Optional[str]
    email_draft: Optional[str]
    messages: List[Dict[str, Any]]


def build_model(model_name: str, temperature: float) -> ChatOpenAI:
    """Build a ChatOpenAI model instance.

    Args:
        model_name: OpenAI chat model id.
        temperature: Sampling temperature.

    Returns:
        Configured ChatOpenAI instance.
    """

    return ChatOpenAI(model=model_name, temperature=temperature)


def read_email(state: EmailState) -> Dict[str, Any]:
    """Node: Log email metadata and proceed.

    Args:
        state: Current workflow state.

    Returns:
        Partial state updates (empty for this node).
    """

    email = state["email"]
    print(
        f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}"
    )
    return {}


def classify_email(state: EmailState, model: ChatOpenAI) -> Dict[str, Any]:
    """Node: Classify email as SPAM or HAM via LLM.

    Args:
        state: Current workflow state.
        model: ChatOpenAI instance used to classify.

    Returns:
        Updates keys: is_spam, messages
    """

    email = state["email"]
    prompt = f"""
As Alfred the butler of Mr Wayne and his SECRET identity Batman, analyze this email and determine if it is spam or legitimate.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

Return only one word: SPAM or HAM.
Answer:
"""
    response = model.invoke([HumanMessage(content=prompt)])
    response_text = str(response.content).lower()
    print(response_text)
    is_spam = "spam" in response_text and "ham" not in response_text

    new_messages = state.get("messages", [])
    if not is_spam:
        new_messages = new_messages + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content},
        ]

    return {"is_spam": is_spam, "messages": new_messages}


def handle_spam(_: EmailState) -> Dict[str, Any]:
    """Node: Handle spam path (log-only)."""

    print("Alfred has marked the email as spam.")
    print("The email has been moved to the spam folder.")
    return {}


def drafting_response(state: EmailState, model: ChatOpenAI) -> Dict[str, Any]:
    """Node: Draft a polite response for legitimate emails.

    Args:
        state: Current workflow state.
        model: ChatOpenAI used to draft a reply.

    Returns:
        Updates keys: email_draft, messages
    """

    email = state["email"]
    prompt = f"""
As Alfred the butler, draft a brief and professional preliminary response to this email for Mr. Wayne to review.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}
"""
    response = model.invoke([HumanMessage(content=prompt)])
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]
    return {"email_draft": response.content, "messages": new_messages}


def notify_mr_wayne(state: EmailState) -> Dict[str, Any]:
    """Node: Present a summary to Mr. Wayne (stdout)."""

    email = state["email"]
    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state.get("email_draft", "<no draft>"))
    print("=" * 50 + "\n")
    return {}


def route_email(state: EmailState) -> str:
    """Router: spam -> 'spam', ham -> 'legitimate'."""

    return "spam" if state.get("is_spam") else "legitimate"


def build_graph(model: ChatOpenAI):
    """Build and compile the LangGraph workflow.

    Args:
        model: ChatOpenAI model bound into nodes that require it.

    Returns:
        Compiled graph ready to invoke.
    """

    graph = StateGraph(EmailState)
    graph.add_node("read_email", read_email)
    graph.add_node("classify_email", lambda s: classify_email(s, model))
    graph.add_node("handle_spam", handle_spam)
    graph.add_node("drafting_response", lambda s: drafting_response(s, model))
    graph.add_node("notify_mr_wayne", notify_mr_wayne)

    graph.add_edge(START, "read_email")
    graph.add_edge("read_email", "classify_email")
    graph.add_conditional_edges("classify_email", route_email, {
        "spam": "handle_spam",
        "legitimate": "drafting_response",
    })
    graph.add_edge("handle_spam", END)
    graph.add_edge("drafting_response", "notify_mr_wayne")
    graph.add_edge("notify_mr_wayne", END)
    return graph.compile()


def run_workflow(email: Dict[str, Any], model_name: str, temperature: float) -> None:
    """Execute the workflow on a single email.

    Args:
        email: Email dictionary with keys sender, subject, body.
        model_name: OpenAI model id.
        temperature: LLM temperature.
    """

    model = build_model(model_name, temperature)
    app = build_graph(model)
    initial_state: EmailState = {
        "email": email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": [],
    }
    app.invoke(initial_state)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph mail sorting workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("run-sample", help="Run on a built-in sample email")
    p_sample.add_argument("--kind", choices=["legitimate", "spam"], default="legitimate")
    p_sample.add_argument("--llm", default="gpt-4o")
    p_sample.add_argument("--temperature", type=float, default=0.0)

    p_run = sub.add_parser("run", help="Run on a provided email")
    p_run.add_argument("--sender", required=True)
    p_run.add_argument("--subject", required=True)
    p_run.add_argument("--body", required=True)
    p_run.add_argument("--llm", default="gpt-4o")
    p_run.add_argument("--temperature", type=float, default=0.0)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "run-sample":
        if args.kind == "legitimate":
            email = {
                "sender": "Joker",
                "subject": "Found you Batman ! ",
                "body": (
                    "Mr. Wayne,I found your secret identity ! I know you're batman ! "
                    "Ther's no denying it, I have proof of that and I'm coming to find you soon."
                ),
            }
        else:
            email = {
                "sender": "Crypto bro",
                "subject": "The best investment of 2025",
                "body": "Mr Wayne, I just launched an ALT coin and want you to buy some !",
            }
        run_workflow(email, args.llm, args.temperature)
        return

    if args.command == "run":
        email = {"sender": args.sender, "subject": args.subject, "body": args.body}
        run_workflow(email, args.llm, args.temperature)
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()


