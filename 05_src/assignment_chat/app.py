from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from assignment_chat.guardrails import apply_input_guardrails
from assignment_chat.main import get_graph

load_dotenv(".env")
load_dotenv(".secrets")

GRAPH = get_graph()
MAX_CONTEXT_TURNS = 12


def _to_langchain_messages(history: list[dict], latest_user_message: str):
    messages = []
    trimmed = history[-MAX_CONTEXT_TURNS:]

    for msg in trimmed:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=latest_user_message))
    return messages


def assignment_chat(message: str, history: list[dict]) -> str:
    blocked = apply_input_guardrails(message)
    if blocked:
        return blocked

    state = {"messages": _to_langchain_messages(history, message)}
    result = GRAPH.invoke(state)
    return result["messages"][-1].content


chat = gr.ChatInterface(
    fn=assignment_chat,
    title="ByteBuddy: Assignment 2 Chat",
    description=(
        "A simplified multi-service AI assistant with API calls, semantic retrieval, "
        "function-calling tools, memory, and assignment guardrails."
    ),
)

if __name__ == "__main__":
    chat.launch()
