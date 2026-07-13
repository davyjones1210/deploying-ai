from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from assignment_chat.prompts import system_prompt
from assignment_chat.services import TOOLS

load_dotenv(".env")
load_dotenv(".secrets")

GATEWAY_URL = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"


def _has_valid_gateway_key() -> bool:
    key = os.getenv("API_GATEWAY_KEY", "").strip()
    if not key:
        return False
    # Avoid common placeholder values from templates.
    if "<" in key or ">" in key:
        return False
    return True


def _build_model():
    if not _has_valid_gateway_key():
        return None
    return init_chat_model(
        "openai:gpt-4o-mini",
        base_url=GATEWAY_URL,
        api_key="any value",
        default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY", "")},
    )


MODEL = _build_model()


def model_setup_message() -> str:
    return (
        "Assignment chat could not call the model because API_GATEWAY_KEY is missing or invalid. "
        "Set API_GATEWAY_KEY in .secrets, then restart the app."
    )


def _call_model(state: MessagesState):
    if MODEL is None:
        raise RuntimeError(model_setup_message())

    response = MODEL.bind_tools(TOOLS).invoke(
        [SystemMessage(content=system_prompt())] + state["messages"]
    )
    return {"messages": [response]}


def get_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", _call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    return builder.compile()
