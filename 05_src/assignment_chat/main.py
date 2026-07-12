from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from assignment_chat.prompts import system_prompt
from assignment_chat.services import TOOLS


MODEL = init_chat_model("openai:gpt-4o-mini")


def _call_model(state: MessagesState):
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
