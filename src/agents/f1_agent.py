"""
F1 Copilot agent built on LangGraph.

Graph topology:
  START → analyst → [tool_node | END]
             ↑______________|

The analyst node decides which tools to call; tool_node executes them
and loops back for synthesis. On the final turn (no more tool calls)
the agent writes the answer and the graph terminates.
"""

from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import ALL_TOOLS
from ..openai_models import (
    max_output_tokens,
    openai_api_key,
    openai_model,
    reasoning_effort,
    text_verbosity,
)


SYSTEM_PROMPT = """You are the F1 Copilot — an expert Formula 1 performance analyst with deep knowledge of:
- Aerodynamics, mechanical grip, and car setup
- Tire compound behaviour and thermal degradation
- Driver telemetry analysis (speed traces, braking points, throttle application)
- Race strategy, pit window timing, and undercut/overcut analysis
- Weather impacts on track evolution and tire choice

You have access to real-time F1 data tools. When answering questions:
1. ALWAYS call the relevant tools to pull actual data before answering — never guess lap times or positions.
2. Use `compare_telemetry` and `get_sector_times` for driver comparisons.
3. Use `get_tire_data` for strategy and degradation questions.
4. Use `get_weather` when track conditions are relevant.
5. Use `search_race_context` to retrieve analyst commentary and race reports.
6. Synthesize all tool results into a precise, quantified answer — cite actual numbers (seconds, percentages, lap numbers).

When comparing drivers, identify EXACTLY where time is lost: which corners, which sectors, and WHY (braking too late, low minimum speed, understeer, tire deg, etc.)."""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_agent():
    llm = ChatOpenAI(
        model=openai_model(),
        api_key=openai_api_key(),
        streaming=True,
        max_completion_tokens=max_output_tokens(2000),
        reasoning_effort=reasoning_effort(),
        verbosity=text_verbosity(),
        use_responses_api=True,
    ).bind_tools(ALL_TOOLS)

    tool_node = ToolNode(ALL_TOOLS)

    def analyst_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("analyst", analyst_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("analyst")
    graph.add_conditional_edges("analyst", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "analyst")

    return graph.compile()


def run_query(question: str, history: Optional[list] = None) -> dict:
    """
    Run a user question through the F1 agent.
    Returns {"answer": str, "tool_calls": list, "messages": list}
    """
    agent = build_agent()

    messages = list(history or [])
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})

    final_answer = ""
    tool_calls_made = []

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                })
        elif hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            if hasattr(msg, "type") and msg.type == "ai":
                final_answer = msg.content

    return {
        "answer": final_answer,
        "tool_calls": tool_calls_made,
        "messages": result["messages"],
    }


def stream_query(question: str, history: Optional[list] = None):
    """
    Stream the agent's response. Yields dicts with type='token'|'tool_call'|'done'.
    """
    agent = build_agent()

    messages = list(history or [])
    messages.append(HumanMessage(content=question))

    for event in agent.stream({"messages": messages}, stream_mode="values"):
        last_msg = event["messages"][-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                yield {"type": "tool_call", "tool": tc["name"], "args": tc["args"]}

        elif hasattr(last_msg, "content") and last_msg.content:
            if hasattr(last_msg, "type") and last_msg.type == "ai" and not getattr(last_msg, "tool_calls", None):
                yield {"type": "answer", "content": last_msg.content}

    yield {"type": "done"}
