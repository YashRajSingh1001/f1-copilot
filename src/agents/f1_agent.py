"""
F1 Copilot agent — LangGraph ReAct loop with conversation memory.

Graph: START → analyst → [tools | END]
                ↑_____________|
"""

from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import ALL_TOOLS
from ..config import get


SYSTEM_PROMPT = """You are the F1 Copilot — an elite Formula 1 performance analyst with deep expertise in:
- Aerodynamics, mechanical grip, and car setup philosophy
- Tire compound behaviour, thermal degradation, and graining
- Driver telemetry: speed traces, braking points, throttle application, gear selection
- Race strategy: pit windows, undercut/overcut, safety car opportunities
- Weather impacts: track evolution, rubber laying, temperature effects on tire choice

TOOL USAGE RULES — always follow these:
1. NEVER answer from memory alone — always call the relevant tools first.
2. For driver comparisons: call BOTH `compare_telemetry` AND `get_sector_times` to pinpoint where time is lost.
3. For strategy questions: call `get_tire_data` for the specific driver(s) asked about.
4. For weather questions: call `get_weather` first, then relate conditions to tire/pace impact.
5. For race overview questions: call `get_race_results` then `search_race_context` for narrative.
6. For lap pace analysis: call `get_lap_times_series` or `compare_race_pace`.
7. Always call `search_race_context` as a supplementary tool to enrich answers with analyst context.

OUTPUT FORMAT:
- Lead with the direct answer (1-2 sentences)
- Then explain the data: cite exact lap times, sector deltas in milliseconds, tire compounds, temperatures
- Identify the root cause: don't just say "Norris was slower", say WHY — corner speed, braking point, tire temp, etc.
- End with a strategic insight or "so what" conclusion"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _history_to_lc(history: list[dict]) -> list[BaseMessage]:
    """Convert UI message dicts to LangChain message objects for memory."""
    result = []
    for msg in history:
        if msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and msg.get("content"):
            result.append(AIMessage(content=msg["content"]))
    return result


def build_agent():
    llm = ChatOpenAI(
        model=get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=get("OPENAI_API_KEY"),
        temperature=0,
        streaming=True,
        model_kwargs={"max_completion_tokens": 2000},
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


def run_query(question: str, history: list[dict] | None = None) -> dict:
    agent = build_agent()
    messages = _history_to_lc(history or []) + [HumanMessage(content=question)]
    result = agent.invoke({"messages": messages})

    final_answer = ""
    tool_calls_made = []

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append({"tool": tc["name"], "args": tc["args"]})
        elif hasattr(msg, "type") and msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
            final_answer = msg.content

    return {"answer": final_answer, "tool_calls": tool_calls_made, "messages": result["messages"]}


def stream_query(question: str, history: list[dict] | None = None):
    """Stream agent events. Yields dicts: type='tool_call'|'answer'|'done'."""
    agent = build_agent()
    messages = _history_to_lc(history or []) + [HumanMessage(content=question)]

    for event in agent.stream({"messages": messages}, stream_mode="values"):
        last_msg = event["messages"][-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                yield {"type": "tool_call", "tool": tc["name"], "args": tc["args"]}

        elif (
            hasattr(last_msg, "type")
            and last_msg.type == "ai"
            and last_msg.content
            and not getattr(last_msg, "tool_calls", None)
        ):
            yield {"type": "answer", "content": last_msg.content}

    yield {"type": "done"}
