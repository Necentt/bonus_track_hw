"""LangGraph negotiation pipeline."""

from langgraph.graph import StateGraph, END

from graph.state import NegotiationState
from graph.nodes import parse_observation, strategize, format_response, fallback


def _route_after_strategize(state: NegotiationState) -> str:
    if state.get("error"):
        return "fallback"
    return "format"


def _route_after_format(state: NegotiationState) -> str:
    if state.get("error"):
        return "fallback"
    return END


def build_negotiation_graph():
    graph = StateGraph(NegotiationState)

    graph.add_node("parse", parse_observation)
    graph.add_node("strategize", strategize)
    graph.add_node("format", format_response)
    graph.add_node("fallback", fallback)

    graph.set_entry_point("parse")

    graph.add_conditional_edges(
        "parse",
        lambda state: "fallback" if state.get("error") and not state.get("observation") else "strategize",
    )
    graph.add_conditional_edges("strategize", _route_after_strategize)
    graph.add_conditional_edges("format", _route_after_format)
    graph.add_edge("fallback", END)

    return graph.compile()
