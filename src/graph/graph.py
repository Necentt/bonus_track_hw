"""LangGraph negotiation pipeline.

v2.0: Deterministic aspiration is the primary strategy. LLM is only a safety
net when the observation cannot be parsed (rare). This keeps our play inside
the Nash equilibrium support and drives MENE Regret → 0.
"""

from langgraph.graph import StateGraph, END

from graph.state import NegotiationState
from graph.nodes import parse_observation, heuristic_decide, format_response, llm_fallback


def build_negotiation_graph():
    graph = StateGraph(NegotiationState)

    graph.add_node("parse", parse_observation)
    graph.add_node("decide", heuristic_decide)
    graph.add_node("format", format_response)
    graph.add_node("llm_fallback", llm_fallback)

    graph.set_entry_point("parse")

    # If we have an observation → deterministic heuristic. Otherwise → LLM fallback.
    graph.add_conditional_edges(
        "parse",
        lambda state: "decide" if state.get("observation") else "llm_fallback",
    )
    # Heuristic is deterministic and always produces valid output → format.
    graph.add_edge("decide", "format")
    graph.add_conditional_edges(
        "format",
        lambda state: "llm_fallback" if state.get("error") else END,
    )
    graph.add_edge("llm_fallback", END)

    return graph.compile()
