"""LangGraph nodes for the negotiation pipeline.

v2.0: Deterministic heuristic is primary. LLM is safety fallback only.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.state import NegotiationState
from strategy.models import (
    Observation,
    ProposalResponse,
    AcceptResponse,
    parse_observation_from_text,
    determine_action_type,
    QUANTITIES,
)
from strategy.heuristics import aspiration_propose, aspiration_accept_or_reject


async def parse_observation(state: NegotiationState) -> NegotiationState:
    raw = state["raw_message"]
    try:
        obs = parse_observation_from_text(raw)
        action_type = "propose" if obs.is_propose else "accept_or_reject"
        if action_type not in ("propose", "accept_or_reject"):
            action_type = determine_action_type(raw)
        return {
            "observation": obs.model_dump(),
            "action_type": action_type,
            "error": None,
        }
    except Exception as e:
        return {
            "observation": None,
            "action_type": determine_action_type(raw),
            "error": f"parse: {e}",
        }


async def heuristic_decide(state: NegotiationState) -> NegotiationState:
    """Deterministic Nash-welfare decision — no LLM."""
    obs = Observation(**state["observation"])
    action_type = state["action_type"]
    history = state.get("history") or []

    if action_type == "propose":
        p = aspiration_propose(obs, history)
        return {
            "response_json": json.dumps({
                "allocation_self": p.allocation_self,
                "allocation_other": p.allocation_other,
                "reason": p.reason,
            }),
            "error": None,
        }

    a = aspiration_accept_or_reject(obs, history)
    return {
        "response_json": json.dumps({
            "accept": a.accept,
            "reason": a.reason,
        }),
        "error": None,
    }


async def format_response(state: NegotiationState) -> NegotiationState:
    resp_json = state.get("response_json", "")
    if not resp_json:
        return {"error": "empty response"}

    try:
        data = json.loads(resp_json)

        if "allocation_self" in data:
            alloc_self = data["allocation_self"]
            alloc_other = data.get("allocation_other")
            if alloc_other is None:
                alloc_other = [q - s for q, s in zip(QUANTITIES, alloc_self)]
                data["allocation_other"] = alloc_other

            for i, (s, o, q) in enumerate(zip(alloc_self, alloc_other, QUANTITIES)):
                if s < 0 or o < 0 or s + o != q:
                    return {"error": f"invalid allocation at item {i}: {s}+{o}!={q}"}

        return {"response_json": json.dumps(data), "error": None}
    except Exception as e:
        return {"error": f"format: {e}"}


async def llm_fallback(state: NegotiationState) -> NegotiationState:
    """Last-resort fallback when the observation couldn't be parsed.

    Produces a safe default response. We don't actually invoke the LLM here
    to keep costs zero — a sensible static response is enough at this rare
    code path.
    """
    action_type = state.get("action_type", "propose")
    if action_type == "accept_or_reject":
        return {
            "response_json": json.dumps({
                "accept": False,
                "reason": "fallback: unparsable observation",
            }),
            "error": None,
        }
    return {
        "response_json": json.dumps({
            "allocation_self": [4, 2, 1],
            "allocation_other": [3, 2, 0],
            "reason": "fallback: unparsable observation",
        }),
        "error": None,
    }
